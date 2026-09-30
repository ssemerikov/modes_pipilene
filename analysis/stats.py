#!/usr/bin/env python3
"""Descriptives, contrasts and the context / period / text comparison.

The five texts are the cases; units are repeated observations within a text.
Nothing here treats 118 units as 118 independent draws from a population of
texts. Intervals are bootstrap intervals over a text's own units and describe
how stable a text's profile is across its chapters. Contrasts between texts are
reported as Cliff's delta with its interval; Mann-Whitney p-values are given for
reference and Holm-adjusted across the 18 measures.

Writes OUT/stats.json
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import kruskal, mannwhitneyu

from common import (BOOK_META, BOOK_ORDER, CONTENT, FEATURES, FORM, boot_ci,
                    boot_delta_ci, cliffs_delta, holm, load_matrix, save_json)

PAIR_TYPES = {
    "same context, same period": "two German texts written during the invasion",
    "same context, different period": "American texts written in different periods",
    "different context, same period": "American and German texts written during the invasion",
    "different context, different period": "American earlier texts against German invasion texts",
}


def pair_type(a: str, b: str) -> str:
    same_ctx = BOOK_META[a]["context"] == BOOK_META[b]["context"]
    same_per = BOOK_META[a]["written"] == BOOK_META[b]["written"]
    return (f"{'same' if same_ctx else 'different'} context, "
            f"{'same' if same_per else 'different'} period")


def describe(df) -> dict:
    out = {}
    for f in FEATURES + ["first_person_authorial", "quoted_share", "perfect_share",
                         "war_weapons_density", "war_military_density"]:
        out[f] = {}
        for b in BOOK_ORDER:
            x = df.loc[df.book == b, f].to_numpy()
            lo, hi = boot_ci(x)
            out[f][b] = dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=1)),
                             median=float(np.median(x)), lo=lo, hi=hi)
        groups = [df.loc[df.book == b, f].to_numpy() for b in BOOK_ORDER]
        h, p = kruskal(*groups)
        n, k = len(df), len(groups)
        out[f]["_omnibus"] = dict(H=float(h), p=float(p),
                                  epsilon2=float((h - k + 1) / (n - k)))
    ps = [out[f]["_omnibus"]["p"] for f in FEATURES]
    for f, adj in zip(FEATURES, holm(ps)):
        out[f]["_omnibus"]["p_holm"] = adj
    return out


def pairwise(df) -> dict:
    out = {}
    for a, b in combinations(BOOK_ORDER, 2):
        key = f"{a}|{b}"
        rows, ps = {}, []
        for f in FEATURES:
            x, y = df.loc[df.book == a, f].to_numpy(), df.loc[df.book == b, f].to_numpy()
            d = cliffs_delta(x, y)
            lo, hi = boot_delta_ci(x, y)
            p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            rows[f] = dict(delta=d, lo=lo, hi=hi, p=p)
            ps.append(p)
        for f, adj in zip(FEATURES, holm(ps)):
            rows[f]["p_holm"] = adj
        out[key] = dict(type=pair_type(a, b), measures=rows)
    return out


def effect_summary(pw: dict) -> dict:
    """Mean |delta| per measure and per block, by type of pair."""
    per_feature = {f: {} for f in FEATURES}
    for t in PAIR_TYPES:
        pairs = [k for k, v in pw.items() if v["type"] == t]
        for f in FEATURES:
            per_feature[f][t] = float(np.mean([abs(pw[k]["measures"][f]["delta"]) for k in pairs]))
    blocks = {}
    for name, cols in (("form", FORM), ("content", CONTENT), ("all", FEATURES)):
        blocks[name] = {t: float(np.mean([per_feature[f][t] for f in cols])) for t in PAIR_TYPES}
    n_pairs = {t: sum(v["type"] == t for v in pw.values()) for t in PAIR_TYPES}
    return dict(per_measure=per_feature, blocks=blocks, n_pairs=n_pairs)


def variance_shares(df) -> dict:
    """Share of unit-level variance lying between contexts, between texts of one
    context, and between units of one text -- each text weighted equally."""
    out = {}
    for f in FEATURES:
        m = {b: df.loc[df.book == b, f].mean() for b in BOOK_ORDER}
        v = {b: df.loc[df.book == b, f].var(ddof=0) for b in BOOK_ORDER}
        ctx = {}
        for c in ("American", "German"):
            ctx[c] = np.mean([m[b] for b in BOOK_ORDER if BOOK_META[b]["context"] == c])
        grand = np.mean(list(m.values()))
        ss_ctx = sum((ctx[BOOK_META[b]["context"]] - grand) ** 2 for b in BOOK_ORDER)
        ss_text = sum((m[b] - ctx[BOOK_META[b]["context"]]) ** 2 for b in BOOK_ORDER)
        ss_within = sum(v.values())
        tot = ss_ctx + ss_text + ss_within
        out[f] = dict(context=float(ss_ctx / tot), text_within_context=float(ss_text / tot),
                      within_text=float(ss_within / tot))
    for name, cols in (("form", FORM), ("content", CONTENT), ("all", FEATURES)):
        out[f"_mean_{name}"] = {k: float(np.mean([out[f][k] for f in cols]))
                                for k in ("context", "text_within_context", "within_text")}
    return out


def within_author(df) -> dict:
    """Period contrasts with author, language and mode held fixed."""
    out = {}
    # Miller: three periods of events inside one book.
    m = df[df.book == "miller"]
    res, ps = {}, []
    for f in FEATURES:
        groups = {p: m.loc[m.period == p, f].to_numpy() for p in ("P1", "P2", "P3")}
        h, p = kruskal(*groups.values())
        res[f] = dict(H=float(h), p=float(p),
                      means={k: float(v.mean()) for k, v in groups.items()},
                      delta_P3_vs_P1=cliffs_delta(groups["P3"], groups["P1"]),
                      delta_P3_vs_P2=cliffs_delta(groups["P3"], groups["P2"]),
                      delta_P2_vs_P1=cliffs_delta(groups["P2"], groups["P1"]))
        ps.append(p)
    for f, adj in zip(FEATURES, holm(ps)):
        res[f]["p_holm"] = adj
    out["miller"] = dict(n={p: int((m.period == p).sum()) for p in ("P1", "P2", "P3")},
                         measures=res)

    # Brumme: diary entries before and from 24 February 2022 (preface excluded).
    b = df[(df.book == "brumme") & df.period.isin(["P2", "P3"])]
    res, ps = {}, []
    for f in FEATURES:
        x, y = b.loc[b.period == "P3", f].to_numpy(), b.loc[b.period == "P2", f].to_numpy()
        p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
        lo, hi = boot_delta_ci(x, y)
        res[f] = dict(delta_P3_vs_P2=cliffs_delta(x, y), lo=lo, hi=hi, p=p,
                      means=dict(P2=float(y.mean()), P3=float(x.mean())))
        ps.append(p)
    for f, adj in zip(FEATURES, holm(ps)):
        res[f]["p_holm"] = adj
    out["brumme"] = dict(n={p: int((b.period == p).sum()) for p in ("P2", "P3")}, measures=res)
    return out


def main() -> None:
    df = load_matrix()
    pw = pairwise(df)
    out = dict(
        n_units={b: int((df.book == b).sum()) for b in BOOK_ORDER},
        n_total=int(len(df)),
        words={b: int(df.loc[df.book == b, "words"].sum()) for b in BOOK_ORDER},
        descriptives=describe(df),
        pairwise=pw,
        pair_types=PAIR_TYPES,
        effect_summary=effect_summary(pw),
        variance=variance_shares(df),
        within_author=within_author(df),
    )
    save_json(out, "stats.json")

    es = out["effect_summary"]["blocks"]
    print("\nmean |Cliff's delta| by type of pair")
    for blk in ("form", "content", "all"):
        print(f"  {blk:8s} " + "  ".join(f"{t}: {es[blk][t]:.2f}" for t in PAIR_TYPES))
    print("\nvariance shares (texts weighted equally)")
    for blk in ("form", "content", "all"):
        v = out["variance"][f"_mean_{blk}"]
        print(f"  {blk:8s} context {v['context']:.2f}  text within context "
              f"{v['text_within_context']:.2f}  within text {v['within_text']:.2f}")


if __name__ == "__main__":
    main()
