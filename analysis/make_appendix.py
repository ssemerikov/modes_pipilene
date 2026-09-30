#!/usr/bin/env python3
"""Tables of the supplementary appendix, from the stored outputs.

Writes source/generated/app_*.tex, read by source/supplementary_appendix.tex.
Like make_numbers.py, nothing here is typed by hand.
"""
from __future__ import annotations

import json

import pandas as pd

import lexicons as L
from common import BOOK_META, BOOK_ORDER, FEATURES, MEASURES, load_json
from make_numbers import PER_K, PERCENT, measure_value, num, pct, signed
from paths import GEN, OUT, check

A = {b: BOOK_META[b]["author"] for b in BOOK_ORDER}
AUX = {"first_person_authorial": "Narrator presence outside quotation",
       "quoted_share": "Share of words inside quotation marks",
       "perfect_share": "Perfect forms, share of verb forms",
       "war_weapons_density": "War vocabulary, weapons",
       "war_military_density": "War vocabulary, military"}


def tex(s: str) -> str:
    return (str(s).replace("\\", "").replace("&", "\\&").replace("%", "\\%")
            .replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")
            .replace("“", "``").replace("”", "''").replace("’", "'").replace("‘", "`"))


def units_table() -> str:
    u = pd.read_csv(OUT / "units.csv")
    rows = []
    for _, r in u.iterrows():
        # six words at most: the locator identifies the chapter, and a full
        # title with its dateline would reproduce the book's table of contents
        words = str(r.label).split()
        label = tex(" ".join(words[:6])) + (" \\ldots" if len(words) > 6 else "")
        per = r.period if isinstance(r.period, str) else "--"
        rows.append(f"{r.unit_id.replace('_', chr(92) + '_')} & {label} & {tex(r.locator)} & {per} & "
                    f"{'yes' if r.ukraine else ''} & {num(int(r.words), 0)} & "
                    f"{'excluded' if r.short else ''} \\\\")
    return "\n".join(rows)


def means_table() -> str:
    s = load_json("stats.json")["descriptives"]
    rows = []
    for f in list(FEATURES) + list(AUX):
        name = MEASURES[f][1] if f in MEASURES else AUX[f]
        unit = " (/1{,}000)" if f in PER_K else (" (\\%)" if f in PERCENT else "")
        cells = []
        for b in BOOK_ORDER:
            m, _ = measure_value(f, s[f][b]["mean"])
            lo, _ = measure_value(f, s[f][b]["lo"])
            hi, _ = measure_value(f, s[f][b]["hi"])
            cells.append(f"{m} {{\\scriptsize[{lo}, {hi}]}}")
        eps = num(s[f]["_omnibus"]["epsilon2"], 2) if "_omnibus" in s[f] else ""
        rows.append(f"{name}{unit} & " + " & ".join(cells) + f" & {eps} \\\\")
        if f == FEATURES[-1]:
            rows.append("\\midrule\n\\multicolumn{7}{@{}l}{\\emph{Auxiliary measures, reported but not "
                        "used by the classifiers}} \\\\")
    return "\n".join(rows)


def pairs_table() -> str:
    pw = load_json("stats.json")["pairwise"]
    keys = list(pw)
    head = " & ".join(f"{A[k.split('|')[0]][:4]}--{A[k.split('|')[1]][:4]}" for k in keys)
    rows = [f"Measure & {head} \\\\", "\\midrule"]
    for f in FEATURES:
        cells = []
        for k in keys:
            m = pw[k]["measures"][f]
            star = "$^{*}$" if m["p_holm"] < 0.05 else ""
            cells.append(f"{signed(m['delta'])}{star}")
        rows.append(f"{MEASURES[f][1]} & " + " & ".join(cells) + " \\\\")
    types = {"same context, same period": "SC/SP", "same context, different period": "SC/DP",
             "different context, same period": "DC/SP", "different context, different period": "DC/DP"}
    rows.append("\\midrule")
    rows.append("Type of pair & " + " & ".join(types[pw[k]["type"]] for k in keys) + " \\\\")
    return "\n".join(rows)


def within_table() -> str:
    s = load_json("stats.json")["within_author"]
    r = load_json("robustness.json")["within_author"]
    rows = []
    for f in FEATURES:
        m = s["miller"]["measures"][f]
        b = s["brumme"]["measures"][f]
        n = r["nicolay:P2_vs_P1"]["measures"][f]
        cells = [measure_value(f, m["means"][p])[0] for p in ("P1", "P2", "P3")]
        cells.append(signed(m["delta_P3_vs_P1"]))
        cells += [measure_value(f, b["means"][p])[0] for p in ("P2", "P3")]
        cells.append(signed(b["delta_P3_vs_P2"]))
        cells += [measure_value(f, n["mean"][p])[0] for p in ("P1", "P2")]
        cells.append(signed(n["delta"]))
        rows.append(f"{MEASURES[f][1]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)


def placement_table() -> str:
    c = load_json("classify.json")
    four = ["applebaum", "nicolay", "brumme", "orth"]
    rows = []
    for blk, name in (("form", "Form"), ("content", "Content"), ("all", "All")):
        for scale, sname in (("raw", "as measured"), ("within_language", "within language")):
            for model, mname in (("random_forest", "RF"), ("lda", "LDA")):
                x = c["placement"][blk][scale][model]
                cells = [f"{x['counts'][b]}" for b in four]
                per = "; ".join(f"{p}: " + "/".join(str(x['by_period'][p][b]) for b in four)
                                for p in ("P1", "P2", "P3"))
                rows.append(f"{name}, {sname} & {mname} & " + " & ".join(cells)
                            + f" & {pct(x['american_share'])} & {{\\scriptsize {per}}} \\\\")
    return "\n".join(rows)


def confusion_table() -> str:
    c = load_json("classify.json")
    x = c["five_texts"]["all"]["random_forest"]["confusion"]
    rows = []
    for b, row in zip(BOOK_ORDER, x):
        rows.append(f"{A[b]} & " + " & ".join(str(v) for v in row) + " \\\\")
    return "\n".join(rows)


def themes_table() -> str:
    t = load_json("topics.json")
    rows = []
    for k in sorted(t["themes"], key=int):
        th = t["themes"][k]
        shares = " & ".join(pct(th["share_by_book"][b]) for b in BOOK_ORDER)
        rows.append(f"{int(k) + 1} & {tex(th['label'])} \\newline {{\\scriptsize {tex(th['group'])}}} & "
                    f"{{\\scriptsize {tex(', '.join(th['terms_en'][:8]))}}} & "
                    f"{{\\scriptsize {tex(', '.join(th['terms_de'][:8]))}}} & {shares} \\\\")
    return "\n".join(rows)


def places_table() -> str:
    g = load_json("geography.json")
    rows = []
    for b in BOOK_ORDER:
        top = ", ".join(f"{tex(n)} {k}" for n, k in g["books"][b]["top_ua"][:8])
        fo = ", ".join(f"{tex(n)} {k}" for n, k in g["books"][b]["top_foreign"][:6])
        rows.append(f"{A[b]} & {{\\scriptsize {top}}} & {{\\scriptsize {fo}}} & "
                    f"{pct(g['coverage'][b]['share'])} \\\\")
    return "\n".join(rows)


def robustness_table() -> str:
    r = load_json("robustness.json")
    rows = []
    blocks = [("all", "All measures"), ("form", "Form"), ("content", "Content"),
              ("without war vocabulary", "All but war vocabulary"),
              ("content without war vocabulary", "Content but war vocabulary")]
    for blk, name in blocks:
        x = r["five_texts"][blk]
        en, de = r["within_language"]["en"][blk], r["within_language"]["de"][blk]
        rows.append(f"{name} & {pct(x['random_forest']['accuracy'], 1)} & {pct(x['lda']['accuracy'], 1)} & "
                    f"{pct(en['random_forest']['accuracy'], 1)} & {pct(en['lda']['accuracy'], 1)} & "
                    f"{pct(de['random_forest']['accuracy'], 1)} & {pct(de['lda']['accuracy'], 1)} \\\\")
    return "\n".join(rows)


def lexicon_table() -> str:
    rows = []
    for name, d, label in (("diary", L.DIARY, "Diary markers"), ("travel", L.TRAVEL, "Travel markers"),
                           ("historical", L.HISTORICAL, "Historical markers"),
                           ("stance", L.STANCE, "Stance markers")):
        rows.append(f"{label} & {{\\scriptsize {tex(', '.join(sorted(d['en'])))}}} & "
                    f"{{\\scriptsize {tex(', '.join(sorted(d['de'])))}}} \\\\")
    for cat in L.WAR_CATEGORIES:
        rows.append(f"War: {cat} & {{\\scriptsize {tex(', '.join(sorted(L.WAR['en'][cat])))}}} & "
                    f"{{\\scriptsize {tex(', '.join(sorted(L.WAR['de'][cat])))}}} \\\\")
    return "\n".join(rows)


def main() -> None:
    check()
    parts = dict(units=units_table(), means=means_table(), pairs=pairs_table(),
                 within=within_table(), placement=placement_table(),
                 confusion=confusion_table(), themes=themes_table(), places=places_table(),
                 robustness=robustness_table(), lexicons=lexicon_table())
    for k, v in parts.items():
        (GEN / f"app_{k}.tex").write_text(
            "%% GENERATED by analysis/make_appendix.py -- do not edit.\n" + v + "\n", encoding="utf8")
    print(f"wrote {len(parts)} appendix tables to {GEN}")


if __name__ == "__main__":
    main()
