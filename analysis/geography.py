#!/usr/bin/env python3
"""Where each text puts Ukraine: which places it names, and which it names together.

Every count comes from matching gazetteer.py against the sentences of a unit.
Three measures, each computed for the whole book and for its Ukraine-set units:

focus             Ukrainian places as a share of all places named
                  (settlements and regions; names of states counted apart)
regional spread   how evenly the Ukrainian mentions fall across four
                  macro-regions (Shannon entropy / ln 4: 0 = one region only,
                  1 = equal shares)
outward linkage   of the sentences that name a Ukrainian place together with
                  another place, the share in which the other place lies
                  outside Ukraine. A "borderland" image links Ukrainian places
                  outward; a "national" image links them to each other.

The first submission inferred the last two from chapter-level co-occurrence of
the first 200 entities of each chapter and a 35-name regional table in which
Chernivtsi was listed as foreign and every unlisted place counted as foreign.

Reads   WORK/parsed/<unit_id>.json, OUT/units.csv
Writes  OUT/geography_units.csv, OUT/geography.json
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

import gazetteer
from common import BOOK_ORDER, cliffs_delta, save_json
from paths import OUT, SEED, WORK

REGIONS = ["West", "Centre", "South", "East"]
WINDOW = 3            # sentences, for the wider co-mention variant

# Where an outward link leads. An outward pair can mean a shared borderland
# history (Lviv with Krakow), a comparison with a European capital (Lviv with
# Venice), the imperial centre or the aggressor (Kyiv with Moscow), or simply
# the writer's own home (Poltava with Berlin); the destination is recorded so
# that the kinds can be told apart.
DEST = {"Russia": "Russia and Belarus", "Belarus": "Russia and Belarus"}
for _a in ("Poland", "Lithuania", "Moldova", "Romania", "Hungary", "Slovakia", "Czechia",
           "Estonia", "Serbia", "Bulgaria", "Croatia", "Slovenia", "Bosnia"):
    DEST[_a] = "Central and Eastern Europe"
for _a in ("Germany", "Austria", "France", "Italy", "United Kingdom", "United States",
           "Netherlands", "Belgium", "Spain", "Switzerland", "Sweden", "Canada"):
    DEST[_a] = "Western Europe and North America"
DEST_GROUPS = ["Russia and Belarus", "Central and Eastern Europe",
               "Western Europe and North America", "Elsewhere"]
_HIST = {"Prussia": "Central and Eastern Europe", "Balkans": "Central and Eastern Europe"}


def dest_group(name: str, area: str) -> str:
    return _HIST.get(name) or DEST.get(area, "Elsewhere")


def scan_unit(parsed: dict, pattern, lookup) -> dict:
    c = Counter()
    places = Counter()
    pair_sent = Counter()
    pair_win = Counter()
    edges = Counter()
    dest = Counter()
    per_sentence = []
    for s in parsed["sentences"]:
        found = [lookup[m.group(1)] for m in pattern.finditer(s["text"])]
        area = {e["name"]: e["area"] for e in found}
        names = set()
        for e in found:
            if e["kind"] == "ua":
                c["ua"] += 1
                c[f"ua_{e['area']}"] += 1
                places[e["name"]] += 1
                names.add((e["name"], "U"))
            elif e["kind"] == "foreign":
                c["foreign"] += 1
                c[f"foreign_{e['area']}"] += 1
                places[e["name"]] += 1
                names.add((e["name"], "F"))
            else:
                c["state_ukraine" if e["name"] == "Ukraine" else "state_other"] += 1
        per_sentence.append(names)
        for (a, ka), (b, kb) in combinations(sorted(names), 2):
            pair_sent["".join(sorted(ka + kb))] += 1
            edges[(a, b)] += 1
            if {ka, kb} == {"U", "F"}:
                f = a if ka == "F" else b
                dest[dest_group(f, area[f])] += 1
    for i in range(len(per_sentence)):
        names = set().union(*per_sentence[i:i + WINDOW])
        first = per_sentence[i]
        for (a, ka), (b, kb) in combinations(sorted(names), 2):
            if (a, ka) in first or (b, kb) in first:        # count a pair where it starts
                pair_win["".join(sorted(ka + kb))] += 1
    return dict(counts=c, places=places, pair_sent=pair_sent, pair_win=pair_win, edges=edges,
                dest=dest)


def measures(rows: pd.DataFrame) -> dict:
    """Pooled measures over a set of unit rows."""
    ua, fo = rows.ua.sum(), rows.foreign.sum()
    reg = np.array([rows[f"ua_{r}"].sum() for r in REGIONS], float)
    p = reg / reg.sum() if reg.sum() else reg
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(REGIONS))) if reg.sum() else float("nan")
    uu, uf, ff = rows.UU.sum(), rows.FU.sum(), rows.FF.sum()
    uuw, ufw = rows.UU_win.sum(), rows.FU_win.sum()
    words = rows.words.sum()
    return dict(
        words=int(words), ua=int(ua), foreign=int(fo),
        focus=float(ua / (ua + fo)) if ua + fo else float("nan"),
        ua_per_1000=float(1000 * ua / words), foreign_per_1000=float(1000 * fo / words),
        region_share={r: float(v) for r, v in zip(REGIONS, p)},
        regional_spread=ent,
        pairs_UU=int(uu), pairs_UF=int(uf), pairs_FF=int(ff),
        outward_linkage=float(uf / (uu + uf)) if uu + uf else float("nan"),
        outward_linkage_window=float(ufw / (uuw + ufw)) if uuw + ufw else float("nan"),
        state_ukraine=int(rows.state_ukraine.sum()), state_other=int(rows.state_other.sum()),
        state_ukraine_share=float(rows.state_ukraine.sum() /
                                  max(rows.state_ukraine.sum() + rows.state_other.sum(), 1)),
        outward_to={g: int(rows[f"UF_{g}"].sum()) for g in DEST_GROUPS},
        outward_to_share={g: float(rows[f"UF_{g}"].sum() / uf) if uf else float("nan")
                          for g in DEST_GROUPS},
    )


def boot(rows: pd.DataFrame, key: str, n: int = 2000) -> tuple[float, float]:
    """Interval for a pooled measure, resampling the text's units."""
    if len(rows) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        sample = rows.iloc[rng.integers(0, len(rows), len(rows))]
        v = measures(sample)[key]
        if v == v:
            vals.append(v)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else (float("nan"),) * 2


def main() -> None:
    pattern, lookup = gazetteer.compile_matcher()
    units = pd.read_csv(OUT / "units.csv")
    rows, places, edges, spacy_hits = [], {}, {}, Counter()
    for _, u in units.iterrows():
        parsed = json.loads((WORK / "parsed" / f"{u.unit_id}.json").read_text(encoding="utf8"))
        r = scan_unit(parsed, pattern, lookup)
        c = r["counts"]
        row = dict(unit_id=u.unit_id, book=u.book, period=u.period, ukraine=bool(u.ukraine),
                   short=bool(u.short), words=int(u.words),
                   ua=c["ua"], foreign=c["foreign"],
                   state_ukraine=c["state_ukraine"], state_other=c["state_other"],
                   UU=r["pair_sent"]["UU"], FU=r["pair_sent"]["FU"], FF=r["pair_sent"]["FF"],
                   UU_win=r["pair_win"]["UU"], FU_win=r["pair_win"]["FU"])
        for reg in REGIONS:
            row[f"ua_{reg}"] = c[f"ua_{reg}"]
        for g in DEST_GROUPS:
            row[f"UF_{g}"] = r["dest"][g]
        rows.append(row)
        places.setdefault(u.book, Counter()).update(r["places"])
        if u.ukraine:
            edges.setdefault(u.book, Counter()).update(r["edges"])
        # coverage diagnostic: how many spans spaCy labelled as places does the gazetteer know?
        for text, label, _ in parsed["entities"]:
            if label in ("GPE", "LOC"):
                spacy_hits[(u.book, bool(pattern.search(text)))] += 1
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "geography_units.csv", index=False)

    out = dict(gazetteer=dict(places=len(gazetteer.entries()),
                              ua=sum(e["kind"] == "ua" for e in gazetteer.entries()),
                              foreign=sum(e["kind"] == "foreign" for e in gazetteer.entries()),
                              state=sum(e["kind"] == "state" for e in gazetteer.entries())),
               books={}, coverage={})
    for b in BOOK_ORDER:
        sub = df[df.book == b]
        res = {}
        for scope, rws in (("volume", sub), ("ukraine", sub[sub.ukraine])):
            m = measures(rws)
            m["n_units"] = int(len(rws))
            for key in ("focus", "regional_spread", "outward_linkage"):
                m[f"{key}_ci"] = boot(rws, key)
            res[scope] = m
        res["distinct_ua_places"] = int(sum(1 for n in places[b]
                                            if lookup_kind(n) == "ua"))
        res["top_ua"] = [(n, k) for n, k in places[b].most_common(200)
                         if lookup_kind(n) == "ua"][:10]
        res["top_foreign"] = [(n, k) for n, k in places[b].most_common(200)
                              if lookup_kind(n) == "foreign"][:10]
        res["edges"] = [dict(a=a, b=c2, w=w, kind=lookup_kind(a)[0] + lookup_kind(c2)[0])
                        for (a, c2), w in edges.get(b, Counter()).most_common(40)]
        out["books"][b] = res
        yes, no = spacy_hits[(b, True)], spacy_hits[(b, False)]
        out["coverage"][b] = dict(model_spans=int(yes + no), in_gazetteer=int(yes),
                                  share=float(yes / max(yes + no, 1)))

    # period contrasts inside one author (Ukraine-set units; pooled counts and unit-level delta)
    within = {}
    for b, periods in (("miller", ["P1", "P2", "P3"]), ("brumme", ["P2", "P3"]),
                       ("nicolay", ["P1", "P2"])):
        sub = df[(df.book == b) & df.ukraine & df.period.isin(periods)]
        within[b] = {p: dict(measures(sub[sub.period == p]), n_units=int((sub.period == p).sum()))
                     for p in periods}
        if b != "nicolay":
            sub = sub[~sub.short].copy()
            sub["ua_rate"] = 1000 * sub.ua / sub.words
            a, c = periods[-1], periods[-2]
            within[b]["delta_ua_rate_last_vs_previous"] = cliffs_delta(
                sub.loc[sub.period == a, "ua_rate"], sub.loc[sub.period == c, "ua_rate"])
    out["within_author"] = within

    # by period written, Ukraine-set units pooled over texts
    save_json(out, "geography.json")
    for b in BOOK_ORDER:
        m = out["books"][b]["ukraine"]
        print(f"{b:10s} Ukraine-set: focus {m['focus']:.3f}  spread {m['regional_spread']:.3f}  "
              f"outward {m['outward_linkage']:.3f} (UU {m['pairs_UU']}, UF {m['pairs_UF']}: "
              + ", ".join(f"{g.split()[0]} {v}" for g, v in m["outward_to"].items()) + ")  "
              f"regions {', '.join(f'{r} {v:.2f}' for r, v in m['region_share'].items())}  "
              f"coverage {out['coverage'][b]['share']:.2f}")


_KIND = None


def lookup_kind(name: str) -> str:
    global _KIND
    if _KIND is None:
        _KIND = {e["name"]: e["kind"] for e in gazetteer.entries()}
    return _KIND[name]


if __name__ == "__main__":
    main()
