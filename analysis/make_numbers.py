#!/usr/bin/env python3
"""Write every number the article prints, as LaTeX macros, from the stored outputs.

The first submission's numbers were typed by hand, and several did not match
the outputs they came from (see REVISION_LOG.md). Here nothing is typed: this
script reads analysis/output/*.json|csv and writes

    source/generated/numbers.tex      one \\newcommand per number
    source/generated/tab_*.tex        the bodies of the article's tables
    analysis/output/numbers.json      the same values, for the response letter

Macro names are built mechanically, so a reader can find any number's source:

    \\nMean<Measure><Text>          text mean of a measure (per 1,000 words for densities)
    \\nDelta<Measure><TextA><TextB> Cliff's delta, TextA against TextB
    \\nEff<Block><PairType>         mean |delta| over the measures of a block, by type of pair
    \\nVar<Part><Block>             share of variance (per cent)
    \\nAcc...  \\nPlace...  \\nGeo...  \\nTheme...  \\nSeg...

audit_numbers.py fails the build if the text uses a macro this file does not
define, or prints a figure that is not a macro.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import BOOK_META, BOOK_ORDER, CONTENT, FEATURES, FORM, MEASURES, load_json
from paths import GEN, OUT, check

T = {b: BOOK_META[b]["author"] for b in BOOK_ORDER}          # Applebaum, ...
M = {  # measure -> macro stem
    "mean_sent_len": "SentLen", "first_person_density": "FirstPerson",
    "noun_ratio": "NounShare", "verb_ratio": "VerbShare", "adj_ratio": "AdjShare",
    "past_tense_ratio": "PastRef", "present_tense_ratio": "PresentRef",
    "mattr": "Mattr", "yules_k": "YulesK", "diary_marker_density": "Diary",
    "travel_marker_density": "Travel", "historical_marker_density": "Historical",
    "positive_ratio": "Positive", "negative_ratio": "Negative", "subjectivity": "Stance",
    "war_total_density": "WarAll", "war_conflict_density": "WarConflict",
    "war_suffering_density": "WarSuffering",
    "first_person_authorial": "FirstPersonAuthorial", "quoted_share": "Quoted",
    "perfect_share": "Perfect", "war_weapons_density": "WarWeapons",
    "war_military_density": "WarMilitary",
}
PER_K = {"first_person_density", "diary_marker_density", "travel_marker_density",
         "historical_marker_density", "subjectivity", "war_total_density",
         "war_conflict_density", "war_suffering_density", "first_person_authorial",
         "war_weapons_density", "war_military_density"}
PERCENT = {"noun_ratio", "verb_ratio", "adj_ratio", "past_tense_ratio", "present_tense_ratio",
           "positive_ratio", "negative_ratio", "quoted_share", "perfect_share"}
PAIR = {"same context, same period": "SameCtxSamePer",
        "same context, different period": "SameCtxDiffPer",
        "different context, same period": "DiffCtxSamePer",
        "different context, different period": "DiffCtxDiffPer"}
PER = {"P1": "POne", "P2": "PTwo", "P3": "PThree"}
BLK = {"all": "All", "form": "Form", "content": "Content",
       "without war vocabulary": "NoWar", "content without war vocabulary": "ContentNoWar",
       "genre markers": "Genre"}
GROUP = {"War": "War", "Politics and nation": "Politics",
         "People and everyday life": "People", "Place and movement": "Place"}

macros: dict[str, str] = {}
values: dict[str, float | int | str] = {}


def put(name: str, text: str, value=None) -> None:
    if not re.fullmatch(r"n[A-Za-z]+", name):
        raise ValueError(f"bad macro name {name!r}")
    if name in macros and macros[name] != text:
        raise ValueError(f"macro {name} defined twice with different values")
    macros[name] = text
    values[name] = value if value is not None else text


def num(x: float, d: int) -> str:
    """Fixed decimals, a real minus sign, thousands separated by {,}."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    s = f"{abs(x):,.{d}f}".replace(",", "{,}")
    return ("$-$" if x < 0 and float(f"{abs(x):.{d}f}") != 0 else "") + s


def signed(x: float, d: int = 2) -> str:
    if float(f"{abs(x):.{d}f}") == 0:
        return num(0.0, d)
    return ("$+$" if x > 0 else "$-$") + f"{abs(x):.{d}f}"


def pct(x: float, d: int = 0) -> str:
    return num(100 * x, d)


def pval(p: float) -> str:
    if p < 0.001:
        return "$<$\\,0.001"
    return f"{p:.3f}"


def measure_value(f: str, v: float) -> tuple[str, float]:
    if f in PER_K:
        return num(1000 * v, 2 if 1000 * v < 1 else 1), 1000 * v
    if f in PERCENT:
        return pct(v), 100 * v
    if f == "mattr":
        return num(v, 2), v
    if f == "yules_k":
        return num(v, 0), v
    return num(v, 1), v                     # sentence length


# ── corpus ──────────────────────────────────────────────────────────────────
def corpus() -> None:
    u = pd.read_csv(OUT / "units.csv")
    fm = pd.read_csv(OUT / "feature_matrix.csv")
    put("nTexts", str(len(BOOK_ORDER)), len(BOOK_ORDER))
    put("nMeasures", str(len(FEATURES)), len(FEATURES))
    put("nMeasuresForm", str(len(FORM)), len(FORM))
    put("nMeasuresContent", str(len(CONTENT)), len(CONTENT))
    put("nUnitsAll", str(len(u)), len(u))
    put("nUnitsTotal", str(len(fm)), len(fm))
    put("nUnitsShort", str(int(u.short.sum())), int(u.short.sum()))
    put("nWordsTotal", num(int(fm.words.sum()), 0), int(fm.words.sum()))
    put("nUnitsFourTexts", str(int(fm.book.isin(["applebaum", "nicolay", "brumme", "orth"]).sum())))
    for b in BOOK_ORDER:
        s, su = fm[fm.book == b], u[u.book == b]
        put(f"nUnits{T[b]}", str(len(s)), len(s))
        put(f"nWords{T[b]}", num(int(s.words.sum()), 0), int(s.words.sum()))
        uk = s[s.ukraine]          # units in the matrix, as in Table 1
        put(f"nUkraineUnits{T[b]}", str(len(uk)), len(uk))
        put(f"nUkraineWords{T[b]}", num(int(uk.words.sum()), 0), int(uk.words.sum()))
        put(f"nUkraineShare{T[b]}", pct(uk.words.sum() / s.words.sum()),
            float(uk.words.sum() / s.words.sum()))
        put(f"nDivisions{T[b]}", str(int(su.divisions.sum())), int(su.divisions.sum()))
        put(f"nYear{T[b]}", str(BOOK_META[b]["year"]))
        for p in ("P1", "P2", "P3"):
            put(f"nUnits{T[b]}{PER[p]}", str(int((s.period == p).sum())), int((s.period == p).sum()))
    # Ukraine-set words over all texts
    uk = fm[fm.ukraine]
    put("nUkraineWordsTotal", num(int(uk.words.sum()), 0), int(uk.words.sum()))
    put("nUnitWordsMin", num(int(fm.words.min()), 0), int(fm.words.min()))
    put("nUnitWordsMax", num(int(fm.words.max()), 0), int(fm.words.max()))
    put("nUnitWordsMedian", num(int(fm.words.median()), 0), int(fm.words.median()))


# ── descriptive statistics and contrasts ────────────────────────────────────
def stats() -> None:
    s = load_json("stats.json")
    d = s["descriptives"]
    for f, stem in M.items():
        for b in BOOK_ORDER:
            txt, v = measure_value(f, d[f][b]["mean"])
            put(f"nMean{stem}{T[b]}", txt, v)
            lo, _ = measure_value(f, d[f][b]["lo"])
            hi, _ = measure_value(f, d[f][b]["hi"])
            put(f"nMeanLo{stem}{T[b]}", lo)
            put(f"nMeanHi{stem}{T[b]}", hi)
        if f in FEATURES:
            o = d[f]["_omnibus"]
            put(f"nEps{stem}", num(o["epsilon2"], 2), o["epsilon2"])
            put(f"nKWp{stem}", pval(o["p_holm"]), o["p_holm"])
    # ratios quoted in the text
    w = {b: d["war_total_density"][b]["mean"] for b in BOOK_ORDER}
    put("nRatioWarMillerApplebaum", num(w["miller"] / w["applebaum"], 1), w["miller"] / w["applebaum"])
    put("nRatioWarBrummeApplebaum", num(w["brumme"] / w["applebaum"], 1), w["brumme"] / w["applebaum"])
    put("nRatioWarMillerNicolay", num(w["miller"] / w["nicolay"], 1), w["miller"] / w["nicolay"])
    # pairwise
    n_sig = {}
    for key, v in s["pairwise"].items():
        a, b = key.split("|")
        n_sig[key] = sum(v["measures"][f]["p_holm"] < 0.05 for f in FEATURES)
        put(f"nSig{T[a]}{T[b]}", str(n_sig[key]), n_sig[key])
        put(f"nSigForm{T[a]}{T[b]}", str(sum(v["measures"][f]["p_holm"] < 0.05 for f in FORM)))
        put(f"nSigContent{T[a]}{T[b]}", str(sum(v["measures"][f]["p_holm"] < 0.05 for f in CONTENT)))
        for f in FEATURES:
            m = v["measures"][f]
            put(f"nDelta{M[f]}{T[a]}{T[b]}", signed(m["delta"]), m["delta"])
            put(f"nDeltaCI{M[f]}{T[a]}{T[b]}", f"[{signed(m['lo'])}, {signed(m['hi'])}]")
    es = s["effect_summary"]
    for blk in ("form", "content", "all"):
        for t, stem in PAIR.items():
            put(f"nEff{BLK[blk]}{stem}", num(es["blocks"][blk][t], 2), es["blocks"][blk][t])
    for t, stem in PAIR.items():
        put(f"nPairs{stem}", str(es["n_pairs"][t]), es["n_pairs"][t])
    for blk in ("form", "content", "all"):
        v = s["variance"][f"_mean_{blk}"]
        put(f"nVarCtx{BLK[blk]}", pct(v["context"]), v["context"])
        put(f"nVarText{BLK[blk]}", pct(v["text_within_context"]), v["text_within_context"])
        put(f"nVarWithin{BLK[blk]}", pct(v["within_text"]), v["within_text"])
    for f in FEATURES:
        v = s["variance"][f]
        put(f"nVarCtx{M[f]}", pct(v["context"]), v["context"])
        put(f"nVarWithin{M[f]}", pct(v["within_text"]), v["within_text"])
    # inside one author
    wa = s["within_author"]
    for p in ("P1", "P2", "P3"):
        put(f"nMillerN{PER[p]}", str(wa["miller"]["n"][p]))
    for f in FEATURES:
        m = wa["miller"]["measures"][f]
        for p in ("P1", "P2", "P3"):
            txt, v = measure_value(f, m["means"][p])
            put(f"nMiller{M[f]}{PER[p]}", txt, v)
        put(f"nMillerDelta{M[f]}PThreePOne", signed(m["delta_P3_vs_P1"]), m["delta_P3_vs_P1"])
        put(f"nMillerDelta{M[f]}PTwoPOne", signed(m["delta_P2_vs_P1"]), m["delta_P2_vs_P1"])
        put(f"nMillerDelta{M[f]}PThreePTwo", signed(m["delta_P3_vs_P2"]), m["delta_P3_vs_P2"])
        put(f"nMillerKWp{M[f]}", pval(m["p_holm"]), m["p_holm"])
    put("nMillerSigMeasures", str(sum(wa["miller"]["measures"][f]["p_holm"] < 0.05 for f in FEATURES)))
    for p in ("P2", "P3"):
        put(f"nBrummeN{PER[p]}", str(wa["brumme"]["n"][p]))
    for f in FEATURES:
        m = wa["brumme"]["measures"][f]
        for p in ("P2", "P3"):
            txt, v = measure_value(f, m["means"][p])
            put(f"nBrumme{M[f]}{PER[p]}", txt, v)
        put(f"nBrummeDelta{M[f]}", signed(m["delta_P3_vs_P2"]), m["delta_P3_vs_P2"])
        put(f"nBrummeDeltaCI{M[f]}", f"[{signed(m['lo'])}, {signed(m['hi'])}]")
    put("nBrummeSigMeasures", str(sum(wa["brumme"]["measures"][f]["p_holm"] < 0.05 for f in FEATURES)))


# ── classification ──────────────────────────────────────────────────────────
def classify() -> None:
    c = load_json("classify.json")
    put("nMajority", pct(c["majority_share"]), c["majority_share"])
    for blk, r in c["five_texts"].items():
        for model, stem in (("random_forest", "RF"), ("lda", "LDA")):
            x = r[model]
            put(f"nAcc{stem}{BLK[blk]}", pct(x["loo_accuracy"], 1), x["loo_accuracy"])
            put(f"nAccBal{stem}{BLK[blk]}", pct(x["loo_balanced_accuracy"], 1))
            put(f"nCorrect{stem}{BLK[blk]}", str(x["loo_correct"]))
            # misattributed units given to a text of the other national context
            ctx = [BOOK_META[b]["context"] for b in c["labels"]]
            cm = x["confusion"]
            cross = sum(cm[i][j] for i in range(len(cm)) for j in range(len(cm)) if ctx[i] != ctx[j])
            put(f"nCrossCtx{stem}{BLK[blk]}", str(cross), cross)
            put(f"nErrors{stem}{BLK[blk]}", str(x["n"] - x["loo_correct"]))
            put(f"nKfold{stem}{BLK[blk]}", pct(x["kfold_mean"], 1), x["kfold_mean"])
            put(f"nKfoldSD{stem}{BLK[blk]}", pct(x["kfold_sd"], 1), x["kfold_sd"])
            if "perm_p" in x:
                put(f"nPermP{stem}", f"{x['perm_p']:.3f}", x["perm_p"])
                # the smallest p a permutation test of this size can return
                put(f"nPermFloor{stem}", f"{1 / (x['perm_n'] + 1):.3f}")
                put(f"nPermAtFloor{stem}", "yes" if abs(x["perm_p"] - 1 / (x["perm_n"] + 1)) < 1e-9 else "no")
                put(f"nPermN{stem}", num(x["perm_n"], 0), x["perm_n"])
                put(f"nPermNull{stem}", pct(x["perm_null_mean"], 1), x["perm_null_mean"])
                put(f"nPermNullMax{stem}", pct(x["perm_null_max"], 1), x["perm_null_max"])
    f4 = c["four_texts"]
    for model, stem in (("random_forest", "RF"), ("lda", "LDA")):
        put(f"nAccFour{stem}", pct(f4[model]["loo_accuracy"], 1), f4[model]["loo_accuracy"])
    for blk, r in c["context"].items():
        for b, v in r["leave_one_text_out"].items():
            put(f"nCtxHeld{BLK[blk]}{T[b]}", pct(v), v)
    for name, r in c["clustering"].items():
        stem = {"text": "Text", "context": "Context", "period_written": "Period"}[name]
        put(f"nARIKmeans{stem}", num(r["kmeans_ari"], 2), r["kmeans_ari"])
        put(f"nARIWard{stem}", num(r["ward_ari"], 2), r["ward_ari"])
    for blk, r in c["placement"].items():
        for scale, sstem in (("raw", "Raw"), ("within_language", "Lang")):
            for model, mstem in (("random_forest", "RF"), ("lda", "LDA")):
                x = r[scale][model]
                for b in ("applebaum", "nicolay", "brumme", "orth"):
                    put(f"nPlace{mstem}{BLK[blk]}{sstem}{T[b]}", pct(x["share"][b]), x["share"][b])
                put(f"nPlace{mstem}{BLK[blk]}{sstem}American", pct(x["american_share"]),
                    x["american_share"])
                put(f"nPlace{mstem}{BLK[blk]}{sstem}German", pct(1 - x["american_share"]),
                    1 - x["american_share"])
                put(f"nPlaceN", str(x["n"]))
    put("nPCOne", pct(c["pca"]["explained"][0]), c["pca"]["explained"][0])
    put("nPCTwo", pct(c["pca"]["explained"][1]), c["pca"]["explained"][1])


# ── robustness ──────────────────────────────────────────────────────────────
def robustness() -> None:
    r = load_json("robustness.json")
    put("nSegWords", num(r["segment_words"], 0), r["segment_words"])
    put("nSegTotal", str(r["n_segments"]), r["n_segments"])
    put("nSegWordsMin", num(r["segment_words_min"], 0), r["segment_words_min"])
    put("nSegWordsMax", num(r["segment_words_max"], 0), r["segment_words_max"])
    put("nSegWordsMean", num(r["segment_words_mean"], 0), r["segment_words_mean"])
    put("nSegMajority", pct(r["majority_share"]), r["majority_share"])
    put("nSegSentMatch", pct(r["sentiment_sentences_matched"], 1), r["sentiment_sentences_matched"])
    for b in BOOK_ORDER:
        put(f"nSeg{T[b]}", str(r["segments_by_book"][b]))
    for blk, x in r["five_texts"].items():
        for model, stem in (("random_forest", "RF"), ("lda", "LDA")):
            put(f"nSegAcc{stem}{BLK[blk]}", pct(x[model]["accuracy"], 1), x[model]["accuracy"])
            put(f"nSegSD{stem}{BLK[blk]}", pct(x[model]["sd"], 1), x[model]["sd"])
    for lang, lstem in (("en", "English"), ("de", "German")):
        x = r["within_language"][lang]
        put(f"nMajority{lstem}", pct(x["majority_share"]), x["majority_share"])
        for blk in [b for b in BLK if b in x]:
            for model, stem in (("random_forest", "RF"), ("lda", "LDA")):
                put(f"nAcc{lstem}{stem}{BLK[blk]}", pct(x[blk][model]["accuracy"], 1),
                    x[blk][model]["accuracy"])
    for b in BOOK_ORDER:
        sc = r["scope"][b]
        put(f"nSegUkraineN{T[b]}", str(sc["n_ukraine"]))
        for f in FEATURES:
            m = sc["measures"][f]
            txt, v = measure_value(f, m["ukraine"])
            put(f"nUkr{M[f]}{T[b]}", txt, v)
            lo, _ = measure_value(f, m["ukraine_ci"][0]) if m["ukraine_ci"][0] == m["ukraine_ci"][0] else ("n/a", 0)
            hi, _ = measure_value(f, m["ukraine_ci"][1]) if m["ukraine_ci"][1] == m["ukraine_ci"][1] else ("n/a", 0)
            put(f"nUkrCI{M[f]}{T[b]}", f"[{lo}, {hi}]")
    if "scope_effects" in r:
        for blk, per in r["scope_effects"].items():
            for t_, stem in PAIR.items():
                if t_ in per:
                    put(f"nUkrEff{BLK[blk]}{stem}", num(per[t_], 2), per[t_])
    u = r["ukraine_only"]
    put("nSegUkraine", str(u["n"]))
    put("nSegUkraineMajority", pct(u["majority_share"]))
    for blk in ("all", "without war vocabulary"):
        for model, stem in (("random_forest", "RF"), ("lda", "LDA")):
            put(f"nAccUkraine{stem}{BLK[blk]}", pct(u[blk][model]["accuracy"], 1))
    for key, v in r["within_author"].items():
        b, pp = key.split(":")
        a, c = pp.split("_vs_")
        stem = f"{T[b]}{PER[a]}{PER[c]}"
        for p, n in v["n"].items():
            put(f"nSegN{T[b]}{PER[p]}", str(n))
        for f in FEATURES:
            m = v["measures"][f]
            put(f"nSegDelta{M[f]}{stem}", signed(m["delta"]), m["delta"])
            for p, mv in m["mean"].items():
                txt, val = measure_value(f, mv)
                put(f"nSeg{M[f]}{T[b]}{PER[p]}", txt, val)


# ── geography ───────────────────────────────────────────────────────────────
def geography() -> None:
    g = load_json("geography.json")
    put("nGazPlaces", str(g["gazetteer"]["places"]))
    put("nGazUkraine", str(g["gazetteer"]["ua"]))
    put("nGazForeign", str(g["gazetteer"]["foreign"]))
    put("nGazStates", str(g["gazetteer"]["state"]))
    for b in BOOK_ORDER:
        for scope, sstem in (("ukraine", ""), ("volume", "Vol")):
            m = g["books"][b][scope]
            put(f"nGeoFocus{sstem}{T[b]}", pct(m["focus"]), m["focus"])
            put(f"nGeoSpread{sstem}{T[b]}", num(m["regional_spread"], 2), m["regional_spread"])
            put(f"nGeoOut{sstem}{T[b]}", num(m["outward_linkage"], 2), m["outward_linkage"])
            put(f"nGeoOutPct{sstem}{T[b]}", pct(m["outward_linkage"]), m["outward_linkage"])
            put(f"nGeoOutWin{sstem}{T[b]}", num(m["outward_linkage_window"], 2), m["outward_linkage_window"])
            put(f"nGeoUU{sstem}{T[b]}", str(m["pairs_UU"]))
            put(f"nGeoUF{sstem}{T[b]}", str(m["pairs_UF"]))
            put(f"nGeoUA{sstem}{T[b]}", num(m["ua"], 0), m["ua"])
            put(f"nGeoUAPerK{sstem}{T[b]}", num(m["ua_per_1000"], 1), m["ua_per_1000"])
            for k in ("focus", "regional_spread", "outward_linkage"):
                lo, hi = m[f"{k}_ci"]
                stem = {"focus": "Focus", "regional_spread": "Spread", "outward_linkage": "Out"}[k]
                f = (lambda x: pct(x)) if k == "focus" else (lambda x: num(x, 2))
                put(f"nGeo{stem}CI{sstem}{T[b]}", f"[{f(lo)}, {f(hi)}]")
            for r_, share in m["region_share"].items():
                put(f"nGeoRegion{r_}{sstem}{T[b]}", pct(share), share)
            for grp, gstem in (("Russia and Belarus", "Russia"),
                               ("Central and Eastern Europe", "CEE"),
                               ("Western Europe and North America", "West"),
                               ("Elsewhere", "Else")):
                put(f"nGeoTo{gstem}{sstem}{T[b]}", str(m["outward_to"][grp]), m["outward_to"][grp])
                sh = m["outward_to_share"][grp]
                put(f"nGeoToShare{gstem}{sstem}{T[b]}", pct(sh) if sh == sh else "n/a", sh)
        put(f"nGeoCoverage{T[b]}", pct(g["coverage"][b]["share"]), g["coverage"][b]["share"])
    for b, per in g["within_author"].items():
        for p, m in per.items():
            if not p.startswith("P"):
                continue
            put(f"nGeoOut{T[b]}{PER[p]}", num(m["outward_linkage"], 2), m["outward_linkage"])
            put(f"nGeoSpread{T[b]}{PER[p]}", num(m["regional_spread"], 2), m["regional_spread"])
            put(f"nGeoFocus{T[b]}{PER[p]}", pct(m["focus"]), m["focus"])
            put(f"nGeoRegionEast{T[b]}{PER[p]}", pct(m["region_share"]["East"]))
            put(f"nGeoUU{T[b]}{PER[p]}", str(m["pairs_UU"]))
            put(f"nGeoUF{T[b]}{PER[p]}", str(m["pairs_UF"]))
            put(f"nGeoUnits{T[b]}{PER[p]}", str(m["n_units"]))


# ── themes ──────────────────────────────────────────────────────────────────
def topics() -> None:
    t = load_json("topics.json")
    put("nThemeK", str(t["k"]))
    put("nThemePassages", num(t["n_passages"], 0), t["n_passages"])
    put("nThemePassageWords", str(t["passage_words"]))
    put("nThemeARI", num(t["seed_stability_ari"]["mean"], 2), t["seed_stability_ari"]["mean"])
    put("nThemeARIMin", num(t["seed_stability_ari"]["min"], 2), t["seed_stability_ari"]["min"])
    put("nThemeSilhouette", num(t["silhouette"][str(t["k"])], 2), t["silhouette"][str(t["k"])])
    for b in BOOK_ORDER:
        for g, gs in GROUP.items():
            x = t["group_share_by_book"][b][g]
            put(f"nTheme{gs}{T[b]}", pct(x["share"]), x["share"])
            put(f"nThemeCI{gs}{T[b]}", f"[{pct(x['lo'])}, {pct(x['hi'])}]")
            put(f"nThemeSeeds{gs}{T[b]}", f"{pct(x['seeds_min'])}--{pct(x['seeds_max'])}")
    for b, per in t["within_author_groups"].items():
        for p, v in per.items():
            for g, gs in GROUP.items():
                put(f"nTheme{gs}{T[b]}{PER[p]}", pct(v[g]), v[g])
    # combined families, in the reference run and at their lowest over the ten
    # alternative runs; and whether a text's largest family is the same in every run
    seeds = t["seed_shares"]
    groups = list(GROUP)
    for b in BOOK_ORDER:
        ref = {g: t["group_share_by_book"][b][g]["share"] for g in groups}
        top = max(ref, key=ref.get)
        stable = all(max(s[b], key=s[b].get) == top for s in seeds)
        put(f"nThemeTop{T[b]}", GROUP[top].lower())
        put(f"nThemeTopStable{T[b]}", "yes" if stable else "no")
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                name = f"{GROUP[g1]}{GROUP[g2]}"
                both = ref[g1] + ref[g2]
                low = min(s[b][g1] + s[b][g2] for s in seeds)
                put(f"nTheme{name}{T[b]}", pct(both), both)
                put(f"nThemeMin{name}{T[b]}", pct(low), low)


# ── the first version's numbers, where the revision quotes them ───────────────
def first_version() -> None:
    """Read from the February outputs, which are read-only and never rewritten.

    The response letter and the deletion notes quote what the first version
    reported; those figures must come from its own outputs, not from memory.
    """
    from paths import CORPUS
    old = CORPUS.parent / "output"
    if not (old / "stage10_validation.json").exists():
        print("  first-version outputs not found; \\nOld... macros not written")
        return
    v = json.loads((old / "stage10_validation.json").read_text())
    s9 = json.loads((old / "stage9_crosslingual.json").read_text())
    s8 = json.loads((old / "stage8_discourse.json").read_text())
    put("nOldAccRF", pct(v["random_forest"]["loo_accuracy"], 1), v["random_forest"]["loo_accuracy"])
    put("nOldAccLDA", pct(v["lda_classification"]["loo_accuracy"], 1))
    put("nOldN", str(v["n_samples"]))
    put("nOldFeatures", str(v["n_features"]))
    put("nOldARI", num(v["clustering"]["kmeans_ari"], 3))
    fp = s9["statistical_tests"]["first_person_density"]
    put("nOldFirstPersonKWp", f"{fp['kruskal_wallis']['p_value']:.3f}")
    put("nOldFirstPersonMWp", f"{fp['mann_whitney']['p_value']:.3f}")
    w = {b: s8[b]["war_vocabulary"]["war_total_density"] for b in s8}
    put("nOldWarBrumme", num(1000 * w["brumme"], 2))
    put("nOldWarRatioBrummeApplebaum", num(w["brumme"] / w["applebaum"], 1))
    put("nOldWarRatioBrummeEnglish", num(w["brumme"] / ((w["applebaum"] + w["nicolay"]) / 2), 1))


def overlap() -> None:
    """Shared eight-word sequences with the two related texts (overlap.py)."""
    f = OUT / "overlap.json"
    if not f.exists():
        return
    o = json.loads(f.read_text())
    for k, stem in (("companion", "Companion"), ("sibling", "Sibling")):
        put(f"nOverlap{stem}", str(o[k]["shared"]), o[k]["shared"])
        put(f"nOverlapShare{stem}", num(100 * o[k]["share"], 2), o[k]["share"])
    put("nOverlapGrams", num(o["ours"], 0), o["ours"])


# ── tables ──────────────────────────────────────────────────────────────────
def tables() -> None:
    u = pd.read_csv(OUT / "units.csv")
    fm = pd.read_csv(OUT / "feature_matrix.csv")
    per_text = {"P1": "before Euromaidan", "P2": "Euromaidan--23 Feb.\\ 2022",
                "P3": "from 24 Feb.\\ 2022"}
    rows = []
    for b in BOOK_ORDER:
        m = BOOK_META[b]
        s, su = fm[fm.book == b], u[u.book == b]
        narrated = [p for p in ("P1", "P2", "P3") if (s.period == p).any()]
        # A diary is written as it narrates: Brumme's entries before 24 February
        # 2022 were written in P2. The book-level code (P3) is used only to type
        # pairs of texts in stats.py.
        written = "--".join(narrated) if m["mode"] == "wartime diary" else m["written"]
        rows.append(
            f"{m['author']} ({m['year']}) & \\emph{{{m['title']}}} & {m['context']} & "
            f"{m['mode'].capitalize()} & {written} & "
            f"{', '.join(narrated)} & {len(s)} & {num(int(s.words.sum()), 0)} & "
            f"{num(int(s[s.ukraine].words.sum()), 0)} \\\\")
    (GEN / "tab_corpus.tex").write_text("\n".join(rows) + "\n", encoding="utf8")

    rows, last = [], None
    for f, (grp, name, definition) in MEASURES.items():
        g = grp if grp != last else ""
        if grp != last and last is not None:
            rows.append("\\addlinespace")
        last = grp
        unit = " (per 1{,}000 words)" if f in PER_K else ""
        rows.append(f"{g} & {name}{unit} & {definition.capitalize()} \\\\")
    (GEN / "tab_measures.tex").write_text("\n".join(rows) + "\n", encoding="utf8")

    c = load_json("classify.json")
    r = load_json("robustness.json")
    rows = []
    for blk, label in (("all", f"All {len(FEATURES)} measures"), ("form", "Form measures only"),
                       ("content", "Content measures only"),
                       ("without war vocabulary", "All but the three war measures"),
                       ("content without war vocabulary", "Content without the war measures"),
                       ("genre markers", "The three genre markers")):
        unit_rf = c["five_texts"][blk]["random_forest"]["loo_accuracy"]
        unit_lda = c["five_texts"][blk]["lda"]["loo_accuracy"]
        if blk not in r["five_texts"]:
            rows.append(f"{label} & {pct(unit_rf, 1)} & {pct(unit_lda, 1)} & & & \\\\")
            continue
        seg = r["five_texts"][blk]
        en, de = r["within_language"]["en"][blk], r["within_language"]["de"][blk]
        rows.append(f"{label} & {pct(unit_rf, 1)} & {pct(unit_lda, 1)} & "
                    f"{pct(seg['random_forest']['accuracy'], 1)} & "
                    f"{pct(en['random_forest']['accuracy'], 1)} & "
                    f"{pct(de['random_forest']['accuracy'], 1)} \\\\")
    rows.append("\\addlinespace")
    rows.append(f"Chance (largest class) & {pct(c['majority_share'], 1)} & {pct(c['majority_share'], 1)} & "
                f"{pct(r['majority_share'], 1)} & "
                f"{pct(r['within_language']['en']['majority_share'], 1)} & "
                f"{pct(r['within_language']['de']['majority_share'], 1)} \\\\")
    (GEN / "tab_classify.tex").write_text("\n".join(rows) + "\n", encoding="utf8")

    g = load_json("geography.json")
    rows = []
    for b in BOOK_ORDER:
        m = g["books"][b]["ukraine"]
        rows.append(
            f"{BOOK_META[b]['author']} & {num(m['words'], 0)} & {num(m['ua'], 0)} & "
            f"{pct(m['focus'])} & {num(m['regional_spread'], 2)} & "
            + " & ".join(pct(m["region_share"][r_]) for r_ in ("West", "Centre", "South", "East"))
            + f" & {m['pairs_UU']} / {m['pairs_UF']} & {num(m['outward_linkage'], 2)} "
            + (f"{{\\footnotesize [{num(m['outward_linkage_ci'][0], 2)}, "
               f"{num(m['outward_linkage_ci'][1], 2)}]}}"
               if m["outward_linkage_ci"][0] == m["outward_linkage_ci"][0] else "")
            + " \\\\")
    (GEN / "tab_geography.tex").write_text("\n".join(rows) + "\n", encoding="utf8")


def main() -> None:
    check()
    corpus()
    stats()
    classify()
    robustness()
    geography()
    topics()
    first_version()
    overlap()
    tables()
    lines = ["%% GENERATED by analysis/make_numbers.py from analysis/output/ -- do not edit.",
             f"%% {len(macros)} macros."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}\\xspace}}" if False else f"\\newcommand{{\\{k}}}{{{v}}}"
              for k, v in sorted(macros.items())]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf8")
    (OUT / "numbers.json").write_text(json.dumps(values, indent=1, ensure_ascii=False,
                                                 default=float), encoding="utf8")
    print(f"wrote {GEN / 'numbers.tex'} ({len(macros)} macros) and 4 table bodies")


if __name__ == "__main__":
    main()
