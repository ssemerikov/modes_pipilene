#!/usr/bin/env python3
"""Does the result depend on how the books were cut up?

The main analysis takes each book's own divisions as units: chapters, or runs of
diary entries. Those units differ in length between books (a chapter of the
longest book is three times a diary unit), and length alone moves several of the
measures. Here every book is cut again into segments of equal length, the
measures are recomputed, and four questions are asked of the segments:

1. Are the texts still told apart, when all segments of one chapter are kept
   together on one side of every train/test split?
2. Are they still told apart when the war vocabulary is left out, and when only
   the measures of form or only those of content are used?
3. Does a text's profile in its Ukraine-set part differ from its profile over
   the whole volume?
4. Inside one author: the two Ukraine chapters of the touring memoir (2012 and
   2014) and the three periods of the reportage, segment by segment.

Reads   WORK/units.json, WORK/sentiment/<unit_id>.json, WORK/parsed/<unit_id>.json
Writes  OUT/segments.csv, OUT/segments_meta.json, OUT/robustness.json

`python3 robustness.py --reuse` skips re-parsing and reads OUT/segments.csv.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

import features
from classify import lda, rf
from common import (BOOK_ORDER, CONTENT, FEATURES, FORM, boot_ci, cliffs_delta, load_matrix,
                    save_json)
from paths import OUT, SEED, WORK, check

JOBS = int(os.environ.get("JOBS", "5"))
SEGMENT_WORDS = 2000
MIN_TAIL = 1000          # a shorter remainder joins the segment before it
NON_WAR = [f for f in FEATURES if not f.startswith("war_")]
NON_WAR_CONTENT = [f for f in CONTENT if not f.startswith("war_")]
BLOCKS = {"all": FEATURES, "form": FORM, "content": CONTENT, "without war vocabulary": NON_WAR,
          "content without war vocabulary": NON_WAR_CONTENT}


def cut(text: str) -> list[str]:
    """Consecutive segments of about SEGMENT_WORDS words, cut at paragraph breaks."""
    out, cur, n = [], [], 0
    for para in text.split("\n\n"):
        cur.append(para)
        n += len(para.split())
        if n >= SEGMENT_WORDS:
            out.append((n, "\n\n".join(cur)))
            cur, n = [], 0
    if cur:
        if n >= MIN_TAIL or not out:
            out.append((n, "\n\n".join(cur)))
        else:
            m, prev = out[-1]
            out[-1] = (m + n, prev + "\n\n" + "\n\n".join(cur))
    return [t for _, t in out]


def build_segments() -> pd.DataFrame:
    units = json.loads((WORK / "units.json").read_text(encoding="utf8"))
    nlp = {"en": spacy.load("en_core_web_sm"), "de": spacy.load("de_core_news_sm")}
    rows, matched, total = [], 0, 0
    for u in units:
        if u["short"]:
            continue
        # sentence labels of the unit, by sentence text, from the cached run
        parsed = json.loads((WORK / "parsed" / f"{u['unit_id']}.json").read_text(encoding="utf8"))
        labels = json.loads((WORK / "sentiment" / f"{u['unit_id']}.json").read_text())
        scored = [s["text"] for s in parsed["sentences"] if s["n"] >= 3]
        assert len(scored) == len(labels), u["unit_id"]
        lookup = dict(zip(scored, labels))
        for i, text in enumerate(cut(u["text"]), 1):
            seg = dict(u, text=text, unit_id=f"{u['unit_id']}_s{i:02d}")
            row, p = features.analyse(seg, nlp[u["lang"]])
            sent = [lookup.get(s["text"]) for s in p["sentences"] if s["n"] >= 3]
            total += len(sent)
            got = [x for x in sent if x]
            matched += len(got)
            n = max(len(got), 1)
            row.update(segment_id=row.pop("unit_id"), unit_id=u["unit_id"], book=u["book"],
                       lang=u["lang"], context=u["context"], period=u["period"],
                       period_written=u["period_written"], ukraine=u["ukraine"],
                       positive_ratio=got.count("positive") / n,
                       negative_ratio=got.count("negative") / n)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["sentiment_match"] = matched / max(total, 1)
    return df


def grouped_cv(df: pd.DataFrame, cols: list[str], labels: list[str]) -> dict:
    X, y, g = df[cols].to_numpy(float), df.book.astype(str).to_numpy(), df.unit_id.to_numpy()
    out = {}
    for name, make in (("random_forest", rf), ("lda", lda)):
        accs, preds = [], None
        for rep in range(10):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
            pred = cross_val_predict(make(), X, y, cv=cv, groups=g, n_jobs=JOBS)
            accs.append(accuracy_score(y, pred))
            if rep == 0:
                preds = pred
        out[name] = dict(accuracy=float(np.mean(accs)), sd=float(np.std(accs, ddof=1)),
                         balanced_accuracy=float(balanced_accuracy_score(y, preds)),
                         confusion=confusion_matrix(y, preds, labels=labels).tolist(),
                         n=int(len(y)))
    return out


def main() -> None:
    check()
    keep = ["segment_id", "unit_id", "book", "lang", "context", "period", "period_written",
            "ukraine", "n_words"] + FEATURES
    if "--reuse" in sys.argv:
        seg = pd.read_csv(OUT / "segments.csv")
        meta = json.loads((OUT / "segments_meta.json").read_text()) \
            if (OUT / "segments_meta.json").exists() else {}
        match = meta.get("sentiment_match", float("nan"))
    else:
        seg = build_segments()
        match = seg.attrs["sentiment_match"]
        seg[keep].to_csv(OUT / "segments.csv", index=False)
        (OUT / "segments_meta.json").write_text(json.dumps(dict(sentiment_match=match)))
    out = dict(segment_words=SEGMENT_WORDS, n_segments=int(len(seg)),
               segments_by_book={b: int((seg.book == b).sum()) for b in BOOK_ORDER},
               segment_words_mean=float(seg.n_words.mean()),
               segment_words_min=int(seg.n_words.min()), segment_words_max=int(seg.n_words.max()),
               sentiment_sentences_matched=float(match),
               majority_share=float(seg.book.value_counts(normalize=True).max()))

    # 1-2. five texts, chapter-grouped folds
    out["five_texts"] = {blk: grouped_cv(seg, cols, BOOK_ORDER) for blk, cols in BLOCKS.items()}

    # within one language: the design Reviewer #2 asked for
    out["within_language"] = {}
    for lang, books in (("en", ["applebaum", "nicolay", "miller"]), ("de", ["brumme", "orth"])):
        sub = seg[seg.lang == lang]
        res = {blk: grouped_cv(sub, cols, books) for blk, cols in BLOCKS.items()}
        res["majority_share"] = float(sub.book.value_counts(normalize=True).max())
        out["within_language"][lang] = res

    # 3. Ukraine-set text against the whole volume
    out["scope"] = {}
    for b in BOOK_ORDER:
        vol, ukr = seg[seg.book == b], seg[(seg.book == b) & seg.ukraine]
        res = dict(n_volume=int(len(vol)), n_ukraine=int(len(ukr)), measures={})
        for f in FEATURES:
            rest = vol.loc[~vol.ukraine, f].to_numpy()
            res["measures"][f] = dict(
                volume=float(vol[f].mean()), ukraine=float(ukr[f].mean()),
                ukraine_ci=boot_ci(ukr[f].to_numpy()),
                delta_ukraine_vs_rest=cliffs_delta(ukr[f].to_numpy(), rest) if len(rest) else None)
        out["scope"][b] = res
    # the comparison of pairs (stats.effect_summary) repeated on Ukraine-set
    # segments, so that the content result does not rest on the parts of the two
    # journeys that lie outside Ukraine
    from itertools import combinations
    from stats import PAIR_TYPES, pair_type
    ukr = seg[seg.ukraine]
    eff = {}
    for blk, cols in (("form", FORM), ("content", CONTENT),
                      ("content without war vocabulary", NON_WAR_CONTENT)):
        per_type = {t: [] for t in PAIR_TYPES}
        for a, b in combinations(BOOK_ORDER, 2):
            x, y = ukr[ukr.book == a], ukr[ukr.book == b]
            per_type[pair_type(a, b)].append(
                np.mean([abs(cliffs_delta(x[f].to_numpy(), y[f].to_numpy())) for f in cols]))
        eff[blk] = {t: float(np.mean(v)) for t, v in per_type.items() if v}
    out["scope_effects"] = eff
    out["ukraine_only"] = {blk: grouped_cv(ukr, cols, BOOK_ORDER)
                           for blk, cols in (("all", FEATURES), ("without war vocabulary", NON_WAR))}
    out["ukraine_only"]["n"] = int(len(ukr))
    out["ukraine_only"]["majority_share"] = float(ukr.book.value_counts(normalize=True).max())

    # 4. inside one author
    within = {}
    for b, a, c in (("nicolay", "P2", "P1"), ("miller", "P3", "P1"), ("miller", "P3", "P2"),
                    ("miller", "P2", "P1"), ("brumme", "P3", "P2")):
        sub = seg[(seg.book == b) & seg.ukraine]
        x, y = sub[sub.period == a], sub[sub.period == c]
        res = dict(n={a: int(len(x)), c: int(len(y))}, measures={})
        for f in FEATURES:
            res["measures"][f] = dict(mean={a: float(x[f].mean()), c: float(y[f].mean())},
                                      delta=cliffs_delta(x[f].to_numpy(), y[f].to_numpy()))
        within[f"{b}:{a}_vs_{c}"] = res
    out["within_author"] = within
    save_json(out, "robustness.json")

    print(f"{len(seg)} segments; sentiment labels matched for {match:.1%} of sentences")
    for blk in BLOCKS:
        r = out["five_texts"][blk]
        print(f"  five texts, {blk:32s} RF {r['random_forest']['accuracy']:.3f}  "
              f"LDA {r['lda']['accuracy']:.3f}")
    for lang in ("en", "de"):
        for blk in BLOCKS:
            r = out["within_language"][lang][blk]
            print(f"  within {lang}, {blk:32s} RF {r['random_forest']['accuracy']:.3f}  "
                  f"LDA {r['lda']['accuracy']:.3f} (majority {out['within_language'][lang]['majority_share']:.2f})")
    r = out["ukraine_only"]
    print(f"  Ukraine-set only (n={r['n']}, majority {r['majority_share']:.2f}): "
          f"RF {r['all']['random_forest']['accuracy']:.3f}, without war "
          f"{r['without war vocabulary']['random_forest']['accuracy']:.3f}")
    for k, v in within.items():
        m = v["measures"]
        print(f"  {k} {v['n']}: war {m['war_total_density']['delta']:+.2f}  first person "
              f"{m['first_person_density']['delta']:+.2f}  negative {m['negative_ratio']['delta']:+.2f}")


if __name__ == "__main__":
    main()
