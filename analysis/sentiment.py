#!/usr/bin/env python3
"""Sentence-level sentiment with one multilingual model for both languages.

The first submission ran an English-only model over the German books and said
so as a limitation. `cardiffnlp/twitter-xlm-roberta-base-sentiment` was trained
on eight languages including English and German, so both are now scored by the
same instrument.

Reads   WORK/parsed/<unit_id>.json
Writes  OUT/sentiment.csv          share of positive / neutral / negative sentences per unit
        WORK/sentiment/<unit_id>.json   per-sentence labels (work tree)
"""
from __future__ import annotations

import csv
import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paths import OUT, WORK, check

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MIN_WORDS = 3            # shorter fragments are headings and OCR debris, not sentences
BATCH = 48
MAX_TOKENS = 128


def main() -> None:
    check()
    torch.manual_seed(0)
    torch.set_num_threads(int(__import__("os").environ.get("TORCH_THREADS", "4")))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    assert set(labels) == {"negative", "neutral", "positive"}, labels

    outdir = WORK / "sentiment"
    outdir.mkdir(exist_ok=True)
    rows = []
    files = sorted((WORK / "parsed").glob("*.json"))
    if not files:
        raise SystemExit("no parsed units; run features.py first")
    for k, f in enumerate(files, 1):
        parsed = json.loads(f.read_text(encoding="utf8"))
        uid = parsed["unit_id"]
        cache = outdir / f"{uid}.json"
        sents = [s["text"] for s in parsed["sentences"] if s["n"] >= MIN_WORDS]
        if cache.exists():
            pred = json.loads(cache.read_text())
            if len(pred) != len(sents):
                pred = None
        else:
            pred = None
        if pred is None:
            pred = []
            # length-sorted batches: far less padding on a CPU
            order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
            out = [None] * len(sents)
            with torch.no_grad():
                for i in range(0, len(order), BATCH):
                    idx = order[i:i + BATCH]
                    enc = tok([sents[j] for j in idx], return_tensors="pt", padding=True,
                              truncation=True, max_length=MAX_TOKENS)
                    arg = model(**enc).logits.argmax(-1).tolist()
                    for j, a in zip(idx, arg):
                        out[j] = labels[a]
            pred = out
            cache.write_text(json.dumps(pred))
        n = max(len(pred), 1)
        rows.append(dict(unit_id=uid, n_scored=len(pred),
                         positive_ratio=pred.count("positive") / n,
                         negative_ratio=pred.count("negative") / n,
                         neutral_ratio=pred.count("neutral") / n))
        print(f"  [{k}/{len(files)}] {uid:14s} {len(pred):5d} sentences "
              f"pos={rows[-1]['positive_ratio']:.3f} neg={rows[-1]['negative_ratio']:.3f}",
              flush=True)

    with open(OUT / "sentiment.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT / 'sentiment.csv'} ({len(rows)} units)")


if __name__ == "__main__":
    sys.exit(main())
