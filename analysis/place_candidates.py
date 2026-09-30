#!/usr/bin/env python3
"""List every string a named-entity model takes for a place, with its frequency.

This is the discovery step behind the gazetteer, not a measurement. spaCy's
English and German models read every sentence, and every span they label as a
location is a candidate; so is every capitalised proper-noun string that occurs
at least three times, whatever the model called it, because the small German
model misses many Ukrainian place names. With `--xlmr` the multilingual
`Davlan/xlm-roberta-base-ner-hrl` is run as well (slow on a CPU; it was used for
the German books). The candidates were then classified by hand into
`gazetteer.py`, and all place counts in the article come from matching that
gazetteer against the text (geography.py). The models therefore decide what gets
looked at, never what gets counted.

Reads   WORK/parsed/<unit_id>.json
Writes  WORK/xlm_ner/<unit_id>.json    location spans per sentence (cache, --xlmr only)
        OUT/place_candidates.tsv       string, total, per-book counts, sources
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

from paths import OUT, WORK, check

MODEL = "Davlan/xlm-roberta-base-ner-hrl"
SPACY_PLACE = {"GPE", "LOC"}
BATCH = 32


def norm(s: str) -> str:
    s = re.sub(r"[’'`]s$", "", s.strip(" .,;:!?\"“”»«()[]"))
    return " ".join(s.split())


PROPN_RUN = re.compile(
    r"\b[A-ZÄÖÜŁŚŻŹĆ][\w’'\-]+(?:[ \-][A-ZÄÖÜŁŚŻŹĆ][\w’'\-]+){0,2}\b")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlmr", nargs="*", default=None, metavar="BOOK",
                    help="also run XLM-R NER on these books (no name: all)")
    args = ap.parse_args()
    check()
    ner = None
    if args.xlmr is not None:
        import torch
        from transformers import pipeline
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
        ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple",
                       device=-1)
    cache_dir = WORK / "xlm_ner"
    cache_dir.mkdir(exist_ok=True)

    counts: dict[str, Counter] = defaultdict(Counter)
    models: dict[str, set] = defaultdict(set)
    caps: dict[str, Counter] = defaultdict(Counter)
    files = sorted((WORK / "parsed").glob("*.json"))
    for k, f in enumerate(files, 1):
        parsed = json.loads(f.read_text(encoding="utf8"))
        uid = parsed["unit_id"]
        book = uid.rsplit("_", 1)[0]
        for text, label, _ in parsed["entities"]:
            if label in SPACY_PLACE:
                n = norm(text)
                if len(n) > 1:
                    counts[n][book] += 1
                    models[n].add("spacy")
        for s in parsed["sentences"]:
            # capitalised strings not at the start of a sentence
            for m in PROPN_RUN.finditer(s["text"]):
                if m.start() > 0:
                    caps[norm(m.group(0))][book] += 1
        cache = cache_dir / f"{uid}.json"
        spans = []
        if cache.exists():
            spans = json.loads(cache.read_text(encoding="utf8"))
        elif ner is not None and (not args.xlmr or book in args.xlmr):
            sents = [s["text"][:600] for s in parsed["sentences"]]
            for i in range(0, len(sents), BATCH):
                for j, res in enumerate(ner(sents[i:i + BATCH], batch_size=BATCH)):
                    for e in res:
                        if e["entity_group"] == "LOC" and e["score"] >= 0.5:
                            spans.append([i + j, e["word"]])
            cache.write_text(json.dumps(spans, ensure_ascii=False), encoding="utf8")
            print(f"  [{k}/{len(files)}] {uid}", flush=True)
        for _, word in spans:
            n = norm(word)
            if len(n) > 1:
                counts[n][book] += 1
                models[n].add("xlmr")

    for n, c in caps.items():
        if sum(c.values()) >= 3 and n not in counts:
            counts[n] = c
            models[n].add("capitalised")

    books = ["applebaum", "nicolay", "miller", "brumme", "orth"]
    rows = sorted(counts.items(), key=lambda kv: -sum(kv[1].values()))
    with open(OUT / "place_candidates.tsv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["string", "total"] + books + ["models"])
        for s, c in rows:
            w.writerow([s, sum(c.values())] + [c[b] for b in books]
                       + ["+".join(sorted(models[s]))])
    print(f"wrote {OUT / 'place_candidates.tsv'}: {len(rows)} strings, "
          f"{sum(sum(c.values()) for c in counts.values())} mentions")


if __name__ == "__main__":
    main()
