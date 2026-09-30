# analysis/ — the revised analysis (release r1)

Rebuilds every number, table and figure of the article from the five books.
Staged by `make_release.py`; regenerate rather than edit.

## What is here

| path | contents |
|---|---|
| `output/units.csv` | the 120 units: book, chapter or entry label (first six words), page or chapter locator, period of the events, Ukraine-set flag, words |
| `output/feature_matrix.csv` | the analysed units (short units excluded) with the eighteen measures and five auxiliary ones |
| `output/stats.json` | text means with bootstrap intervals, pairwise Cliff's delta, Holm-adjusted tests, pair-type summaries, variance shares, within-author contrasts |
| `output/classify.json`, `pca_coords.csv` | classification, permutation tests, clustering, placement of the fifth text |
| `output/robustness.json`, `segments.csv` | the same questions on segments of about 2,000 words, on Ukraine-set text only and without the war vocabulary |
| `output/geography*.{json,csv}` | place counts, focus, regional spread, outward linkage and where outward links lead |
| `output/topics.json`, `topic_*.csv` | the twelve themes, their distinctive words, the four families, shares by text and by period, and the ten-run stability check |
| `output/numbers.json` | every number printed in the article, under the name of its LaTeX macro |
| `output/environment.json` | versions of Python, libraries and models |
| `lexicons.py`, `gazetteer.py` | the word lists and the place gazetteer |

The placement test's readings were fixed before it was run; the article
(Section 3.5) states them and which variants were added afterwards.

## What is withheld, and why

The books are in copyright. Every text-bearing intermediate (unit texts, parsed
sentences, OCR pages, model predictions) is written to a separate work
directory, never here. Before release, every file is checked against the books:
none may share an eight-word sequence with them beyond a few chapter titles.
Reproducing the analysis needs the books, identified by edition in the
article's reference list.

## Reproducing

```bash
export CORPUS_DIR=/path/to/corpus        # plain text of the five books, one directory per book
export ELS23_WORK=/path/to/work          # text-bearing intermediates go here
./run_all.sh                             # about 40 minutes on 8 CPU cores
```

`run_all.sh` runs, in order: `units.py`, `features.py`, `sentiment.py`,
`geography.py`, `topics.py`, `assemble.py`, `stats.py`, `classify.py`,
`robustness.py`, `environment.py`, `make_numbers.py`, `make_appendix.py`,
`figures.py`. `audit_numbers.py` checks that the manuscript prints no number
that is not generated. The two scanned books are recognised page by page with
`ocr_page.sh` (Tesseract).

Requires Python 3.10+, spaCy 3.8 with `en_core_web_sm` and `de_core_news_sm`,
scikit-learn 1.6, pandas, scipy, matplotlib, PyTorch, transformers and
sentence-transformers, and Tesseract 4 with English and German data. Model
revisions are in `output/environment.json`. Two consecutive clean runs gave
identical values for every number (with the model predictions cached after the
first run).

## Contents as staged

- analysis/: 23 programs
- analysis/output/: 17 files
- docs/tables_S1-S8.pdf
