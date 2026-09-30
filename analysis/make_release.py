#!/usr/bin/env python3
"""Stage the public release of the R1 analysis for the public repository.

The article's data statement points to one place only, the public repository;
there is no separate supplementary file. This program stages what goes there
and refuses to stage anything that breaks one of three rules:

1. **No text from the books.** They are in copyright. Nothing staged may share an
   eight-word sequence with any of the five volumes, except the files listed in
   ALLOW_SHARED, each capped at the overlap found when it was checked by hand.
   The PDF of Tables S1--S8 is checked through its extracted text.
2. **No identifying strings** beyond the repository's own address: author names,
   the companion article, the sibling submission and local paths are read from
   a list kept in the work tree (FORBIDDEN_FILE), so that this program, which is
   itself published, does not carry them.
3. **Only programs, generated outputs, the tables and READMEs.**

Layout staged under analysis/release/, mirroring the repository:

    README.md                 what is where
    analysis/                 the R1 programs (run_all.sh runs them in order)
    analysis/output/          every output the article's numbers come from
    docs/tables_S1-S8.pdf     the tables the article cites as S1-S8
    scripts/                  (not touched) the first submission's pipeline

Usage:  python3 make_release.py                 stage and check
        python3 make_release.py --into REPO     also copy into a clone of the repository
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from paths import CORPUS, OUT, SOURCE, WORK

HERE = Path(__file__).resolve().parent
DEST = HERE / "release"
TABLES_PDF = SOURCE / "supplementary_appendix.pdf"
FORBIDDEN_FILE = WORK / "supplement_forbidden.txt"
# Not part of the analysis: overlap.py measures wording shared with the authors'
# related texts, for the cover letter; make_supplement.py was its predecessor.
NOT_RELEASED = {"overlap.py", "make_supplement.py"}
BOOK_FILES = {"applebaum": "applebaum", "nicolay": "nicolay", "miller": "miller",
              "brumme": "brumme", "orth": "orth_digital"}
OUTPUTS = ["units.csv", "features_lexical.csv", "sentiment.csv", "feature_matrix.csv",
           "stats.json", "classify.json", "pca_coords.csv", "robustness.json",
           "segments.csv", "segments_meta.json", "geography.json", "geography_units.csv",
           "topics.json", "topic_units.csv", "topic_passages.csv", "numbers.json",
           "environment.json"]
# Not released: place_candidates.tsv (capitalised strings harvested from the
# books to build the gazetteer -- a diagnostic that carries fragments of text).
ALLOW_SHARED = {
    "analysis/output/units.csv": (2, "two chapter titles, shortened to six words"),
    "analysis/units.py": (12, "chapter titles from the tables of contents, used as unit labels"),
    "docs/tables_S1-S8.txt": (16, "Table S1 lists consecutive chapters: their titles, cut to six words, run together in the extracted text"),
}


def anonymise(s: str) -> str:
    """Local absolute paths become placeholders."""
    s = s.replace(str(CORPUS.parent), "/path/to/corpus_root")
    s = s.replace(str(WORK), "/path/to/corpus_root/work")
    return re.sub(r"corpus_root", "corpus_root", s, flags=re.I)


def stage() -> list[str]:
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "analysis" / "output").mkdir(parents=True)
    (DEST / "docs").mkdir()
    log = []
    code = sorted(list(HERE.glob("*.py")) + [HERE / "run_all.sh"])
    for f in code:
        if f.name in NOT_RELEASED:
            continue
        (DEST / "analysis" / f.name).write_text(anonymise(f.read_text(encoding="utf8")),
                                                  encoding="utf8")
    ocr = WORK / "ocr" / "ocr_page.sh"
    if ocr.exists():
        (DEST / "analysis" / "ocr_page.sh").write_text(anonymise(ocr.read_text()), encoding="utf8")
    for f in ("run_all.sh", "ocr_page.sh"):
        p = DEST / "analysis" / f
        if p.exists():
            p.chmod(0o755)
    log.append(f"analysis/: {len(list((DEST / 'analysis').glob('*.*')))} programs")

    missing = [n for n in OUTPUTS if not (OUT / n).exists()]
    if missing:
        raise SystemExit(f"outputs missing: {missing}; run the analysis first")
    for name in OUTPUTS:
        src, dst = OUT / name, DEST / "analysis" / "output" / name
        if name == "units.csv":
            df = pd.read_csv(src)
            df["label"] = [" ".join(str(x).split()[:6]) for x in df.label]
            df.to_csv(dst, index=False)
        elif name == "topic_passages.csv":
            df = pd.read_csv(src)
            df.drop(columns=[c for c in ("text",) if c in df.columns]).to_csv(dst, index=False)
        else:
            dst.write_text(anonymise(src.read_text(encoding="utf8")), encoding="utf8")
    log.append(f"analysis/output/: {len(OUTPUTS)} files")

    if not TABLES_PDF.exists():
        raise SystemExit(f"{TABLES_PDF} missing: build source/supplementary_appendix.tex")
    shutil.copy2(TABLES_PDF, DEST / "docs" / "tables_S1-S8.pdf")
    log.append("docs/tables_S1-S8.pdf")
    (DEST / "README.md").write_text(readme_top(), encoding="utf8")
    (DEST / "analysis" / "README.md").write_text(readme_analysis(log), encoding="utf8")
    return log


def check() -> None:
    if not FORBIDDEN_FILE.exists():
        raise SystemExit(f"missing {FORBIDDEN_FILE}: one forbidden string per line")
    forbidden = [l.strip().lower() for l in FORBIDDEN_FILE.read_text(encoding="utf8").splitlines()
                 if l.strip() and not l.startswith("#")]
    grams: set[str] = set()
    for f in BOOK_FILES.values():
        src = CORPUS / f / f"{f}.txt"
        if not src.exists():
            raise SystemExit(f"cannot check against {src}")
        w = re.findall(r"\w+", src.read_text(encoding="utf8", errors="replace").lower())
        grams |= {" ".join(w[i:i + 8]) for i in range(len(w) - 7)}
    print(f"  copyright check against {len(grams):,} eight-word sequences of {len(BOOK_FILES)} books")

    texts = {}
    for p in sorted(DEST.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(DEST).as_posix()
        if p.suffix == ".pdf":
            texts[rel[:-4] + ".txt"] = subprocess.run(
                ["pdftotext", "-q", str(p), "-"], capture_output=True, text=True).stdout
        else:
            texts[rel] = p.read_text(encoding="utf8", errors="replace")
    bad, allowed = [], []
    for rel, t in texts.items():
        low = t.lower()
        bad += [f"{rel}: forbidden string no. {i + 1}" for i, n in enumerate(forbidden) if n in low]
        w = re.findall(r"\w+", low)
        shared = {" ".join(w[i:i + 8]) for i in range(len(w) - 7)} & grams
        cap, why = ALLOW_SHARED.get(rel, (0, ""))
        if len(shared) > cap:
            bad.append(f"{rel}: {len(shared)} eight-word sequences from the books, cap {cap}: "
                       f"{sorted(shared)[:2]}")
        elif shared:
            allowed.append(f"{rel}: {len(shared)}/{cap} ({why})")
    for a in allowed:
        print(f"  reviewed overlap  {a}")
    for b in bad:
        print(f"  FAIL {b}")
    if bad:
        raise SystemExit("release refused")
    print(f"  {len(texts)} files checked: no book text, no identifying string")


def copy_into(repo: Path) -> None:
    if not (repo / ".git").exists():
        raise SystemExit(f"{repo} is not a clone of the repository")
    for sub in ("analysis", "docs"):
        if (repo / sub).exists():
            shutil.rmtree(repo / sub)
        shutil.copytree(DEST / sub, repo / sub)
    shutil.copy2(DEST / "README.md", repo / "README.md")
    print(f"  copied into {repo} (scripts/ left as it was)")


def readme_top() -> str:
    return """# Modes of cultural mediation — data and programs

Data and programs for the article *Modes of cultural mediation: How documentary
genres shape the construction of Ukraine in American and German nonfiction*
(manuscript SSHO-D-26-02576, *Social Sciences & Humanities Open*).

| directory | what |
|---|---|
| `analysis/` | **the revised analysis (release `r1`)**: every number, table and figure of the article is computed by these programs; `analysis/README.md` explains how |
| `analysis/output/` | the outputs the article's numbers are read from, including `numbers.json`, which lists every number printed in the article under the name of the LaTeX macro that prints it |
| `docs/tables_S1-S8.pdf` | the tables the article cites as Tables S1–S8 |
| `scripts/` | the pipeline of the first submission (February 2026), kept unchanged as its record; it is superseded by `analysis/` |

The five books analysed are commercially published and in copyright; no file here
carries text from them (see `analysis/README.md`).
"""


def readme_analysis(log: list[str]) -> str:
    n_units = len(pd.read_csv(OUT / "units.csv"))
    return f"""# analysis/ — the revised analysis (release r1)

Rebuilds every number, table and figure of the article from the five books.
Staged by `make_release.py`; regenerate rather than edit.

## What is here

| path | contents |
|---|---|
| `output/units.csv` | the {n_units} units: book, chapter or entry label (first six words), page or chapter locator, period of the events, Ukraine-set flag, words |
| `output/feature_matrix.csv` | the analysed units (short units excluded) with the eighteen measures and five auxiliary ones |
| `output/stats.json` | text means with bootstrap intervals, pairwise Cliff's delta, Holm-adjusted tests, pair-type summaries, variance shares, within-author contrasts |
| `output/classify.json`, `pca_coords.csv` | classification, permutation tests, clustering, placement of the fifth text |
| `output/robustness.json`, `segments.csv` | the same questions on segments of about 2,000 words, on Ukraine-set text only and without the war vocabulary |
| `output/geography*.{{json,csv}}` | place counts, focus, regional spread, outward linkage and where outward links lead |
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

{chr(10).join('- ' + l for l in log)}
"""


def main() -> None:
    log = stage()
    print("\n".join("  " + l for l in log))
    check()
    if "--into" in sys.argv:
        copy_into(Path(sys.argv[sys.argv.index("--into") + 1]))


if __name__ == "__main__":
    main()
