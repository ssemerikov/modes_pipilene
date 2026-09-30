#!/usr/bin/env python3
"""Where everything lives. Imported by every script in this directory.

Three trees, kept apart on purpose:

* **Corpus** -- `corpus_root/corpus_expanded/`, read-only here. The books are in
  copyright and are not redistributable.
* **Work** -- `corpus_root/els23_work/`. Every intermediate that still carries the
  books' wording (unit texts, parsed tokens, sentences) goes here and nowhere
  else. It sits beside the corpus, outside the paper folder, because the paper
  folder syncs to Dropbox and is what gets packaged for the journal.
* **Output** -- `analysis/output/` in the paper folder. Counts, labels and
  statistics only; this is what ships as supplementary material.

Adapted from the path module of the companion analysis (same corpus, different
manuscript). One constant per tree, one place to change it.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
OUT = HERE / "output"
SOURCE = PAPER / "source"
GEN = SOURCE / "generated"          # numbers.tex and table bodies, written by make_numbers.py
FIGDIR = SOURCE                     # figures sit beside the .tex files, as fig_*.pdf

# Override with CORPUS_DIR=/some/path when the tree moves again.
CORPUS = Path(os.environ.get(
    "CORPUS_DIR", "/path/to/corpus_root/corpus_expanded"))
RAW = CORPUS / "_raw"
SUPERSEDED = CORPUS / "_superseded"     # February extraction; chapter boundaries only

WORK = Path(os.environ.get("ELS23_WORK", str(CORPUS.parent / "els23_work")))

SEED = 20260930


def check() -> None:
    """Fail loudly and early rather than producing an empty analysis."""
    missing = [str(p) for p in (CORPUS, RAW) if not p.exists()]
    if missing:
        raise SystemExit(
            "corpus not found: " + ", ".join(missing)
            + "\nSet CORPUS_DIR if the tree has moved.")
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    check()
    for name in ("PAPER", "OUT", "CORPUS", "RAW", "SUPERSEDED", "WORK", "GEN"):
        p = globals()[name]
        print(f"{name:11s} {'ok ' if p.exists() else 'MISSING'} {p}")
