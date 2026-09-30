#!/usr/bin/env python3
"""Fail if the manuscript prints a number that does not come from the outputs.

Two checks over source/draft/*.tex:

1. Every \\n<Name> macro the text uses is defined in source/generated/numbers.tex
   (so no number can be a stale macro from an earlier run).
2. In the parts that report or interpret results -- the abstract (front.tex),
   findings.tex, discussion.tex, conclusion.tex, and every table in any part --
   no digit survives once macros,
   references, citations, labels, change tags and an explicit short allowlist
   (years, dates, research-question and period names) are removed.

The same scan runs over the other parts (introduction, theory, methods)
and lists what it finds as notes: design constants such as "500
trees" or "2,000-word segments" are allowed there, results are not, and the
list makes the difference easy to check by eye.

Also scans response_to_reviewers.tex and cover_letter_r1.tex for check 1.

Exit status 1 on any failure, including finding no text to check.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from paths import GEN, SOURCE

STRICT = {"front", "findings", "discussion", "conclusion"}
DRAFT = SOURCE / "draft"

# Removed before looking for digits. Order matters.
STRIP = [
    (r"(?<!\\)%.*", ""),                                      # comments
    (r"\\cmdel\{[^}]*\}\{[^}]*\}\{(?:[^{}]|\{[^{}]*\})*\}", " "),  # deletion notes quote old text
    (r"\\cm(?:text)?\{[^}]*\}\{[^}]*\}", " "),                 # change tags R1.Q1.3
    (r"\\(?:label|ref|eqref|pageref|input|include|includegraphics|cite[a-z]*|companiont?|url|href)"
     r"(?:\[[^\]]*\])*\{[^}]*\}", " "),
    (r"\\begin\{(?:tabular\*?|tabularx|table\*?|figure\*?)\}(?:\{[^}]*\})*(?:\[[^\]]*\])?(?:\{[^}]*\})*", " "),
    (r"\\(?:hspace|vspace|rule|setlength|addlinespace|arraystretch|cmidrule)\*?(?:\([^)]*\))?(?:\[[^\]]*\])?(?:\{[^}]*\})*", " "),
    (r"\\multicolumn\{\d+\}\{[^}]*\}", " "),                  # table layout
    (r"\\cmidrule(?:\([^)]*\))?\{[\d-]+\}", " "),
    (r"[pm]\{[\d.]+(?:cm|mm|em|pt)\}", " "),                  # column widths
    (r"\\n[A-Za-z]+", " "),                                  # number macros
    # The short allowlist of design constants and names, nothing else:
    (r"per\s+1\{,\}000|1\{,\}000\s+words", " "),                 # the unit of every density
    (r"Reviewer~?\\?#\d", " "),                               # Reviewer #2
    (r"Tables?(?:~|\s)+S\d+(?:--S\d+)?", " "),                        # supplementary tables
    (r"\(0[: ]|(?<=\s)1(?=: they| equal shares)", " "),      # endpoints of a scale being defined
    (r"\\[A-Za-z]+\*?", " "),                                # any other command name
    (r"\b(?:1[89]\d\d|20[0-4]\d)(?:s|/\d{4})?\b", " "),      # years, 1990s, 1994/2015
    (r"\b\d{1,2}(?:~|\s)(?:January|February|March|April|May|June|July|August|September|"
     r"October|November|December)\b", " "),                  # 24 February
    (r"\b(?:RQ|P|Q|R|H)\d\b", " "),                           # RQ1, P3, Q4, R2
    (r"\bQ\d\.\d\b", " "),                                    # Q1.3
    (r"\b(?:first|second|third|fourth|fifth)\b", " "),
    (r"\[\s*[\d.]+(?:em|pt|cm|mm|ex)\s*\]", " "),
    (r"\b[\d.]+\\?(?:linewidth|textwidth|columnwidth)\b", " "),
]
DIGIT = re.compile(r"\d[\d.,{}]*")


def clean(text: str) -> str:
    for pat, rep in STRIP:
        text = re.sub(pat, rep, text)
    return text


def tables(text: str) -> list[str]:
    return re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, re.S)


def defined_macros() -> set[str]:
    f = GEN / "numbers.tex"
    if not f.exists():
        sys.exit(f"FAIL: {f} missing; run make_numbers.py")
    return set(re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}", f.read_text(encoding="utf8")))


def main() -> int:
    defined = defined_macros()
    files = sorted(DRAFT.glob("*.tex"))
    files = [f for f in files if f.stem != "changemarks"]
    if not files:
        print(f"FAIL: no .tex files under {DRAFT}")
        return 1
    extra = [SOURCE / "response_to_reviewers.tex", SOURCE / "cover_letter_r1.tex",
             SOURCE / "highlights.tex"]
    fails, notes, n_used = [], [], 0
    for f in files + [e for e in extra if e.exists()]:
        text = f.read_text(encoding="utf8")
        body = re.sub(r"(?<!\\)%.*", "", text)
        used = set(re.findall(r"\\(n[A-Z][A-Za-z]*)", body))
        n_used += len(used)
        for m in sorted(used - defined):
            fails.append(f"{f.name}: macro \\{m} is not defined in generated/numbers.tex")
        for m in re.finditer(r"XXX|TODO|\?\?", body):
            fails.append(f"{f.name}: placeholder '{m.group(0)}' left in the text")
        if f.parent != DRAFT:
            continue
        strict_parts = [body] if f.stem in STRICT else tables(body)
        for part in strict_parts:
            for line in clean(part).splitlines():
                for m in DIGIT.finditer(line):
                    fails.append(f"{f.name}: literal number '{m.group(0)}' in: {line.strip()[:110]}")
        if f.stem not in STRICT:
            rest = body
            for tb in tables(body):
                rest = rest.replace(tb, "")
            for line in clean(rest).splitlines():
                for m in DIGIT.finditer(line):
                    notes.append(f"{f.name}: '{m.group(0)}' in: {line.strip()[:100]}")
    for n in notes:
        print("note ", n)
    for x in fails:
        print("FAIL ", x)
    print(f"{len(files)} draft files, {n_used} macro uses, {len(defined)} macros defined; "
          f"{len(fails)} failures, {len(notes)} notes")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
