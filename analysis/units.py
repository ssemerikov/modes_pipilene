#!/usr/bin/env python3
"""Build the units of observation for all five books, by one rule.

A unit is the book's own narrative division: a chapter where the book has
chapters, a run of consecutive dated entries where it is a diary. Every unit
holds at least MIN_WORDS words; a shorter division is merged with the next one,
and merging never crosses a period boundary, so each unit is set in one period.

Sources, per book:

* Applebaum, Nicolay -- single-page scans. The embedded text layer of both PDFs
  splits words ("dow ntow n ar t world"), so the pages were re-recognised with
  Tesseract 4.1.1 at 300 dpi (`WORK/ocr/<book>/pNNN.txt`, written by
  `WORK/ocr/ocr_page.sh`). Chapters are cut at the PDF pages on which the
  books' own tables of contents place them.
* Brumme -- the Tesseract (German) text of the repaired corpus; cut at the
  diary's datelines.
* Orth, Miller -- the digital editions of the repaired corpus, cut at their
  chapter markers.

Front and back matter, part-title pages, maps and plates are excluded.

Output
------
WORK/units.json        units with their text (never leaves the work tree)
OUT/units.csv          the same units without text: what was analysed, from where
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter

import corpus
from paths import OUT, WORK, check

MIN_WORDS = 1000

# Periods are defined by the events narrated, not by the language of the book.
PERIODS = {
    "P1": "before Euromaidan (to November 2013)",
    "P2": "Euromaidan to 23 February 2022",
    "P3": "from 24 February 2022",
}

BOOK_META = {
    "applebaum": dict(author="Applebaum", year=1994, lang="en", context="American",
                      mode="historical analysis", written="P1",
                      title="Between East and West"),
    "nicolay": dict(author="Nicolay", year=2016, lang="en", context="American",
                    mode="experiential testimony", written="P2",
                    title="The Humorless Ladies of Border Control"),
    "miller": dict(author="Miller", year=2023, lang="en", context="American",
                   mode="war reportage", written="P3",
                   title="The War Came to Us"),
    "brumme": dict(author="Brumme", year=2022, lang="de", context="German",
                   mode="wartime diary", written="P3",
                   title="Im Schatten des Krieges"),
    "orth": dict(author="Orth", year=2024, lang="de", context="German",
                 mode="travel reportage", written="P3",
                 title="Couchsurfing in der Ukraine"),
}
BOOK_ORDER = list(BOOK_META)

# ── Applebaum: (label, first pdf page, last pdf page, set in Ukraine) ─────────
# Printed page = pdf page - 35. Starts follow the book's table of contents and
# were checked against the heading on each opening page. The 2015 introduction
# (pdf 20-23) is excluded: it was written two decades after the text.
APPLEBAUM = [
    ("Introduction", 24, 35, False),
    ("Prelude", 36, 45, False),
    ("Part One: Germans (essay)", 48, 55, False),
    ("Kaliningrad/Königsberg", 56, 75, False),
    ("Part Two: Poles and Lithuanians (essay)", 82, 93, False),
    ("Vilnius/Wilno", 94, 106, False),
    ("Paberžė", 107, 116, False),
    ("Perloja", 117, 125, False),
    ("Eišiškės", 126, 130, False),
    ("Radun", 131, 134, False),
    ("Hermaniszki", 135, 140, False),
    ("Bieniakonie", 141, 160, False),
    ("Nowogródek", 161, 171, False),
    ("Part Three: Russians, Belarusians, and Ukrainians (essay)", 178, 196, False),
    ("Minsk", 197, 208, False),
    ("Brest", 209, 219, False),
    ("Kobrin", 220, 224, False),
    ("A Memory", 225, 228, False),
    ("L'viv/Lvov/Lwów", 229, 247, True),
    ("Woroniaki", 248, 252, True),
    ("Drohobych", 253, 259, True),
    ("Across the Carpathians", 260, 272, True),
    ("Chernivtsi/Czernowitz", 278, 286, True),
    ("Kamenets Podolsky", 287, 295, True),
    ("Kishinev/Chișinău", 296, 308, False),
    ("Odessa", 309, 319, True),
    ("Epilogue", 320, 331, False),
]

# ── Nicolay: printed page = pdf page - 15. Periods from the book's own
# Itinerary (pp. 369-371): Ukraine in June 2012, Russia and Mongolia June-July
# 2012, the Balkans and Central Europe May 2012 and March-April 2013, Ukraine
# again in July 2014.
NICOLAY = [
    ("Introduction", 16, 22, False, None),
    ("The Humorless Ladies of Border Control (Ukraine)", 26, 48, True, "P1"),
    ("Party for Everybody (Rostov-on-Don to Saint Petersburg)", 49, 74, False, "P1"),
    ("A Real Lenin of Our Time (Moscow)", 75, 84, False, "P1"),
    ("God-Forget-It House (Trans-Siberian)", 85, 104, False, "P1"),
    ("The Knout and the Pierogi (Tomsk to Baikal)", 105, 157, False, "P1"),
    ("The Hall of Sufficient Looking (Trans-Mongolian)", 158, 186, False, "P1"),
    ("Drunk Nihilists Make a Good Audience (Croatia, Slovenia, Serbia)", 190, 234, False, "P1"),
    ("A Fur Coat with Morsels (Hungary, Poland)", 235, 247, False, "P1"),
    ("Poor, but They Have Style (Romania)", 248, 262, False, "P1"),
    ("You Are an Asshole Big Time (Bulgaria)", 263, 296, False, "P1"),
    ("Don't Bring Your Beer in Church (Bucharest to Vienna)", 297, 306, False, "P1"),
    ("Changing the Country, We Apologize (Ukraine After the Flood)", 310, 374, True, "P2"),
]

# ── Miller: period of the events in each numbered chapter, from the dateline in
# its title. Chapters not listed are set from 24 February 2022 (P3).
MILLER_P1 = {1, 2, 3, 4}                                   # 2010-2013
MILLER_P2 = set(range(5, 31)) | {40, 41, 49}               # Nov 2013 - 23 Feb 2022
MILLER_SKIP = re.compile(
    r"^(dedication|contents|author[’']?s note|a brief history of ukraine|"
    r"acknowledge?ments?|index|copyright|title page|also by|about the author)\b", re.I)

# ── Orth (digital edition): chapters 3-17 are the narrative; 1-2 and 18-20 are
# publisher's matter, contents, thanks and plates.
ORTH_CHAPTERS = range(3, 18)
ORTH_LABELS = ["Kyjiw", "Iwano-Frankiwsk", "Drohobytsch", "Charkiw", "Poltawa", "Odesa",
               "Dnipro", "Saporischschja", "Tscherniwzi", "Winnyzja", "Lukaschiwka",
               "Tschernihiw", "Dnipro (2)", "Kostjantyniwka", "Oblast Donezk"]

# ── Brumme ────────────────────────────────────────────────────────────────────
DATELINE = re.compile(
    r"^[ \t]*[A-ZÄÖÜ][\wäöüßıİ’\-]+,\s*(?:Montag|Dienstag|Mittwoch|Donnerstag|"
    r"Freitag|Sonnabend|Samstag|Sonntag),?\s*(\d{1,2}|l)\.\s*([A-Za-zäöü]+)\s*(\d{4})?[ \t]*$",
    re.M)
MONTHS = {"januar": 1, "februar": 2, "märz": 3, "marz": 3, "april": 4, "aprit": 4,
          "mai": 5}
BRUMME_LAST_FOLIO = 108          # the diary ends on printed page 108; ads follow


# ── Cleaning ──────────────────────────────────────────────────────────────────
def _is_prose_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True                      # blank lines carry paragraph breaks
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 2:
        return False                     # folios, rules, OCR specks
    if sum(c.isalpha() or c.isspace() for c in s) / len(s) < 0.6:
        return False                     # map scales, plate debris
    if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        return False                     # running heads, map labels
    return True


def _norm_head(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zäöüß ]", "", line.lower())).strip()


def clean_pages(pages: list[str]) -> str:
    """Scanned pages -> running prose.

    Drops running heads (the first line of a page when the same line, folio
    aside, opens several pages, or when it is a short line carrying a folio),
    lines that are not prose, and the short display lines that precede the
    first line of prose on a page (chapter titles, map labels).
    """
    firsts = Counter()
    split = []
    for p in pages:
        lines = p.split("\n")
        idx = next((i for i, l in enumerate(lines) if l.strip()), None)
        split.append((lines, idx))
        if idx is not None and len(lines[idx].split()) <= 9:
            firsts[_norm_head(lines[idx])] += 1
    heads = {h for h, n in firsts.items() if n >= 4 and h}

    out_pages = []
    for lines, idx in split:
        if idx is not None:
            first = lines[idx].strip()
            if (_norm_head(first) in heads
                    or (len(first.split()) <= 9
                        and (re.match(r"^\W*[\dIlOo]{1,3}\s+\S", first)
                             or re.search(r"\S\s+[\dIlOoJ]{1,3}\W*$", first)))):
                lines = lines[:idx] + lines[idx + 1:]
        lines = [l for l in lines if _is_prose_line(l)]
        # display lines above the first prose line of the page
        k = 0
        while k < len(lines) and len(lines[k].split()) < 6:
            k += 1
        lead = [l for l in lines[:k] if not l.strip()]
        lines = lead + lines[k:]
        out_pages.append("\n".join(lines))
    return tidy("\n".join(out_pages))


def tidy(text: str) -> str:
    """Rejoin hyphenated line breaks and unwrap lines; keep paragraph breaks."""
    text = text.replace("\x0c", "\n")
    text = "\n".join(l for l in text.split("\n") if _is_prose_line(l))
    text = re.sub(r"(\w)[-‐‑]\n\s*(?=[a-zäöüß])", r"\1", text)      # com-\nplained
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", " ", text)                       # paragraph mark
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\s* \s*", "\n\n", text).strip()


def tidy_digital(text: str) -> str:
    """Digital editions set one paragraph per line; keep them as paragraphs."""
    lines = [" ".join(l.split()) for l in text.replace("\x0c", "\n").split("\n")]
    return "\n\n".join(l for l in lines if l)


def drop_heading(text: str, min_words: int) -> str:
    """Drop the display lines that open a division: number, title, dateline.

    Everything before the first paragraph of at least `min_words` words goes.
    A chapter that opens on a one-line paragraph loses that line; the loss is a
    few words per chapter and falls on every book alike.
    """
    paras = text.split("\n\n")
    k = 0
    while k < len(paras) - 1 and len(paras[k].split()) < min_words:
        k += 1
    return "\n\n".join(paras[k:])


# ── Per-book unit builders ────────────────────────────────────────────────────
def ocr_pages(key: str) -> dict[int, str]:
    d = WORK / "ocr" / key
    pages = {}
    for f in sorted(d.glob("p*.txt")):
        pages[int(f.stem[1:])] = f.read_text(encoding="utf8", errors="ignore")
    if not pages:
        raise SystemExit(f"no OCR pages in {d}; run WORK/ocr/ocr_page.sh first")
    return pages


def build_applebaum() -> list[dict]:
    pages = ocr_pages("applebaum")
    units = []
    for label, a, b, ua in APPLEBAUM:
        text = drop_heading(clean_pages([pages[p] for p in range(a, b + 1)]), 15)
        units.append(dict(label=label, text=text, ukraine=ua, period="P1",
                          locator=(f"pp. {a - 35}-{b - 35}" if a > 35
                                   else "front matter (1994 introduction)")))
    return units


def build_nicolay() -> list[dict]:
    pages = ocr_pages("nicolay")
    units = []
    for label, a, b, ua, period in NICOLAY:
        text = drop_heading(clean_pages([pages[p] for p in range(a, b + 1)]), 15)
        units.append(dict(label=label, text=text, ukraine=ua, period=period,
                          locator=f"pp. {a - 15}-{b - 15}"))
    return units


def _chapter_spans(book: corpus.Book) -> list[tuple[int, str, int, int]]:
    """(chapter number, title, start, end) from chapter markers, pages merged."""
    spans = []
    for i, (pos, info) in enumerate(book.anchors):
        if "chapter" not in info:
            continue
        n = int(info["chapter"])
        if spans and spans[-1][0] == n:
            continue
        spans.append([n, info.get("title", ""), pos, None])
    for i, s in enumerate(spans):
        s[3] = spans[i + 1][2] if i + 1 < len(spans) else len(book.text)
    return [tuple(s) for s in spans]


def build_miller() -> list[dict]:
    book = corpus.load("miller")
    units = []
    for n, title, a, b in _chapter_spans(book):
        if MILLER_SKIP.match(title.strip()):
            continue
        m = re.match(r"^(\d{1,2})\s+(.*)$", title)
        if m:
            num, label = int(m.group(1)), f"{m.group(1)} {m.group(2)}"
            period = "P1" if num in MILLER_P1 else "P2" if num in MILLER_P2 else "P3"
        elif title.lower().startswith("prologue"):
            label, period = title, "P3"
        else:
            continue                                    # part titles, epigraphs
        text = drop_heading(tidy_digital(book.text[a:b]), 15)
        if len(text.split()) < 50:
            continue
        units.append(dict(label=label[:80], text=text, ukraine=True, period=period,
                          locator=f"ch. {n}"))
    return units


def build_orth() -> list[dict]:
    book = corpus.load("orth_digital")
    units = []
    for n, title, a, b in _chapter_spans(book):
        if n not in ORTH_CHAPTERS:
            continue
        text = drop_heading(tidy_digital(book.text[a:b]), 15)
        label = ORTH_LABELS[n - 3]
        units.append(dict(label=f"{n - 2} {label}", text=text, ukraine=True,
                          period="P3", locator=f"ch. {n}"))
    return units


def build_brumme() -> list[dict]:
    book = corpus.load("brumme")
    text = book.text
    start = text.find("Vorwort")
    if start < 0:
        raise SystemExit("Brumme: preface heading not found")
    hits = list(DATELINE.finditer(text))
    if len(hits) < 40:
        raise SystemExit(f"Brumme: only {len(hits)} datelines found")
    m = re.search(rf"\n\s*{BRUMME_LAST_FOLIO}\s*\n", text[hits[-1].end():])
    end = hits[-1].end() + (m.start() if m else 0)
    if not m:
        raise SystemExit("Brumme: end of diary not found")

    units = [dict(label="Vorwort", text=tidy(text[start + len("Vorwort"):hits[0].start()]),
                  ukraine=True, period=None, locator="preface")]
    for i, h in enumerate(hits):
        nxt = hits[i + 1].start() if i + 1 < len(hits) else end
        day = 1 if h.group(1) == "l" else int(h.group(1))
        month = MONTHS.get(h.group(2).lower())
        if month is None:
            raise SystemExit(f"Brumme: unreadable month in dateline {h.group(0)!r}")
        period = "P2" if (month, day) < (2, 24) else "P3"
        units.append(dict(label=f"2022-{month:02d}-{day:02d}", text=tidy(text[h.end():nxt]),
                          ukraine=True, period=period,
                          locator=book.locator_at(h.start())))
    return units


BUILDERS = dict(applebaum=build_applebaum, nicolay=build_nicolay, miller=build_miller,
                brumme=build_brumme, orth=build_orth)


# ── The merging rule ──────────────────────────────────────────────────────────
def merge_short(units: list[dict]) -> list[dict]:
    """Merge divisions under MIN_WORDS forward, within runs of one period.

    A run is a maximal sequence of consecutive divisions set in the same period.
    Inside a run, divisions accumulate until the unit reaches MIN_WORDS; a short
    remainder joins the unit before it. A run that is short as a whole stays as
    one unit and is flagged `short`; it is kept in the record and left out of the
    feature matrix.
    """
    runs, cur = [], []
    for u in units:
        if cur and u["period"] != cur[-1]["period"]:
            runs.append(cur)
            cur = []
        cur.append(u)
    if cur:
        runs.append(cur)

    merged = []
    for run in runs:
        out, acc = [], []
        for u in run:
            acc.append(u)
            if sum(len(x["text"].split()) for x in acc) >= MIN_WORDS:
                out.append(acc)
                acc = []
        if acc:
            if out:
                out[-1].extend(acc)
            else:
                out.append(acc)
        for group in out:
            text = "\n\n".join(g["text"] for g in group)
            label = (group[0]["label"] if len(group) == 1
                     else f"{group[0]['label']} .. {group[-1]['label']}")
            merged.append(dict(label=label, text=text, ukraine=all(g["ukraine"] for g in group),
                               period=group[0]["period"],
                               locator=(group[0]["locator"] if len(group) == 1
                                        else f"{group[0]['locator']} .. {group[-1]['locator']}"),
                               divisions=len(group),
                               short=len(text.split()) < MIN_WORDS))
    return merged


def main() -> None:
    check()
    all_units = []
    for key in BOOK_ORDER:
        raw_units = BUILDERS[key]()
        units = merge_short(raw_units)
        meta = BOOK_META[key]
        for i, u in enumerate(units, 1):
            u.update(book=key, unit_id=f"{key}_{i:02d}", lang=meta["lang"],
                     context=meta["context"], mode=meta["mode"],
                     period_written=meta["written"], words=len(u["text"].split()))
        all_units += units
        kept = [u for u in units if not u["short"]]
        print(f"{key:10s} divisions={len(raw_units):3d} units={len(units):3d} "
              f"in matrix={len(kept):3d} words={sum(u['words'] for u in units):7,d} "
              f"Ukraine-set words={sum(u['words'] for u in units if u['ukraine']):7,d} "
              f"min/median/max={min(u['words'] for u in kept)}/"
              f"{sorted(u['words'] for u in kept)[len(kept) // 2]}/{max(u['words'] for u in kept)}")

    (WORK / "units.json").write_text(json.dumps(all_units, ensure_ascii=False), encoding="utf8")
    cols = ["unit_id", "book", "lang", "context", "mode", "period_written", "period",
            "ukraine", "short", "divisions", "words", "locator", "label"]
    with open(OUT / "units.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for u in all_units:
            w.writerow(u)
    print(f"\nwrote {WORK / 'units.json'} and {OUT / 'units.csv'}: {len(all_units)} units, "
          f"{sum(not u['short'] for u in all_units)} in the matrix")


if __name__ == "__main__":
    main()
