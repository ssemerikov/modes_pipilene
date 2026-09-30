#!/usr/bin/env python3
"""Load the repaired corpus with locators, a quotation mask and a furniture mask.

Adapted from the loader written for the companion analysis of the same corpus
(clause-level coding); the masks and the marker parsing are unchanged, the
clause-specific helpers are dropped.

Scripts read `_raw/<key>.pages.txt`, not the flat file, so every unit keeps the
printed page (or chapter) it came from.

Two masks matter for chapter-level features:

**Furniture.** Running heads, chapter datelines and repeated chapter titles are
set on the page but are not prose, and the scans splice them into whatever
sentence spans the page break. Left in, a running head repeated 140 times lowers
vocabulary diversity and adds capitalised pseudo-places to the toponym counts.

**Quotation.** First-person pronouns inside quoted speech are other people's,
not the narrator's. The narrator-presence measure is therefore reported both
over the whole text and over the text outside quotation marks.

Quotation conventions differ per book and were read off the texts, not assumed:
Applebaum single British quotes, Nicolay straight doubles, Miller typographic
doubles, Brumme and Orth German guillemets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from paths import RAW

# key -> language, quotation style
BOOKS: dict[str, dict] = {
    "applebaum":    dict(lang="en", quotes="single"),
    "nicolay":      dict(lang="en", quotes="double"),
    "miller":       dict(lang="en", quotes="double"),
    "brumme":       dict(lang="de", quotes="guillemet"),
    "orth":         dict(lang="de", quotes="guillemet"),
    "orth_digital": dict(lang="de", quotes="guillemet"),
}

MARKER = re.compile(r"^\[\[([^\]]*)\]\]\s*$", re.M)
# A quote run longer than this is a parse failure (an unbalanced mark), not a
# quotation; treat it as authorial rather than swallowing half a chapter.
MAX_QUOTE_CHARS = 1500


def _parse_marker(body: str) -> dict:
    out = {}
    for m in re.finditer(r"(\w+)=([^\s\]]+)", body):
        out[m.group(1)] = m.group(2)
    # title= runs to the end of the marker and contains spaces, so the generic
    # key=value pattern above truncates it at the first word.
    m = re.search(r"\btitle=(.*)$", body)
    if m:
        out["title"] = m.group(1).strip()
    m = re.search(r"\bp\.(\S+)", body)
    if m:
        out["page"] = m.group(1)
        if "title" in out:
            out["title"] = re.sub(r"\s*p\.\S+$", "", out["title"]).strip()
    return out


def _quote_spans(text: str, style: str) -> list[tuple[int, int]]:
    """Character spans of quoted material, [start, end)."""
    if style == "guillemet":
        # Toggle on either guillemet: the scans confuse the two glyphs, so
        # direction cannot be trusted, but alternation can.
        marks = [m.start() for m in re.finditer(r"[»«]", text)]
    elif style == "double":
        marks = [m.start() for m in re.finditer(r'["“”]', text)]
    elif style == "single":
        # Single quotes collide with apostrophes. Only a mark that opens like a
        # quote can open one, and only a mark that closes like one can close it.
        marks = []
        for m in re.finditer(r"['‘’]", text):
            i = m.start()
            before = text[i - 1] if i else " "
            after = text[i + 1] if i + 1 < len(text) else " "
            opens = (not before.isalnum()) and after.isalnum()
            closes = (not before.isspace() and before not in "([{—-"
                      and not after.isalnum())
            if opens or closes:
                marks.append(i)
    else:
        return []

    spans, open_at = [], None
    for i in marks:
        if open_at is None:
            open_at = i
        elif i - open_at <= MAX_QUOTE_CHARS:
            spans.append((open_at, i + 1))
            open_at = None
        else:
            open_at = i          # the previous mark never closed; restart here
    return spans


# ── Page furniture ───────────────────────────────────────────────────────────
# A running head is an all-caps string that repeats. Legitimate display capitals
# do not occur 20 times; a book title in the running head occurs on every page.
CAPS_RUN = re.compile(r"(?:\b[A-ZÄÖÜ][A-ZÄÖÜ'’\-]{1,}\b[ ,]+){1,7}\b[A-ZÄÖÜ][A-ZÄÖÜ'’\-]{1,}\b")
MIN_HEAD_REPEATS = 5

DATELINES = [
    # Brumme's diary: place, weekday, date. The OCR reads "1." as "l." once.
    re.compile(r"\b[A-ZÄÖÜ][\wäöüß’\-]+,\s*(?:Montag|Dienstag|Mittwoch|"
               r"Donnerstag|Freitag|Sonnabend|Samstag|Sonntag),?\s*"
               r"(?:\d{1,2}|l)\.\s*[A-Za-zäöü]*\s*\d{0,4}"),
    # Miller's chapter datelines: place, oblast, date.
    re.compile(r"\b[A-Z][\w’\-]+,\s*(?:[\w’\-]+\s+)?(?:Oblast|oblast|"
               r"Ukraine|Crimea|Russia)[^.\n]{0,40}?\d{1,2}\s+"
               r"(?:January|February|March|April|May|June|July|August|"
               r"September|October|November|December)[^.\n]{0,20}\d{4}"),
]

# Mixed-case running heads (Nicolay's book title, set ~140 times and spliced
# mid-sentence): repeated title-like word runs that recur far more often than
# any phrase in running prose does.
HEAD_NGRAM_LEN = 5
HEAD_NGRAM_REPEATS = 10
HEAD_CAP_SHARE = 0.6
WORD = re.compile(r"\S+")


def _title_ngram_spans(text: str) -> list[tuple[int, int]]:
    toks = [(m.start(), m.end(), m.group(0)) for m in WORD.finditer(text)]
    counts: dict[str, int] = {}
    for i in range(len(toks) - HEAD_NGRAM_LEN):
        run = toks[i:i + HEAD_NGRAM_LEN]
        words = [w for _, _, w in run]
        caps = sum(1 for w in words if w[:1].isupper())
        if caps / len(words) < HEAD_CAP_SHARE:
            continue
        key = " ".join(words).lower()
        counts[key] = counts.get(key, 0) + 1
    heads = {k for k, n in counts.items() if n >= HEAD_NGRAM_REPEATS}
    if not heads:
        return []
    out = []
    for i in range(len(toks) - HEAD_NGRAM_LEN):
        run = toks[i:i + HEAD_NGRAM_LEN]
        key = " ".join(w for _, _, w in run).lower()
        if key in heads:
            out.append((run[0][0], run[-1][1]))
    return out


def _furniture_spans(text: str, anchors) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = list(_title_ngram_spans(text))

    counts: dict[str, int] = {}
    for m in CAPS_RUN.finditer(text):
        counts[m.group(0).strip()] = counts.get(m.group(0).strip(), 0) + 1
    heads = {h for h, n in counts.items() if n >= MIN_HEAD_REPEATS}
    for m in CAPS_RUN.finditer(text):
        if m.group(0).strip() in heads:
            a, b = m.start(), m.end()
            # swallow an adjacent folio: "154 RUSSIANS, ..." or "... WEST 154"
            pre = re.search(r"\d{1,4}\s*$", text[max(0, a - 6):a])
            if pre:
                a -= len(pre.group(0))
            post = re.match(r"\s*\d{1,4}\b", text[b:b + 6])
            if post:
                b += len(post.group(0))
            spans.append((a, b))

    for rx in DATELINES:
        spans += [(m.start(), m.end()) for m in rx.finditer(text)]

    # A chapter heading repeated as the first words of the chapter body.
    for at, info in anchors:
        title = info.get("title", "")
        if len(title) < 8:
            continue
        head = " ".join(title.split())[:60]
        window = text[at:at + len(head) + 40]
        norm = " ".join(window.split())
        if norm.startswith(head[:30]):
            spans.append((at, at + len(head) + 10))

    merged: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


@dataclass
class Book:
    key: str
    lang: str = ""
    text: str = ""
    anchors: list[tuple[int, dict]] = field(default_factory=list)
    quoted: bytearray = field(default_factory=bytearray)
    furniture: bytearray = field(default_factory=bytearray)

    @property
    def words(self) -> int:
        return len(self.text.split())

    def locator_at(self, i: int) -> str:
        """'p. 152' or 'ch. 7' for a character offset."""
        lo, hi = 0, len(self.anchors) - 1
        if hi < 0:
            return "?"
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.anchors[mid][0] <= i:
                lo = mid
            else:
                hi = mid - 1
        info = self.anchors[lo][1]
        if "page" in info:
            return f"p. {info['page']}"
        if "chapter" in info:
            return f"ch. {info['chapter']}"
        return "?"


@lru_cache(maxsize=None)
def load(key: str) -> Book:
    path = RAW / f"{key}.pages.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf8", errors="ignore")

    # Strip marker lines, remembering where each one applied.
    out, anchors, pos, last = [], [], 0, 0
    for m in MARKER.finditer(raw):
        out.append(raw[last:m.start()])
        pos += m.start() - last
        anchors.append((pos, _parse_marker(m.group(1))))
        last = m.end()
    out.append(raw[last:])
    text = "".join(out)

    meta = BOOKS.get(key, dict(lang="en", quotes="double"))
    quoted = bytearray(len(text))
    for a, b in _quote_spans(text, meta["quotes"]):
        for j in range(a, min(b, len(text))):
            quoted[j] = 1
    furniture = bytearray(len(text))
    for a, b in _furniture_spans(text, anchors):
        for j in range(max(0, a), min(b, len(text))):
            furniture[j] = 1

    return Book(key=key, lang=meta["lang"], text=text, anchors=anchors,
                quoted=quoted, furniture=furniture)


if __name__ == "__main__":
    print(f"{'key':14s} {'lang':5s} {'words':>8s} {'quoted%':>8s} {'furniture%':>10s}"
          f"  first / last locator")
    for k in BOOKS:
        b = load(k)
        q = 100 * sum(b.quoted) / max(len(b.text), 1)
        f = 100 * sum(b.furniture) / max(len(b.text), 1)
        print(f"{k:14s} {b.lang:5s} {b.words:8,d} {q:7.1f}% {f:9.1f}%"
              f"  {b.locator_at(0)} / {b.locator_at(len(b.text) - 1)}")
