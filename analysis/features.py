#!/usr/bin/env python3
"""Parse every unit once and compute the lexical and grammatical measures.

Whole units are parsed (in chunks cut at paragraph breaks); nothing is
truncated. The first version of the pipeline analysed only the first 100,000
characters of a chapter.

Output
------
OUT/features_lexical.csv       one row per unit, counts and densities
WORK/parsed/<unit_id>.json     sentences and named entities, for sentiment.py
                               and geography.py (carries book text: work tree)

Definitions are in MEASURES at the foot of this file; make_numbers.py prints
the same text into the article's table of measures.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter

import numpy as np
import spacy

import corpus
import lexicons as L
from paths import OUT, WORK, check

CHUNK_CHARS = 40_000
MATTR_WINDOW = 500

QUOTE_STYLE = {"applebaum": "single", "nicolay": "double", "miller": "double",
               "brumme": "guillemet", "orth": "guillemet"}
DE_PARTICIPLE_TAGS = {"VVPP", "VAPP", "VMPP"}


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    out, cur, n = [], [], 0
    for para in text.split("\n\n"):
        if n + len(para) > size and cur:
            out.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(para)
        n += len(para) + 2
    if cur:
        out.append("\n\n".join(cur))
    return out


def mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float:
    """Moving-average type-token ratio (Covington & McFall 2010)."""
    if len(tokens) < window:
        return float("nan")
    counts = Counter(tokens[:window])
    types = len(counts)
    total = types
    for i in range(window, len(tokens)):
        out_tok, in_tok = tokens[i - window], tokens[i]
        counts[out_tok] -= 1
        if counts[out_tok] == 0:
            types -= 1
        if counts[in_tok] == 0:
            types += 1
        counts[in_tok] += 1
        total += types
    return total / ((len(tokens) - window + 1) * window)


def yules_k(tokens: list[str]) -> float:
    freqs = Counter(tokens)
    n = len(tokens)
    if n == 0:
        return float("nan")
    m2 = sum(f * f for f in freqs.values())
    return 10_000 * (m2 - n) / (n * n)


def in_lex(tok, lex: set[str]) -> bool:
    return tok.lemma_.lower() in lex or tok.lower_ in lex


def is_first_person(tok, lang: str) -> bool:
    low = tok.lower_
    if lang == "en":
        if tok.text == "US":
            return False
        if low == "mine":
            return tok.pos_ == "PRON"
        return low in L.FIRST_PERSON["en"]
    if low in L.FIRST_PERSON["de"]:
        return True
    return low.startswith(L.FIRST_PERSON_PREFIX_DE) and tok.pos_ in ("DET", "PRON")


def tense_class(tok, lang: str) -> str | None:
    """'past', 'perfect', 'present' for a finite verb; None otherwise.

    A present-tense auxiliary that forms a perfect ("has seen", "hat gesehen",
    "ist gegangen") is counted as past reference. Without this, German
    narrative -- which reports the past in the perfect far more often than
    English does -- reads as present-tense for a purely grammatical reason.
    """
    if lang == "en":
        if tok.tag_ == "VBD":
            return "past"
        if tok.tag_ in ("VBP", "VBZ"):
            if tok.lemma_ == "have" and tok.dep_ == "aux" and tok.head.tag_ == "VBN":
                return "perfect"
            return "present"
        return None
    morph = tok.morph
    if "Fin" not in morph.get("VerbForm"):
        return None
    tense = morph.get("Tense")
    if "Past" in tense:
        return "past"
    if "Pres" in tense:
        if tok.lemma_.lower() in ("haben", "sein") and any(
                c.tag_ in DE_PARTICIPLE_TAGS for c in tok.children):
            return "perfect"
        return "present"
    return None


def war_category(tok, lang: str, prev, nxt) -> str | None:
    lemma, low = tok.lemma_.lower(), tok.lower_
    for cat in L.WAR_CATEGORIES:
        lex = L.WAR[lang][cat]
        hit = low if low in lex else lemma if lemma in lex else None
        if hit:
            if hit in L.WAR_NOUN_ONLY[lang] and tok.pos_ not in ("NOUN", "PROPN"):
                return None
            return cat
    if lang == "en" and low == "front" and tok.pos_ in ("NOUN", "PROPN"):
        if prev is not None and prev.lower_ == "the" and (nxt is None or nxt.lower_ != "of"):
            return "military"
    if lang == "de" and tok.pos_ in ("NOUN", "PROPN") and len(low) >= 8:
        base = re.sub(r"(en|es|er|e|n|s)$", "", low)
        for cat in L.WAR_CATEGORIES:                       # head of the compound first
            if any(base.endswith(s) or low.endswith(s) for s in L.WAR_COMPOUND_STEMS_DE[cat]):
                return cat
        for cat in L.WAR_CATEGORIES:
            if any(low.startswith(s) for s in L.WAR_COMPOUND_STEMS_DE[cat]):
                return cat
    return None


def analyse(unit: dict, nlp) -> tuple[dict, dict]:
    lang, book = unit["lang"], unit["book"]
    c = Counter()
    sent_lens, alpha, sents_out, ents_out = [], [], [], []
    phrases = dates = 0

    for chunk in chunks(unit["text"]):
        quoted = bytearray(len(chunk))
        for a, b in corpus._quote_spans(chunk, QUOTE_STYLE[book]):
            for j in range(a, min(b, len(chunk))):
                quoted[j] = 1
        phrases += sum(len(re.findall(p, chunk, flags=re.I)) for p in L.DIARY_PHRASES[lang])
        dates += len(re.findall(L.DATE_PATTERN[lang], chunk))

        doc = nlp(chunk)
        toks = [t for t in doc if not t.is_space]
        for i, t in enumerate(toks):
            if t.is_punct:
                continue
            q = bool(quoted[t.idx]) if t.idx < len(quoted) else False
            c["words"] += 1
            c["words_authorial"] += not q
            if t.is_alpha:
                alpha.append(t.lower_)
            c[f"pos_{t.pos_}"] += 1
            if is_first_person(t, lang):
                c["first_person"] += 1
                c["first_person_authorial"] += not q
            if t.pos_ in ("VERB", "AUX"):
                c["verbs"] += 1
                tc = tense_class(t, lang)
                if tc:
                    c[f"tense_{tc}"] += 1
            if in_lex(t, L.DIARY[lang]) and not (
                    t.lower_ in L.DIARY_ADV_ONLY[lang] and (t.pos_ != "ADV" or t.text[:1].isupper())):
                c["diary"] += 1
            c["travel"] += in_lex(t, L.TRAVEL[lang])
            c["historical"] += in_lex(t, L.HISTORICAL[lang])
            c["stance"] += in_lex(t, L.STANCE[lang])
            prev = toks[i - 1] if i else None
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            cat = war_category(t, lang, prev, nxt)
            if cat:
                c[f"war_{cat}"] += 1
                c["war_total"] += 1

        for s in doc.sents:
            n = sum(1 for t in s if not t.is_punct and not t.is_space)
            if n == 0:
                continue
            sent_lens.append(n)
            text = " ".join(s.text.split())
            mid = s.start_char + len(s.text) // 2
            sents_out.append(dict(text=text, quoted=bool(quoted[min(mid, len(quoted) - 1)]),
                                  n=n))
            idx = len(sents_out) - 1
            for e in s.ents:
                ents_out.append([" ".join(e.text.split()), e.label_, idx])

    w = max(c["words"], 1)
    v = max(c["verbs"], 1)
    row = dict(
        unit_id=unit["unit_id"], n_words=c["words"], n_sentences=len(sent_lens),
        mean_sent_len=float(np.mean(sent_lens)) if sent_lens else float("nan"),
        first_person_density=c["first_person"] / w,
        first_person_authorial=c["first_person_authorial"] / max(c["words_authorial"], 1),
        quoted_share=1 - c["words_authorial"] / w,
        noun_ratio=c["pos_NOUN"] / w, verb_ratio=c["pos_VERB"] / w, adj_ratio=c["pos_ADJ"] / w,
        past_tense_ratio=(c["tense_past"] + c["tense_perfect"]) / v,
        present_tense_ratio=c["tense_present"] / v,
        perfect_share=c["tense_perfect"] / v,
        mattr=mattr(alpha), yules_k=yules_k(alpha),
        diary_marker_density=(c["diary"] + phrases + dates) / w,
        travel_marker_density=c["travel"] / w,
        historical_marker_density=c["historical"] / w,
        subjectivity=c["stance"] / w,
        war_total_density=c["war_total"] / w,
        war_conflict_density=c["war_conflict"] / w,
        war_suffering_density=c["war_suffering"] / w,
        war_weapons_density=c["war_weapons"] / w,
        war_military_density=c["war_military"] / w,
    )
    return row, dict(unit_id=unit["unit_id"], sentences=sents_out, entities=ents_out)


def main() -> None:
    check()
    units = json.loads((WORK / "units.json").read_text(encoding="utf8"))
    nlp = {"en": spacy.load("en_core_web_sm"), "de": spacy.load("de_core_news_sm")}
    for n in nlp.values():
        n.max_length = 2_000_000
    (WORK / "parsed").mkdir(exist_ok=True)

    rows = []
    for u in units:
        row, parsed = analyse(u, nlp[u["lang"]])
        rows.append(row)
        (WORK / "parsed" / f"{u['unit_id']}.json").write_text(
            json.dumps(parsed, ensure_ascii=False), encoding="utf8")
        print(f"  {u['unit_id']:14s} {row['n_words']:6d} words {row['n_sentences']:5d} sentences")

    with open(OUT / "features_lexical.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT / 'features_lexical.csv'} ({len(rows)} units); "
          f"spaCy {spacy.__version__}, en_core_web_sm {nlp['en'].meta['version']}, "
          f"de_core_news_sm {nlp['de'].meta['version']}")


# The 18 measures of the feature matrix: column -> (group, plain name, definition).
MEASURES = {
    "mean_sent_len": ("Form", "Sentence length", "mean number of words per sentence"),
    "first_person_density": ("Form", "Narrator presence",
                             "first-person pronouns and possessives per word"),
    "noun_ratio": ("Form", "Noun share", "common nouns as a share of all words"),
    "verb_ratio": ("Form", "Verb share", "lexical verbs as a share of all words"),
    "adj_ratio": ("Form", "Adjective share", "adjectives as a share of all words"),
    "past_tense_ratio": ("Form", "Past reference",
                         "finite verbs in the past tense or the perfect, as a share of all verb forms"),
    "present_tense_ratio": ("Form", "Present reference",
                            "finite verbs in the present tense (perfects excluded), as a share of all verb forms"),
    "mattr": ("Form", "Vocabulary diversity (MATTR)",
              "share of distinct words in a 500-word window, averaged over all windows"),
    "yules_k": ("Form", "Vocabulary repetition (Yule's K)",
                "how strongly the text reuses the same words; higher means more repetition"),
    "diary_marker_density": ("Genre markers", "Diary markers",
                             "words anchored in the day of writing (today, yesterday, now) and calendar dates, per word"),
    "travel_marker_density": ("Genre markers", "Travel markers",
                              "verbs of movement and sensory observation, per word"),
    "historical_marker_density": ("Genre markers", "Historical markers",
                                  "words of temporal distance (century, decade, once, memory), per word"),
    "positive_ratio": ("Evaluation", "Positive sentences",
                       "share of sentences a multilingual sentiment classifier labels positive"),
    "negative_ratio": ("Evaluation", "Negative sentences",
                       "share of sentences the same classifier labels negative"),
    "subjectivity": ("Evaluation", "Stance markers",
                     "evaluative adjectives, modal verbs, hedges and intensifiers, per word"),
    "war_total_density": ("War vocabulary", "War vocabulary, all",
                          "words from all four war lists (conflict, weapons, suffering, military), per word"),
    "war_conflict_density": ("War vocabulary", "War vocabulary, conflict",
                             "the conflict list only (war, battle, attack, invasion), per word"),
    "war_suffering_density": ("War vocabulary", "War vocabulary, suffering",
                              "the suffering list only (death, wounded, refugee, destruction), per word"),
}
FEATURES = list(MEASURES)
FORM = [k for k, v in MEASURES.items() if v[0] == "Form"]
CONTENT = [k for k, v in MEASURES.items() if v[0] != "Form"]


if __name__ == "__main__":
    main()
