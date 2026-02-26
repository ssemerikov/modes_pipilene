#!/usr/bin/env python3
"""
Stage 2: Linguistic Feature Extraction
POS distributions, tense, pronoun density, NER, sentence complexity.
"""
import json
import re
import warnings
warnings.filterwarnings("ignore")

import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import Counter

from utils import OUTPUT_DIR, BOOK_META, load_json, save_json

# First-person pronouns
FP_PRONOUNS_EN = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
FP_PRONOUNS_DE = {"ich", "mir", "mich", "mein", "meine", "meinem", "meinen", "meiner",
                   "meines", "wir", "uns", "unser", "unsere", "unserem", "unseren", "unserer"}


def load_spacy_models():
    print("Loading spaCy models...")
    nlp_en = spacy.load("en_core_web_sm", disable=["textcat"])
    nlp_de = spacy.load("de_core_news_sm", disable=["textcat"])
    # Increase max length for long texts
    nlp_en.max_length = 2_000_000
    nlp_de.max_length = 2_000_000
    return nlp_en, nlp_de


def load_ner_pipeline():
    print("Loading XLM-RoBERTa NER pipeline...")
    ner_pipe = pipeline(
        "ner",
        model="Davlan/xlm-roberta-base-ner-hrl",
        aggregation_strategy="simple",
        device=-1,  # CPU
    )
    return ner_pipe


def extract_pos_distribution(doc):
    """Extract POS tag distribution from spaCy doc."""
    pos_counts = Counter()
    for token in doc:
        if not token.is_punct and not token.is_space:
            pos_counts[token.pos_] += 1
    total = sum(pos_counts.values())
    if total == 0:
        return {}, {}
    ratios = {pos: count / total for pos, count in pos_counts.items()}
    return dict(pos_counts), ratios


def extract_tense_distribution(doc, lang="en"):
    """Extract tense distribution from morphological features."""
    tense_counts = Counter()
    for token in doc:
        if token.pos_ in ("VERB", "AUX"):
            morph = token.morph.to_dict()
            tense = morph.get("Tense", "Unknown")
            if isinstance(tense, list):
                tense = tense[0]
            tense_counts[tense] += 1
    total = sum(tense_counts.values())
    if total == 0:
        return {}, {}
    ratios = {t: c / total for t, c in tense_counts.items()}
    return dict(tense_counts), ratios


def extract_pronoun_density(doc, lang="en"):
    """Compute first-person pronoun density."""
    fp_set = FP_PRONOUNS_EN if lang == "en" else FP_PRONOUNS_DE
    total_tokens = 0
    fp_count = 0
    for token in doc:
        if not token.is_punct and not token.is_space:
            total_tokens += 1
            if token.text.lower() in fp_set:
                fp_count += 1
    density = fp_count / total_tokens if total_tokens > 0 else 0
    return {"first_person_count": fp_count, "total_tokens": total_tokens,
            "first_person_density": density}


def extract_sentence_stats(doc):
    """Extract sentence length and complexity statistics."""
    sent_lengths = []
    for sent in doc.sents:
        tokens = [t for t in sent if not t.is_punct and not t.is_space]
        if tokens:
            sent_lengths.append(len(tokens))

    if not sent_lengths:
        return {"mean_sent_len": 0, "std_sent_len": 0, "num_sentences": 0}

    import numpy as np
    arr = np.array(sent_lengths)
    return {
        "mean_sent_len": float(arr.mean()),
        "std_sent_len": float(arr.std()),
        "median_sent_len": float(np.median(arr)),
        "num_sentences": len(sent_lengths),
        "sent_lengths": sent_lengths,
    }


def extract_spacy_ner(doc):
    """Extract NER from spaCy doc."""
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


def extract_xlm_ner(text, ner_pipe, max_chunk=512):
    """Extract NER using XLM-RoBERTa, processing in chunks."""
    # Split text into manageable chunks at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    all_entities = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) > max_chunk * 4:  # ~4 chars per token
            if current_chunk:
                try:
                    ents = ner_pipe(current_chunk)
                    all_entities.extend(ents)
                except Exception:
                    pass
            current_chunk = sent
        else:
            current_chunk += " " + sent if current_chunk else sent

    if current_chunk:
        try:
            ents = ner_pipe(current_chunk)
            all_entities.extend(ents)
        except Exception:
            pass

    # Deduplicate and format
    formatted = []
    for e in all_entities:
        formatted.append({
            "text": e["word"],
            "label": e["entity_group"],
            "score": float(e["score"]),
        })
    return formatted


def process_chapter(text, nlp, ner_pipe, lang):
    """Process a single chapter through all linguistic analyses."""
    # Truncate very long texts for spaCy (process first 100k chars)
    trunc_text = text[:100_000] if len(text) > 100_000 else text
    doc = nlp(trunc_text)

    pos_counts, pos_ratios = extract_pos_distribution(doc)
    tense_counts, tense_ratios = extract_tense_distribution(doc, lang)
    pronoun_stats = extract_pronoun_density(doc, lang)
    sent_stats = extract_sentence_stats(doc)
    spacy_ner = extract_spacy_ner(doc)

    # XLM-RoBERTa NER on a sample (first 20k chars)
    xlm_ner = extract_xlm_ner(text[:20_000], ner_pipe)

    return {
        "pos_counts": pos_counts,
        "pos_ratios": pos_ratios,
        "tense_counts": tense_counts,
        "tense_ratios": tense_ratios,
        "pronoun_stats": pronoun_stats,
        "sentence_stats": {k: v for k, v in sent_stats.items() if k != "sent_lengths"},
        "sent_lengths": sent_stats.get("sent_lengths", []),
        "spacy_ner": spacy_ner[:200],  # Limit for JSON size
        "xlm_ner": xlm_ner[:200],
    }


def main():
    print("=" * 60)
    print("STAGE 2: Linguistic Feature Extraction")
    print("=" * 60)

    corpus = load_json(OUTPUT_DIR / "stage1_corpus.json")
    nlp_en, nlp_de = load_spacy_models()
    ner_pipe = load_ner_pipeline()

    results = {}

    for book_id, book_data in corpus.items():
        lang = book_data["metadata"]["lang"]
        nlp = nlp_en if lang == "en" else nlp_de
        print(f"\nProcessing {book_id} ({lang})...")

        book_results = []
        for i, ch in enumerate(book_data["chapters"]):
            print(f"  Chapter {i+1}/{len(book_data['chapters'])}: {ch['chapter'][:50]}...")
            features = process_chapter(ch["text"], nlp, ner_pipe, lang)
            features["chapter"] = ch["chapter"]
            features["word_count"] = len(ch["text"].split())
            book_results.append(features)

        results[book_id] = {
            "metadata": book_data["metadata"],
            "chapters": book_results,
        }

    save_json(results, OUTPUT_DIR / "stage2_linguistic_features.json")
    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
