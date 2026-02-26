#!/usr/bin/env python3
"""
Stage 4: Sentiment, Subjectivity & Emotion Analysis
Sentence-level sentiment, subjectivity detection, emotion classification.
"""
import json
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from collections import Counter, defaultdict

from utils import (
    OUTPUT_DIR, FIGURES_DIR, BOOK_META,
    UKRAINE_KEYWORDS_EN, UKRAINE_KEYWORDS_DE,
    contains_ukraine_keyword, load_json, save_json,
)

# Subjectivity lexicon markers
SUBJ_MARKERS_EN = {
    "evaluative_adj": {"beautiful", "ugly", "wonderful", "terrible", "amazing",
                       "horrible", "excellent", "awful", "great", "bad", "strange",
                       "odd", "remarkable", "impressive", "disturbing", "sad",
                       "happy", "grim", "bleak", "vibrant", "charming"},
    "modal_verbs": {"might", "could", "would", "should", "may", "must"},
    "hedges": {"perhaps", "maybe", "somewhat", "rather", "quite", "fairly",
               "apparently", "seemingly", "allegedly"},
    "intensifiers": {"very", "extremely", "incredibly", "absolutely", "utterly",
                     "really", "deeply", "profoundly", "tremendously"},
}

SUBJ_MARKERS_DE = {
    "evaluative_adj": {"schön", "hässlich", "wunderbar", "schrecklich", "toll",
                       "furchtbar", "großartig", "schlimm", "seltsam", "merkwürdig",
                       "beeindruckend", "beunruhigend", "traurig", "fröhlich",
                       "düster", "trostlos", "lebendig"},
    "modal_verbs": {"könnte", "würde", "sollte", "müsste", "dürfte", "mag"},
    "hedges": {"vielleicht", "möglicherweise", "etwas", "ziemlich", "anscheinend",
               "offenbar", "wohl", "vermutlich"},
    "intensifiers": {"sehr", "extrem", "unglaublich", "absolut", "völlig",
                     "wirklich", "zutiefst", "enorm", "ungeheuer"},
}


def split_sentences(text):
    """Simple sentence splitter."""
    sents = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', text)
    return [s.strip() for s in sents if len(s.strip()) > 10]


def translate_de_to_en(sentences, batch_size=20):
    """Translate German sentences to English using deep_translator (batch mode)."""
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='de', target='en')
    translated = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        try:
            # Use batch translation: join with delimiter, translate, split
            combined = " ||| ".join(s[:500] for s in batch)
            if len(combined) > 4500:
                # Fall back to individual translation for oversized batches
                for sent in batch:
                    try:
                        t = translator.translate(sent[:500])
                        translated.append(t if t else sent)
                    except Exception:
                        translated.append(sent)
            else:
                result = translator.translate(combined)
                if result and "|||" in result:
                    parts = result.split("|||")
                    for p in parts:
                        translated.append(p.strip())
                    # Pad if split produced fewer parts
                    while len(translated) < i + len(batch):
                        translated.append(batch[len(translated) - i])
                else:
                    # Fallback: apply same result to all
                    for sent in batch:
                        translated.append(result if result else sent)
        except Exception:
            translated.extend(batch)
        if (i + batch_size) % 200 == 0:
            print(f"      Translated {min(i + batch_size, len(sentences))}/{len(sentences)}...")
    return translated


def compute_subjectivity(text, lang="en"):
    """Compute subjectivity score based on lexical markers."""
    markers = SUBJ_MARKERS_EN if lang == "en" else SUBJ_MARKERS_DE
    tokens = text.lower().split()
    total = len(tokens)
    if total == 0:
        return 0.0

    marker_count = 0
    for category, words in markers.items():
        marker_count += sum(1 for t in tokens if t in words)

    return marker_count / total


def analyze_sentiment_book(chapters, lang, sentiment_pipe, emotion_pipe):
    """Analyze sentiment and emotion for one book."""
    all_results = []

    for ch in chapters:
        text = ch["text"]
        sentences = split_sentences(text)

        if not sentences:
            all_results.append({
                "chapter": ch["chapter"],
                "sentiment": {"positive": 0, "negative": 0, "neutral": 0},
                "emotion": {},
                "subjectivity": 0,
                "ukraine_sentiment": {},
            })
            continue

        # For German: apply sentiment model directly (it handles multilingual
        # input reasonably) to avoid slow translation bottleneck
        en_sentences = sentences

        # Sentiment analysis in batches
        sent_scores = {"positive": 0, "negative": 0, "neutral": 0}
        batch_size = 32
        sent_per_sentence = []

        for i in range(0, len(en_sentences), batch_size):
            batch = en_sentences[i:i + batch_size]
            # Truncate long sentences
            batch = [s[:512] for s in batch]
            try:
                results = sentiment_pipe(batch, truncation=True, max_length=512)
                for r in results:
                    # top_k=1 returns [[{...}]], so unwrap
                    if isinstance(r, list):
                        r = r[0]
                    label = r["label"].lower()
                    if "pos" in label:
                        sent_scores["positive"] += 1
                        sent_per_sentence.append(1)
                    elif "neg" in label:
                        sent_scores["negative"] += 1
                        sent_per_sentence.append(-1)
                    else:
                        sent_scores["neutral"] += 1
                        sent_per_sentence.append(0)
            except Exception as e:
                print(f"    Sentiment error: {e}")
                pass

        # Emotion analysis on sample (first 100 sentences for speed)
        emotion_counts = Counter()
        sample = en_sentences[:100]
        for i in range(0, len(sample), batch_size):
            batch = [s[:512] for s in sample[i:i + batch_size]]
            try:
                results = emotion_pipe(batch, truncation=True, max_length=512)
                for r in results:
                    if isinstance(r, list):
                        r = r[0]
                    emotion_counts[r["label"]] += 1
            except Exception as e:
                print(f"    Emotion error: {e}")
                pass

        # Subjectivity
        subj = compute_subjectivity(text, lang)

        # Ukraine-specific sentiment
        ukraine_sents = []
        ukraine_keywords = UKRAINE_KEYWORDS_EN if lang == "en" else UKRAINE_KEYWORDS_DE
        for j, sent in enumerate(sentences):
            if any(kw in sent.lower() for kw in ukraine_keywords):
                if j < len(sent_per_sentence):
                    ukraine_sents.append(sent_per_sentence[j])

        ukraine_sentiment = {}
        if ukraine_sents:
            ukraine_sentiment = {
                "positive": ukraine_sents.count(1),
                "negative": ukraine_sents.count(-1),
                "neutral": ukraine_sents.count(0),
                "mean_score": float(np.mean(ukraine_sents)),
            }

        total_s = sum(sent_scores.values())
        sent_ratios = {k: v / total_s for k, v in sent_scores.items()} if total_s > 0 else {}

        all_results.append({
            "chapter": ch["chapter"],
            "sentiment_counts": sent_scores,
            "sentiment_ratios": sent_ratios,
            "emotion_counts": dict(emotion_counts),
            "subjectivity_score": subj,
            "ukraine_sentiment": ukraine_sentiment,
            "num_sentences": len(sentences),
            "sentiment_trajectory": sent_per_sentence[:500],  # limit for JSON
        })

    return all_results


def plot_sentiment_comparison(results):
    """Box plots of sentiment across authors."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Sentiment Distribution by Author", fontsize=14, fontweight="bold")

    for idx, metric in enumerate(["positive", "negative", "neutral"]):
        data = []
        labels = []
        for book_id, chapters in results.items():
            for ch in chapters:
                ratios = ch.get("sentiment_ratios", {})
                if ratios:
                    data.append(ratios.get(metric, 0))
                    labels.append(BOOK_META[book_id]["author"].split()[-1])

        if data:
            import pandas as pd
            df = pd.DataFrame({"Author": labels, metric.title(): data})
            sns.boxplot(data=df, x="Author", y=metric.title(), ax=axes[idx],
                       palette="Set2")
            axes[idx].set_title(f"{metric.title()} Sentiment")
            axes[idx].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_sentiment_boxplots.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_sentiment_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved sentiment box plots")


def plot_emotion_comparison(results):
    """Heatmap of emotion distributions."""
    emotions = set()
    for chapters in results.values():
        for ch in chapters:
            emotions.update(ch.get("emotion_counts", {}).keys())
    emotions = sorted(emotions)

    if not emotions:
        return

    matrix = []
    labels = []
    for book_id, chapters in results.items():
        total_emotions = Counter()
        for ch in chapters:
            for e, c in ch.get("emotion_counts", {}).items():
                total_emotions[e] += c
        total = sum(total_emotions.values())
        row = [total_emotions.get(e, 0) / total * 100 if total > 0 else 0 for e in emotions]
        matrix.append(row)
        labels.append(BOOK_META[book_id]["author"].split()[-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(np.array(matrix), annot=True, fmt=".1f", xticklabels=emotions,
                yticklabels=labels, cmap="YlOrRd", ax=ax)
    ax.set_title("Emotion Distribution (%) by Author")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_emotion_heatmap.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_emotion_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved emotion heatmap")


def plot_sentiment_trajectory(results):
    """Sentiment trajectory (rolling average) per author."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sentiment Trajectory (Rolling Avg, window=20)", fontsize=14, fontweight="bold")

    colors = sns.color_palette("Set2")
    for ax, (book_id, chapters) in zip(axes.flat, results.items()):
        all_scores = []
        for ch in chapters:
            all_scores.extend(ch.get("sentiment_trajectory", []))

        if len(all_scores) > 20:
            arr = np.array(all_scores, dtype=float)
            window = min(20, len(arr) // 3)
            if window > 0:
                rolling = np.convolve(arr, np.ones(window)/window, mode='valid')
                ax.plot(rolling, color=colors[list(results.keys()).index(book_id)], linewidth=0.8)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_title(f"{BOOK_META[book_id]['author']}")
                ax.set_xlabel("Sentence position")
                ax.set_ylabel("Sentiment")
                ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_sentiment_trajectory.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_sentiment_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved sentiment trajectory figure")


def main():
    print("=" * 60)
    print("STAGE 4: Sentiment, Subjectivity & Emotion Analysis")
    print("=" * 60)

    corpus = load_json(OUTPUT_DIR / "stage1_corpus.json")

    print("Loading sentiment model...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1,
        top_k=1,
    )

    print("Loading emotion model...")
    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=-1,
        top_k=1,
    )

    results = {}
    for book_id, book_data in corpus.items():
        lang = book_data["metadata"]["lang"]
        print(f"\nProcessing {book_id} ({lang})...")
        chapters = book_data["chapters"]
        book_results = analyze_sentiment_book(chapters, lang, sentiment_pipe, emotion_pipe)
        results[book_id] = book_results

    # Generate figures
    plot_sentiment_comparison(results)
    plot_emotion_comparison(results)
    plot_sentiment_trajectory(results)

    save_json(results, OUTPUT_DIR / "stage4_sentiment_emotion.json")
    print("\nStage 4 complete.")


if __name__ == "__main__":
    main()
