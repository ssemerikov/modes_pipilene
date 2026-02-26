#!/usr/bin/env python3
"""
Stage 7: Stylometric & Genre Analysis
Vocabulary richness, sentence distributions, function words, genre markers, PCA.
"""
import json
import re
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import OUTPUT_DIR, FIGURES_DIR, BOOK_META, load_json, save_json


# Function words
FUNCTION_WORDS_EN = [
    "the", "of", "and", "a", "to", "in", "is", "was", "that", "it",
    "for", "on", "are", "as", "with", "his", "they", "be", "at", "one",
    "have", "this", "from", "or", "had", "by", "not", "but", "what", "all",
    "were", "we", "when", "your", "can", "said", "there", "each", "which", "do",
    "their", "if", "will", "up", "about", "out", "them", "then", "she", "many",
]

FUNCTION_WORDS_DE = [
    "der", "die", "das", "und", "in", "den", "von", "zu", "ist", "mit",
    "sich", "des", "auf", "für", "nicht", "ein", "eine", "dem", "es", "er",
    "sie", "sind", "war", "auch", "als", "an", "noch", "wie", "nach", "aus",
    "bei", "nur", "so", "aber", "am", "um", "hat", "oder", "vor", "zur",
    "bis", "über", "haben", "dass", "dann", "wenn", "schon", "ich", "wir", "man",
]

# Genre markers
DIARY_MARKERS_EN = {
    "temporal_deictics": {"today", "yesterday", "tomorrow", "tonight", "this morning",
                          "this evening", "this afternoon", "now", "later", "earlier"},
    "date_pattern": r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}',
}

DIARY_MARKERS_DE = {
    "temporal_deictics": {"heute", "gestern", "morgen", "jetzt", "vorhin", "später",
                          "gerade", "eben", "soeben", "nunmehr", "derzeit"},
    "date_pattern": r'\b\d{1,2}\.\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)',
}

TRAVEL_MARKERS_EN = {
    "movement": {"arrive", "arrived", "depart", "departed", "travel", "traveled",
                 "journey", "drive", "drove", "walk", "walked", "cross", "crossed",
                 "reach", "reached", "leave", "left", "pass", "border"},
    "sensory": {"see", "saw", "seen", "hear", "heard", "smell", "taste", "feel",
                "look", "sound", "touch", "beautiful", "cold", "warm", "dark", "bright"},
}

TRAVEL_MARKERS_DE = {
    "movement": {"ankommen", "angekommen", "abfahren", "reisen", "fahren", "fuhr",
                 "gefahren", "gehen", "ging", "gegangen", "überqueren", "erreichen",
                 "verlassen", "grenze", "passieren"},
    "sensory": {"sehen", "gesehen", "hören", "gehört", "riechen", "schmecken",
                "fühlen", "aussehen", "klingen", "schön", "kalt", "warm", "dunkel", "hell"},
}

HISTORICAL_MARKERS_EN = {
    "temporal_distance": {"century", "centuries", "year", "years", "decade", "decades",
                          "ancient", "medieval", "historical", "formerly", "once",
                          "tradition", "heritage", "memory", "remember"},
}

HISTORICAL_MARKERS_DE = {
    "temporal_distance": {"jahrhundert", "jahrhunderte", "jahr", "jahre", "jahrzehnt",
                          "antik", "mittelalterlich", "historisch", "ehemals", "einst",
                          "tradition", "erbe", "erinnerung", "erinnern"},
}


def compute_mattr(tokens, window=500):
    """Moving Average Type-Token Ratio."""
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens) if tokens else 0

    ttrs = []
    for i in range(len(tokens) - window + 1):
        w = tokens[i:i + window]
        ttrs.append(len(set(w)) / window)

    return float(np.mean(ttrs))


def compute_yules_k(tokens):
    """Yule's K measure of vocabulary richness."""
    freqs = Counter(tokens)
    freq_of_freqs = Counter(freqs.values())
    n = len(tokens)
    if n == 0:
        return 0

    m1 = n
    m2 = sum(i * i * v for i, v in freq_of_freqs.items())

    if m1 == 0 or m1 == m2:
        return 0

    k = 10000 * (m2 - m1) / (m1 * m1)
    return float(k)


def compute_hapax_ratio(tokens):
    """Ratio of words appearing exactly once."""
    freqs = Counter(tokens)
    hapax = sum(1 for v in freqs.values() if v == 1)
    return hapax / len(freqs) if freqs else 0


def compute_guiraud(tokens):
    """Guiraud's Root TTR."""
    n = len(tokens)
    v = len(set(tokens))
    return v / math.sqrt(n) if n > 0 else 0


def compute_function_word_freqs(text, lang):
    """Compute function word frequencies."""
    words = FUNCTION_WORDS_EN if lang == "en" else FUNCTION_WORDS_DE
    tokens = [t.lower() for t in text.split() if t.isalpha()]
    total = len(tokens)
    if total == 0:
        return {}
    counts = Counter(tokens)
    return {w: counts.get(w, 0) / total for w in words}


def compute_genre_markers(text, lang):
    """Compute genre marker densities."""
    tokens = text.lower().split()
    total = len(tokens)
    if total == 0:
        return {}

    results = {}

    # Diary markers
    diary_m = DIARY_MARKERS_EN if lang == "en" else DIARY_MARKERS_DE
    diary_count = sum(1 for t in tokens if t in diary_m["temporal_deictics"])
    date_pat = diary_m["date_pattern"]
    diary_count += len(re.findall(date_pat, text, re.IGNORECASE))
    results["diary_marker_density"] = diary_count / total

    # Travel markers
    travel_m = TRAVEL_MARKERS_EN if lang == "en" else TRAVEL_MARKERS_DE
    travel_count = sum(1 for t in tokens if t in travel_m["movement"])
    travel_count += sum(1 for t in tokens if t in travel_m["sensory"])
    results["travel_marker_density"] = travel_count / total

    # Historical markers
    hist_m = HISTORICAL_MARKERS_EN if lang == "en" else HISTORICAL_MARKERS_DE
    hist_count = sum(1 for t in tokens if t in hist_m["temporal_distance"])
    results["historical_marker_density"] = hist_count / total

    return results


def analyze_book_stylometry(chapters, lang):
    """Full stylometric analysis for one book."""
    # Combine all text
    full_text = " ".join(ch["text"] for ch in chapters)
    tokens = [t.lower() for t in full_text.split() if t.isalpha()]

    # Per-chapter analysis
    chapter_results = []
    for ch in chapters:
        ch_tokens = [t.lower() for t in ch["text"].split() if t.isalpha()]

        # Sentence lengths
        sents = re.split(r'(?<=[.!?])\s+', ch["text"])
        sent_lengths = [len(s.split()) for s in sents if len(s.split()) > 2]

        genre = compute_genre_markers(ch["text"], lang)

        chapter_results.append({
            "chapter": ch["chapter"],
            "mattr": compute_mattr(ch_tokens),
            "yules_k": compute_yules_k(ch_tokens),
            "hapax_ratio": compute_hapax_ratio(ch_tokens),
            "guiraud": compute_guiraud(ch_tokens),
            "sent_lengths": sent_lengths,
            "mean_sent_len": float(np.mean(sent_lengths)) if sent_lengths else 0,
            "genre_markers": genre,
            "word_count": len(ch_tokens),
        })

    # Book-level
    book_level = {
        "mattr": compute_mattr(tokens),
        "yules_k": compute_yules_k(tokens),
        "hapax_ratio": compute_hapax_ratio(tokens),
        "guiraud": compute_guiraud(tokens),
        "function_word_freqs": compute_function_word_freqs(full_text, lang),
        "genre_markers": compute_genre_markers(full_text, lang),
    }

    return {"book_level": book_level, "chapters": chapter_results}


def plot_pca(results):
    """PCA of stylometric features, scatter colored by author."""
    features = []
    labels = []

    for book_id, data in results.items():
        for ch in data["chapters"]:
            vec = [
                ch["mattr"],
                ch["yules_k"],
                ch["hapax_ratio"],
                ch["guiraud"],
                ch["mean_sent_len"],
                ch["genre_markers"].get("diary_marker_density", 0),
                ch["genre_markers"].get("travel_marker_density", 0),
                ch["genre_markers"].get("historical_marker_density", 0),
            ]
            features.append(vec)
            labels.append(book_id)

    if not features:
        return

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("Set2", 4)
    book_ids = list(BOOK_META.keys())

    for i, book_id in enumerate(book_ids):
        mask = [j for j, l in enumerate(labels) if l == book_id]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=[colors[i]], label=BOOK_META[book_id]["author"].split()[-1],
                  alpha=0.6, s=50)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA of Stylometric Features by Author")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_stylometric_pca.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_stylometric_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved PCA scatter plot")


def plot_genre_radar(results):
    """Radar chart of genre marker densities."""
    categories = ["Diary markers", "Travel markers", "Historical markers"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("Set2", 4)

    for i, (book_id, data) in enumerate(results.items()):
        gm = data["book_level"]["genre_markers"]
        values = [
            gm.get("diary_marker_density", 0) * 1000,
            gm.get("travel_marker_density", 0) * 1000,
            gm.get("historical_marker_density", 0) * 1000,
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', color=colors[i],
                label=BOOK_META[book_id]["author"].split()[-1], linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title("Genre Marker Density (per 1000 words)", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_genre_radar.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_genre_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved genre radar chart")


def plot_sentence_violin(results):
    """Violin plots of sentence length per author."""
    import pandas as pd

    data = []
    for book_id, book_data in results.items():
        for ch in book_data["chapters"]:
            for sl in ch["sent_lengths"]:
                data.append({
                    "Author": BOOK_META[book_id]["author"].split()[-1],
                    "Sentence Length": sl,
                })

    if not data:
        return

    df = pd.DataFrame(data)
    # Cap for visualization
    df = df[df["Sentence Length"] < 80]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="Author", y="Sentence Length", palette="Set2",
                   inner="box", ax=ax)
    ax.set_title("Sentence Length Distribution by Author")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_sentence_violin.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_sentence_violin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved sentence violin plot")


def main():
    print("=" * 60)
    print("STAGE 7: Stylometric & Genre Analysis")
    print("=" * 60)

    corpus = load_json(OUTPUT_DIR / "stage1_corpus.json")

    results = {}
    for book_id, book_data in corpus.items():
        lang = book_data["metadata"]["lang"]
        print(f"\nAnalyzing {book_id} ({lang})...")
        result = analyze_book_stylometry(book_data["chapters"], lang)
        results[book_id] = result

        bl = result["book_level"]
        print(f"  MATTR: {bl['mattr']:.4f}")
        print(f"  Yule's K: {bl['yules_k']:.2f}")
        print(f"  Hapax ratio: {bl['hapax_ratio']:.4f}")
        print(f"  Genre markers: {bl['genre_markers']}")

    # Generate figures
    plot_pca(results)
    plot_genre_radar(results)
    plot_sentence_violin(results)

    save_json(results, OUTPUT_DIR / "stage7_stylometrics.json")
    print("\nStage 7 complete.")


if __name__ == "__main__":
    main()
