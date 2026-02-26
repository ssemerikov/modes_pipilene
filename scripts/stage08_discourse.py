#!/usr/bin/env python3
"""
Stage 8: Discourse & Representation Analysis
KWIC, self/other patterns, war vocabulary, collocate analysis, wordclouds.
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
from collections import Counter, defaultdict
from wordcloud import WordCloud

from utils import (
    OUTPUT_DIR, FIGURES_DIR, BOOK_META,
    UKRAINE_KEYWORDS_EN, UKRAINE_KEYWORDS_DE,
    WAR_VOCAB_EN, WAR_VOCAB_DE,
    kwic, contains_ukraine_keyword, load_json, save_json,
)

# Self/Other pronouns
SELF_PRONOUNS_EN = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
OTHER_PRONOUNS_EN = {"they", "them", "their", "theirs", "he", "him", "his", "she", "her"}
SELF_PRONOUNS_DE = {"ich", "mir", "mich", "mein", "meine", "wir", "uns", "unser", "unsere"}
OTHER_PRONOUNS_DE = {"sie", "ihnen", "ihr", "ihre", "er", "ihm", "sein", "seine"}

# Here/There deictics
HERE_EN = {"here", "this"}
THERE_EN = {"there", "that"}
HERE_DE = {"hier", "dies", "diese", "dieses"}
THERE_DE = {"dort", "jene", "jener", "jenes", "da", "drüben"}


def compute_war_vocab_density(text, lang):
    """Compute war vocabulary density by subcategory."""
    tokens = [t.lower() for t in text.split() if t.isalpha()]
    total = len(tokens)
    if total == 0:
        return {}

    vocab = WAR_VOCAB_EN if lang == "en" else WAR_VOCAB_DE
    densities = {}
    total_war = 0

    for category, words in vocab.items():
        count = sum(1 for t in tokens if t in words)
        densities[f"war_{category}_density"] = count / total
        total_war += count

    densities["war_total_density"] = total_war / total
    return densities


def compute_self_other_ratio(text, lang, context_keywords=None):
    """Compute self/other pronoun ratio, optionally in keyword contexts."""
    tokens = text.lower().split()

    if lang == "en":
        self_set, other_set = SELF_PRONOUNS_EN, OTHER_PRONOUNS_EN
    else:
        self_set, other_set = SELF_PRONOUNS_DE, OTHER_PRONOUNS_DE

    if context_keywords:
        # Only count in windows around keywords
        window = 10
        self_count = 0
        other_count = 0
        for i, t in enumerate(tokens):
            if any(kw in t for kw in context_keywords):
                window_tokens = tokens[max(0, i-window):i+window+1]
                self_count += sum(1 for wt in window_tokens if wt in self_set)
                other_count += sum(1 for wt in window_tokens if wt in other_set)
    else:
        self_count = sum(1 for t in tokens if t in self_set)
        other_count = sum(1 for t in tokens if t in other_set)

    ratio = self_count / other_count if other_count > 0 else float(self_count)
    return {"self_count": self_count, "other_count": other_count, "self_other_ratio": ratio}


def compute_collocates(text, target_keywords, window=5, min_freq=3):
    """Find words co-occurring with target keywords within window."""
    tokens = text.lower().split()
    collocate_counts = Counter()

    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                 "of", "is", "was", "are", "were", "be", "been", "have", "has", "had",
                 "der", "die", "das", "und", "in", "von", "zu", "mit", "auf", "für",
                 "ist", "war", "ein", "eine", "nicht", "den", "dem", "des", "sich"}

    for i, t in enumerate(tokens):
        if any(kw in t for kw in target_keywords):
            context = tokens[max(0, i-window):i] + tokens[i+1:i+window+1]
            for c in context:
                if c.isalpha() and len(c) > 2 and c not in stopwords and c not in target_keywords:
                    collocate_counts[c] += 1

    return {w: c for w, c in collocate_counts.most_common(50) if c >= min_freq}


def detect_similes(text, lang):
    """Detect simile patterns."""
    if lang == "en":
        patterns = [
            r'\blike\s+a\s+\w+',
            r'\bas\s+\w+\s+as\s+',
            r'\bresembl\w+\s+',
        ]
    else:
        patterns = [
            r'\bwie\s+ein[e]?\s+\w+',
            r'\bals\s+ob\s+',
            r'\bähnlich\s+\w+',
        ]

    similes = []
    for pat in patterns:
        matches = re.finditer(pat, text, re.IGNORECASE)
        for m in matches:
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            similes.append(text[start:end].strip())

    return similes[:50]


def generate_wordcloud(text, keywords, lang, title, filename):
    """Generate wordcloud from Ukraine-context passages."""
    # Extract sentences containing keywords
    sents = re.split(r'(?<=[.!?])\s+', text)
    context_text = " ".join(s for s in sents if any(kw in s.lower() for kw in keywords))

    if len(context_text) < 50:
        return

    # Remove the keywords themselves
    for kw in keywords:
        context_text = re.sub(rf'\b{re.escape(kw)}\b', '', context_text, flags=re.IGNORECASE)

    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                 "of", "is", "was", "are", "were", "be", "been", "have", "has", "had",
                 "it", "that", "this", "with", "from", "not", "as", "by", "they", "we",
                 "der", "die", "das", "und", "in", "von", "zu", "mit", "auf", "für",
                 "ist", "war", "ein", "eine", "nicht", "den", "dem", "des", "sich",
                 "auch", "noch", "wie", "dann", "aber", "oder", "nach", "aus"}

    wc = WordCloud(width=800, height=400, background_color="white",
                   max_words=100, stopwords=stopwords,
                   colormap="viridis", random_state=42)

    try:
        wc.generate(context_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=13)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close()
    except ValueError:
        pass


def analyze_book_discourse(chapters, lang):
    """Full discourse analysis for one book."""
    full_text = " ".join(ch["text"] for ch in chapters)
    keywords = UKRAINE_KEYWORDS_EN if lang == "en" else UKRAINE_KEYWORDS_DE

    # War vocabulary
    war_density = compute_war_vocab_density(full_text, lang)

    # Self/Other in general and Ukraine contexts
    self_other_general = compute_self_other_ratio(full_text, lang)
    self_other_ukraine = compute_self_other_ratio(full_text, lang, keywords)

    # Collocates
    collocates = compute_collocates(full_text, keywords)

    # Similes
    similes = detect_similes(full_text, lang)

    # KWIC for Ukraine
    ukraine_kwic = kwic(full_text, r'\bukrain', window=7, max_results=50)

    # Per-chapter war density
    chapter_war = []
    for ch in chapters:
        wd = compute_war_vocab_density(ch["text"], lang)
        chapter_war.append({"chapter": ch["chapter"], **wd})

    return {
        "war_vocabulary": war_density,
        "chapter_war_density": chapter_war,
        "self_other_general": self_other_general,
        "self_other_ukraine_context": self_other_ukraine,
        "collocates": collocates,
        "similes": similes[:20],
        "ukraine_kwic": [{"left": l, "match": m, "right": r} for l, m, r in ukraine_kwic[:30]],
    }


def plot_war_vocab_comparison(results):
    """Bar chart comparing war vocabulary density."""
    categories = ["conflict", "weapons", "suffering", "military"]
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results))
    width = 0.2
    colors = sns.color_palette("Reds", len(categories))

    for i, cat in enumerate(categories):
        vals = [results[b]["war_vocabulary"].get(f"war_{cat}_density", 0) * 1000
                for b in results]
        ax.bar(x + i * width, vals, width, label=cat.title(), color=colors[i])

    ax.set_ylabel("Density (per 1000 words)")
    ax.set_title("War Vocabulary Density by Author and Category")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([BOOK_META[b]["author"].split()[-1] for b in results])
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_war_vocab.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_war_vocab.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved war vocabulary figure")


def plot_collocate_bars(results):
    """Top collocates of Ukraine per author."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Top 15 Collocates of 'Ukraine' by Author", fontsize=14, fontweight="bold")

    for ax, (book_id, data) in zip(axes.flat, results.items()):
        colls = data["collocates"]
        top = dict(list(colls.items())[:15])
        if top:
            ax.barh(list(reversed(top.keys())), list(reversed(top.values())),
                    color=sns.color_palette("Set2")[list(results.keys()).index(book_id)])
        ax.set_title(f"{BOOK_META[book_id]['author']}", fontsize=11)
        ax.set_xlabel("Frequency")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_collocates.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_collocates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved collocate figure")


def main():
    print("=" * 60)
    print("STAGE 8: Discourse & Representation Analysis")
    print("=" * 60)

    corpus = load_json(OUTPUT_DIR / "stage1_corpus.json")

    results = {}
    for book_id, book_data in corpus.items():
        lang = book_data["metadata"]["lang"]
        print(f"\nAnalyzing {book_id} ({lang})...")
        result = analyze_book_discourse(book_data["chapters"], lang)
        results[book_id] = result

        wd = result["war_vocabulary"]
        print(f"  War vocab density: {wd.get('war_total_density', 0)*1000:.2f} per 1000 words")
        print(f"  Self/Other ratio (general): {result['self_other_general']['self_other_ratio']:.2f}")
        print(f"  Self/Other ratio (Ukraine): {result['self_other_ukraine_context']['self_other_ratio']:.2f}")
        print(f"  Top collocates: {list(result['collocates'].items())[:5]}")

    # Generate wordclouds
    for book_id, book_data in corpus.items():
        lang = book_data["metadata"]["lang"]
        keywords = UKRAINE_KEYWORDS_EN if lang == "en" else UKRAINE_KEYWORDS_DE
        full_text = " ".join(ch["text"] for ch in book_data["chapters"])
        generate_wordcloud(
            full_text, keywords, lang,
            f"Ukraine Context: {BOOK_META[book_id]['author']}",
            f"fig_wordcloud_{book_id}.png"
        )

    # Figures
    plot_war_vocab_comparison(results)
    plot_collocate_bars(results)

    save_json(results, OUTPUT_DIR / "stage8_discourse.json")
    print("\nStage 8 complete.")


if __name__ == "__main__":
    main()
