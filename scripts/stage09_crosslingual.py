#!/usr/bin/env python3
"""
Stage 9: Cross-Linguistic Comparative Analysis
Language-neutral features, statistical tests, cross-lingual comparison.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
from collections import defaultdict

from utils import OUTPUT_DIR, FIGURES_DIR, BOOK_META, load_json, save_json


def gather_chapter_features(ling_data, stylo_data, geo_data, sent_data, disc_data):
    """Gather language-neutral features per chapter."""
    rows = []

    for book_id in BOOK_META:
        lang = BOOK_META[book_id]["lang"]
        lang_group = "English" if lang == "en" else "German"

        ling_chapters = ling_data.get(book_id, {}).get("chapters", [])
        stylo_chapters = stylo_data.get(book_id, {}).get("chapters", [])
        sent_chapters = sent_data.get(book_id, [])
        disc_chapter_war = disc_data.get(book_id, {}).get("chapter_war_density", [])

        n = min(len(ling_chapters), len(stylo_chapters))
        for i in range(n):
            lc = ling_chapters[i]
            sc = stylo_chapters[i] if i < len(stylo_chapters) else {}
            se = sent_chapters[i] if i < len(sent_chapters) else {}
            dw = disc_chapter_war[i] if i < len(disc_chapter_war) else {}

            row = {
                "book_id": book_id,
                "author": BOOK_META[book_id]["author"],
                "lang_group": lang_group,
                "model": BOOK_META[book_id]["model"],
                "chapter": lc.get("chapter", ""),
                # Linguistic
                "mean_sent_len": lc.get("sentence_stats", {}).get("mean_sent_len", 0),
                "first_person_density": lc.get("pronoun_stats", {}).get("first_person_density", 0),
                "noun_ratio": lc.get("pos_ratios", {}).get("NOUN", 0),
                "verb_ratio": lc.get("pos_ratios", {}).get("VERB", 0),
                "adj_ratio": lc.get("pos_ratios", {}).get("ADJ", 0),
                "past_tense_ratio": lc.get("tense_ratios", {}).get("Past", 0),
                "present_tense_ratio": lc.get("tense_ratios", {}).get("Pres", 0),
                # Stylometric
                "mattr": sc.get("mattr", 0),
                "yules_k": sc.get("yules_k", 0),
                "diary_marker_density": sc.get("genre_markers", {}).get("diary_marker_density", 0),
                "travel_marker_density": sc.get("genre_markers", {}).get("travel_marker_density", 0),
                "historical_marker_density": sc.get("genre_markers", {}).get("historical_marker_density", 0),
                # Sentiment
                "positive_ratio": se.get("sentiment_ratios", {}).get("positive", 0),
                "negative_ratio": se.get("sentiment_ratios", {}).get("negative", 0),
                "subjectivity": se.get("subjectivity_score", 0),
                # War
                "war_total_density": dw.get("war_total_density", 0),
                "war_conflict_density": dw.get("war_conflict_density", 0),
                "war_suffering_density": dw.get("war_suffering_density", 0),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def run_statistical_tests(df):
    """Run Mann-Whitney U (EN vs DE) and Kruskal-Wallis (4 authors)."""
    features = [
        "mean_sent_len", "first_person_density", "mattr", "yules_k",
        "positive_ratio", "negative_ratio", "subjectivity",
        "war_total_density", "diary_marker_density", "travel_marker_density",
        "historical_marker_density",
    ]

    test_results = {}

    for feat in features:
        vals = df[feat].dropna()
        if vals.std() == 0:
            continue

        # Mann-Whitney U: English vs German
        en_vals = df[df["lang_group"] == "English"][feat].dropna()
        de_vals = df[df["lang_group"] == "German"][feat].dropna()

        mw_result = {}
        if len(en_vals) > 2 and len(de_vals) > 2:
            try:
                u_stat, p_val = mannwhitneyu(en_vals, de_vals, alternative="two-sided")
                mw_result = {
                    "U_statistic": float(u_stat),
                    "p_value": float(p_val),
                    "significant": p_val < 0.05,
                    "en_median": float(en_vals.median()),
                    "de_median": float(de_vals.median()),
                }
            except Exception:
                pass

        # Kruskal-Wallis: 4 authors
        kw_result = {}
        groups = [df[df["book_id"] == b][feat].dropna().values for b in BOOK_META]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 3:
            try:
                h_stat, p_val = kruskal(*groups)
                kw_result = {
                    "H_statistic": float(h_stat),
                    "p_value": float(p_val),
                    "significant": p_val < 0.05,
                }
            except Exception:
                pass

        test_results[feat] = {
            "mann_whitney": mw_result,
            "kruskal_wallis": kw_result,
        }

    return test_results


def plot_language_comparison(df):
    """Box plots comparing English vs German features."""
    features = ["mean_sent_len", "first_person_density", "mattr",
                "war_total_density", "positive_ratio", "subjectivity"]
    nice_names = ["Sentence Length", "1st Person Density", "MATTR",
                  "War Vocab Density", "Positive Sentiment", "Subjectivity"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("English vs. German: Feature Comparison", fontsize=14, fontweight="bold")

    for ax, feat, name in zip(axes.flat, features, nice_names):
        sns.boxplot(data=df, x="lang_group", y=feat, palette="Set2", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_crosslingual_boxplots.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_crosslingual_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved cross-lingual box plots")


def plot_author_comparison(df):
    """Grouped bar charts for key features by author."""
    features = ["mean_sent_len", "first_person_density", "war_total_density",
                "diary_marker_density", "travel_marker_density", "historical_marker_density"]
    nice_names = ["Sentence Length", "1st Person Density", "War Vocab",
                  "Diary Markers", "Travel Markers", "Historical Markers"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Feature Comparison Across Four Authors", fontsize=14, fontweight="bold")

    for ax, feat, name in zip(axes.flat, features, nice_names):
        means = df.groupby("book_id")[feat].mean()
        stds = df.groupby("book_id")[feat].std()
        authors = [BOOK_META[b]["author"].split()[-1] for b in means.index]
        colors = [sns.color_palette("Set2")[list(BOOK_META.keys()).index(b)] for b in means.index]

        ax.bar(authors, means.values, yerr=stds.values, color=colors, capsize=3)
        ax.set_title(name)
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_author_comparison.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_author_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved author comparison figure")


def main():
    print("=" * 60)
    print("STAGE 9: Cross-Linguistic Comparative Analysis")
    print("=" * 60)

    # Load all previous stage results
    ling_data = load_json(OUTPUT_DIR / "stage2_linguistic_features.json")
    stylo_data = load_json(OUTPUT_DIR / "stage7_stylometrics.json")
    sent_data = load_json(OUTPUT_DIR / "stage4_sentiment_emotion.json")
    disc_data = load_json(OUTPUT_DIR / "stage8_discourse.json")

    # Also load geographic data
    geo_data = load_json(OUTPUT_DIR / "stage3_geographic.json")

    print("Gathering chapter-level features...")
    df = gather_chapter_features(ling_data, stylo_data, geo_data, sent_data, disc_data)
    print(f"  Feature matrix: {df.shape[0]} chapters x {df.shape[1]} features")
    print(f"  By language: {df['lang_group'].value_counts().to_dict()}")

    print("\nRunning statistical tests...")
    test_results = run_statistical_tests(df)

    # Print significant results
    print("\nSignificant results (p < 0.05):")
    for feat, tests in test_results.items():
        mw = tests.get("mann_whitney", {})
        kw = tests.get("kruskal_wallis", {})
        if mw.get("significant"):
            print(f"  {feat}: Mann-Whitney p={mw['p_value']:.4f} "
                  f"(EN median={mw['en_median']:.4f}, DE median={mw['de_median']:.4f})")
        if kw.get("significant"):
            print(f"  {feat}: Kruskal-Wallis p={kw['p_value']:.4f}")

    # Generate plots
    plot_language_comparison(df)
    plot_author_comparison(df)

    # Descriptive stats - convert tuple keys to strings
    desc = df.groupby("book_id")[["mean_sent_len", "first_person_density", "mattr",
                                   "war_total_density", "positive_ratio"]].describe()
    desc_dict = {}
    for col in desc.columns:
        key = f"{col[0]}_{col[1]}" if isinstance(col, tuple) else str(col)
        desc_dict[key] = desc[col].to_dict()

    # Save
    results = {
        "statistical_tests": test_results,
        "descriptive_stats": desc_dict,
        "n_chapters": len(df),
    }
    save_json(results, OUTPUT_DIR / "stage9_crosslingual.json")

    # Also save the feature matrix
    df.to_csv(OUTPUT_DIR / "stage9_feature_matrix.csv", index=False)
    print(f"  Saved feature matrix: {OUTPUT_DIR / 'stage9_feature_matrix.csv'}")

    print("\nStage 9 complete.")


if __name__ == "__main__":
    main()
