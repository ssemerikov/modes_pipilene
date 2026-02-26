#!/usr/bin/env python3
"""
Stage 10: Computational Validation of 4 Documentary Models
K-Means, Agglomerative clustering, LDA, Random Forest feature importance.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import adjusted_rand_score, classification_report
from scipy.cluster.hierarchy import dendrogram, linkage

from utils import OUTPUT_DIR, FIGURES_DIR, BOOK_META, load_json, save_json


FEATURE_COLS = [
    "mean_sent_len", "first_person_density", "noun_ratio", "verb_ratio", "adj_ratio",
    "past_tense_ratio", "present_tense_ratio",
    "mattr", "yules_k",
    "diary_marker_density", "travel_marker_density", "historical_marker_density",
    "positive_ratio", "negative_ratio", "subjectivity",
    "war_total_density", "war_conflict_density", "war_suffering_density",
]


def load_feature_matrix():
    """Load chapter-level feature matrix from Stage 9."""
    df = pd.read_csv(OUTPUT_DIR / "stage9_feature_matrix.csv")
    return df


def run_clustering(X, true_labels, n_clusters=4):
    """Run K-Means and Agglomerative clustering."""
    # K-Means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)
    km_ari = adjusted_rand_score(true_labels, km_labels)

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    agg_labels = agg.fit_predict(X)
    agg_ari = adjusted_rand_score(true_labels, agg_labels)

    return {
        "kmeans_ari": float(km_ari),
        "agglomerative_ari": float(agg_ari),
        "kmeans_labels": km_labels.tolist(),
        "agglomerative_labels": agg_labels.tolist(),
    }


def run_lda_classification(X, y):
    """Linear Discriminant Analysis with LOO cross-validation."""
    lda = LinearDiscriminantAnalysis()
    loo = LeaveOneOut()
    scores = cross_val_score(lda, X, y, cv=loo, scoring="accuracy")

    # Fit on full data for analysis
    lda.fit(X, y)

    return {
        "loo_accuracy": float(scores.mean()),
        "loo_std": float(scores.std()),
        "n_components": lda.n_components if hasattr(lda, 'n_components') else None,
    }


def run_random_forest(X, y, feature_names):
    """Random Forest for feature importance."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    feature_importance = {}
    for idx in indices:
        feature_importance[feature_names[idx]] = float(importances[idx])

    # Also get LOO accuracy
    loo = LeaveOneOut()
    scores = cross_val_score(rf, X, y, cv=loo, scoring="accuracy")

    return {
        "feature_importance": feature_importance,
        "loo_accuracy": float(scores.mean()),
        "loo_std": float(scores.std()),
    }


def validate_qualitative_predictions(df):
    """Compare observed features against qualitative model predictions."""
    # Expected patterns:
    # Brumme (wartime_diary): highest war vocab, highest diary markers, high 1st person
    # Orth (travel_documentary): highest geographic breadth, highest travel markers
    # Applebaum (historical_documentary): highest historical markers, highest past tense
    # Nicolay (experiential_documentary): high 1st person, varied sentence length

    predictions = {}
    book_means = df.groupby("book_id")[FEATURE_COLS].mean()

    checks = [
        ("brumme", "war_total_density", "highest", "Brumme has highest war vocabulary"),
        ("brumme", "diary_marker_density", "highest", "Brumme has highest diary markers"),
        ("brumme", "first_person_density", "high", "Brumme has high 1st person density"),
        ("orth", "travel_marker_density", "highest", "Orth has highest travel markers"),
        ("applebaum", "historical_marker_density", "highest", "Applebaum has highest historical markers"),
        ("applebaum", "past_tense_ratio", "highest", "Applebaum has highest past tense ratio"),
    ]

    for book_id, feature, expected, description in checks:
        if feature not in book_means.columns:
            predictions[description] = {"result": "N/A", "note": "Feature not available"}
            continue

        val = book_means.loc[book_id, feature]
        rank = book_means[feature].rank(ascending=False)
        book_rank = int(rank.loc[book_id])

        if expected == "highest":
            confirmed = book_rank == 1
        elif expected == "high":
            confirmed = book_rank <= 2
        else:
            confirmed = True

        predictions[description] = {
            "value": float(val),
            "rank": book_rank,
            "confirmed": confirmed,
            "all_values": book_means[feature].to_dict(),
        }

    return predictions


def plot_dendrogram(X, labels, author_labels):
    """Hierarchical clustering dendrogram."""
    Z = linkage(X, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    color_map = {b: sns.color_palette("Set2")[i] for i, b in enumerate(BOOK_META.keys())}
    leaf_colors = [color_map.get(l, "gray") for l in author_labels]

    dendrogram(Z, labels=[l[:15] for l in labels], ax=ax,
               leaf_rotation=90, leaf_font_size=6)
    ax.set_title("Hierarchical Clustering of Chapters (Ward Linkage)", fontsize=13)
    ax.set_ylabel("Distance")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_dendrogram.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved dendrogram")


def plot_feature_importance(rf_results):
    """Bar chart of Random Forest feature importance."""
    fi = rf_results["feature_importance"]
    sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(list(reversed(sorted_fi.keys())), list(reversed(sorted_fi.values())),
            color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance for Documentary Model Classification")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_feature_importance.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved feature importance chart")


def plot_model_radar(df):
    """Radar chart comparing the 4 documentary models."""
    features_for_radar = [
        "first_person_density", "war_total_density", "diary_marker_density",
        "travel_marker_density", "historical_marker_density", "mattr",
        "mean_sent_len", "subjectivity",
    ]
    nice_names = [
        "1st Person", "War Vocab", "Diary", "Travel",
        "Historical", "Vocab Richness", "Sent Length", "Subjectivity",
    ]

    means = df.groupby("book_id")[features_for_radar].mean()
    # Normalize to 0-1 for radar
    mins = means.min()
    maxs = means.max()
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normalized = (means - mins) / ranges

    N = len(features_for_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = sns.color_palette("Set2", 4)

    for i, book_id in enumerate(BOOK_META.keys()):
        if book_id in normalized.index:
            values = normalized.loc[book_id].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', color=colors[i],
                    label=BOOK_META[book_id]["author"].split()[-1], linewidth=2)
            ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), nice_names)
    ax.set_title("Documentary Model Feature Profiles (Normalized)", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_model_radar.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_model_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved model radar chart")


def main():
    print("=" * 60)
    print("STAGE 10: Computational Validation of Documentary Models")
    print("=" * 60)

    df = load_feature_matrix()
    print(f"Feature matrix: {df.shape}")

    # Prepare features
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_raw = df[available_cols].fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    le = LabelEncoder()
    y = le.fit_transform(df["book_id"].values)
    labels = df["chapter"].values
    author_labels = df["book_id"].values

    print(f"Features: {len(available_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Classes: {le.classes_}")

    # Clustering
    print("\nRunning clustering...")
    cluster_results = run_clustering(X, y)
    print(f"  K-Means ARI: {cluster_results['kmeans_ari']:.3f}")
    print(f"  Agglomerative ARI: {cluster_results['agglomerative_ari']:.3f}")

    # LDA
    print("\nRunning LDA classification...")
    lda_results = run_lda_classification(X, y)
    print(f"  LOO Accuracy: {lda_results['loo_accuracy']:.3f} ± {lda_results['loo_std']:.3f}")

    # Random Forest
    print("\nRunning Random Forest...")
    rf_results = run_random_forest(X, y, available_cols)
    print(f"  LOO Accuracy: {rf_results['loo_accuracy']:.3f} ± {rf_results['loo_std']:.3f}")
    print("  Top 5 features:")
    for feat, imp in list(rf_results["feature_importance"].items())[:5]:
        print(f"    {feat}: {imp:.4f}")

    # Qualitative validation
    print("\nValidating qualitative predictions...")
    predictions = validate_qualitative_predictions(df)
    for desc, result in predictions.items():
        status = "CONFIRMED" if result.get("confirmed") else "NOT CONFIRMED"
        print(f"  {status}: {desc} (rank={result.get('rank', 'N/A')})")

    # Plots
    plot_dendrogram(X, labels, author_labels)
    plot_feature_importance(rf_results)
    plot_model_radar(df)

    # Save
    results = {
        "clustering": cluster_results,
        "lda_classification": lda_results,
        "random_forest": rf_results,
        "qualitative_validation": predictions,
        "n_features": len(available_cols),
        "n_samples": len(X),
        "feature_names": available_cols,
    }
    save_json(results, OUTPUT_DIR / "stage10_validation.json")

    print("\nStage 10 complete.")


if __name__ == "__main__":
    main()
