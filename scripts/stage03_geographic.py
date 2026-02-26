#!/usr/bin/env python3
"""
Stage 3: Toponym & Geographic Analysis
Extract, normalize, classify geographic entities; build co-mention networks.
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
import networkx as nx
from collections import Counter, defaultdict
from itertools import combinations
from scipy.stats import entropy

from utils import (
    OUTPUT_DIR, FIGURES_DIR, BOOK_META,
    normalize_toponym, get_region, UKRAINE_REGIONS,
    load_json, save_json,
)


def extract_geographic_entities(ner_data):
    """Extract location entities from NER results."""
    locations = []
    for ent in ner_data:
        label = ent.get("label", "")
        if label in ("LOC", "GPE", "Location"):
            text = ent["text"].strip()
            if len(text) > 1:
                locations.append(text)
    return locations


def analyze_book_geography(book_chapters):
    """Analyze geographic references for one book."""
    all_locations = []
    chapter_locations = {}

    for ch in book_chapters:
        ch_name = ch["chapter"]
        # Combine spaCy and XLM-RoBERTa NER
        locs = extract_geographic_entities(ch.get("spacy_ner", []))
        locs += extract_geographic_entities(ch.get("xlm_ner", []))

        # Normalize
        normalized = [normalize_toponym(l) for l in locs]
        chapter_locations[ch_name] = normalized
        all_locations.extend(normalized)

    # Count and classify
    location_counts = Counter(all_locations)
    region_counts = Counter()
    ukraine_count = 0
    non_ukraine_count = 0

    for loc, count in location_counts.items():
        region = get_region(loc)
        region_counts[region] += count
        if region != "Non-Ukraine" and region != "Other":
            ukraine_count += count
        else:
            non_ukraine_count += count

    total = ukraine_count + non_ukraine_count
    ukraine_focus = ukraine_count / total if total > 0 else 0

    # Geographic breadth (entropy of regional distribution)
    ukraine_regions = {k: v for k, v in region_counts.items()
                       if k not in ("Non-Ukraine", "Other")}
    if ukraine_regions:
        vals = np.array(list(ukraine_regions.values()), dtype=float)
        geo_entropy = float(entropy(vals / vals.sum()))
    else:
        geo_entropy = 0.0

    return {
        "location_counts": dict(location_counts.most_common(50)),
        "region_counts": dict(region_counts),
        "ukraine_focus_ratio": ukraine_focus,
        "geographic_breadth_entropy": geo_entropy,
        "total_toponyms": len(all_locations),
        "unique_toponyms": len(location_counts),
        "chapter_locations": {k: Counter(v).most_common(10) for k, v in chapter_locations.items()},
    }


def build_comention_network(book_chapters, min_weight=2):
    """Build place co-mention network (places in same chapter)."""
    edges = Counter()
    for ch in book_chapters:
        locs = extract_geographic_entities(ch.get("spacy_ner", []))
        locs += extract_geographic_entities(ch.get("xlm_ner", []))
        normalized = list(set(normalize_toponym(l) for l in locs))
        for a, b in combinations(sorted(normalized), 2):
            edges[(a, b)] += 1

    G = nx.Graph()
    for (a, b), w in edges.items():
        if w >= min_weight:
            G.add_edge(a, b, weight=w)
    return G


def plot_toponym_comparison(results):
    """Bar chart comparing top toponyms across authors."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Top 15 Toponyms by Author", fontsize=14, fontweight="bold")

    for ax, (book_id, data) in zip(axes.flat, results.items()):
        locs = data["location_counts"]
        top = dict(list(locs.items())[:15])
        if top:
            bars = ax.barh(list(reversed(top.keys())), list(reversed(top.values())),
                          color=sns.color_palette("Set2")[list(results.keys()).index(book_id)])
            ax.set_title(f"{BOOK_META[book_id]['author']}", fontsize=11)
            ax.set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_toponyms_by_author.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_toponyms_by_author.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved toponym comparison figure")


def plot_regional_coverage(results):
    """Stacked bar chart of regional coverage."""
    regions = ["Western", "Central", "Eastern", "Southern"]
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results))
    width = 0.6
    bottom = np.zeros(len(results))
    colors = sns.color_palette("Set2", len(regions))

    for i, region in enumerate(regions):
        vals = []
        for book_id in results:
            rc = results[book_id]["region_counts"]
            total = sum(rc.get(r, 0) for r in regions)
            vals.append(rc.get(region, 0) / total * 100 if total > 0 else 0)
        ax.bar(x, vals, width, bottom=bottom, label=region, color=colors[i])
        bottom += vals

    ax.set_ylabel("Percentage of Ukraine toponyms")
    ax.set_title("Regional Coverage of Ukraine by Author")
    ax.set_xticks(x)
    ax.set_xticklabels([BOOK_META[b]["author"].split()[-1] for b in results])
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_regional_coverage.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_regional_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved regional coverage figure")


def plot_comention_network(networks, results):
    """Plot co-mention networks for each author."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Place Co-mention Networks", fontsize=14, fontweight="bold")

    for ax, (book_id, G) in zip(axes.flat, networks.items()):
        if len(G.nodes()) == 0:
            ax.set_title(f"{BOOK_META[book_id]['author']}: No network")
            ax.axis("off")
            continue

        pos = nx.spring_layout(G, k=2, seed=42)
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w = max(weights) if weights else 1

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3,
                               width=[w / max_w * 3 for w in weights])
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=100,
                               node_color="steelblue", alpha=0.7)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
        ax.set_title(f"{BOOK_META[book_id]['author']}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_comention_networks.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_comention_networks.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved co-mention network figure")


def main():
    print("=" * 60)
    print("STAGE 3: Toponym & Geographic Analysis")
    print("=" * 60)

    ling_data = load_json(OUTPUT_DIR / "stage2_linguistic_features.json")

    results = {}
    networks = {}

    for book_id, book_data in ling_data.items():
        print(f"\nAnalyzing {book_id}...")
        geo = analyze_book_geography(book_data["chapters"])
        results[book_id] = geo
        networks[book_id] = build_comention_network(book_data["chapters"])

        print(f"  Toponyms: {geo['total_toponyms']} total, {geo['unique_toponyms']} unique")
        print(f"  Ukraine focus: {geo['ukraine_focus_ratio']:.2%}")
        print(f"  Geographic breadth: {geo['geographic_breadth_entropy']:.3f}")
        print(f"  Top 5: {list(geo['location_counts'].items())[:5]}")

    # Generate figures
    plot_toponym_comparison(results)
    plot_regional_coverage(results)
    plot_comention_network(networks, results)

    save_json(results, OUTPUT_DIR / "stage3_geographic.json")
    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
