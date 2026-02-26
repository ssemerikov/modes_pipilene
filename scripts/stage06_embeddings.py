#!/usr/bin/env python3
"""
Stage 6: Semantic Embedding Analysis
Multilingual paragraph embeddings, UMAP projection, HDBSCAN clustering,
cross-text semantic bridges.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN

from utils import OUTPUT_DIR, FIGURES_DIR, BOOK_META, load_json, save_json


def get_paragraphs_with_meta(corpus):
    """Get paragraphs with full metadata."""
    paragraphs = []
    meta = []
    for book_id, book_data in corpus.items():
        for ch_idx, ch in enumerate(book_data["chapters"]):
            paras = [p.strip() for p in ch["text"].split('\n\n') if len(p.strip()) > 50]
            for p_idx, para in enumerate(paras):
                paragraphs.append(para)
                meta.append({
                    "book_id": book_id,
                    "chapter": ch["chapter"],
                    "lang": book_data["metadata"]["lang"],
                    "author": book_data["metadata"]["author"],
                    "model": book_data["metadata"]["model"],
                })
    return paragraphs, meta


def compute_book_similarity(embeddings, meta):
    """Compute book-level cosine similarity matrix."""
    book_ids = list(BOOK_META.keys())
    book_embeddings = {}

    for book_id in book_ids:
        indices = [i for i, m in enumerate(meta) if m["book_id"] == book_id]
        if indices:
            book_embeddings[book_id] = embeddings[indices].mean(axis=0)

    n = len(book_ids)
    sim_matrix = np.zeros((n, n))
    for i, a in enumerate(book_ids):
        for j, b in enumerate(book_ids):
            if a in book_embeddings and b in book_embeddings:
                sim = cosine_similarity(
                    book_embeddings[a].reshape(1, -1),
                    book_embeddings[b].reshape(1, -1)
                )[0, 0]
                sim_matrix[i, j] = sim

    return sim_matrix, book_ids


def find_semantic_bridges(paragraphs, embeddings, meta, top_k=20):
    """Find most similar passages across different books."""
    n = len(paragraphs)
    bridges = []

    # Sample for efficiency: compare centroids or use batch cosine
    # Group by book
    book_indices = {}
    for i, m in enumerate(meta):
        book_indices.setdefault(m["book_id"], []).append(i)

    book_ids = list(book_indices.keys())

    for a_idx in range(len(book_ids)):
        for b_idx in range(a_idx + 1, len(book_ids)):
            a_id = book_ids[a_idx]
            b_id = book_ids[b_idx]
            a_inds = book_indices[a_id]
            b_inds = book_indices[b_id]

            # Compute pairwise similarities (sample if too large)
            max_sample = 200
            if len(a_inds) > max_sample:
                a_sample = np.random.choice(a_inds, max_sample, replace=False)
            else:
                a_sample = a_inds
            if len(b_inds) > max_sample:
                b_sample = np.random.choice(b_inds, max_sample, replace=False)
            else:
                b_sample = b_inds

            sims = cosine_similarity(embeddings[a_sample], embeddings[b_sample])
            top_pairs = np.unravel_index(np.argsort(sims.ravel())[-5:], sims.shape)

            for ai, bi in zip(top_pairs[0], top_pairs[1]):
                real_a = a_sample[ai]
                real_b = b_sample[bi]
                bridges.append({
                    "book_a": a_id,
                    "book_b": b_id,
                    "similarity": float(sims[ai, bi]),
                    "text_a": paragraphs[real_a][:200],
                    "text_b": paragraphs[real_b][:200],
                    "chapter_a": meta[real_a]["chapter"],
                    "chapter_b": meta[real_b]["chapter"],
                })

    bridges.sort(key=lambda x: -x["similarity"])
    return bridges[:top_k]


def plot_umap_scatter(embeddings_2d, meta):
    """UMAP 2D scatter plot colored by author."""
    fig, ax = plt.subplots(figsize=(12, 9))

    colors = sns.color_palette("Set2", 4)
    book_ids = list(BOOK_META.keys())

    for i, book_id in enumerate(book_ids):
        indices = [j for j, m in enumerate(meta) if m["book_id"] == book_id]
        ax.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            c=[colors[i]],
            label=BOOK_META[book_id]["author"].split()[-1],
            alpha=0.4,
            s=8,
        )

    ax.legend(fontsize=11, markerscale=3)
    ax.set_title("UMAP Projection of Paragraph Embeddings (Multilingual)", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_umap_scatter.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_umap_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved UMAP scatter plot")


def plot_similarity_heatmap(sim_matrix, book_ids):
    """Heatmap of inter-book cosine similarity."""
    labels = [BOOK_META[b]["author"].split()[-1] for b in book_ids]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(sim_matrix, annot=True, fmt=".3f",
                xticklabels=labels, yticklabels=labels,
                cmap="YlOrRd", vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title("Inter-Book Semantic Similarity (Cosine)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_book_similarity.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_book_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved similarity heatmap")


def main():
    print("=" * 60)
    print("STAGE 6: Semantic Embedding Analysis")
    print("=" * 60)

    corpus = load_json(OUTPUT_DIR / "stage1_corpus.json")
    paragraphs, meta = get_paragraphs_with_meta(corpus)
    print(f"Total paragraphs: {len(paragraphs)}")

    # Load embeddings from Stage 5 if available, otherwise compute
    emb_path = OUTPUT_DIR / "stage5_embeddings.npz"
    if emb_path.exists():
        print("Loading pre-computed embeddings from Stage 5...")
        data = np.load(emb_path)
        embeddings = data["embeddings"]
        if len(embeddings) != len(paragraphs):
            print(f"  Mismatch: {len(embeddings)} embeddings vs {len(paragraphs)} paragraphs")
            print("  Recomputing embeddings...")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            texts = [p[:1000] for p in paragraphs]
            embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    else:
        print("Computing embeddings...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        texts = [p[:1000] for p in paragraphs]
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    print(f"Embeddings shape: {embeddings.shape}")

    # UMAP 2D projection
    print("Computing UMAP 2D projection...")
    umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.1,
                    metric="cosine", random_state=42)
    embeddings_2d = umap_2d.fit_transform(embeddings)

    # HDBSCAN clustering
    print("Running HDBSCAN clustering...")
    clusterer = HDBSCAN(min_cluster_size=20, min_samples=5)
    cluster_labels = clusterer.fit_predict(embeddings_2d)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"  Found {n_clusters} clusters")

    # Book similarity
    print("Computing book-level similarity...")
    sim_matrix, book_ids = compute_book_similarity(embeddings, meta)

    # Semantic bridges
    print("Finding cross-text semantic bridges...")
    bridges = find_semantic_bridges(paragraphs, embeddings, meta)

    # Plots
    plot_umap_scatter(embeddings_2d, meta)
    plot_similarity_heatmap(sim_matrix, book_ids)

    # Save results
    results = {
        "n_paragraphs": len(paragraphs),
        "embedding_dim": int(embeddings.shape[1]),
        "n_clusters": n_clusters,
        "similarity_matrix": sim_matrix.tolist(),
        "similarity_labels": book_ids,
        "semantic_bridges": bridges[:10],
    }

    save_json(results, OUTPUT_DIR / "stage6_embeddings.json")

    # Save full embeddings
    np.savez_compressed(
        OUTPUT_DIR / "stage6_embeddings.npz",
        embeddings=embeddings,
        embeddings_2d=embeddings_2d,
        cluster_labels=cluster_labels,
    )

    print("\nStage 6 complete.")


if __name__ == "__main__":
    main()
