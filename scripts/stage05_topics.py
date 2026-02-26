#!/usr/bin/env python3
"""
Stage 5: Topic Modeling
LDA (gensim) + BERTopic with multilingual embeddings.
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
import spacy
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from collections import defaultdict

from utils import OUTPUT_DIR, FIGURES_DIR, BOOK_META, load_json, save_json


def get_paragraphs(corpus):
    """Extract paragraph-level documents with metadata."""
    paragraphs = []
    para_meta = []  # (book_id, chapter_idx, para_idx)

    for book_id, book_data in corpus.items():
        for ch_idx, ch in enumerate(book_data["chapters"]):
            paras = [p.strip() for p in ch["text"].split('\n\n') if len(p.strip()) > 50]
            for p_idx, para in enumerate(paras):
                paragraphs.append(para)
                para_meta.append({
                    "book_id": book_id,
                    "chapter": ch["chapter"],
                    "lang": book_data["metadata"]["lang"],
                })

    return paragraphs, para_meta


def lemmatize_for_lda(paragraphs, para_meta):
    """Lemmatize paragraphs using spaCy, keeping nouns/verbs/adjectives."""
    print("  Loading spaCy models for lemmatization...")
    nlp_en = spacy.load("en_core_web_sm", disable=["ner", "textcat", "parser"])
    nlp_de = spacy.load("de_core_news_sm", disable=["ner", "textcat", "parser"])
    nlp_en.max_length = 2_000_000
    nlp_de.max_length = 2_000_000

    keep_pos = {"NOUN", "VERB", "ADJ"}
    docs = []

    for i, (para, meta) in enumerate(zip(paragraphs, para_meta)):
        nlp = nlp_en if meta["lang"] == "en" else nlp_de
        doc = nlp(para[:10000])
        tokens = [
            token.lemma_.lower() for token in doc
            if token.pos_ in keep_pos
            and not token.is_stop
            and len(token.text) > 2
            and token.is_alpha
        ]
        docs.append(tokens)
        if (i + 1) % 500 == 0:
            print(f"    Lemmatized {i+1}/{len(paragraphs)} paragraphs...")

    return docs


def run_lda(docs, k_range=(5, 26)):
    """Run LDA with coherence optimization."""
    print("  Building LDA dictionary and corpus...")
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    print(f"  Dictionary: {len(dictionary)} terms, {len(bow_corpus)} documents")

    # Sweep k values
    coherence_scores = {}
    best_k = k_range[0]
    best_score = -1
    best_model = None

    for k in range(k_range[0], min(k_range[1], 21), 5):  # 5, 10, 15, 20
        print(f"    Training LDA k={k}...")
        model = models.LdaModel(
            bow_corpus, num_topics=k, id2word=dictionary,
            passes=10, random_state=42, alpha="auto", eta="auto",
        )
        cm = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence="c_v")
        score = cm.get_coherence()
        coherence_scores[k] = score
        print(f"      c_v coherence: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    return best_model, dictionary, bow_corpus, coherence_scores, best_k


def run_bertopic(paragraphs, para_meta):
    """Run BERTopic with multilingual embeddings."""
    print("  Running BERTopic...")
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Truncate long paragraphs for embedding
    texts = [p[:1000] for p in paragraphs]

    print("    Computing embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=False, batch_size=64)

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5,
                            metric="euclidean", prediction_data=True)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics="auto",
        verbose=False,
    )

    topics, probs = topic_model.fit_transform(texts, embeddings)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    topic_words = {}
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        topic_words[int(topic_id)] = [(w, float(s)) for w, s in words[:10]]

    return topics, probs, topic_words, embeddings


def compute_author_topic_distribution(topics, para_meta, n_topics):
    """Compute topic distribution per author."""
    author_topics = defaultdict(lambda: defaultdict(int))
    for topic, meta in zip(topics, para_meta):
        if topic >= 0:
            author_topics[meta["book_id"]][topic] += 1

    # Normalize
    distributions = {}
    for book_id, counts in author_topics.items():
        total = sum(counts.values())
        dist = {t: counts.get(t, 0) / total for t in range(n_topics)}
        distributions[book_id] = dist

    return distributions


def plot_coherence(coherence_scores):
    """Plot coherence vs number of topics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(coherence_scores.keys())
    scores = [coherence_scores[k] for k in ks]
    ax.plot(ks, scores, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel("Number of Topics (k)")
    ax.set_ylabel("Coherence Score (c_v)")
    ax.set_title("LDA Topic Coherence")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_lda_coherence.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_lda_coherence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved coherence plot")


def plot_topic_heatmap(distributions, topic_words):
    """Heatmap of topic distributions per author."""
    n_topics = len(topic_words)
    if n_topics == 0:
        return

    # Use top topics (up to 15)
    display_topics = min(n_topics, 15)

    matrix = []
    authors = []
    for book_id in distributions:
        dist = distributions[book_id]
        row = [dist.get(t, 0) * 100 for t in range(display_topics)]
        matrix.append(row)
        authors.append(BOOK_META[book_id]["author"].split()[-1])

    # Topic labels
    topic_labels = []
    for t in range(display_topics):
        if t in topic_words:
            words = [w for w, s in topic_words[t][:3]]
            topic_labels.append(f"T{t}: {', '.join(words)}")
        else:
            topic_labels.append(f"T{t}")

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(np.array(matrix), annot=True, fmt=".1f",
                xticklabels=topic_labels, yticklabels=authors,
                cmap="YlOrRd", ax=ax)
    ax.set_title("BERTopic Distribution (%) by Author")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_topic_heatmap.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_topic_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved topic heatmap")


def main():
    print("=" * 60)
    print("STAGE 5: Topic Modeling")
    print("=" * 60)

    corpus = load_json(OUTPUT_DIR / "stage1_corpus.json")
    paragraphs, para_meta = get_paragraphs(corpus)
    print(f"Total paragraphs: {len(paragraphs)}")

    # LDA
    print("\nRunning LDA...")
    docs = lemmatize_for_lda(paragraphs, para_meta)
    lda_model, dictionary, bow_corpus, coherence_scores, best_k = run_lda(docs)

    lda_topics = []
    for t in range(best_k):
        words = lda_model.show_topic(t, topn=10)
        lda_topics.append([(w, float(s)) for w, s in words])

    plot_coherence(coherence_scores)

    # BERTopic
    print("\nRunning BERTopic...")
    bt_topics, bt_probs, bt_topic_words, embeddings = run_bertopic(paragraphs, para_meta)

    n_bt_topics = max(bt_topics) + 1 if bt_topics else 0
    bt_distributions = compute_author_topic_distribution(bt_topics, para_meta, n_bt_topics)
    plot_topic_heatmap(bt_distributions, bt_topic_words)

    # Save results
    results = {
        "lda": {
            "best_k": best_k,
            "coherence_scores": {str(k): v for k, v in coherence_scores.items()},
            "topics": {str(i): lda_topics[i] for i in range(len(lda_topics))},
        },
        "bertopic": {
            "n_topics": n_bt_topics,
            "topic_words": {str(k): v for k, v in bt_topic_words.items()},
            "author_distributions": {k: {str(t): v for t, v in d.items()}
                                     for k, d in bt_distributions.items()},
            "topic_assignments": [int(t) for t in bt_topics[:1000]],  # sample
        },
        "n_paragraphs": len(paragraphs),
    }

    save_json(results, OUTPUT_DIR / "stage5_topics.json")

    # Save embeddings for Stage 6
    np.savez_compressed(
        OUTPUT_DIR / "stage5_embeddings.npz",
        embeddings=embeddings,
    )
    print(f"  Saved embeddings: {embeddings.shape}")

    print("\nStage 5 complete.")


if __name__ == "__main__":
    main()
