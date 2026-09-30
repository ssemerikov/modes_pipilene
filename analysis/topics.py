#!/usr/bin/env python3
"""What the Ukraine-set text is about, beyond war: themes shared across languages.

Reviewer #1 asked for content differences other than war vocabulary. The
Ukraine-set units are cut into passages of about a hundred words, each passage
is embedded with a multilingual sentence model, and the passages are clustered.
A cluster is a theme if passages from both languages fall into it.

Two safeguards against the clusters simply rediscovering the language or the
book:

* each language's mean vector is subtracted before clustering, which removes
  the component of the embedding that encodes "this is German";
* for every theme the output reports how its passages divide between the two
  languages and among the five books, so a theme carried by one book is visible
  as such.

Themes are named by hand from their most distinctive words (THEME_LABELS); the
words themselves are in the output for anyone who would name them differently.

Reads   WORK/parsed/<unit_id>.json, OUT/units.csv
Writes  WORK/topic_embeddings.npz (cache), OUT/topics.json, OUT/topic_passages.csv
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import BOOK_ORDER, save_json
from paths import OUT, SEED, WORK

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PASSAGE_WORDS = 100
K = 12
K_RANGE = (8, 10, 12, 14, 16)

# Named after reading the distinctive words of each cluster (K = 12, SEED as in
# paths.py). theme number -> (label, group). The grouping into four families is
# ours; the twelve themes and their words are reported so that it can be checked.
GROUPS = ["War", "Politics and nation", "People and everyday life", "Place and movement"]
THEME_LABELS: dict[int, tuple[str, str]] = {
    0: ("Encounters: language, school, acquaintances", "People and everyday life"),
    1: ("Russia and geopolitics", "Politics and nation"),
    2: ("Family and personal stories", "People and everyday life"),
    3: ("Townscape and history", "Place and movement"),
    4: ("Combat and bombardment", "War"),
    5: ("Nation, language, independence", "Politics and nation"),
    6: ("Conversation and small scenes", "People and everyday life"),
    7: ("Music, leisure, daily routine", "People and everyday life"),
    8: ("Transport and lodging", "Place and movement"),
    9: ("Military operations", "War"),
    10: ("Soldiers and casualties", "War"),
    11: ("Domestic politics", "Politics and nation"),
}
# The first two English and German words each label was read from; the build
# stops if a re-run no longer produces these clusters.
THEME_ANCHORS = {0: ("english", "school"), 1: ("russia", "putin"), 3: ("kamenets", "city"),
                 4: ("fire", "soldiers"), 8: ("train", "station"), 9: ("troops", "forces"),
                 11: ("yanukovych", "zelensky")}


def passages(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, u in units.iterrows():
        parsed = json.loads((WORK / "parsed" / f"{u.unit_id}.json").read_text(encoding="utf8"))
        cur, n = [], 0
        for s in parsed["sentences"]:
            cur.append(s["text"])
            n += s["n"]
            if n >= PASSAGE_WORDS:
                rows.append(dict(unit_id=u.unit_id, book=u.book, lang=u.lang, period=u.period,
                                 words=n, text=" ".join(cur)))
                cur, n = [], 0
        if n >= PASSAGE_WORDS // 2:
            rows.append(dict(unit_id=u.unit_id, book=u.book, lang=u.lang, period=u.period,
                             words=n, text=" ".join(cur)))
    return pd.DataFrame(rows)


def embed(texts: list[str]) -> np.ndarray:
    cache = WORK / "topic_embeddings.npz"
    if cache.exists():
        z = np.load(cache)
        if z["n"] == len(texts):
            return z["emb"]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")
    emb = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    np.savez_compressed(cache, emb=emb, n=len(texts))
    return emb


def distinctive_terms(df: pd.DataFrame, labels: np.ndarray, k: int, top: int = 15) -> dict:
    """Log-odds of each word in a theme against the rest of the same language."""
    stop = {"en": spacy.blank("en").Defaults.stop_words, "de": spacy.blank("de").Defaults.stop_words}
    tok = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")
    counts = {lang: [Counter() for _ in range(k)] for lang in ("en", "de")}
    for text, lang, lab in zip(df.text, df.lang, labels):
        words = [w.lower() for w in tok.findall(text)]
        counts[lang][lab].update(w for w in words if w not in stop[lang])
    out = {}
    for c in range(k):
        out[c] = {}
        for lang in ("en", "de"):
            tot = Counter()
            for cc in counts[lang]:
                tot.update(cc)
            n_c, n_all = sum(counts[lang][c].values()), sum(tot.values())
            scores = []
            for w, a in counts[lang][c].items():
                if a < 5:
                    continue
                b = tot[w] - a
                lo = math.log((a + 0.5) / (n_c - a + 0.5)) - math.log((b + 0.5) / (n_all - n_c - b + 0.5))
                scores.append((lo / math.sqrt(1 / (a + 0.5) + 1 / (b + 0.5)), w))
            out[c][lang] = [w for _, w in sorted(scores, reverse=True)[:top]]
    return out


def main() -> None:
    units = pd.read_csv(OUT / "units.csv")
    units = units[units.ukraine & ~units.short]
    df = passages(units)
    emb = embed(df.text.tolist())
    # remove the language component, then renormalise
    X = emb.copy()
    for lang in ("en", "de"):
        m = (df.lang == lang).to_numpy()
        X[m] -= X[m].mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    sil = {}
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
        sil[k] = float(silhouette_score(X, lab, sample_size=2000, random_state=SEED))
    labels = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit_predict(X)
    df["theme"] = labels
    terms = distinctive_terms(df, labels, K)
    for c, anchors in THEME_ANCHORS.items():
        if not set(anchors) <= set(terms[c]["en"][:6]):
            raise SystemExit(f"theme {c} no longer matches its label: {terms[c]['en'][:6]}")
    df["group"] = [THEME_LABELS[c][1] for c in labels]

    # how far the partition depends on the random start
    from sklearn.metrics import adjusted_rand_score
    ref_group = np.array([THEME_LABELS[c][1] for c in labels])
    ari, alt_shares = [], []
    for i in range(1, 11):
        alt = KMeans(n_clusters=K, n_init=20, random_state=SEED + i).fit_predict(X)
        ari.append(adjusted_rand_score(labels, alt))
        # give each cluster of the alternative run the family most of its passages
        # have in the reference run, and recompute the family shares per text
        fam = np.empty(len(alt), dtype=object)
        for c in range(K):
            m = alt == c
            fam[m] = Counter(ref_group[m]).most_common(1)[0][0]
        w = df.words.to_numpy(float)
        alt_shares.append({b: {g: float(w[(df.book == b).to_numpy() & (fam == g)].sum()
                                         / w[(df.book == b).to_numpy()].sum())
                               for g in GROUPS} for b in BOOK_ORDER})

    words_by_book = df.groupby("book").words.sum()
    themes = {}
    for c in range(K):
        sub = df[df.theme == c]
        share = {b: float(sub.loc[sub.book == b, "words"].sum() / words_by_book[b])
                 for b in BOOK_ORDER}
        lang_mix = {l: float(sub.loc[sub.lang == l, "words"].sum() / sub.words.sum())
                    for l in ("en", "de")}
        label, group = THEME_LABELS[c]
        themes[c] = dict(label=label, group=group, passages=int(len(sub)),
                         share_by_book=share, language_mix=lang_mix,
                         terms_en=terms[c]["en"], terms_de=terms[c]["de"])
    # within-author period shares (Miller, Brumme)
    within = {}
    for b in ("miller", "brumme"):
        sub = df[df.book == b]
        within[b] = {}
        for p in sorted(sub.period.dropna().unique()):
            sp = sub[sub.period == p]
            within[b][p] = {int(c): float(sp.loc[sp.theme == c, "words"].sum() / sp.words.sum())
                            for c in range(K)}
    # the four families: share of each text's Ukraine-set words, with an interval
    # from resampling the text's units
    unit_tab = (df.pivot_table(index=["book", "unit_id", "period"], columns="group",
                               values="words", aggfunc="sum", fill_value=0, dropna=False)
                  .reset_index())
    unit_tab = unit_tab[unit_tab[GROUPS].sum(axis=1) > 0]
    unit_tab.to_csv(OUT / "topic_units.csv", index=False)
    rng = np.random.default_rng(SEED)
    groups = {}
    for b in BOOK_ORDER:
        sub = unit_tab[unit_tab.book == b]
        tot = sub[GROUPS].to_numpy(float)
        share = tot.sum(0) / tot.sum()
        boots = []
        for _ in range(2000):
            t = tot[rng.integers(0, len(tot), len(tot))]
            boots.append(t.sum(0) / t.sum())
        boots = np.array(boots)
        groups[b] = {g: dict(share=float(share[i]), lo=float(np.quantile(boots[:, i], 0.025)),
                             hi=float(np.quantile(boots[:, i], 0.975)),
                             seeds_min=float(min(a[b][g] for a in alt_shares)),
                             seeds_max=float(max(a[b][g] for a in alt_shares)))
                     for i, g in enumerate(GROUPS)}
    within_groups = {}
    for b in ("miller", "brumme"):
        sub = unit_tab[unit_tab.book == b]
        within_groups[b] = {}
        for p in sorted(sub.period.dropna().unique()):
            t = sub.loc[sub.period == p, GROUPS].to_numpy(float)
            within_groups[b][p] = {g: float(v) for g, v in zip(GROUPS, t.sum(0) / t.sum())}

    out = dict(groups=GROUPS, group_share_by_book=groups, within_author_groups=within_groups,
               seed_stability_ari=dict(mean=float(np.mean(ari)), min=float(np.min(ari))),
               model=MODEL, passage_words=PASSAGE_WORDS, k=K, n_passages=int(len(df)),
               passages_by_book={b: int((df.book == b).sum()) for b in BOOK_ORDER},
               silhouette=sil, themes=themes, within_author=within,
               # family shares per text in each of the ten alternative runs, so
               # that any claim about an ordering can be checked run by run
               seed_shares=alt_shares)
    save_json(out, "topics.json")
    df.drop(columns="text").to_csv(OUT / "topic_passages.csv", index=False)

    for c in range(K):
        t = themes[c]
        print(f"\n[{c}] {t['label']}  passages={t['passages']}  EN/DE={t['language_mix']['en']:.2f}/"
              f"{t['language_mix']['de']:.2f}")
        print("   by book: " + "  ".join(f"{b[:4]} {t['share_by_book'][b]:.3f}" for b in BOOK_ORDER))
        print("   EN: " + ", ".join(t["terms_en"]))
        print("   DE: " + ", ".join(t["terms_de"]))
    print("\nsilhouette by k:", {k: round(v, 3) for k, v in sil.items()},
          "| ARI across seeds: mean", round(float(np.mean(ari)), 3))
    for b in BOOK_ORDER:
        print(f"  {b:10s} " + "  ".join(f"{g}: {groups[b][g]['share']:.2f} "
              f"[{groups[b][g]['lo']:.2f}, {groups[b][g]['hi']:.2f}] "
              f"seeds {groups[b][g]['seeds_min']:.2f}-{groups[b][g]['seeds_max']:.2f}" for g in GROUPS))
    for b, w in within_groups.items():
        for p, v in w.items():
            print(f"  {b} {p}: " + "  ".join(f"{g}: {x:.2f}" for g, x in v.items()))


if __name__ == "__main__":
    main()
