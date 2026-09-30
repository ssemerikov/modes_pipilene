#!/usr/bin/env python3
"""How separable are the texts, and where does the fifth text fall?

Three questions, in this order:

1. Can a classifier tell the five texts apart from the measures of a single
   unit? With one text per mode this is identification of a *text*; it cannot
   separate mode from author, and the article says so.
2. Do the measures cluster by national context without being told the labels?
3. A model trained on the four texts of the first submission has never seen the
   fifth. Which of the four does it take the fifth text's units for -- on form
   measures, and on content measures? The readings of each outcome were fixed
   before the model was run (see the plan recorded in REVISION_LOG.md).

Scaling is fitted inside every training fold (sklearn Pipeline); the first
version of the pipeline scaled all rows before cross-validating.

Writes OUT/classify.json and OUT/pca_coords.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             balanced_accuracy_score, confusion_matrix)
from sklearn.model_selection import (LeaveOneOut, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score, permutation_test_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (BOOK_ORDER, CONTENT, FEATURES, FORM, ORIGINAL_FOUR, load_matrix,
                    save_json)
from paths import OUT, SEED

NO_WAR = [f for f in FEATURES if not f.startswith("war_")]
GENRE = ["diary_marker_density", "travel_marker_density", "historical_marker_density"]
# The last three blocks answer Reviewer #1 (Q1.4): is a content result anything
# more than the war vocabulary?
BLOCKS = {"all": FEATURES, "form": FORM, "content": CONTENT,
          "without war vocabulary": NO_WAR,
          "content without war vocabulary": [f for f in CONTENT if f in NO_WAR],
          "genre markers": GENRE}
JOBS = int(__import__("os").environ.get("JOBS", "5"))   # parallel folds, not trees


def rf():
    return make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=1))


def lda():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())


MODELS = {"random_forest": rf, "lda": lda}


def evaluate(X, y, labels, n_perm: dict) -> dict:
    out = {}
    for name, make in MODELS.items():
        pred = cross_val_predict(make(), X, y, cv=LeaveOneOut(), n_jobs=JOBS)
        rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
        folds = cross_val_score(make(), X, y, cv=rkf, scoring="accuracy", n_jobs=JOBS)
        res = dict(loo_accuracy=float(accuracy_score(y, pred)),
                   loo_correct=int((pred == y).sum()), n=int(len(y)),
                   loo_balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                   kfold_mean=float(folds.mean()), kfold_sd=float(folds.std(ddof=1)),
                   confusion=confusion_matrix(y, pred, labels=labels).tolist())
        if n_perm.get(name):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            score, perm, p = permutation_test_score(
                make(), X, y, cv=cv, n_permutations=n_perm[name], random_state=SEED,
                scoring="accuracy", n_jobs=JOBS)
            res.update(perm_score=float(score), perm_null_mean=float(perm.mean()),
                       perm_null_max=float(perm.max()), perm_p=float(p),
                       perm_n=int(n_perm[name]))
        out[name] = res
    return out


def placement(df, cols, within_language: bool) -> dict:
    """Train on the original four texts, place the units of the fifth."""
    train = df[df.book.isin(ORIGINAL_FOUR)]
    test = df[df.book == "miller"]
    Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
    if within_language:
        # Remove each language's own level before comparing profiles: centre and
        # scale the English training texts together, the German ones together,
        # and express the fifth (English) text on the English scale.
        Xtr = Xtr.copy()
        for lang in ("en", "de"):
            mask = (train.lang == lang).to_numpy()
            mu, sd = Xtr[mask].mean(0), Xtr[mask].std(0, ddof=0)
            sd[sd == 0] = 1
            Xtr[mask] = (Xtr[mask] - mu) / sd
            if lang == "en":
                Xte = (Xte - mu) / sd
    y = train.book.astype(str).to_numpy()
    out = {}
    for name, make in MODELS.items():
        model = make().fit(Xtr, y)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte).mean(0)
        classes = list(model.classes_)
        by_period = {}
        for p in ("P1", "P2", "P3"):
            m = (test.period == p).to_numpy()
            by_period[p] = {b: int((pred[m] == b).sum()) for b in ORIGINAL_FOUR}
        out[name] = dict(counts={b: int((pred == b).sum()) for b in ORIGINAL_FOUR},
                         share={b: float((pred == b).mean()) for b in ORIGINAL_FOUR},
                         mean_probability={b: float(proba[classes.index(b)])
                                           for b in ORIGINAL_FOUR},
                         by_period=by_period, n=int(len(pred)))
        out[name]["american_share"] = float(np.isin(pred, ["applebaum", "nicolay"]).mean())
    return out


def main() -> None:
    df = load_matrix()
    y = df.book.astype(str).to_numpy()
    out = dict(n=int(len(df)), majority_share=float(pd.Series(y).value_counts(normalize=True).max()),
               labels=BOOK_ORDER)

    # 1. five texts
    out["five_texts"] = {}
    for blk, cols in BLOCKS.items():
        n_perm = {"lda": 1000, "random_forest": 200} if blk == "all" else {}
        out["five_texts"][blk] = evaluate(df[cols].to_numpy(float), y, BOOK_ORDER, n_perm)
        print(blk, {m: round(r["loo_accuracy"], 3) for m, r in out["five_texts"][blk].items()})

    # the four texts of the first submission, for comparison with its 96.4 %
    four = df[df.book.isin(ORIGINAL_FOUR)]
    out["four_texts"] = evaluate(four[FEATURES].to_numpy(float),
                                 four.book.astype(str).to_numpy(), ORIGINAL_FOUR, {})
    out["four_texts"]["n"] = int(len(four))

    # national context, two classes, and whether it generalises to an unseen text
    ctx = df.context.to_numpy()
    out["context"] = {}
    for blk, cols in BLOCKS.items():
        held = {}
        for b in BOOK_ORDER:
            tr, te = df[df.book != b], df[df.book == b]
            model = lda().fit(tr[cols].to_numpy(float), tr.context.to_numpy())
            held[b] = float((model.predict(te[cols].to_numpy(float)) == te.context.to_numpy()).mean())
        out["context"][blk] = dict(leave_one_text_out=held)

    # 2. unsupervised structure
    Z = StandardScaler().fit_transform(df[FEATURES].to_numpy(float))
    clu = {}
    for k, target, name in ((5, y, "text"), (2, ctx, "context"),
                            (3, df.period_written.to_numpy(), "period_written")):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Z)
        wa = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
        clu[name] = dict(k=k, kmeans_ari=float(adjusted_rand_score(target, km)),
                         ward_ari=float(adjusted_rand_score(target, wa)))
        if name == "context":
            clu[name]["kmeans_crosstab"] = pd.crosstab(df.book.astype(str), km).reindex(
                BOOK_ORDER).to_dict("index")
    out["clustering"] = clu

    # 3. placement of the fifth text
    out["placement"] = {}
    for blk, cols in BLOCKS.items():
        out["placement"][blk] = dict(raw=placement(df, cols, False),
                                     within_language=placement(df, cols, True))
        r = out["placement"][blk]
        print(f"placement {blk:8s} raw RF {r['raw']['random_forest']['counts']} | "
              f"within-language RF {r['within_language']['random_forest']['counts']}")

    # Feature importance is not reported. The first version's importance figure
    # ranked measures by how a forest fitted to all units used them, which says
    # little about a text; the article reports effect sizes per measure instead.

    # PCA for the figure
    pca = PCA(n_components=3, random_state=SEED).fit(Z)
    coords = pca.transform(Z)
    pd.DataFrame(dict(unit_id=df.unit_id, book=df.book.astype(str), context=df.context,
                      period=df.period, pc1=coords[:, 0], pc2=coords[:, 1],
                      pc3=coords[:, 2])).to_csv(OUT / "pca_coords.csv", index=False)
    out["pca"] = dict(explained=[float(v) for v in pca.explained_variance_ratio_],
                      loadings={f: [float(pca.components_[i, j]) for i in range(3)]
                                for j, f in enumerate(FEATURES)})
    save_json(out, "classify.json")


if __name__ == "__main__":
    main()
