#!/usr/bin/env python3
"""Shared helpers: the assembled matrix, effect sizes, resampling, multiplicity."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from features import CONTENT, FEATURES, FORM, MEASURES  # noqa: F401  (re-exported)
from paths import OUT, SEED
from units import BOOK_META, BOOK_ORDER, PERIODS  # noqa: F401

AUTHOR = {k: v["author"] for k, v in BOOK_META.items()}
CONTEXT = {k: v["context"] for k, v in BOOK_META.items()}
ORIGINAL_FOUR = ["applebaum", "nicolay", "brumme", "orth"]

# One colour per text, used in every figure (colour-blind-safe Okabe-Ito set).
COLOUR = {"applebaum": "#0072B2", "nicolay": "#56B4E9", "miller": "#009E73",
          "brumme": "#D55E00", "orth": "#E69F00"}
MARKER = {"American": "o", "German": "s"}


def load_matrix() -> pd.DataFrame:
    """Units in the feature matrix (short units excluded), with all measures."""
    df = pd.read_csv(OUT / "feature_matrix.csv")
    df["book"] = pd.Categorical(df["book"], BOOK_ORDER, ordered=True)
    return df.sort_values(["book", "unit_id"]).reset_index(drop=True)


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b): +1 when every a exceeds every b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def boot_ci(x, stat=np.mean, n: int = 5000, seed: int = SEED, alpha: float = 0.05):
    """Percentile bootstrap over units."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = stat(x[idx], axis=1)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def boot_delta_ci(a, b, n: int = 2000, seed: int = SEED, alpha: float = 0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    vals = [cliffs_delta(a[rng.integers(0, len(a), len(a))],
                         b[rng.integers(0, len(b), len(b))]) for _ in range(n)]
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def save_json(obj, name: str) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"wrote {OUT / name}")


def load_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf8"))
