#!/usr/bin/env python3
"""Join unit metadata, lexical measures and sentiment into the feature matrix.

OUT/feature_matrix.csv has one row per unit of at least 1,000 words: eight
columns describing the unit, the 18 measures, and four auxiliary columns that
are reported but do not enter the classifier.
"""
from __future__ import annotations

import pandas as pd

from features import FEATURES
from paths import OUT

META = ["unit_id", "book", "context", "lang", "mode", "period_written", "period",
        "ukraine", "words"]
AUX = ["first_person_authorial", "quoted_share", "perfect_share",
       "war_weapons_density", "war_military_density"]


def main() -> None:
    units = pd.read_csv(OUT / "units.csv")
    lex = pd.read_csv(OUT / "features_lexical.csv")
    sent = pd.read_csv(OUT / "sentiment.csv")
    df = units.merge(lex, on="unit_id", validate="1:1").merge(sent, on="unit_id", validate="1:1")
    n_all = len(df)
    df = df[~df["short"]].copy()
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"measures missing from the inputs: {missing}")
    if df[FEATURES].isna().any().any():
        raise SystemExit("missing values in the feature matrix:\n"
                         + df.loc[df[FEATURES].isna().any(axis=1), ["unit_id"] + FEATURES].to_string())
    df[META + FEATURES + AUX].to_csv(OUT / "feature_matrix.csv", index=False)
    print(f"wrote {OUT / 'feature_matrix.csv'}: {len(df)} units x {len(FEATURES)} measures "
          f"({n_all - len(df)} short units left out)")
    print(df.groupby("book", sort=False).size().to_string())


if __name__ == "__main__":
    main()
