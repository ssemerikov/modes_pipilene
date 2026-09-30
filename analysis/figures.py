#!/usr/bin/env python3
"""Every figure of the article, from the stored outputs only.

Reviewer #1 could not read several labels of the first submission's figures
("MATTR", "yules K", "war total density" beside "war conflict density"). Every
axis here carries the plain name of the measure from features.MEASURES, the
same name the article's table of measures defines. No figure carries a title
inside the plot: the caption is in the manuscript.

Writes source/fig_*.pdf (vector) and analysis/output/figures/fig_*.png (preview)
"""
from __future__ import annotations

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (BOOK_META, BOOK_ORDER, COLOUR, CONTENT, FEATURES, FORM, MEASURES,
                    load_json, load_matrix)
from paths import FIGDIR, OUT

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})
PREVIEW = OUT / "figures"
NAME = {b: BOOK_META[b]["author"] for b in BOOK_ORDER}
LABEL = {b: f"{BOOK_META[b]['author']} ({BOOK_META[b]['year']})" for b in BOOK_ORDER}
PERIOD_SHORT = {"P1": "before\nEuromaidan", "P2": "Euromaidan to\n23 Feb 2022",
                "P3": "from\n24 Feb 2022"}
PLAIN = {k: v[1] for k, v in MEASURES.items()}
ABBR = {"applebaum": "Appl.", "nicolay": "Nico.", "miller": "Mill.", "brumme": "Brum.", "orth": "Orth"}


def save(fig, name: str) -> None:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(PREVIEW / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  {name}")


# 1 ─ profiles ────────────────────────────────────────────────────────────────
def fig_profiles(df: pd.DataFrame) -> None:
    """Heat map: each text's mean on each measure, as a z-score over all units;
    the cell text is the mean itself in the measure's own unit."""
    means = df.groupby("book", observed=True)[FEATURES].mean().reindex(BOOK_ORDER)
    z = (means - df[FEATURES].mean()) / df[FEATURES].std(ddof=0)
    fig, ax = plt.subplots(figsize=(6.3, 6.4))
    im = ax.imshow(z.T.to_numpy(), cmap="RdBu_r", vmin=-1.8, vmax=1.8, aspect="auto")
    for i, f in enumerate(FEATURES):
        for j, b in enumerate(BOOK_ORDER):
            v = means.loc[b, f]
            txt = f"{v:.1f}" if f in ("mean_sent_len", "yules_k") else (
                f"{v:.2f}" if f in ("mattr", "past_tense_ratio", "present_tense_ratio",
                                    "noun_ratio", "verb_ratio", "adj_ratio",
                                    "positive_ratio", "negative_ratio")
                else f"{1000 * v:.1f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if abs(z.loc[b, f]) > 1.2 else "black")
    per_k = {"first_person_density", "diary_marker_density", "travel_marker_density",
             "historical_marker_density", "subjectivity", "war_total_density",
             "war_conflict_density", "war_suffering_density"}
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([PLAIN[f] + (" (per 1,000 words)" if f in per_k else "") for f in FEATURES])
    ax.set_xticks(range(len(BOOK_ORDER)))
    ax.set_xticklabels([f"{LABEL[b]}\n{BOOK_META[b]['context']}" for b in BOOK_ORDER])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for y in (len(FORM) - 0.5, len(FORM) + 2.5, len(FORM) + 5.5):
        ax.axhline(y, color="white", lw=3)
    ax.axvline(2.5, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("text mean, standard deviations from the mean of all units")
    save(fig, "fig_profiles")


# 2 ─ what separates texts: context, period, or neither ──────────────────────
def fig_effects(stats: dict) -> None:
    """Mean |Cliff's delta| per measure for three kinds of pair."""
    pw = stats["pairwise"]
    kinds = [
        ("same context, same period", "German diary vs German travel reportage, both written during the invasion", "#666666", "o"),
        ("same context, different period", "American texts written in different periods", "#0072B2", "s"),
        ("different context, same period", "American vs German texts written during the invasion", "#D55E00", "D"),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    ys = np.arange(len(FEATURES))[::-1]
    for k, (t, label, col, mk) in enumerate(kinds):
        pairs = [p for p, v in pw.items() if v["type"] == t]
        vals = [np.mean([abs(pw[p]["measures"][f]["delta"]) for p in pairs]) for f in FEATURES]
        ax.scatter(vals, ys + (k - 1) * 0.22, color=col, marker=mk, s=22, zorder=3,
                   label=f"{label} ({len(pairs)} pair{'s' if len(pairs) > 1 else ''})")
    ax.set_yticks(ys)
    ax.set_yticklabels([PLAIN[f] for f in FEATURES])
    ax.axhline(len(CONTENT) - 0.5, color="#bbbbbb", lw=0.8, ls="--")
    ax.text(1.0, len(CONTENT) - 0.35, "form", ha="right", va="bottom", fontsize=7.5, color="#777777")
    ax.text(1.0, len(CONTENT) - 0.65, "content", ha="right", va="top", fontsize=7.5, color="#777777")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("size of the difference between two texts, mean |Cliff's δ|\n"
                  "(0 = the units of the two texts overlap completely, 1 = not at all)")
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.14), frameon=False, ncol=1)
    save(fig, "fig_effects")


# 3 ─ PCA ────────────────────────────────────────────────────────────────────
def fig_pca(cls: dict) -> None:
    pc = pd.read_csv(OUT / "pca_coords.csv")
    ev = cls["pca"]["explained"]
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    for b in BOOK_ORDER:
        s = pc[pc.book == b]
        ax.scatter(s.pc1, s.pc2, s=26 if b != "miller" else 30, color=COLOUR[b],
                   marker="o" if BOOK_META[b]["context"] == "American" else "s",
                   edgecolor="black" if b == "miller" else "none", linewidth=0.4,
                   alpha=0.85, label=f"{LABEL[b]}, {BOOK_META[b]['mode']}")
    ax.set_xlabel(f"first component ({100 * ev[0]:.0f}% of variance)")
    ax.set_ylabel(f"second component ({100 * ev[1]:.0f}% of variance)")
    ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.6, zorder=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    save(fig, "fig_pca")


# 4 ─ where the fifth text is placed ──────────────────────────────────────────
def fig_placement(cls: dict) -> None:
    rows = []
    for blk, name in (("form", "form"), ("content", "content"),
                      ("content without war vocabulary", "content without war words"),
                      ("genre markers", "genre markers only"), ("all", "all eighteen")):
        for model, mname in (("random_forest", "random forest"), ("lda", "discriminant")):
            r = cls["placement"][blk]["raw"][model]
            rows.append((f"{name}, {mname}", r["share"]))
    four = ["applebaum", "nicolay", "brumme", "orth"]
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for b in four:
        v = np.array([r[1][b] for r in rows])
        ax.barh(y, v, left=left, color=COLOUR[b], edgecolor="white", height=0.7,
                label=f"{LABEL[b]}, {BOOK_META[b]['mode']}")
        for yi, l, w in zip(y, left, v):
            if w >= 0.08:
                ax.text(l + w / 2, yi, f"{100 * w:.0f}%", ha="center", va="center",
                        fontsize=7, color="white" if b in ("applebaum", "brumme") else "black")
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"share of the {cls['placement']['all']['raw']['random_forest']['n']} units of "
                  "Miller (2023) assigned to each of the four original texts")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2)
    save(fig, "fig_placement")


# 5 ─ war vocabulary: by text, and by period inside one author ───────────────
def fig_war(df: pd.DataFrame, seg: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.3),
                                 gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.42))
    rng = np.random.default_rng(1)
    for i, b in enumerate(BOOK_ORDER):
        v = 1000 * df.loc[df.book == b, "war_total_density"].to_numpy()
        a1.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=COLOUR[b], alpha=0.7)
        a1.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color="black", lw=1.2)
    # the two journeys that leave Ukraine: mean of their Ukraine-set segments
    for i, b in enumerate(BOOK_ORDER):
        s = seg[(seg.book == b) & seg.ukraine]
        if b in ("applebaum", "nicolay") and len(s):
            a1.scatter([i + 0.36], [1000 * s.war_total_density.mean()], marker="D", s=22,
                       facecolor="white", edgecolor="black", linewidth=0.8, zorder=4)
    a1.set_xticks(range(len(BOOK_ORDER)))
    a1.set_xticklabels([f"{ABBR[b]}\n{BOOK_META[b]['year']}" for b in BOOK_ORDER], fontsize=7)
    a1.set_ylabel("war words per 1,000 words\n(one dot per unit)")
    # right: Ukraine-set segments of the three authors whose books span periods
    groups = [("nicolay", "P1"), ("nicolay", "P2"), ("miller", "P1"), ("miller", "P2"),
              ("miller", "P3"), ("brumme", "P2"), ("brumme", "P3")]
    xs, labels, x = [], [], 0
    for k, (b, p) in enumerate(groups):
        if k and groups[k - 1][0] != b:
            x += 0.6
        v = 1000 * seg.loc[(seg.book == b) & seg.ukraine & (seg.period == p),
                           "war_total_density"].to_numpy()
        a2.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=COLOUR[b], alpha=0.7)
        if len(v):
            a2.plot([x - 0.25, x + 0.25], [v.mean()] * 2, color="black", lw=1.2)
        xs.append(x)
        labels.append(f"{ABBR[b]}\n{p}")
        x += 1
    a2.set_xticks(xs)
    a2.set_xticklabels(labels, fontsize=7)
    a2.set_ylabel("war words per 1,000 words\n(one dot per 2,000-word segment)")
    top = max(a1.get_ylim()[1], a2.get_ylim()[1])
    a1.set_ylim(0, top)
    a2.set_ylim(0, top)
    for ax, t in ((a1, "(a) all units, by text"), (a2, "(b) Ukraine-set text, by period narrated")):
        ax.text(0.0, 1.03, t, transform=ax.transAxes, fontsize=8.5, ha="left")
    save(fig, "fig_war")


# 6 ─ what the Ukraine-set text is about ─────────────────────────────────────
def fig_themes(top: dict) -> None:
    groups = top["groups"]
    cols = {"War": "#CC3311", "Politics and nation": "#EE7733",
            "People and everyday life": "#009988", "Place and movement": "#0077BB"}
    rows = [(LABEL[b], top["group_share_by_book"][b]) for b in BOOK_ORDER]
    rows.append(None)
    for b, per in top["within_author_groups"].items():
        for p, v in per.items():
            rows.append((f"{NAME[b]}, {p}", {g: dict(share=v[g]) for g in groups}))
    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    y, ylab, pos = len(rows), [], []
    for r in rows:
        y -= 1
        if r is None:
            continue
        left = 0
        for g in groups:
            w = r[1][g]["share"]
            ax.barh(y, w, left=left, color=cols[g], edgecolor="white", height=0.72,
                    label=g if r is rows[0] else None)
            if w >= 0.07:
                ax.text(left + w / 2, y, f"{100 * w:.0f}", ha="center", va="center",
                        fontsize=7, color="white")
            left += w
        ylab.append(r[0])
        pos.append(y)
    ax.set_yticks(pos)
    ax.set_yticklabels(ylab)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the Ukraine-set text (per cent of words) in each family of themes")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=4)
    save(fig, "fig_themes")


# 7 ─ geography ──────────────────────────────────────────────────────────────
def fig_geography(geo: dict) -> None:
    fig = plt.figure(figsize=(6.3, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 0.8], hspace=0.75,
                          wspace=0.45)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    x = np.arange(len(BOOK_ORDER))
    for i, b in enumerate(BOOK_ORDER):
        m = geo["books"][b]["ukraine"]
        lo, hi = m["outward_linkage_ci"]
        if lo == lo:
            a1.errorbar(i, m["outward_linkage"], yerr=[[m["outward_linkage"] - lo],
                                                       [hi - m["outward_linkage"]]],
                        fmt="o", color=COLOUR[b], capsize=3, ms=5)
        else:                                        # too few units to resample
            a1.plot(i, m["outward_linkage"], "o", color=COLOUR[b], ms=5, mfc="white")
        a1.text(i, 0.03, f"n={m['pairs_UU'] + m['pairs_UF']}", ha="center", va="bottom",
                fontsize=6.5, color="#555555")
    a1.set_xticks(x)
    a1.set_xticklabels([f"{ABBR[b]}\n{BOOK_META[b]['year']}" for b in BOOK_ORDER], fontsize=7)
    a1.set_ylim(0, 1)
    a1.set_ylabel("outward linkage: share of place\npairs with one place abroad")
    groups = ["Russia and Belarus", "Central and Eastern Europe",
              "Western Europe and North America", "Elsewhere"]
    gc = {"Russia and Belarus": "#AA3377", "Central and Eastern Europe": "#CCBB44",
          "Western Europe and North America": "#4477AA", "Elsewhere": "#BBBBBB"}
    left = np.zeros(len(BOOK_ORDER))
    for g in groups:
        v = np.array([geo["books"][b]["ukraine"]["outward_to"][g] for b in BOOK_ORDER], float)
        a2.barh(x[::-1], v, left=left, color=gc[g], edgecolor="white", height=0.7, label=g)
        left += v
    a2.set_yticks(x[::-1])
    a2.set_yticklabels([NAME[b] for b in BOOK_ORDER])
    a2.set_xlabel("outward pairs, by where the foreign place lies")
    a2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.3), ncol=2, fontsize=7)
    regions = ["West", "Centre", "South", "East"]
    rc = {"West": "#88CCEE", "Centre": "#44AA99", "South": "#DDCC77", "East": "#AA4499"}
    left = np.zeros(len(BOOK_ORDER))
    for r in regions:
        v = np.array([geo["books"][b]["ukraine"]["region_share"][r] for b in BOOK_ORDER])
        a3.barh(x[::-1], v, left=left, color=rc[r], edgecolor="white", height=0.7, label=r)
        for yi, l, w in zip(x[::-1], left, v):
            if w >= 0.08:
                a3.text(l + w / 2, yi, f"{100 * w:.0f}", ha="center", va="center", fontsize=6.5)
        left += v
    a3.set_yticks(x[::-1])
    a3.set_yticklabels([NAME[b] for b in BOOK_ORDER])
    a3.set_xlim(0, 1)
    a3.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    a3.set_xlabel("share of Ukrainian place mentions by macro-region (per cent)")
    a3.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    for ax, t in ((a1, "(a) links outward"), (a2, "(b) where outward links lead"),
                  (a3, "(c) regional spread")):
        ax.text(0.0, 1.04, t, transform=ax.transAxes, fontsize=8.5, ha="left")
    save(fig, "fig_geography")


def main() -> None:
    df = load_matrix()
    stats, cls = load_json("stats.json"), load_json("classify.json")
    geo, top = load_json("geography.json"), load_json("topics.json")
    seg = pd.read_csv(OUT / "segments.csv")
    fig_profiles(df)
    fig_effects(stats)
    # fig_pca(cls) is not in the article: the profiles, the pair comparison and
    # the placement show what the principal components showed, in plain terms.
    fig_placement(cls)
    fig_war(df, seg)
    fig_themes(top)
    fig_geography(geo)


if __name__ == "__main__":
    main()
