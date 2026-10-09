#!/usr/bin/env python3
"""Analysis 1 (preregistered): Spearman correlation of every rubric feature
with cites_per_year, per year band, plus the heatmap figure.

Writes: data/analysis/rubric_spearman.csv  (feature, year_band, rho, p, n, weak_p)
        paper/figs/fig_rubric_spearman.png

Skips politely when the rubric or outcomes tables are absent, so a
rubric-less pipeline run still completes.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from analysis.rubric_features import (
    TIER1_FEATURES, TIER2_FEATURES, load_joined, weak_p,
)
from sample.stratify import year_band

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("AA_DATA", os.path.join(REPO, "data"))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_band"] = df["year"].map(year_band)
    features = [f for f in TIER1_FEATURES + TIER2_FEATURES if f in df.columns]
    rows = []
    bands = sorted(df["year_band"].unique()) + ["all"]
    for band in bands:
        sub = df if band == "all" else df[df["year_band"] == band]
        for feat in features:
            pair = sub[[feat, "cites_per_year"]].dropna()
            if len(pair) < 10 or pair[feat].nunique() < 2:
                rows.append({"feature": feat, "year_band": band, "rho": np.nan,
                             "p": np.nan, "n": len(pair), "weak_p": False})
                continue
            rho, p = spearmanr(pair[feat], pair["cites_per_year"])
            rows.append({"feature": feat, "year_band": band,
                         "rho": round(float(rho), 6), "p": float(p),
                         "n": len(pair), "weak_p": weak_p(p)})
    return pd.DataFrame(rows)


def heatmap(table: pd.DataFrame, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = table.pivot(index="feature", columns="year_band", values="rho")
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(pivot.columns),
                                    1.2 + 0.5 * len(pivot.index)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Rubric feature vs cites/year — Spearman ρ by year band")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    df = load_joined(DATA)
    if df is None:
        print("  rubric_abstract or outcomes missing; rubric_spearman skipped")
        return
    table = compute(df)
    out_dir = os.path.join(DATA, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "rubric_spearman.csv")
    table.to_csv(out_csv, index=False)
    heatmap(table, os.path.join(REPO, "paper", "figs", "fig_rubric_spearman.png"))
    print(f"  {len(table)} rows → {out_csv}")


if __name__ == "__main__":
    main()
