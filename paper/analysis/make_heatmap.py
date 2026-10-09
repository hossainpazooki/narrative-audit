"""Regenerate the judge/metric correlation heatmap at a legible type size.

The canonical generator, src/analysis/judge_feature_correlation.py, writes
into manuscript/figs for the NeurIPS paper and sizes its type for a
single-column float. Rendered at the width this paper uses, its row labels
come out near 3 pt. This script re-plots the same committed correlations
with type sized for a two-column float, and writes only into
uncertainlp/figs so the other manuscripts are untouched.

Reads  data/pgfplots/judge_metric_corr.csv  (written by the canonical
       generator; columns: metric, label, r_pooled, r_within, one per model)
Writes paper/figs/fig_judge_metric_heatmap.png

Run from the repo root with the venv active:

    python paper/analysis/make_heatmap.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MODELS = [
    "Gemma-3-27B-Instruct",
    "Gemma-4-31B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct",
    "Mixtral-8x7B-Instruct",
    "Qwen2.5-32B-Instruct",
]

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, os.environ.get("AA_DATA", "data"), "pgfplots", "judge_metric_corr.csv")
DEST = os.path.join(REPO, "paper", "figs", "fig_judge_metric_heatmap.png")

# Sized for a figure* placed at 0.72\textwidth. At that width the scale
# factor is about 0.63, so 11 pt here renders near 7 pt on the page and the
# cell values near 5.7 pt. The in-figure title is dropped because the LaTeX
# caption already carries it.
FIGSIZE = (7.2, 6.4)
FS_YTICK = 11
FS_XTICK = 11
FS_CELL = 9
FS_CBAR = 10
DPI = 200


def main() -> None:
    corr = pd.read_csv(SRC).sort_values("r_pooled")
    H = corr.set_index("label")[MODELS]
    short = [m.replace("-Instruct", "") for m in MODELS]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(H.values, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=FS_XTICK)
    ax.set_yticks(range(len(H.index)))
    ax.set_yticklabels(H.index, fontsize=FS_YTICK)
    ax.tick_params(length=2, pad=2)

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H.values[i, j]:+.2f}", ha="center", va="center",
                    fontsize=FS_CELL)

    cbar = fig.colorbar(im, label="Spearman $r$", shrink=0.65, pad=0.02)
    cbar.ax.tick_params(labelsize=FS_CBAR)
    cbar.set_label("Spearman $r$", fontsize=FS_CBAR)

    fig.tight_layout()
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    fig.savefig(DEST, dpi=DPI, bbox_inches="tight")
    print(f"wrote {DEST}  ({len(H)} metrics x {len(MODELS)} models)")


if __name__ == "__main__":
    main()
