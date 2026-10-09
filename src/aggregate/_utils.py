"""Shared helper for per-metric aggregation.

Trimmed for this release, which covers NeurIPS only. The upstream version
also aggregated arXiv by primary category and three further corpora; that
branch is removed here rather than left dead, so every line in this file
runs on the shipped data.

Paths resolve from AA_DATA (default "data") so a caller can redirect the
whole tree without editing this module.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

DATA_DIR      = os.environ.get("AA_DATA", "data")
PER_PAPER_DIR = os.path.join(DATA_DIR, "per_paper")
OUT_DIR       = os.path.join(DATA_DIR, "aggregate")
# Aggregate only NeurIPS (the shipped corpus). validate.py's VALID_VENUES is
# broader {"neurips","iclr","icml","arxiv"}; adding a venue there without
# adding it here will pass validation but silently skip aggregation.
VENUES        = ["neurips"]


def aggregate(
    metric: str,
    cols: list[str],
    out_name: str | None = None,
    venues: list[str] | None = None,
    per_paper_dir: str | None = None,
    out_dir: str | None = None,
) -> str:
    """Load {per_paper_dir}/{metric}/{venue}.csv for each venue, group by year,
    compute mean, std and count for each col, merge venues into one wide CSV.

    Returns the output path.
    """
    venues        = venues if venues is not None else VENUES
    per_paper_dir = per_paper_dir or PER_PAPER_DIR
    out_dir       = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    for venue in venues:
        path = os.path.join(per_paper_dir, metric, f"{venue}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {venue}", file=sys.stderr)
            continue
        df = pd.read_csv(path, low_memory=False)
        present = [c for c in cols if c in df.columns]
        missing = set(cols) - set(present)
        if missing:
            print(f"  WARNING: {metric}/{venue} missing {missing}", file=sys.stderr)

        agg = df.groupby("year")[present].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["year"] + [f"{venue}_{c}_{s}" for c, s in agg.columns[1:]]
        frames.append(agg)

    if not frames:
        raise RuntimeError(f"No data found for metric '{metric}'")

    result = frames[0]
    for other in frames[1:]:
        result = result.merge(other, on="year", how="outer")
    result = result.sort_values("year").reset_index(drop=True)

    out_path = os.path.join(out_dir, f"{out_name or metric}.csv")
    result.to_csv(out_path, index=False)
    print(f"  {len(result)} rows -> {out_path}")
    return out_path
