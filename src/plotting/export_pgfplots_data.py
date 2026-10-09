#!/usr/bin/env python3
"""Export plot-ready CSV files for the pgfplots figures.

Each metric becomes a CSV of `year` plus one column per venue present in the
aggregate. This release covers NeurIPS only, so the files are `year,neurips`;
the upstream project emitted four further corpus columns, and those are not
padded in here as empty columns.

Outputs go to data/pgfplots/.

Run:
  python src/plotting/export_pgfplots_data.py
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from aggregate._utils import VENUES

_D = os.environ.get("AA_DATA", "data")
AGG  = os.path.join(_D, "aggregate")
OUT  = os.path.join(_D, "pgfplots")

COMPLETE_MODELS = [
    "Gemma-3-27B-Instruct",
    "Gemma-4-31B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct",
    "Mixtral-8x7B-Instruct",
    "Qwen2.5-32B-Instruct",
]




def save(df: pd.DataFrame, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False, na_rep="nan")
    print(f"  {path}  ({len(df)} rows x {len(df.columns)} cols)")




def export_metric(metric_name: str, col: str, out_name: str | None = None) -> None:
    """Write year plus one column per venue present in the aggregate."""
    main_csv = f"{AGG}/{metric_name}.csv"
    if not os.path.exists(main_csv):
        print(f"  SKIP {main_csv} (not found)")
        return

    df = pd.read_csv(main_csv)
    result = df[["year"]].copy()
    for venue in VENUES:
        mean_col = f"{venue}_{col}_mean"
        if mean_col in df.columns:
            result[venue] = df[mean_col]

    series = [c for c in result.columns if c != "year"]
    if not series:
        print(f"  SKIP {metric_name}/{col} (no venue columns)")
        return
    result = result.dropna(how="all", subset=series).reset_index(drop=True)

    stem = out_name or f"{metric_name}_{col}"
    save(result, f"{stem}.csv")




def export_readability() -> None:
    metrics = [
        "flesch_ease", "flesch_kincaid", "gunning_fog", "smog",
        "dale_chall", "spache", "coleman_liau", "ari",
        "linsear_write", "lix", "rix", "forcast",
        "powers_sumner_kearl", "avg_sentence_length", "avg_syllables_per_word",
    ]
    for m in metrics:
        export_metric("readability", m, f"readability_{m}")


def export_llm_scores() -> None:
    # neurips keeps the un-suffixed filenames (back-compat with the current
    # figure); iclr/icml emit venue-suffixed files when their aggregates exist.
    for venue in ("neurips", "iclr", "icml"):
        suffix = "" if venue == "neurips" else f"_{venue}"
        for prompt in ("simple", "ascb", "own_reasoning"):
            csv = f"{AGG}/llm_scores_{venue}_{prompt}.csv"
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            cols = ["year"]
            for m in COMPLETE_MODELS:
                col = f"{m}_mean"
                if col in df.columns:
                    cols.append(col)
            save(df[cols], f"llm_scores{suffix}_{prompt}.csv")









def export_standardized_llm() -> None:
    """Standardize per-model LLM scores to z-scores (pre-2022 baseline) for all 3 prompts.

    neurips keeps the un-suffixed filenames; iclr/icml emit venue-suffixed
    files when their per-venue score exports exist.
    """
    for venue in ("neurips", "iclr", "icml"):
        suffix = "" if venue == "neurips" else f"_{venue}"
        for prompt in ("simple", "ascb", "own_reasoning"):
            src = f"{OUT}/llm_scores{suffix}_{prompt}.csv"
            if not os.path.exists(src):
                if venue == "neurips":
                    print(f"  SKIP {src} (not found)")
                continue
            df = pd.read_csv(src)
            model_cols = [f"{m}_mean" for m in COMPLETE_MODELS if f"{m}_mean" in df.columns]
            result = df[["year"]].copy()
            for col in model_cols:
                b = df.loc[df["year"] <= 2022, col].dropna()
                if len(b) < 3:
                    continue
                result[col.replace("_mean", "_z")] = (df[col] - b.mean()) / b.std(ddof=1)
            save(result, f"llm_scores_standardized{suffix}_{prompt}.csv")






def main() -> None:
    print("Exporting pgfplots CSVs")
    export_readability()
    export_llm_scores()
    export_standardized_llm()
    print("Done.")


if __name__ == "__main__":
    main()
