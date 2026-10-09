#!/usr/bin/env python3
"""
Compute model-averaged LLM-as-judge z-scores for each prompt.

Algorithm (per the manuscript review):
  1. For each model and prompt, z-score the raw per-year mean relative to its
     own pre-2022 baseline:
         z = (score - mean_{year<=2022}) / std_{year<=2022}
     This step is already done by export_pgfplots_data.export_standardized_llm(),
     which writes data/pgfplots/llm_scores_standardized_{prompt}.csv.
  2. Average the z-scores over all complete models for each year.
  3. Write data/pgfplots/llm_scores_model_avg_{prompt}.csv with columns:
         year, avg_z, std_z, n_models

Reads:  data/pgfplots/llm_scores_standardized_{prompt}.csv
Writes: data/pgfplots/llm_scores_model_avg_{prompt}.csv

Run (from repo root with venv active):
    python src/aggregate/aggregate_llm_model_avg.py
    python src/aggregate/aggregate_llm_model_avg.py --prompts simple ascb
    python src/aggregate/aggregate_llm_model_avg.py --in-dir data/pgfplots --out-dir data/pgfplots
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROMPTS = ["simple", "ascb", "own_reasoning"]

MODELS = [
    "Gemma-3-27B-Instruct",
    "Gemma-4-31B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct",
    "Mixtral-8x7B-Instruct",
    "Qwen2.5-32B-Instruct",
]


def process_prompt(prompt: str, in_dir: str, out_dir: str, venue: str = "neurips") -> str:
    # neurips keeps the un-suffixed filenames; iclr/icml read/write suffixed ones.
    suffix = "" if venue == "neurips" else f"_{venue}"
    src = os.path.join(in_dir, f"llm_scores_standardized{suffix}_{prompt}.csv")
    if not os.path.exists(src):
        print(f"  SKIP {src} (not found)", file=sys.stderr)
        return ""

    df = pd.read_csv(src)

    z_cols = [f"{m}_z" for m in MODELS if f"{m}_z" in df.columns]
    missing = [m for m in MODELS if f"{m}_z" not in df.columns]
    if missing:
        print(f"  WARNING [{prompt}]: missing models: {missing}", file=sys.stderr)
    if not z_cols:
        print(f"  SKIP [{prompt}]: no z-score columns found", file=sys.stderr)
        return ""

    result = df[["year"]].copy()
    z_data = df[z_cols]

    result["avg_z"]   = z_data.mean(axis=1)
    result["std_z"]   = z_data.std(axis=1, ddof=1)
    result["n_models"] = z_data.notna().sum(axis=1)

    # Round to 6 decimal places to keep CSV compact
    result["avg_z"] = result["avg_z"].round(6)
    result["std_z"] = result["std_z"].round(6)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"llm_scores_model_avg{suffix}_{prompt}.csv")
    result.to_csv(out_path, index=False, na_rep="")

    n_years = len(result)
    n_models = int(result["n_models"].max())
    avg_pre  = result.loc[result["year"] <= 2022, "avg_z"].mean()
    avg_post = result.loc[result["year"] >  2022, "avg_z"].mean()
    print(
        f"  [{prompt}] {n_years} years, {n_models} models averaged | "
        f"pre-2022 avg_z={avg_pre:.3f}  post-2022 avg_z={avg_post:.3f} | "
        f"→ {out_path}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts", nargs="+", default=PROMPTS,
        choices=PROMPTS,
        help="Which prompts to process (default: all three).",
    )
    parser.add_argument(
        "--in-dir",  default=os.path.join(os.environ.get("AA_DATA","data"), "pgfplots"),
        help="Directory containing llm_scores_standardized_*.csv files.",
    )
    parser.add_argument(
        "--out-dir", default=os.path.join(os.environ.get("AA_DATA","data"), "pgfplots"),
        help="Directory to write llm_scores_model_avg_*.csv files.",
    )
    parser.add_argument(
        "--venue", default="neurips",
        help="neurips (un-suffixed filenames, default) or iclr/icml "
             "(reads/writes venue-suffixed standardized + model_avg files).",
    )
    args = parser.parse_args()

    print(f"Computing model-averaged LLM z-scores (venue={args.venue}) …")
    for prompt in args.prompts:
        process_prompt(prompt, args.in_dir, args.out_dir, args.venue)
    print("Done.")


if __name__ == "__main__":
    main()
