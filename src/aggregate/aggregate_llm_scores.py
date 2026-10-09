#!/usr/bin/env python3
"""
Aggregate LLM readability scores by year.

Reads:  judge_scores/{venue}/{model}.csv
Writes: data/aggregate/llm_scores_{venue}.csv

Columns: year, {model}_mean, {model}_std
  - mean/std computed over all papers × prompts × runs for that year
  - also writes a separate breakdown per prompt: llm_scores_{venue}_{prompt}.csv

Usage:
  python src/aggregate/aggregate_llm_scores.py
  python src/aggregate/aggregate_llm_scores.py --venue neurips
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
from collections import defaultdict
import statistics

VENUE    = "neurips"
_D = os.environ.get("AA_DATA", "data")
IN_BASE  = os.environ.get("AA_JUDGE", "judge_scores")
OUT_BASE = os.path.join(_D, "aggregate")
PROMPTS  = ["simple", "ascb", "own_reasoning"]


def load_scores(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                score = float(row["score"])
            except (ValueError, TypeError):
                continue
            rows.append({"paper_id": row["paper_id"], "year": int(row["year"]),
                         "prompt_name": row["prompt_name"], "score": score})
    return rows


def aggregate_by_year(rows: list[dict], prompt_filter: str | None = None
                      ) -> dict[int, list[float]]:
    by_year: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if prompt_filter and r["prompt_name"] != prompt_filter:
            continue
        by_year[r["year"]].append(r["score"])
    return by_year


def write_aggregate(out_path: str, model_data: dict[str, dict[int, list[float]]]):
    all_years = sorted({y for scores in model_data.values() for y in scores})
    model_names = sorted(model_data.keys())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = ["year"]
    for m in model_names:
        fieldnames += [f"{m}_mean", f"{m}_std", f"{m}_count"]

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for year in all_years:
            row = {"year": year}
            for m in model_names:
                vals = model_data[m].get(year, [])
                row[f"{m}_mean"]  = round(statistics.mean(vals), 4) if vals else ""
                row[f"{m}_std"]   = round(statistics.stdev(vals), 4) if len(vals) > 1 else ""
                row[f"{m}_count"] = len(vals)
            writer.writerow(row)
    print(f"  → {out_path}  ({len(all_years)} years, {len(model_names)} models)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default=VENUE)
    parser.add_argument("--in-base",  default=IN_BASE)
    parser.add_argument("--out-base", default=OUT_BASE)
    args = parser.parse_args()

    model_files = glob.glob(os.path.join(args.in_base, args.venue, "*.csv"))
    if not model_files:
        print(f"No score files found in {args.in_base}/{args.venue}/"); return

    # Load all models
    all_rows: dict[str, list[dict]] = {}
    for path in sorted(model_files):
        model = os.path.splitext(os.path.basename(path))[0]
        rows  = load_scores(path)
        all_rows[model] = rows
        print(f"  {model}: {len(rows):,} scored rows")

    # Aggregate: all prompts combined
    model_data = {m: aggregate_by_year(rows) for m, rows in all_rows.items()}
    write_aggregate(os.path.join(args.out_base, f"llm_scores_{args.venue}.csv"), model_data)

    # Aggregate: per prompt
    for prompt in PROMPTS:
        model_data_p = {m: aggregate_by_year(rows, prompt) for m, rows in all_rows.items()}
        if any(model_data_p.values()):
            write_aggregate(
                os.path.join(args.out_base, f"llm_scores_{args.venue}_{prompt}.csv"),
                model_data_p,
            )


if __name__ == "__main__":
    main()
