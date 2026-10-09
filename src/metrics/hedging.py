#!/usr/bin/env python3
"""
Per-paper hedging metric — count of hedging words/phrases per 100 words.

Based on Hohmann, Barnett, King & Connell (2025), Scientometrics 130:3349-3366.
DOI: https://doi.org/10.1007/s11192-025-05353-8
Source R code: https://github.com/agbarnett/narrator

Hedging words are useful for expressing uncertainty but also make readers less
confident in the writing. Hedging has decreased in scientific literature since
the 1950s (Hohmann et al., 2025).

Word list sourced from 99_key_characters_words_phrases.R lines 43–87.

Output: data/per_paper/hedging/{venue}.csv
Columns: paper_id, venue, year, n_words, hedging_count, hedging_per_100

Usage:
  python src/metrics/hedging.py
  python src/metrics/hedging.py --venues neurips iclr
"""

from __future__ import annotations
import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from metrics._narrator_utils import (
    preprocess, word_count, count_hedging, run_venue_metric,
)

VENUES = ["neurips", "iclr", "icml", "arxiv"]


def _compute_row(paper: dict) -> dict:
    abstract = preprocess(paper.get("abstract", ""))
    nw = word_count(abstract)
    count = count_hedging(abstract)
    return {
        "paper_id":        paper["paper_id"],
        "venue":           paper["venue"],
        "year":            paper["year"],
        "n_words":         nw,
        "hedging_count":   count,
        "hedging_per_100": round(count / nw * 100, 4) if nw > 0 else 0.0,
    }


def process_venue(venue: str, processed_dir: str, out_dir: str) -> int:
    return run_venue_metric(venue, processed_dir, out_dir, _compute_row)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=os.path.join(repo, "data", "processed"))
    parser.add_argument("--out-dir",       default=os.path.join(repo, "data", "per_paper", "hedging"))
    parser.add_argument("--venues", nargs="+", default=VENUES)
    args = parser.parse_args()

    total = 0
    for venue in args.venues:
        total += process_venue(venue, args.processed_dir, args.out_dir)
    print(f"\nTotal: {total} papers")


if __name__ == "__main__":
    main()
