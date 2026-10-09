#!/usr/bin/env python3
"""
Per-paper signposting metric — count of signposting words/phrases per 100 words.

Based on Hohmann, Barnett, King & Connell (2025), Scientometrics 130:3349-3366.
DOI: https://doi.org/10.1007/s11192-025-05353-8
Source R code: https://github.com/agbarnett/narrator

Signposting words denote order and/or cause-and-effect, helping the audience
keep track of how ideas relate to one another (Lindsay, 2011; Montgomery, 2003).
Signposting has remained broadly stable in the scientific literature since the
1950s (Hohmann et al., 2025).

Word list sourced from 99_key_characters_words_phrases.R lines 93–129.

Output: data/per_paper/signposting/{venue}.csv
Columns: paper_id, venue, year, n_words, signposting_count, signposting_per_100

Usage:
  python src/metrics/signposting.py
  python src/metrics/signposting.py --venues neurips iclr
"""

from __future__ import annotations
import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from metrics._narrator_utils import (
    preprocess, word_count, count_signposting, run_venue_metric,
)

VENUES = ["neurips", "iclr", "icml", "arxiv"]


def _compute_row(paper: dict) -> dict:
    abstract = preprocess(paper.get("abstract", ""))
    nw = word_count(abstract)
    count = count_signposting(abstract)
    return {
        "paper_id":            paper["paper_id"],
        "venue":               paper["venue"],
        "year":                paper["year"],
        "n_words":             nw,
        "signposting_count":   count,
        "signposting_per_100": round(count / nw * 100, 4) if nw > 0 else 0.0,
    }


def process_venue(venue: str, processed_dir: str, out_dir: str) -> int:
    return run_venue_metric(venue, processed_dir, out_dir, _compute_row)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=os.path.join(repo, "data", "processed"))
    parser.add_argument("--out-dir",       default=os.path.join(repo, "data", "per_paper", "signposting"))
    parser.add_argument("--venues", nargs="+", default=VENUES)
    args = parser.parse_args()

    total = 0
    for venue in args.venues:
        total += process_venue(venue, args.processed_dir, args.out_dir)
    print(f"\nTotal: {total} papers")


if __name__ == "__main__":
    main()
