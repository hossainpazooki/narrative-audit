#!/usr/bin/env python3
"""
Per-paper active narration metric — binary presence of first-person language.

Based on Hohmann, Barnett, King & Connell (2025), Scientometrics 130:3349-3366.
DOI: https://doi.org/10.1007/s11192-025-05353-8
Source R code: https://github.com/agbarnett/narrator

Active narration = personal language using "We/we/Us/us", which helps audiences
relate to the text (Hillier et al., 2016; Pinker, 2014; Sword, 2012). This is a
binary YES/NO outcome per paper (i.e. does the abstract use first-person plural?).

Detection: case-sensitive whole-word match on "We", "we", "Us", "us".
"US" (uppercase) is intentionally excluded to avoid false positives from
the country abbreviation.

Output: data/per_paper/active_narration/{venue}.csv
Columns: paper_id, venue, year, n_words, has_narrator (0/1)

Usage:
  python src/metrics/active_narration.py
  python src/metrics/active_narration.py --venues neurips iclr
"""

from __future__ import annotations
import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from metrics._narrator_utils import (
    preprocess, word_count, has_narrator, run_venue_metric,
)

VENUES = ["neurips", "iclr", "icml", "arxiv"]


def _compute_row(paper: dict) -> dict:
    abstract = preprocess(paper.get("abstract", ""))
    nw = word_count(abstract)
    return {
        "paper_id":     paper["paper_id"],
        "venue":        paper["venue"],
        "year":         paper["year"],
        "n_words":      nw,
        "has_narrator": int(has_narrator(abstract)),
    }


def process_venue(venue: str, processed_dir: str, out_dir: str) -> int:
    return run_venue_metric(venue, processed_dir, out_dir, _compute_row)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=os.path.join(repo, "data", "processed"))
    parser.add_argument("--out-dir",       default=os.path.join(repo, "data", "per_paper", "active_narration"))
    parser.add_argument("--venues", nargs="+", default=VENUES)
    args = parser.parse_args()

    total = 0
    for venue in args.venues:
        total += process_venue(venue, args.processed_dir, args.out_dir)
    print(f"\nTotal: {total} papers")


if __name__ == "__main__":
    main()
