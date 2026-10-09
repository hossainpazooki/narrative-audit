#!/usr/bin/env python3
"""
Per-paper sensational language metric — hype word counts per 100 words.

Based on Hohmann, Barnett, King & Connell (2025), Scientometrics 130:3349-3366.
DOI: https://doi.org/10.1007/s11192-025-05353-8
Source R code: https://github.com/agbarnett/narrator

Sensational/hype language is often used to convince readers of the importance
of research. While it can attract readers, over-reliance can lead to spin and
inaccuracy (Mammola, 2020; Millar et al., 2022).

Nine categories based on Millar, Batalo & Budgell (2022), JAMA Network Open:
  1. importance  — compelling, crucial, vital, pivotal, ...
  2. novelty     — novel, groundbreaking, revolutionary, unprecedented, ...
  3. rigor       — robust, rigorous, sophisticated, advanced, ...
  4. scale       — comprehensive, vast, massive, inter-disciplinary, ...
  5. utility     — effective, scalable, transformative, user-friendly, ...
  6. quality     — prestigious, exceptional, talented, renowned, ...
  7. attitude    — exciting, remarkable, outstanding, intriguing, ...
  8. problem     — alarming, dire, elusive, devastating, ...
  9. additional  — (words in Hohmann et al. not in original Millar list)

Word lists sourced from 99_key_characters_words_phrases.R lines 412–457.

Output: data/per_paper/sensational_language/{venue}.csv
Columns: paper_id, venue, year, n_words,
         hype_importance, hype_novelty, hype_rigor, hype_scale,
         hype_utility, hype_quality, hype_attitude, hype_problem,
         hype_additional, hype_total,
         hype_importance_per_100, ..., hype_total_per_100

Usage:
  python src/metrics/sensational_language.py
  python src/metrics/sensational_language.py --venues neurips iclr
"""

from __future__ import annotations
import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from metrics._narrator_utils import (
    preprocess, word_count, count_hype, HYPE_CATEGORIES, run_venue_metric,
)

VENUES = ["neurips", "iclr", "icml", "arxiv"]
_CATS = list(HYPE_CATEGORIES.keys())


def _compute_row(paper: dict) -> dict:
    abstract = preprocess(paper.get("abstract", ""))
    nw = word_count(abstract)
    counts = count_hype(abstract)
    hype_total = sum(counts.values())
    row: dict = {
        "paper_id": paper["paper_id"],
        "venue":    paper["venue"],
        "year":     paper["year"],
        "n_words":  nw,
    }
    for cat in _CATS:
        row[f"hype_{cat}"] = counts[cat]
    row["hype_total"] = hype_total
    denom = nw if nw > 0 else 1
    for cat in _CATS:
        row[f"hype_{cat}_per_100"] = round(counts[cat] / denom * 100, 4)
    row["hype_total_per_100"] = round(hype_total / denom * 100, 4)
    return row


def process_venue(venue: str, processed_dir: str, out_dir: str) -> int:
    return run_venue_metric(venue, processed_dir, out_dir, _compute_row)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=os.path.join(repo, "data", "processed"))
    parser.add_argument("--out-dir",       default=os.path.join(repo, "data", "per_paper", "sensational_language"))
    parser.add_argument("--venues", nargs="+", default=VENUES)
    args = parser.parse_args()

    total = 0
    for venue in args.venues:
        total += process_venue(venue, args.processed_dir, args.out_dir)
    print(f"\nTotal: {total} papers")


if __name__ == "__main__":
    main()
