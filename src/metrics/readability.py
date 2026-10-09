#!/usr/bin/env python3
"""
Classical readability metrics for all papers in data/processed/{venue}/.

Implements the full set of readability matrices listed in Hohmann et al. (2025)
Appendix I, Table 1, plus additional metrics available from textstat.

Reference:
  Hohmann MH, Barnett AG, King N, Connell SD. (2025). The evolution of
  scientific writing: an analysis of 20 million abstracts over 70 years in
  health and medical science. Scientometrics 130:3349–3366.
  https://doi.org/10.1007/s11192-025-05353-8

Requires: pip install textstat

Metrics implemented (from Table 1):
  - Flesch Reading Ease             (textstat)
  - Flesch-Kincaid Grade Level      (textstat)
  - Gunning Fog Index               (textstat)
  - SMOG Index                      (textstat)
  - Dale-Chall Readability Score    (textstat)
  - Spache Readability              (textstat)
  - Coleman-Liau Index              (textstat)
  - Automated Readability Index     (textstat)
  - Linsear Write Formula           (textstat)
  - FORCAST Formula                 (manual — monosyllabic words / 10)
  - Powers-Sumner-Kearl             (manual — grade-level variant of Flesch)
  - LIX                             (textstat)
  - RIX                             (textstat)

Not implemented (reasons noted):
  - Fry Readability Formula  — chart-based; not reducible to a closed formula
  - Lexile Framework         — proprietary; requires licensed word-frequency DB

Helper quantities stored per paper:
  word_count, sentence_count, syllable_count, polysyllable_count,
  monosyllable_count, difficult_words_count,
  avg_sentence_length, avg_syllables_per_word

Output: data/per_paper/readability/{venue}.csv
  One row per paper; arxiv_primary_category is empty string for non-arXiv venues.

Run:
  python src/metrics/readability.py
  python src/metrics/readability.py --venues neurips iclr icml
"""

from __future__ import annotations
import argparse
import os
import re
import sys

import pandas as pd
import textstat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from schema import load_processed, resume_year_counts, resume_year_ids

VENUES = ["neurips", "iclr", "icml", "arxiv"]


# ── FORCAST (manual) ─────────────────────────────────────────────────────────

def _forcast(text: str) -> float | None:
    """
    FORCAST Formula (Caylor & Sticht, 1973):
      Grade = 20 - (monosyllabic_words_in_150_word_sample / 10)

    For texts longer than 150 words we use all words (proportionally scaled):
      Grade = 20 - (total_monosyllabic / total_words * 150 / 10)
    = 20 - (monosyllabic_rate * 15)
    """
    words = text.split()
    if not words:
        return None
    mono = textstat.monosyllabcount(text)
    rate = mono / len(words)  # proportion of monosyllabic words
    return round(20 - rate * 15, 3)


# ── Powers-Sumner-Kearl (manual) ─────────────────────────────────────────────

def _powers_sumner_kearl(text: str) -> float | None:
    """
    Powers-Sumner-Kearl Formula (1958) — re-calibration of Flesch.
      Grade = 0.0778 * avg_sentence_length + 0.0455 * (syllables_per_100_words) - 2.2029
    Note: the second term uses syllables per 100 words (not per word).
    """
    try:
        words = textstat.lexicon_count(text, removepunct=True)
        if words == 0:
            return None
        asl = textstat.words_per_sentence(text)
        syllables_per_100 = textstat.syllable_count(text) / words * 100
        return round(0.0778 * asl + 0.0455 * syllables_per_100 - 2.2029, 3)
    except Exception:
        return None


# ── Main compute function ─────────────────────────────────────────────────────

_MIN_WORDS = 10


def compute(abstract: str) -> dict:
    """Compute all readability metrics for one abstract."""
    if not abstract or len(abstract.split()) < _MIN_WORDS:
        return {k: None for k in _COLUMNS}

    try:
        r: dict = {
            # ── Helper quantities ──────────────────────────────────────────
            "word_count":          textstat.lexicon_count(abstract, removepunct=True),
            "sentence_count":      textstat.sentence_count(abstract),
            "syllable_count":      textstat.syllable_count(abstract),
            "polysyllable_count":  textstat.polysyllabcount(abstract),
            "monosyllable_count":  textstat.monosyllabcount(abstract),
            "difficult_words_count": textstat.difficult_words(abstract),
            "avg_sentence_length": round(textstat.words_per_sentence(abstract), 3),
            "avg_syllables_per_word": round(textstat.avg_syllables_per_word(abstract), 3),

            # ── From Appendix Table 1 (textstat) ──────────────────────────
            "flesch_ease":         round(textstat.flesch_reading_ease(abstract), 3),
            "flesch_kincaid":      round(textstat.flesch_kincaid_grade(abstract), 3),
            "gunning_fog":         round(textstat.gunning_fog(abstract), 3),
            "smog":                round(textstat.smog_index(abstract), 3),
            "dale_chall":          round(textstat.dale_chall_readability_score(abstract), 3),
            "spache":              round(textstat.spache_readability(abstract), 3),
            "coleman_liau":        round(textstat.coleman_liau_index(abstract), 3),
            "ari":                 round(textstat.automated_readability_index(abstract), 3),
            "linsear_write":       round(textstat.linsear_write_formula(abstract), 3),
            "lix":                 round(textstat.lix(abstract), 3),
            "rix":                 round(textstat.rix(abstract), 3),

            # ── From Appendix Table 1 (manual implementations) ────────────
            "forcast":             _forcast(abstract),
            "powers_sumner_kearl": _powers_sumner_kearl(abstract),
        }
    except Exception:
        return {k: None for k in _COLUMNS}

    return r


# Column order for the CSV (helper quantities first, then metrics)
_COLUMNS = [
    "word_count", "sentence_count", "syllable_count",
    "polysyllable_count", "monosyllable_count", "difficult_words_count",
    "avg_sentence_length", "avg_syllables_per_word",
    "flesch_ease", "flesch_kincaid", "gunning_fog", "smog",
    "dale_chall", "spache", "coleman_liau", "ari", "linsear_write",
    "lix", "rix", "forcast", "powers_sumner_kearl",
]


# ── Venue processing ──────────────────────────────────────────────────────────

def process_venue(venue: str, processed_dir: str, out_dir: str) -> int:
    venue_dir = os.path.join(processed_dir, venue)
    if not os.path.isdir(venue_dir):
        print(f"  [{venue}] processed dir not found: {venue_dir}")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{venue}.csv")

    done_counts = resume_year_counts(out_path)
    if done_counts:
        print(f"  [{venue}] resuming — {len(done_counts)} years present in output")

    header_written = os.path.exists(out_path)
    total = 0
    for fname in sorted(f for f in os.listdir(venue_dir) if f.endswith(".json")):
        m = re.search(r'(\d{4})', fname)
        year = int(m.group(1)) if m else -1
        papers = load_processed(os.path.join(venue_dir, fname))
        if done_counts.get(year, 0) >= len(papers):
            print(f"  [{venue}] {fname}: skip (complete)"); continue
        if done_counts.get(year, 0) > 0:
            done = resume_year_ids(out_path, year)
            papers = [p for p in papers if p["paper_id"] not in done]
            print(f"  [{venue}] {fname}: partial year — {len(papers)} papers to top up")
        rows = []
        for paper in papers:
            metrics = compute(paper.get("abstract", ""))
            rows.append({
                "paper_id": paper["paper_id"],
                "venue":    paper["venue"],
                "year":     paper["year"],
                **metrics,
                "arxiv_primary_category": paper.get("arxiv_primary_category") or "",
            })
        if rows:
            pd.DataFrame(rows).to_csv(out_path, mode='a', header=not header_written, index=False)
            header_written = True
            total += len(rows)
        print(f"  [{venue}] {fname}: {len(papers)} papers → {len(rows)} rows (total {total})")

    print(f"  [{venue}] done — {total} new rows → {out_path}")
    return total


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir",
                        default=os.path.join(repo, "data", "processed"))
    parser.add_argument("--out-dir",
                        default=os.path.join(repo, "data", "per_paper", "readability"))
    parser.add_argument("--venues", nargs="+", default=VENUES)
    args = parser.parse_args()

    total = 0
    for venue in args.venues:
        total += process_venue(venue, args.processed_dir, args.out_dir)
    print(f"\nTotal: {total} papers")


if __name__ == "__main__":
    main()
