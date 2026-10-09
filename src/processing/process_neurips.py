#!/usr/bin/env python3
"""
Convert raw NeurIPS JSON files to canonical processed per-year JSON files.

Filtering applied (raw → processed):
  - Drop records with empty title, short abstract (< 5 words), or no authors

Text sanitization:
  - sanitize_text() applied to title and abstract
  - OCR artifact removal: "(cid:NNN)" tokens stripped (all years); leading
    "Abstract:" labels stripped
  - Pre-2008 only: some scraped "abstracts" are OCR'd full-text prefixes.
    A leading page-number + ALL-CAPS title/author header block is stripped,
    and the text is capped at 350 words, cut at the last sentence boundary,
    so acknowledgements and mid-sentence tails do not enter the metrics.

Output fields: paper_id, venue, year, title, abstract, authors

Note: NeurIPS raw files come from scrape_neurips.py which scrapes all
accepted papers from proceedings.neurips.cc — no reject filtering needed.

Raw input:  data/raw/neurips/neurips_{year}_data.json
Output:     data/processed/neurips/neurips_{year}.json

Usage:
  python src/processing/process_neurips.py
  python src/processing/process_neurips.py --years 2020 2021 2022
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from schema import make_paper, save_processed, sanitize_text

# Pre-2008 abstracts were scraped from pages that sometimes bundle OCR'd full
# text into the abstract field. Cap those at this many words (cut at the last
# sentence boundary) so funding acknowledgements and body text do not leak
# into the writing metrics. Modern abstracts are far below this cap.
_PRE2008_WORD_CAP = 350

_CID_RE        = re.compile(r"\(cid:\d+\)")
_ABS_LABEL_RE  = re.compile(r"^\s*ABSTRACT[:.\s]+", re.IGNORECASE)
_PAGENUM_RE    = re.compile(r"^\d{1,4}\s+(?=[A-Z])")


def _looks_like_prose(sentence: str) -> bool:
    """A sentence of mostly-lowercase words of reasonable length."""
    words = sentence.split()
    if len(words) < 6:
        return False
    lower = sum(1 for w in words if w[:1].islower())
    return lower / len(words) > 0.5


def clean_ocr_abstract(abstract: str, year: int) -> str:
    """Strip OCR artifacts; cap pre-2008 full-text prefixes at a sentence end."""
    text = _CID_RE.sub(" ", abstract)
    text = _ABS_LABEL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if year >= 2008:
        return text

    # Leading page-number + ALL-CAPS title/author header: drop sentences from
    # the front until the first one that reads as prose.
    if _PAGENUM_RE.match(text):
        text = _PAGENUM_RE.sub("", text)
        parts = re.split(r"(?<=[.!?])\s+", text)
        start = 0
        for i, s in enumerate(parts):
            if _looks_like_prose(s):
                start = i
                break
        text = " ".join(parts[start:])

    words = text.split()
    if len(words) > _PRE2008_WORD_CAP:
        head = " ".join(words[:_PRE2008_WORD_CAP])
        # cut back to the last full sentence inside the cap
        cut = max(head.rfind(". "), head.rfind(".") if head.endswith(".") else -1)
        if cut > 100:
            head = head[:cut + 1]
        text = head
    return text


def best_abstract(record: dict, year: int) -> str:
    """Return the cleaned abstract, falling back to a full-text prefix.

    67 pre-2008 records carry "Abstract Unavailable" (or nothing) in the
    abstract field but have OCR'd full text. Dropping them would distort the
    per-year paper counts (e.g. NeurIPS 1987 has 90 papers, 13 of them
    fallback-only), so we take a 600-word full-text prefix and pass it through
    the same OCR cleaner (header strip + 350-word sentence-boundary cap) as
    every other pre-2008 abstract.
    """
    abstract = sanitize_text(record.get("abstract") or "")
    if abstract.lower().startswith("abstract unavailable"):
        abstract = ""
    if len(abstract.split()) < 20:
        full = sanitize_text(record.get("full_text") or "")
        if len(full.split()) >= 20:
            abstract = " ".join(full.split()[:600])
    return clean_ocr_abstract(abstract, year)


def parse_authors(raw) -> list[str]:
    if isinstance(raw, list):
        return [a.strip() for a in raw if isinstance(a, str) and a.strip()]
    if isinstance(raw, str):
        return [a.strip() for a in raw.split(",") if a.strip()]
    return []


def process_year(year: int, raw_dir: str, out_dir: str) -> tuple[int, int]:
    path = os.path.join(raw_dir, f"neurips_{year}_data.json")
    if not os.path.exists(path):
        print(f"  [NeurIPS {year}] not found: {path}")
        return 0, 0

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    papers = []
    skipped = 0
    for i, r in enumerate(raw):
        title    = sanitize_text(r.get("title") or "")
        abstract = best_abstract(r, year)
        authors  = parse_authors(r.get("authors", []))

        if not title:
            skipped += 1
            continue

        if len(abstract.split()) < 5:
            skipped += 1
            continue

        if not authors:
            skipped += 1
            continue

        raw_id = str(r.get("id") or r.get("source_id") or "").strip()
        # Prefix with year: pre-2020 and post-2020 NeurIPS use different hash
        # namespaces that can collide, so year+hash is the stable unique ID.
        paper_id = f"{year}_{raw_id}" if raw_id else f"neurips_{year}_{i}"

        papers.append(make_paper(
            paper_id=paper_id,
            venue="neurips",
            year=int(r.get("year", year)),
            title=title,
            abstract=abstract,
            authors=authors,
        ))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"neurips_{year}.json")
    save_processed(papers, out_path)
    print(f"  [NeurIPS {year}] {len(papers)} papers  |  {skipped} skipped")
    return len(papers), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir",       default="data/raw/neurips")
    parser.add_argument("--processed-dir", default="data/processed/neurips")
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(1987, 2026)))
    args = parser.parse_args()

    total_papers = total_skipped = 0
    for year in args.years:
        p, s = process_year(year, args.raw_dir, args.processed_dir)
        total_papers  += p
        total_skipped += s

    print(f"\nNeurIPS total: {total_papers} papers  |  {total_skipped} skipped")


if __name__ == "__main__":
    main()
