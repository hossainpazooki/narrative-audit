"""
Canonical processed-paper schema shared across all venues.

Design principles:
- Only fields with actual data are stored; no None/empty placeholders.
- Fields that require external API calls (n_citations, n_references,
  author_affiliations) are NOT present in the initial processed files.
  They are added by src/metrics/citations.py as a separate merge step.
- arxiv_categories / arxiv_primary_category are only present for arXiv papers.

Always-present fields (all venues):
    paper_id, venue, year, title, abstract, authors

arXiv-only fields (added by process_arxiv.py):
    arxiv_categories, arxiv_primary_category

Added later by citations.py:
    n_citations, n_references, author_affiliations
"""

from __future__ import annotations
import json
import os
import re
import unicodedata


# ── Text sanitization ─────────────────────────────────────────────────────────

def sanitize_text(text: str) -> str:
    """
    Basic text cleaning applied to every title and abstract during processing.

    Steps (more venue-specific cleaning can be added in each process_*.py):
    1. Normalise unicode to NFC (composed form, e.g. é not e + combining accent)
    2. Remove null bytes and ASCII control characters (keep \\t, \\n, space)
    3. Collapse runs of whitespace / newlines into a single space
    4. Strip leading / trailing whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # Step 2: remove control characters (keep printable + whitespace)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Step 3: collapse whitespace (including embedded newlines in abstracts)
    text = re.sub(r"\s+", " ", text)

    # Step 4: strip
    return text.strip()


# ── Schema helpers ────────────────────────────────────────────────────────────

def make_paper(
    paper_id: str,
    venue: str,
    year: int,
    title: str,
    abstract: str,
    authors: list[str],
    *,
    arxiv_categories: list[str] | None = None,
    arxiv_primary_category: str | None = None,
    biorxiv_category: str | None = None,
) -> dict:
    """
    Build a processed-paper dict containing only the fields that have data.
    Call sanitize_text() on title and abstract before passing them here.
    """
    record: dict = {
        "paper_id": paper_id,
        "venue":    venue,
        "year":     year,
        "title":    title,
        "abstract": abstract,
        "authors":  authors,
    }
    # Only add arXiv fields if they carry data
    if arxiv_categories:
        record["arxiv_categories"] = arxiv_categories
    if arxiv_primary_category:
        record["arxiv_primary_category"] = arxiv_primary_category
    if biorxiv_category:
        record["biorxiv_category"] = biorxiv_category
    return record


def load_processed(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_processed(papers: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(papers, fh, ensure_ascii=False, indent=2)


def resume_year_counts(csv_path: str) -> dict:
    """Per-year row counts from a metric output CSV.

    Used for resume bookkeeping: a year is complete only when the output CSV
    holds as many rows for it as the processed file has papers. (The previous
    convention treated a year as done if ANY row existed, so a partially
    written year silently dropped the remaining papers forever.)
    """
    import csv as _csv
    counts: dict = {}
    if not os.path.exists(csv_path):
        return counts
    with open(csv_path, encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            y = r.get("year")
            if y:
                counts[int(y)] = counts.get(int(y), 0) + 1
    return counts


_resume_id_cache: dict = {}


def resume_year_ids(csv_path: str, year: int) -> set:
    """paper_ids already present for one year of a metric output CSV.

    Builds a full {year: set(ids)} index on the first call per path so that
    repeated calls for different partial years cost one scan total, not one
    scan per year.
    """
    import csv as _csv
    if not os.path.exists(csv_path):
        return set()
    if csv_path not in _resume_id_cache:
        index: dict[int, set] = {}
        with open(csv_path, encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                y, pid = r.get("year"), r.get("paper_id")
                if y and pid:
                    index.setdefault(int(y), set()).add(pid)
        _resume_id_cache[csv_path] = index
    return _resume_id_cache[csv_path].get(year, set())
