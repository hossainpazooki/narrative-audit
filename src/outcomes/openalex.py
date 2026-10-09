#!/usr/bin/env python3
"""Outcomes stage: match every processed paper to OpenAlex and write
citations, age-normalized citations, subfield and arXiv linkage.

Reads:  data/processed/neurips/neurips_{year}.json
Writes: data/outcomes/outcomes_raw.csv      one row per paper, resumable
        data/outcomes/outcomes.parquet      finalized table (normalize.finalize)
        data/outcomes/outcomes_report.json  match counts, unmatched by year

Matching: title search filtered to publication_year within one year of the
venue year, picked by normalize.pick_match; a record carrying a DOI falls
back to a direct DOI lookup when the title search fails. Unmatched papers
keep a null row and are counted in the report, never dropped.

The raw CSV is the resume point (same convention as the metric stages):
paper_ids already present are skipped, so rate-limit interruptions lose at
most the in-flight paper. The parquet and report are rebuilt from the full
CSV at the end of every run.

Usage:
  python src/outcomes/openalex.py
  python src/outcomes/openalex.py --venues neurips --mailto you@example.org
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse

import pandas as pd

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from schema import load_processed
from outcomes.normalize import (
    finalize, normalize_title, pick_match, stage_report, work_to_row,
)

API = "https://api.openalex.org/works"
FIELDS = ("id,ids,title,display_name,publication_year,cited_by_count,"
          "authorships,concepts,locations,first_online_date,doi")
FIELDNAMES = ["paper_id", "year", "match_method", "openalex_id", "citations",
              "author_count", "subfield", "arxiv_id", "arxiv_v1_date"]

MAX_RETRIES = 6


def _get(session, url: str, params: dict) -> dict | None:
    """GET with exponential backoff on rate limits, 5xx and timeouts."""
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code not in (429, 500, 502, 503, 504):
                print(f"  HTTP {r.status_code} for {r.url}", file=sys.stderr)
                return None
        except Exception as exc:                       # timeout, conn reset
            print(f"  request error: {exc}", file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    return None


def fetch_candidates(session, title: str, year: int, mailto: str | None) -> list[dict]:
    q = normalize_title(title)
    if not q:
        return []
    params = {
        "filter": f"title.search:{q},publication_year:{year - 1}|{year}|{year + 1}",
        "select": FIELDS,
        "per-page": 10,
    }
    if mailto:
        params["mailto"] = mailto
    data = _get(session, API, params)
    return (data or {}).get("results", [])


def fetch_by_doi(session, doi: str, mailto: str | None) -> dict | None:
    url = API + "/" + urllib.parse.quote(f"https://doi.org/{doi}", safe=":/")
    params = {"select": FIELDS}
    if mailto:
        params["mailto"] = mailto
    return _get(session, url, params)


def match_paper(session, paper: dict, mailto: str | None) -> dict:
    """One paper -> one output row; the fallback order the spec names."""
    candidates = fetch_candidates(session, paper.get("title", ""), paper["year"], mailto)
    work, method = pick_match(paper, candidates)
    if work is None and paper.get("doi"):
        work = fetch_by_doi(session, paper["doi"], mailto)
        if work is not None:
            method = "doi"
    return work_to_row(paper, work, method)


def done_ids(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, encoding="utf-8") as fh:
        return {r["paper_id"] for r in csv.DictReader(fh)}


def run_venue(venue: str, processed_dir: str, out_dir: str, mailto: str | None) -> int:
    venue_dir = os.path.join(processed_dir, venue)
    if not os.path.isdir(venue_dir):
        print(f"  [{venue}] not found")
        return 0
    import requests
    session = requests.Session()

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "outcomes_raw.csv")
    done = done_ids(csv_path)
    if done:
        print(f"  [{venue}] resuming — {len(done)} papers already matched")

    header_written = os.path.exists(csv_path)
    out_fh = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=FIELDNAMES)
    if not header_written:
        writer.writeheader()

    total = 0
    for fname in sorted(os.listdir(venue_dir)):
        if not fname.endswith(".json"):
            continue
        papers = [p for p in load_processed(os.path.join(venue_dir, fname))
                  if p["paper_id"] not in done]
        for p in papers:
            writer.writerow(match_paper(session, p, mailto))
            total += 1
            if total % 25 == 0:
                out_fh.flush()
                print(f"  [{venue}] {fname}: {total} new rows", flush=True)
        out_fh.flush()
    out_fh.close()
    print(f"  [{venue}] done — {total} new rows → {csv_path}")
    return total


def write_outputs(out_dir: str) -> None:
    csv_path = os.path.join(out_dir, "outcomes_raw.csv")
    if not os.path.exists(csv_path):
        print("  no outcomes_raw.csv; nothing to finalize")
        return
    raw = pd.read_csv(csv_path)
    df = finalize(raw).sort_values("paper_id").reset_index(drop=True)
    pq_path = os.path.join(out_dir, "outcomes.parquet")
    df.to_parquet(pq_path, index=False)
    report = stage_report(df)
    with open(os.path.join(out_dir, "outcomes_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"  {report['matched']}/{report['papers']} matched "
          f"({report['unmatched']} unmatched, counted) → {pq_path}")


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=os.path.join(repo, "data", "processed"))
    ap.add_argument("--out-dir",       default=os.path.join(repo, "data", "outcomes"))
    ap.add_argument("--venues", nargs="+", default=["neurips"])
    ap.add_argument("--mailto", default=os.environ.get("OPENALEX_MAILTO"),
                    help="contact email for the OpenAlex polite pool")
    args = ap.parse_args()

    total = 0
    for venue in args.venues:
        total += run_venue(venue, args.processed_dir, args.out_dir, args.mailto)
    write_outputs(args.out_dir)
    print(f"\nTotal: {total} papers matched this run")


if __name__ == "__main__":
    main()
