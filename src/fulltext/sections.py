#!/usr/bin/env python3
"""The fulltext stage: sampled papers -> data/fulltext/{paper_id}.json.

Per paper: arXiv PDF fetch, GROBID parse (primary), pymupdf heading
heuristic (fallback). The output JSON is

  {paper_id, arxiv_id, source: grobid|fallback|null,
   fulltext_status: ok|fetch_failed|parse_failed,
   sections: [{heading, normalized, text}], figure_captions, urls}

A paper whose fetch or parse fails keeps its JSON with the failure status:
it is excluded from Tier 3 and counted in data/fulltext/fulltext_report.json,
and the stage promotes the sampler's pre-drawn same-stratum replacements
(up to the 10% overdraw already in sample.csv) to take its place.

Resume: an existing JSON with status ok is never redone; failed ones are
retried only under --retry-failed.

Usage:
  python src/fulltext/sections.py
  python src/fulltext/sections.py --retry-failed --sleep 3
  python src/fulltext/sections.py --v1-abstracts     # reverse-causation data
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import pandas as pd

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from fulltext.fetch import fetch_pdf, fetch_v1_abstract
from fulltext.grobid import DEFAULT_URL, grobid_sections
from fulltext.fallback import fallback_sections

# heading -> canonical section name used by Tier 3's SECTION_TARGETS
_NORMALIZE = [
    (re.compile(r"abstract", re.I),                              "abstract"),
    (re.compile(r"introduction", re.I),                          "introduction"),
    (re.compile(r"related\s+work|prior\s+work|background", re.I), "related_work"),
    (re.compile(r"limitation", re.I),                            "limitations"),
    (re.compile(r"experiment|evaluation|empirical", re.I),       "experiments"),
    (re.compile(r"result", re.I),                                "results"),
    (re.compile(r"method|approach|architecture|our\s+model|algorithm|setup", re.I),
                                                                 "methods"),
    (re.compile(r"discussion|analysis|broader\s+impact", re.I),  "discussion"),
    (re.compile(r"conclusion|summary|future\s+work", re.I),      "conclusion"),
    (re.compile(r"appendix|supplement", re.I),                   "appendix"),
    (re.compile(r"reference|bibliograph", re.I),                 "references"),
    (re.compile(r"acknowledg", re.I),                            "other"),
]


def normalize_heading(heading: str) -> str:
    h = re.sub(r"^(?:\d+|[IVX]+)[.\s]+", "", heading or "").strip()
    for pat, name in _NORMALIZE:
        if pat.search(h):
            return name
    return "other"


def build_doc(paper_id: str, arxiv_id: str, parsed: dict | None,
              source: str | None, status: str) -> dict:
    sections = []
    for sec in (parsed or {}).get("sections", []):
        sections.append({
            "heading":    sec.get("heading", ""),
            "normalized": normalize_heading(sec.get("heading", "")),
            "text":       sec.get("text", ""),
        })
    return {
        "paper_id":        paper_id,
        "arxiv_id":        arxiv_id,
        "source":          source,
        "fulltext_status": status,
        "sections":        sections,
        "figure_captions": (parsed or {}).get("figure_captions", []),
        "urls":            (parsed or {}).get("urls", []),
    }


def process_paper(paper_id: str, arxiv_id: str, pdf_dir: str,
                  grobid_url: str) -> dict:
    pdf = fetch_pdf(arxiv_id, pdf_dir)
    if pdf is None:
        return build_doc(paper_id, arxiv_id, None, None, "fetch_failed")
    parsed, source = grobid_sections(pdf, grobid_url), "grobid"
    if parsed is None:
        parsed, source = fallback_sections(pdf), "fallback"
    if parsed is None:
        return build_doc(paper_id, arxiv_id, None, None, "parse_failed")
    return build_doc(paper_id, arxiv_id, parsed, source, "ok")


def _existing_status(out_dir: str, paper_id: str) -> str | None:
    path = os.path.join(out_dir, f"{paper_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("fulltext_status")
    except (json.JSONDecodeError, OSError):
        return None


def run(sample: pd.DataFrame, arxiv_ids: dict[str, str], out_dir: str,
        pdf_dir: str, grobid_url: str, sleep: float,
        retry_failed: bool) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    repl = sample["is_replacement"].astype(str).str.lower().isin(("true", "1"))
    status_of: dict[str, str] = {}

    def handle(paper_id: str) -> str:
        prior = _existing_status(out_dir, paper_id)
        if prior == "ok" or (prior is not None and not retry_failed):
            status_of[paper_id] = prior
            return prior
        arxiv_id = arxiv_ids.get(paper_id)
        doc = (build_doc(paper_id, "", None, None, "fetch_failed")
               if not arxiv_id else
               process_paper(paper_id, arxiv_id, pdf_dir, grobid_url))
        with open(os.path.join(out_dir, f"{paper_id}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=1)
        status_of[paper_id] = doc["fulltext_status"]
        if arxiv_id:
            time.sleep(sleep)
        return doc["fulltext_status"]

    done = 0
    for _, row in sample[~repl].iterrows():
        handle(row["paper_id"])
        done += 1
        if done % 25 == 0:
            print(f"  {done} primary papers processed", flush=True)

    # Promote pre-drawn replacements, in stratum_rank order, one per failure.
    promoted = {}
    for (band, terc), grp in sample[~repl].groupby(["year_band", "tercile"]):
        failures = sum(status_of.get(pid) != "ok" for pid in grp["paper_id"])
        pool = sample[repl & (sample["year_band"] == band)
                      & (sample["tercile"] == terc)].sort_values("stratum_rank")
        used = 0
        for _, row in pool.iterrows():
            if used >= failures:
                break
            if handle(row["paper_id"]) == "ok":
                used += 1
        promoted[f"{band}/t{terc}"] = used

    counts: dict[str, int] = {}
    for st in status_of.values():
        counts[st] = counts.get(st, 0) + 1
    return {"processed": len(status_of), "by_status": counts,
            "replacements_promoted": promoted}


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",   default=os.path.join(repo, "data", "sample", "sample.csv"))
    ap.add_argument("--outcomes", default=os.path.join(repo, "data", "outcomes", "outcomes.parquet"))
    ap.add_argument("--out-dir",  default=os.path.join(repo, "data", "fulltext"))
    ap.add_argument("--pdf-dir",  default=os.path.join(repo, "data", "fulltext", "pdf"))
    ap.add_argument("--grobid-url", default=DEFAULT_URL)
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="seconds between arXiv fetches")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--v1-abstracts", action="store_true",
                    help="also fetch arXiv v1 abstracts for sampled papers "
                         "(reverse-causation check)")
    args = ap.parse_args()

    if not os.path.exists(args.sample):
        print("  no sample.csv; fulltext stage skipped")
        return
    sample = pd.read_csv(args.sample)
    outcomes = pd.read_parquet(args.outcomes)
    arxiv_ids = dict(outcomes.dropna(subset=["arxiv_id"])
                     [["paper_id", "arxiv_id"]].itertuples(index=False))

    report = run(sample, arxiv_ids, args.out_dir, args.pdf_dir,
                 args.grobid_url, args.sleep, args.retry_failed)

    if args.v1_abstracts:
        path = os.path.join(args.out_dir, "v1_abstracts.json")
        v1 = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                v1 = json.load(fh)
        todo = [pid for pid in sample["paper_id"]
                if pid in arxiv_ids and pid not in v1]
        for pid in todo:
            v1[pid] = fetch_v1_abstract(arxiv_ids[pid])
            time.sleep(args.sleep)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(v1, fh, ensure_ascii=False, indent=1, sort_keys=True)
        report["v1_abstracts"] = sum(1 for v in v1.values() if v)

    with open(os.path.join(args.out_dir, "fulltext_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"  {report['processed']} papers → {args.out_dir} "
          f"({report['by_status']})")


if __name__ == "__main__":
    main()
