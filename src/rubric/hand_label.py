#!/usr/bin/env python3
"""Hand-labeling workflow for the judge-validity guard.

The upstream study (docs/related_work.md) shows a judge panel can pass
every agreement check while being directionally wrong, so both judge
tiers are anchored to human labels: Tier 3 on 10% of the sample (the
design spec) and Tier 2 on a seeded 100-abstract validation set
(preregistration Amendment 1). This module owns the three mechanical
steps so the labeled sets are preregistered draws, not ad-hoc picks:

  draw-tier2    seeded draw over the processed corpus ->
                data/hand_labels/tier2_sheet.csv
                (paper_id, abstract, empty n_claims / motivation /
                evidence_standard columns, with the scale in the header
                comment of this file and docs/rubric.md)
  draw-tier3    seeded 10% of the primary sample ->
                data/hand_labels/tier3_sheet.csv
                (one row per paper x feature, with the question and the
                section text the judge saw, empty value column)
  validate      check a filled sheet and convert it to the canonical
                long format judge_validity.py reads
                (tier2.csv: paper_id, prompt, value;
                 tier3.csv: paper_id, feature, value)

Draws are deterministic in (corpus/sample, seed): ids are sorted before
the seeded permutation, so input order cannot change the drawn set.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from schema import load_processed
from rubric.judge_prompts import PROMPTS as TIER2_PROMPTS
from rubric.tier3 import FEATURES as TIER3_FEATURES, QUESTIONS, feature_text, load_fulltext

SEED = 20261008
TIER2_N = 100
TIER3_SHARE = 0.10


def _seeded_draw(ids: list[str], n: int, seed: int) -> list[str]:
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    return [ids[i] for i in order[:n]]


def load_abstracts(processed_dir: str, venues: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for venue in venues:
        venue_dir = os.path.join(processed_dir, venue)
        if not os.path.isdir(venue_dir):
            continue
        for fname in sorted(os.listdir(venue_dir)):
            if fname.endswith(".json"):
                for p in load_processed(os.path.join(venue_dir, fname)):
                    out[p["paper_id"]] = p.get("abstract", "")
    return out


def draw_tier2(processed_dir: str, venues: list[str], out_dir: str,
               n: int = TIER2_N, seed: int = SEED) -> pd.DataFrame:
    abstracts = load_abstracts(processed_dir, venues)
    if not abstracts:
        raise SystemExit("no processed papers; run the process stage first")
    chosen = _seeded_draw(list(abstracts), min(n, len(abstracts)), seed)
    sheet = pd.DataFrame({
        "paper_id": chosen,
        "abstract": [abstracts[pid] for pid in chosen],
        "n_claims": "", "motivation": "", "evidence_standard": "",
    })
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tier2_sheet.csv")
    sheet.to_csv(path, index=False)
    print(f"  {len(sheet)} abstracts → {path}")
    print("  scales: n_claims 0-5 (5 = five or more); motivation 0/1/2; "
          "evidence_standard 0/1/2 — definitions in docs/rubric.md")
    return sheet


def draw_tier3(sample_csv: str, fulltext_dir: str, out_dir: str,
               share: float = TIER3_SHARE, seed: int = SEED) -> pd.DataFrame:
    if not os.path.exists(sample_csv):
        raise SystemExit(f"{sample_csv} not found — run the sample stage first")
    sample = pd.read_csv(sample_csv)
    repl = sample["is_replacement"].astype(str).str.lower().isin(("true", "1"))
    primary = sorted(sample.loc[~repl, "paper_id"])
    n = max(1, int(round(len(primary) * share)))
    chosen = _seeded_draw(primary, n, seed)

    rows = []
    for pid in chosen:
        doc = load_fulltext(fulltext_dir, pid)
        usable = doc is not None and doc.get("fulltext_status") == "ok"
        for feat in TIER3_FEATURES:
            rows.append({
                "paper_id": pid,
                "feature": feat,
                "question": QUESTIONS[feat],
                "section_text": feature_text(doc, feat)[:4000] if usable else "",
                "fulltext_ok": usable,
                "value": "",
            })
    sheet = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tier3_sheet.csv")
    sheet.to_csv(path, index=False)
    missing = int((~sheet["fulltext_ok"]).sum() // max(len(TIER3_FEATURES), 1))
    print(f"  {len(chosen)} papers × {len(TIER3_FEATURES)} features → {path}"
          + (f" ({missing} papers without full text; label from the PDF)"
             if missing else ""))
    return sheet


def validate_tier2(sheet_path: str, out_dir: str) -> pd.DataFrame:
    sheet = pd.read_csv(sheet_path, dtype={"paper_id": str})
    bounds = {name: (lo, hi) for name, (_t, lo, hi) in TIER2_PROMPTS.items()}
    errors, rows = [], []
    for _, row in sheet.iterrows():
        for prompt, (lo, hi) in bounds.items():
            raw = row.get(prompt)
            if pd.isna(raw) or str(raw).strip() == "":
                errors.append(f"{row['paper_id']}: {prompt} is empty")
                continue
            try:
                val = int(float(raw))
            except (TypeError, ValueError):
                errors.append(f"{row['paper_id']}: {prompt}={raw!r} not an integer")
                continue
            if not lo <= val <= hi:
                errors.append(f"{row['paper_id']}: {prompt}={val} outside [{lo},{hi}]")
                continue
            rows.append({"paper_id": row["paper_id"], "prompt": prompt,
                         "value": val})
    if errors:
        raise SystemExit("tier2 sheet invalid:\n  " + "\n  ".join(errors[:20])
                         + (f"\n  … and {len(errors) - 20} more" if len(errors) > 20 else ""))
    out = pd.DataFrame(rows).sort_values(["paper_id", "prompt"]).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tier2.csv")
    out.to_csv(path, index=False)
    print(f"  {len(out)} labels → {path}")
    return out


_TRUE = {"true", "1", "yes", "y"}
_FALSE = {"false", "0", "no", "n"}


def validate_tier3(sheet_path: str, out_dir: str) -> pd.DataFrame:
    sheet = pd.read_csv(sheet_path, dtype={"paper_id": str})
    errors, rows = [], []
    for _, row in sheet.iterrows():
        feat = row.get("feature")
        if feat not in TIER3_FEATURES:
            errors.append(f"{row['paper_id']}: unknown feature {feat!r}")
            continue
        raw = str(row.get("value")).strip().lower()
        if raw in _TRUE:
            val = True
        elif raw in _FALSE:
            val = False
        else:
            errors.append(f"{row['paper_id']}: {feat}={row.get('value')!r} "
                          "not a boolean (true/false)")
            continue
        rows.append({"paper_id": row["paper_id"], "feature": feat, "value": val})
    if errors:
        raise SystemExit("tier3 sheet invalid:\n  " + "\n  ".join(errors[:20])
                         + (f"\n  … and {len(errors) - 20} more" if len(errors) > 20 else ""))
    out = pd.DataFrame(rows).sort_values(["paper_id", "feature"]).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tier3.csv")
    out.to_csv(path, index=False)
    print(f"  {len(out)} labels → {path}")
    return out


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d2 = sub.add_parser("draw-tier2")
    d2.add_argument("--processed-dir", default=os.path.join(repo, "data", "processed"))
    d2.add_argument("--venues", nargs="+", default=["neurips"])
    d2.add_argument("--n", type=int, default=TIER2_N)
    d2.add_argument("--seed", type=int, default=SEED)

    d3 = sub.add_parser("draw-tier3")
    d3.add_argument("--sample", default=os.path.join(repo, "data", "sample", "sample.csv"))
    d3.add_argument("--fulltext-dir", default=os.path.join(repo, "data", "fulltext"))
    d3.add_argument("--share", type=float, default=TIER3_SHARE)
    d3.add_argument("--seed", type=int, default=SEED)

    va = sub.add_parser("validate")
    va.add_argument("tier", choices=["tier2", "tier3"])
    va.add_argument("--sheet", default=None)

    for p in (d2, d3, va):
        p.add_argument("--out-dir", default=os.path.join(repo, "data", "hand_labels"))

    args = ap.parse_args()
    if args.cmd == "draw-tier2":
        draw_tier2(args.processed_dir, args.venues, args.out_dir, args.n, args.seed)
    elif args.cmd == "draw-tier3":
        draw_tier3(args.sample, args.fulltext_dir, args.out_dir, args.share, args.seed)
    else:
        sheet = args.sheet or os.path.join(args.out_dir, f"{args.tier}_sheet.csv")
        if args.tier == "tier2":
            validate_tier2(sheet, args.out_dir)
        else:
            validate_tier3(sheet, args.out_dir)


if __name__ == "__main__":
    main()
