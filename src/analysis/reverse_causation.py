#!/usr/bin/env python3
"""Reverse-causation guard (design spec §5): score the arXiv v1 abstract
where one exists and report the delta against the camera-ready abstract.

If rubric features changed between v1 (written before reception was known)
and camera-ready, the camera-ready features partly reflect the paper's
reception — a reason to prefer v1-based scores. The deltas are reported
either way.

Reads:  data/fulltext/v1_abstracts.json  ({paper_id: v1 abstract or null},
        written by `src/fulltext/sections.py --v1-abstracts`)
        data/processed/neurips/*.json    (camera-ready abstracts)
Writes: data/analysis/reverse_causation.csv — per numeric feature: mean
        camera-ready value, mean v1 value, mean delta, share changed; and
        for claim_strength the share of papers whose class changed.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from schema import load_processed
from rubric import tier1
from analysis.rubric_features import CLAIM_ORDER

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("AA_DATA", os.path.join(REPO, "data"))


def camera_ready_abstracts(processed_dir: str, paper_ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for venue in sorted(os.listdir(processed_dir)) if os.path.isdir(processed_dir) else []:
        venue_dir = os.path.join(processed_dir, venue)
        if not os.path.isdir(venue_dir):
            continue
        for fname in sorted(os.listdir(venue_dir)):
            if fname.endswith(".json"):
                for p in load_processed(os.path.join(venue_dir, fname)):
                    if p["paper_id"] in paper_ids:
                        out[p["paper_id"]] = p.get("abstract", "")
    return out


def main() -> None:
    v1_path = os.path.join(DATA, "fulltext", "v1_abstracts.json")
    if not os.path.exists(v1_path):
        print("  no v1_abstracts.json; reverse_causation skipped")
        return
    with open(v1_path, encoding="utf-8") as fh:
        v1 = {k: v for k, v in json.load(fh).items() if v}
    if not v1:
        print("  v1_abstracts.json holds no abstracts; reverse_causation skipped")
        return

    camera = camera_ready_abstracts(os.path.join(DATA, "processed"), set(v1))
    pairs = [(pid, camera[pid], v1[pid]) for pid in v1 if camera.get(pid)]
    if not pairs:
        print("  no overlapping camera-ready abstracts; reverse_causation skipped")
        return

    numeric = {
        "concrete_number":    lambda t: float(tier1.concrete_number(t)),
        "claim_strength_ord": lambda t: float(CLAIM_ORDER[tier1.claim_strength(t)]),
        "role_sequence_dist": lambda t: float(tier1.role_sequence_dist(t)),
        "jargon_density":     lambda t: tier1.jargon_density(t),
    }
    rows = []
    for name, fn in numeric.items():
        cr = np.array([fn(c) for _, c, _ in pairs])
        v = np.array([fn(x) for _, _, x in pairs])
        rows.append({
            "feature":          name,
            "n":                len(pairs),
            "mean_camera_ready": round(float(cr.mean()), 6),
            "mean_v1":           round(float(v.mean()), 6),
            "mean_delta":        round(float((cr - v).mean()), 6),
            "share_changed":     round(float((cr != v).mean()), 6),
        })
    changed = np.mean([tier1.claim_strength(c) != tier1.claim_strength(x)
                       for _, c, x in pairs])
    rows.append({"feature": "claim_strength_class", "n": len(pairs),
                 "mean_camera_ready": np.nan, "mean_v1": np.nan,
                 "mean_delta": np.nan, "share_changed": round(float(changed), 6)})

    out_dir = os.path.join(DATA, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "reverse_causation.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  {len(pairs)} v1/camera-ready pairs → {out_csv}")


if __name__ == "__main__":
    main()
