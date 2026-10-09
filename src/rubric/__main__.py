"""Single-paper rubric debugging.

    PYTHONPATH=src python -m rubric score <paper_id>

Prints the Tier-1 features, the sentence-role labeling behind
role_sequence_dist, and whatever Tier-2/Tier-3 judge values exist for the
paper, as JSON. Reads the processed corpus, so the `process` stage must
have run (or the paper's year file must exist under data/processed/).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from schema import load_processed
from rubric import tier1

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_paper(processed_dir: str, paper_id: str) -> dict | None:
    year = paper_id.split("_", 1)[0]
    candidates = glob.glob(os.path.join(processed_dir, "*", f"*{year}*.json")) or \
                 glob.glob(os.path.join(processed_dir, "*", "*.json"))
    for path in sorted(candidates):
        for paper in load_processed(path):
            if paper["paper_id"] == paper_id:
                return paper
    return None


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m rubric")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="print rubric features for one paper")
    sc.add_argument("paper_id")
    sc.add_argument("--processed-dir", default=os.path.join(REPO, "data", "processed"))
    args = ap.parse_args()

    paper = find_paper(args.processed_dir, args.paper_id)
    if paper is None:
        sys.exit(f"paper_id {args.paper_id!r} not found under {args.processed_dir}")

    abstract = paper.get("abstract", "")
    sentences = tier1.split_sentences(abstract)
    out = {
        "paper_id": paper["paper_id"],
        "title":    paper.get("title"),
        "tier1":    {k: v for k, v in tier1.compute_row(paper).items()
                     if k not in ("paper_id", "venue", "year")},
        "roles":    [{"sentence": s, "role": r}
                     for s, r in zip(sentences, tier1.label_roles(sentences))],
    }

    tier2 = tier1.load_tier2_medians(os.path.join(REPO, "judge_scores", "rubric"))
    if tier2 is not None:
        hit = tier2[tier2["paper_id"] == paper["paper_id"]]
        if not hit.empty:
            out["tier2"] = hit.iloc[0].drop("paper_id").to_dict()

    ft_pq = os.path.join(REPO, "data", "rubric", "rubric_fulltext.parquet")
    if os.path.exists(ft_pq):
        import pandas as pd
        ft = pd.read_parquet(ft_pq)
        hit = ft[ft["paper_id"] == paper["paper_id"]]
        if not hit.empty:
            out["tier3"] = hit.iloc[0].drop("paper_id").to_dict()

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
