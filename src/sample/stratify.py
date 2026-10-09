#!/usr/bin/env python3
"""The sample stage: a stratified full-text sample from the outcomes table.

Strata are year-band x within-band citation tercile. Allocation is
proportional to stratum size (largest-remainder rounding), the target is
1,500 primary papers, and the same draw also pre-orders a replacement list
of up to 10% overdraw per the design spec: when a primary paper later fails
full-text fetch or parse, the fulltext stage takes the next unused
replacement from the same stratum.

Eligibility: matched outcomes (citations present) AND an arXiv id, because
full texts come from arXiv PDFs. Ineligible papers are counted in
data/sample/sample_report.json, never silently dropped.

Everything is a function of (outcomes table, seed), so the same seed yields
a byte-identical sample.csv — a property the tests pin.

Output: data/sample/sample.csv with columns
  paper_id, year, year_band, tercile, stratum_rank, is_replacement

Usage:
  python src/sample/stratify.py
  python src/sample/stratify.py --target 1500 --seed 20261008
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

YEAR_BANDS = [(1987, 1999), (2000, 2009), (2010, 2017), (2018, 2021), (2022, 2025)]
DEFAULT_SEED = 20261008
DEFAULT_TARGET = 1500
OVERDRAW = 0.10


def year_band(year: int) -> str:
    for lo, hi in YEAR_BANDS:
        if lo <= year <= hi:
            return f"{lo}-{hi}"
    return "out_of_range"


def assign_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Add year_band and within-band cites_per_year tercile (0 low..2 high).

    Terciles are per band, not global: citation scales differ by decades,
    and the point of the stratum is coverage within each era.
    """
    out = df.copy()
    out["year_band"] = out["year"].map(year_band)

    def terciles(s: pd.Series) -> pd.Series:
        if len(s) < 3:                      # a band too small to cut
            return pd.Series(1, index=s.index)
        return pd.qcut(s.rank(method="first"), 3, labels=False)

    out["tercile"] = (
        out.groupby("year_band")["cites_per_year"].transform(terciles).astype(int)
    )
    return out


def _allocate(sizes: pd.Series, target: int) -> dict:
    """Largest-remainder proportional allocation, capped at stratum size."""
    total = int(sizes.sum())
    if total <= target:
        return {k: int(v) for k, v in sizes.items()}
    exact = sizes * target / total
    base = exact.astype(int).clip(upper=sizes)
    remainder = (exact - base).sort_values(ascending=False)
    short = target - int(base.sum())
    for k in remainder.index:
        if short <= 0:
            break
        if base[k] < sizes[k]:
            base[k] += 1
            short -= 1
    return {k: int(v) for k, v in base.items()}


def draw(eligible: pd.DataFrame, target: int, seed: int,
         overdraw: float = OVERDRAW) -> pd.DataFrame:
    """The stratified draw. Deterministic in (eligible table, args).

    Within each stratum the paper order is a seeded permutation of the
    paper_ids sorted first — sorting makes the draw independent of input
    row order. The first n_primary are the sample; the next
    ceil(n_primary * overdraw) are the ordered replacement queue.
    """
    df = assign_strata(eligible)
    sizes = df.groupby(["year_band", "tercile"])["paper_id"].count().sort_index()
    alloc = _allocate(sizes, target)

    rng = np.random.default_rng(seed)
    pieces = []
    for (band, terc) in sizes.index:
        ids = sorted(df.loc[(df["year_band"] == band) & (df["tercile"] == terc),
                            "paper_id"])
        order = rng.permutation(len(ids))
        n_primary = alloc[(band, terc)]
        n_extra = min(len(ids) - n_primary, int(np.ceil(n_primary * overdraw)))
        take = [ids[i] for i in order[:n_primary + n_extra]]
        for rank, pid in enumerate(take):
            pieces.append({
                "paper_id":      pid,
                "year_band":     band,
                "tercile":       terc,
                "stratum_rank":  rank,
                "is_replacement": rank >= n_primary,
            })
    out = pd.DataFrame(pieces).merge(df[["paper_id", "year"]], on="paper_id")
    cols = ["paper_id", "year", "year_band", "tercile", "stratum_rank", "is_replacement"]
    return out[cols].sort_values(["year_band", "tercile", "stratum_rank"]).reset_index(drop=True)


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default=os.path.join(repo, "data", "outcomes", "outcomes.parquet"))
    ap.add_argument("--out-dir",  default=os.path.join(repo, "data", "sample"))
    ap.add_argument("--target",   type=int, default=DEFAULT_TARGET)
    ap.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if not os.path.exists(args.outcomes):
        raise SystemExit(f"{args.outcomes} not found — run the outcomes stage first")
    outcomes = pd.read_parquet(args.outcomes)

    eligible = outcomes.dropna(subset=["citations", "arxiv_id"])
    sample = draw(eligible, args.target, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "sample.csv")
    sample.to_csv(csv_path, index=False)

    report = {
        "papers":        int(len(outcomes)),
        "eligible":      int(len(eligible)),
        "ineligible_no_match": int(outcomes["citations"].isna().sum()),
        "ineligible_no_arxiv": int((outcomes["citations"].notna()
                                    & outcomes["arxiv_id"].isna()).sum()),
        "primary":       int((~sample["is_replacement"]).sum()),
        "replacements":  int(sample["is_replacement"].sum()),
        "seed":          args.seed,
        "target":        args.target,
        "by_stratum":    {f"{b}/t{t}": int(n) for (b, t), n in
                          sample[~sample["is_replacement"]]
                          .groupby(["year_band", "tercile"])["paper_id"].count().items()},
    }
    with open(os.path.join(args.out_dir, "sample_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"  {report['primary']} primary + {report['replacements']} replacements "
          f"→ {csv_path}")


if __name__ == "__main__":
    main()
