#!/usr/bin/env python3
"""Judge-validity guard (design spec §5): three agreement readings,
reported regardless of direction.

  judge-judge   Spearman between each model pair's per-paper prompt medians
                (Tier 2), and raw agreement + Cohen's kappa between model
                pairs on Tier-3 booleans
  judge-human   Cohen's kappa per Tier-3 feature against the hand labels
                (data/hand_labels/tier3.csv: paper_id, feature, value)
  judge-tier1   Spearman of each Tier-2 judge score against each Tier-1
                deterministic feature

Writes data/analysis/judge_validity.csv with one row per comparison:
  kind, a, b, stat, value, n
Sections that lack inputs (no judge scores yet, no hand labels yet) are
skipped with a note; whatever exists is reported.
"""

from __future__ import annotations

import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from analysis.rubric_features import TIER1_FEATURES, load_joined
from rubric.tier3 import FEATURES as T3_FEATURES, parse_tier3_json

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("AA_DATA", os.path.join(REPO, "data"))


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two boolean/categorical series (aligned index)."""
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return np.nan
    po = float((a == b).mean())
    pe = sum(float((a == k).mean()) * float((b == k).mean())
             for k in set(a) | set(b))
    if pe == 1.0:
        return np.nan
    return (po - pe) / (1 - pe)


def tier2_model_medians(judge_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(os.path.join(judge_dir, "*.csv"))):
        df = pd.read_csv(path).dropna(subset=["score"])
        med = (df.groupby(["paper_id", "prompt_name"])["score"]
                 .median().unstack("prompt_name"))
        out[os.path.splitext(os.path.basename(path))[0]] = med
    return out


def tier3_model_verdicts(judge_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    import json
    for path in sorted(glob.glob(os.path.join(judge_dir, "*.jsonl"))):
        votes: dict[tuple[str, str], list[bool]] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                parsed = parse_tier3_json(rec.get("raw_output", ""))
                if parsed is not None:
                    votes.setdefault((rec["paper_id"], rec["feature"]),
                                     []).append(parsed["value"])
        rows = [{"paper_id": pid, "feature": feat,
                 "value": (sum(v) * 2 > len(v)) if sum(v) * 2 != len(v) else None}
                for (pid, feat), v in votes.items()]
        if rows:
            df = pd.DataFrame(rows).pivot(index="paper_id", columns="feature",
                                          values="value")
            out[os.path.splitext(os.path.basename(path))[0]] = df
    return out


def main() -> None:
    rows = []

    # judge-judge, Tier 2
    t2 = tier2_model_medians(os.path.join(REPO, "judge_scores", "rubric"))
    for (ma, da), (mb, db) in itertools.combinations(t2.items(), 2):
        joined = da.join(db, lsuffix="_a", rsuffix="_b", how="inner")
        for prompt in {c[:-2] for c in joined.columns if c.endswith("_a")}:
            pair = joined[[prompt + "_a", prompt + "_b"]].dropna()
            if len(pair) >= 10:
                rho, _ = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                rows.append({"kind": "judge-judge", "a": f"{ma}:{prompt}",
                             "b": f"{mb}:{prompt}", "stat": "spearman_rho",
                             "value": round(float(rho), 6), "n": len(pair)})

    # judge-judge, Tier 3
    t3 = tier3_model_verdicts(os.path.join(REPO, "judge_scores", "tier3"))
    for (ma, da), (mb, db) in itertools.combinations(t3.items(), 2):
        for feat in set(da.columns) & set(db.columns):
            joined = pd.concat([da[feat], db[feat]], axis=1, join="inner")
            k = cohens_kappa(joined.iloc[:, 0], joined.iloc[:, 1])
            rows.append({"kind": "judge-judge", "a": f"{ma}:{feat}",
                         "b": f"{mb}:{feat}", "stat": "cohens_kappa",
                         "value": round(k, 6) if not np.isnan(k) else np.nan,
                         "n": int((joined.notna().all(axis=1)).sum())})

    # judge-human: hand labels vs aggregated Tier-3 verdicts
    hand_path = os.path.join(DATA, "hand_labels", "tier3.csv")
    pq_path = os.path.join(DATA, "rubric", "rubric_fulltext.parquet")
    if os.path.exists(hand_path) and os.path.exists(pq_path):
        hand = pd.read_csv(hand_path)
        agg = pd.read_parquet(pq_path).set_index("paper_id")
        for feat in T3_FEATURES:
            sub = hand[hand["feature"] == feat].set_index("paper_id")["value"]
            if sub.empty or feat not in agg.columns:
                continue
            joined = pd.concat([sub.astype("boolean"),
                                agg[feat].astype("boolean")], axis=1,
                               join="inner")
            k = cohens_kappa(joined.iloc[:, 0], joined.iloc[:, 1])
            rows.append({"kind": "judge-human", "a": "hand", "b": feat,
                         "stat": "cohens_kappa",
                         "value": round(k, 6) if not np.isnan(k) else np.nan,
                         "n": int(joined.notna().all(axis=1).sum())})
    else:
        print("  no hand labels (data/hand_labels/tier3.csv); judge-human skipped")

    # judge-tier1
    df = load_joined(DATA)
    if df is not None:
        for jcol in ("judge_n_claims", "judge_motivation", "judge_evidence_standard"):
            if jcol not in df.columns or df[jcol].notna().sum() < 10:
                continue
            for feat in TIER1_FEATURES:
                pair = df[[jcol, feat]].dropna()
                if len(pair) >= 10 and pair[feat].nunique() > 1:
                    rho, _ = spearmanr(pair[jcol], pair[feat])
                    rows.append({"kind": "judge-tier1", "a": jcol, "b": feat,
                                 "stat": "spearman_rho",
                                 "value": round(float(rho), 6), "n": len(pair)})

    if not rows:
        print("  no judge scores present; judge_validity skipped")
        return
    out_dir = os.path.join(DATA, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "judge_validity.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  {len(rows)} comparisons → {out_csv}")


if __name__ == "__main__":
    main()
