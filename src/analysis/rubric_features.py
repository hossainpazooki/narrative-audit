"""Shared feature loading/encoding for the rubric analysis scripts.

CLAIM_ORDER encodes claim_strength as ordinal claim *scope* for
correlation and regression: an existence proof claims the least territory,
a guarantee the most. docs/rubric.md states the encoding; treating the
classes as unordered is the robustness check reported in the paper, not a
pipeline concern.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

CLAIM_ORDER = {"existence": 0, "hedged": 1, "systematic": 2, "guarantee": 3}

TIER1_FEATURES = ["concrete_number", "claim_strength_ord",
                  "role_sequence_dist", "jargon_density"]
TIER2_FEATURES = ["judge_n_claims", "judge_motivation", "judge_evidence_standard"]
TIER3_FEATURES = ["baseline_present", "ablation_present", "limitations_present",
                  "variance_reported", "code_link", "prepost_disclosure",
                  "fig1_is_diagram"]

# The design spec flags .005 < p < .05 as weak, after the source post's
# warning that findings in that band often fail to replicate.
WEAK_P_LOW, WEAK_P_HIGH = 0.005, 0.05


def weak_p(p: float) -> bool:
    return bool(WEAK_P_LOW < p < WEAK_P_HIGH)


def load_joined(data_dir: str) -> pd.DataFrame | None:
    """rubric_abstract x outcomes, with encodings applied; None if either
    table is missing (the stage then skips politely)."""
    rub_path = os.path.join(data_dir, "rubric", "rubric_abstract.parquet")
    out_path = os.path.join(data_dir, "outcomes", "outcomes.parquet")
    if not (os.path.exists(rub_path) and os.path.exists(out_path)):
        return None
    rub = pd.read_parquet(rub_path)
    out = pd.read_parquet(out_path).drop(columns=["year"], errors="ignore")
    df = rub.merge(out, on="paper_id", how="inner")
    df["claim_strength_ord"] = df["claim_strength"].map(CLAIM_ORDER)
    df["concrete_number"] = df["concrete_number"].astype(float)
    df["log1p_citations"] = np.log1p(df["citations"])
    # Abstract word count: the negative-control confound. Taken from the
    # hedging metric CSV (n_words), which every pipeline run produces.
    hedging = os.path.join(data_dir, "per_paper", "hedging", "neurips.csv")
    if os.path.exists(hedging):
        wc = pd.read_csv(hedging)[["paper_id", "n_words"]]
        df = df.merge(wc.rename(columns={"n_words": "abstract_word_count"}),
                      on="paper_id", how="left")
    else:
        df["abstract_word_count"] = np.nan
    return df


def load_tier3(data_dir: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, "rubric", "rubric_fulltext.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    for feat in TIER3_FEATURES:
        if feat in df.columns:
            # nullable booleans (None = counted null) -> float with NaN
            df[feat] = df[feat].map({True: 1.0, False: 0.0})
    return df[["paper_id"] + [f for f in TIER3_FEATURES if f in df.columns]]
