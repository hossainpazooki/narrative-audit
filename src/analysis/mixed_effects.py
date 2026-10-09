#!/usr/bin/env python3
"""Analyses 2 and 3 (preregistered): mixed-effects regressions.

Model 2, all abstracts:
    log1p(citations) ~ Tier-1 + Tier-2 features + confounds + (1 | year)

Model 3, Tier-3 subset: model 2's terms plus the full-text booleans, on
the sampled papers with full text.

Confounds: abstract_word_count (also the negative control), author_count,
preprint_before_conference, and subfield as fixed effects over the ten
largest subfields (rarer ones pool into 'other' — OpenAlex concepts have
hundreds of levels and singleton dummies estimate nothing). Year enters as
the random intercept, not a fixed effect.

Writes: data/analysis/mixed_effects_abstract.csv
        data/analysis/mixed_effects_fulltext.csv  (when Tier 3 exists)
Each row: term, coef, se, ci_low, ci_high, p, weak_p — effect sizes first,
with the .005 < p < .05 band flagged weak per the design spec.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from analysis.rubric_features import (
    TIER1_FEATURES, TIER2_FEATURES, TIER3_FEATURES,
    load_joined, load_tier3, weak_p,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("AA_DATA", os.path.join(REPO, "data"))

CONFOUNDS = ["abstract_word_count", "author_count", "preprint_before_conference"]
TOP_SUBFIELDS = 10


def prepare(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Complete-case matrix with pooled subfield dummies; returns (data, terms)."""
    df = df.copy()
    df["preprint_before_conference"] = df["preprint_before_conference"].map(
        {True: 1.0, False: 0.0})
    top = df["subfield"].value_counts().head(TOP_SUBFIELDS).index
    df["subfield_pooled"] = np.where(df["subfield"].isin(top),
                                     df["subfield"], "other")

    cols = [f for f in features if f in df.columns and df[f].notna().any()]
    confounds = [c for c in CONFOUNDS if df[c].notna().any()]
    keep = df.dropna(subset=cols + confounds + ["log1p_citations", "year"])
    dummies = pd.get_dummies(keep["subfield_pooled"], prefix="subfield",
                             drop_first=True, dtype=float)
    data = pd.concat([keep[cols + confounds + ["log1p_citations", "year"]],
                      dummies], axis=1)
    terms = cols + confounds + list(dummies.columns)
    return data, terms


def fit(data: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    import statsmodels.api as sm

    exog = sm.add_constant(data[terms].astype(float))
    model = sm.MixedLM(data["log1p_citations"].astype(float), exog,
                       groups=data["year"])
    res = model.fit(reml=True)
    ci = res.conf_int()
    rows = []
    for term in exog.columns:
        p = float(res.pvalues[term])
        rows.append({
            "term":    term,
            "coef":    round(float(res.params[term]), 6),
            "se":      round(float(res.bse[term]), 6),
            "ci_low":  round(float(ci.loc[term, 0]), 6),
            "ci_high": round(float(ci.loc[term, 1]), 6),
            "p":       p,
            "weak_p":  weak_p(p),
        })
    rows.append({"term": "__n_obs", "coef": float(len(data)), "se": np.nan,
                 "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "weak_p": False})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_joined(DATA)
    if df is None:
        print("  rubric_abstract or outcomes missing; mixed_effects skipped")
        return
    out_dir = os.path.join(DATA, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    features = TIER1_FEATURES + TIER2_FEATURES
    data, terms = prepare(df, features)
    if len(data) < 50:
        print(f"  only {len(data)} complete cases; mixed_effects skipped")
        return
    table = fit(data, terms)
    path = os.path.join(out_dir, "mixed_effects_abstract.csv")
    table.to_csv(path, index=False)
    print(f"  model 2: {len(data)} papers → {path}")

    tier3 = load_tier3(DATA)
    if tier3 is not None:
        sub = df.merge(tier3, on="paper_id", how="inner")
        data3, terms3 = prepare(sub, features + TIER3_FEATURES)
        if len(data3) >= 50:
            table3 = fit(data3, terms3)
            path3 = os.path.join(out_dir, "mixed_effects_fulltext.csv")
            table3.to_csv(path3, index=False)
            print(f"  model 3: {len(data3)} papers → {path3}")
        else:
            print(f"  model 3: only {len(data3)} complete cases; skipped")


if __name__ == "__main__":
    main()
