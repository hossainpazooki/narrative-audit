#!/usr/bin/env python3
"""Analyses 2 and 3 (preregistered, as amended — see preregistration.md
Amendment 1): regressions of log1p(citations) on the rubric features.

The specification is reconciled with the published precedent (Appendix E
of Rangarajan & Krishnan, *Measurable Writing Standards for AI-Native
Venues*; see docs/related_work.md), so coefficients land on a
comparable scale:

- confounds enter as log(abstract word count) and log(author count);
- every continuous predictor is z-scored on the estimation sample, so
  coefficients are standardized (binary predictors stay 0/1);
- the most recent year in the data is excluded as citation-immature;
- the primary subfield control is the corpus-derived topic
  (data/outcomes/topics.parquet) as fixed effects; the pooled OpenAlex
  concept is the robustness alternative when topics are absent.

Primary models (year as random intercept, REML):
  model 2  all abstracts:  log1p(citations) ~ Tier-1 + Tier-2 + confounds
  model 3  Tier-3 subset:  model 2's terms + the full-text booleans

Robustness specs for model 2, written to one table with a `spec` column:
  fe_cluster  OLS with year fixed effects, SEs clustered by year
              (the precedent's exact design)
  pre2023     the primary spec restricted to papers from before 2023
              (judge-familiarity confound and citation immaturity both
              concentrate after 2022; see docs/related_work.md)

Writes: data/analysis/mixed_effects_abstract.csv
        data/analysis/mixed_effects_fulltext.csv   (when Tier 3 exists)
        data/analysis/mixed_effects_robustness.csv
Each row: term, coef, se, ci_low, ci_high, p, weak_p — effect sizes
first, with the .005 < p < .05 band flagged weak per the design spec.
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

CONFOUNDS = ["log_abstract_word_count", "log_author_count",
             "preprint_before_conference"]
# Predictors that stay 0/1; everything else continuous gets z-scored.
BINARY = {"concrete_number", "preprint_before_conference", *TIER3_FEATURES}
TOP_SUBFIELDS = 10


def load_topics(data_dir: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, "outcomes", "topics.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def prepare(df: pd.DataFrame, features: list[str],
            topics: pd.DataFrame | None,
            max_year: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Complete-case standardized design matrix; returns (data, terms).

    max_year caps the sample (inclusive). When None, the most recent year
    present is dropped as citation-immature, mirroring the precedent.
    """
    df = df.copy()
    df["preprint_before_conference"] = df["preprint_before_conference"].map(
        {True: 1.0, False: 0.0})
    df["log_abstract_word_count"] = np.log(
        df["abstract_word_count"].where(df["abstract_word_count"] > 0))
    df["log_author_count"] = np.log(
        df["author_count"].where(df["author_count"] > 0))

    if max_year is None:
        max_year = int(df["year"].max()) - 1
    df = df[df["year"] <= max_year]

    if topics is not None:
        df = df.merge(topics, on="paper_id", how="left")
        df["subfield_ctrl"] = df["topic"].map(lambda t: f"topic_{int(t)}"
                                              if pd.notna(t) else np.nan)
    else:
        top = df["subfield"].value_counts().head(TOP_SUBFIELDS).index
        df["subfield_ctrl"] = np.where(df["subfield"].isin(top),
                                       df["subfield"], "other")

    cols = [f for f in features if f in df.columns and df[f].notna().any()]
    confounds = [c for c in CONFOUNDS if df[c].notna().any()]
    keep = df.dropna(subset=cols + confounds
                     + ["log1p_citations", "year", "subfield_ctrl"]).copy()

    # Standardize continuous predictors on the estimation sample. A
    # predictor with no variance in the sample carries no information
    # and would make the design singular; it is dropped, not zeroed.
    kept_cols = []
    for col in cols + confounds:
        s = keep[col].astype(float)
        if s.nunique() < 2:
            print(f"  dropping constant predictor: {col}")
            continue
        if col not in BINARY:
            keep[col] = (s - s.mean()) / s.std(ddof=1)
        kept_cols.append(col)

    dummies = pd.get_dummies(keep["subfield_ctrl"], prefix="subfield",
                             drop_first=True, dtype=float)
    data = pd.concat([keep[kept_cols + ["log1p_citations", "year"]],
                      dummies], axis=1)
    terms = kept_cols + list(dummies.columns)
    return data, terms


def _rows(params, bse, ci, pvalues, terms, n_obs) -> pd.DataFrame:
    rows = []
    for term in terms:
        p = float(pvalues[term])
        rows.append({
            "term":    term,
            "coef":    round(float(params[term]), 6),
            "se":      round(float(bse[term]), 6),
            "ci_low":  round(float(ci.loc[term, 0]), 6),
            "ci_high": round(float(ci.loc[term, 1]), 6),
            "p":       p,
            "weak_p":  weak_p(p),
        })
    rows.append({"term": "__n_obs", "coef": float(n_obs), "se": np.nan,
                 "ci_low": np.nan, "ci_high": np.nan, "p": np.nan,
                 "weak_p": False})
    return pd.DataFrame(rows)


def fit(data: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """Primary spec: random intercept per year, REML."""
    import statsmodels.api as sm

    exog = sm.add_constant(data[terms].astype(float))
    res = sm.MixedLM(data["log1p_citations"].astype(float), exog,
                     groups=data["year"]).fit(reml=True)
    return _rows(res.params, res.bse, res.conf_int(), res.pvalues,
                 list(exog.columns), len(data))


def fit_fe_cluster(data: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """Robustness spec: OLS with year fixed effects, SEs clustered by
    year — the precedent's design. Year dummies are absorbed and not
    reported."""
    import statsmodels.api as sm

    year_d = pd.get_dummies(data["year"].astype(int), prefix="year",
                            drop_first=True, dtype=float)
    exog = sm.add_constant(pd.concat([data[terms].astype(float), year_d],
                                     axis=1))
    res = sm.OLS(data["log1p_citations"].astype(float), exog).fit(
        cov_type="cluster", cov_kwds={"groups": data["year"]})
    report = ["const"] + terms
    return _rows(res.params, res.bse, res.conf_int(), res.pvalues,
                 report, len(data))


def main() -> None:
    df = load_joined(DATA)
    if df is None:
        print("  rubric_abstract or outcomes missing; mixed_effects skipped")
        return
    topics = load_topics(DATA)
    print(f"  subfield control: "
          f"{'corpus topics' if topics is not None else 'OpenAlex (pooled)'}")
    out_dir = os.path.join(DATA, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    features = TIER1_FEATURES + TIER2_FEATURES
    data, terms = prepare(df, features, topics)
    if len(data) < 50:
        print(f"  only {len(data)} complete cases; mixed_effects skipped")
        return
    table = fit(data, terms)
    path = os.path.join(out_dir, "mixed_effects_abstract.csv")
    table.to_csv(path, index=False)
    print(f"  model 2: {len(data)} papers → {path}")

    # Robustness specs for model 2.
    robust = []
    fe = fit_fe_cluster(data, terms)
    fe.insert(0, "spec", "fe_cluster")
    robust.append(fe)
    if int(data["year"].max()) >= 2023:
        d23, t23 = prepare(df, features, topics, max_year=2022)
        if len(d23) >= 50:
            pre = fit(d23, t23)
            pre.insert(0, "spec", "pre2023")
            robust.append(pre)
    rpath = os.path.join(out_dir, "mixed_effects_robustness.csv")
    pd.concat(robust, ignore_index=True).to_csv(rpath, index=False)
    print(f"  robustness ({', '.join(r['spec'].iloc[0] for r in robust)}) → {rpath}")

    tier3 = load_tier3(DATA)
    if tier3 is not None:
        sub = df.merge(tier3, on="paper_id", how="inner")
        data3, terms3 = prepare(sub, features + TIER3_FEATURES, topics)
        if len(data3) >= 50:
            table3 = fit(data3, terms3)
            path3 = os.path.join(out_dir, "mixed_effects_fulltext.csv")
            table3.to_csv(path3, index=False)
            print(f"  model 3: {len(data3)} papers → {path3}")
        else:
            print(f"  model 3: only {len(data3)} complete cases; skipped")


if __name__ == "__main__":
    main()
