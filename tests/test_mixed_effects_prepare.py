"""The regression design matrix (preregistration Amendment 1): immature
most-recent year excluded, continuous predictors standardized, binary
predictors untouched, log confounds, topic fixed effects preferred over
the OpenAlex pooled fallback. Plus a fit smoke test on both specs."""

import numpy as np
import pandas as pd

from analysis.mixed_effects import fit, fit_fe_cluster, prepare
from analysis.rubric_features import TIER1_FEATURES, TIER2_FEATURES

FEATURES = TIER1_FEATURES + TIER2_FEATURES


def _joined(n=300, years=(2018, 2024), seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "paper_id": [f"p{i:04d}" for i in range(n)],
        "year": rng.integers(years[0], years[1] + 1, n),
        "concrete_number": rng.integers(0, 2, n).astype(float),
        "claim_strength_ord": rng.integers(0, 4, n).astype(float),
        "role_sequence_dist": rng.integers(0, 5, n).astype(float),
        "jargon_density": rng.random(n),
        "judge_n_claims": rng.integers(0, 6, n).astype(float),
        "judge_motivation": rng.integers(0, 3, n).astype(float),
        "judge_evidence_standard": rng.integers(0, 3, n).astype(float),
        "abstract_word_count": rng.integers(80, 300, n).astype(float),
        "author_count": rng.integers(1, 9, n).astype(float),
        "preprint_before_conference": rng.integers(0, 2, n).astype(bool),
        "subfield": rng.choice(["NLP", "Vision", "Theory"], n),
    })
    df["log1p_citations"] = (0.3 * df["concrete_number"]
                             + rng.normal(0, 1, n) + 2)
    return df


def _topics(df, k=4, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"paper_id": df["paper_id"],
                         "topic": rng.integers(0, k, len(df))})


def test_most_recent_year_excluded():
    data, _ = prepare(_joined(), FEATURES, topics=None)
    assert data["year"].max() == 2023


def test_max_year_override():
    data, _ = prepare(_joined(), FEATURES, topics=None, max_year=2022)
    assert data["year"].max() == 2022


def test_continuous_standardized_binary_untouched():
    data, _ = prepare(_joined(), FEATURES, topics=None)
    for col in ("jargon_density", "judge_n_claims", "log_abstract_word_count"):
        assert abs(data[col].mean()) < 1e-9
        assert abs(data[col].std(ddof=1) - 1) < 1e-9
    assert set(data["concrete_number"].unique()) <= {0.0, 1.0}
    assert set(data["preprint_before_conference"].unique()) <= {0.0, 1.0}


def test_log_confounds_replace_raw():
    data, terms = prepare(_joined(), FEATURES, topics=None)
    assert "log_abstract_word_count" in terms
    assert "log_author_count" in terms
    assert "abstract_word_count" not in terms


def test_topics_preferred_as_subfield_control():
    df = _joined()
    data, terms = prepare(df, FEATURES, topics=_topics(df, k=4))
    topic_dummies = [t for t in terms if t.startswith("subfield_topic_")]
    assert len(topic_dummies) == 3           # k - 1 after drop_first
    assert not any(t.startswith("subfield_NLP") for t in terms)


def test_openalex_fallback_without_topics():
    data, terms = prepare(_joined(), FEATURES, topics=None)
    assert any(t.startswith("subfield_") and "topic" not in t for t in terms)


def test_fit_both_specs_recover_planted_effect():
    df = _joined(n=500)
    data, terms = prepare(df, FEATURES, topics=_topics(df))
    for fitter in (fit, fit_fe_cluster):
        table = fitter(data, terms)
        row = table[table["term"] == "concrete_number"].iloc[0]
        assert 0.1 < row["coef"] < 0.5       # planted 0.3
        assert {"coef", "se", "ci_low", "ci_high", "p", "weak_p"} <= set(table.columns)
        assert table[table["term"] == "__n_obs"]["coef"].iloc[0] == len(data)
    # year dummies are absorbed, not reported, in the FE spec
    fe = fit_fe_cluster(data, terms)
    assert not any(str(t).startswith("year_") for t in fe["term"])
