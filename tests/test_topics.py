"""Corpus-derived topics: determinism (the control enters a preregistered
model, so the same corpus must yield the same topics), input-order
independence, and the small-corpus edges."""

import numpy as np
import pandas as pd

from outcomes.topics import clean, fit_topics


def _corpus(n_per_group=30, seed=3):
    rng = np.random.default_rng(seed)
    vocab = {
        "vision": ["image", "convolution", "pixel", "segmentation", "visual",
                   "object", "detection", "camera"],
        "rl": ["reward", "policy", "agent", "environment", "exploration",
               "q-learning", "bandit", "regret"],
        "theory": ["bound", "convergence", "convex", "regret", "complexity",
                   "proof", "generalization", "pac"],
    }
    rows = []
    for g, words in vocab.items():
        for i in range(n_per_group):
            text = " ".join(rng.choice(words, size=30))
            rows.append({"paper_id": f"{g}_{i:03d}", "text": text})
    return pd.DataFrame(rows)


def test_clean_strips_latex_urls_math():
    assert clean(r"We study \emph{CNNs} at https://x.y/z with $x^2$ terms") == \
        "we study cnns at with terms"


def test_same_corpus_same_topics():
    corpus = _corpus()
    a, _ = fit_topics(corpus, k=3, seed=1)
    b, _ = fit_topics(corpus, k=3, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_input_order_does_not_matter():
    corpus = _corpus()
    a, _ = fit_topics(corpus, k=3, seed=1)
    b, _ = fit_topics(corpus.sample(frac=1, random_state=9), k=3, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_separable_groups_land_in_distinct_topics():
    topics, _ = fit_topics(_corpus(), k=3, seed=1)
    topics["group"] = topics["paper_id"].str.split("_").str[0]
    # each synthetic group should be dominated by one topic
    for _, grp in topics.groupby("group"):
        assert grp["topic"].value_counts().iloc[0] >= 0.8 * len(grp)


def test_report_structure():
    _, report = fit_topics(_corpus(), k=3, seed=1)
    assert report["k"] == 3
    assert sum(report["topic_sizes"].values()) == report["papers"]
    assert all(len(v) > 0 for v in report["top_terms"].values())
    assert "adjusted_rand" in report["stability"]


def test_k_capped_by_corpus_size():
    corpus = _corpus(n_per_group=2)          # 6 papers
    topics, report = fit_topics(corpus, k=15, seed=1)
    assert report["k"] == 6
    assert topics["topic"].nunique() <= 6
