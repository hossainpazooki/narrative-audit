#!/usr/bin/env python3
"""Corpus-derived topic controls, following the recipe in Rangarajan &
Krishnan, *Measurable Writing Standards for AI-Native Venues*
(docs/related_work.md, Appendix E of the paper), which in turn follows
Weißer et al. (2020).

Why: that paper shows a Simpson's reversal in the readability-citation
association — across topics the least readable topics are the most cited,
within topics the association is positive — so the subfield control
decides the sign of the answer. An external taxonomy (the OpenAlex
concept in outcomes.parquet) is coarse for NeurIPS; topics derived from
the corpus itself are the primary control, OpenAlex the robustness
alternative.

Recipe: clean each paper's title + abstract (LaTeX commands and URLs
removed), TF-IDF over words and word pairs, truncated SVD to SVD_DIMS,
k-means at K_TOPICS. Everything is seeded and single-init, so the same
corpus yields the same topics.parquet bytes; sklearn is pinned in
requirements.txt because its internals are part of the result.

Reads:  data/processed/{venue}/*.json
Writes: data/outcomes/topics.parquet      paper_id, topic
        data/outcomes/topics_report.json  per-topic size and top terms,
                                          plus a held-out stability check
                                          (adjusted Rand / AMI, as the
                                          source paper reports)

Usage:
  python src/outcomes/topics.py
  python src/outcomes/topics.py --k 20 --venues neurips
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from schema import load_processed

K_TOPICS = 15
SVD_DIMS = 150
SEED = 20261008
TOP_TERMS = 8
MIN_DF = 5

_URL = re.compile(r"https?://\S+")
_LATEX = re.compile(r"\\[a-zA-Z]+")
_MATH = re.compile(r"\$[^$]*\$")


def clean(text: str) -> str:
    text = _URL.sub(" ", text or "")
    text = _MATH.sub(" ", text)
    text = _LATEX.sub(" ", text)
    text = re.sub(r"[{}\\$^_~]", " ", text)   # markup leftovers
    return re.sub(r"\s+", " ", text).strip().lower()


def load_corpus(processed_dir: str, venues: list[str]) -> pd.DataFrame:
    rows = []
    for venue in venues:
        venue_dir = os.path.join(processed_dir, venue)
        if not os.path.isdir(venue_dir):
            continue
        for fname in sorted(os.listdir(venue_dir)):
            if fname.endswith(".json"):
                for p in load_processed(os.path.join(venue_dir, fname)):
                    rows.append({
                        "paper_id": p["paper_id"],
                        "text": clean(f"{p.get('title', '')} {p.get('abstract', '')}"),
                    })
    return pd.DataFrame(rows)


def _vectorize(texts: pd.Series, svd_dims: int, seed: int):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    # The document-frequency floor assumes a real corpus; a tiny one
    # (tests, debugging) would prune every term.
    min_df = MIN_DF if len(texts) >= 100 else 1
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df,
                          stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(texts)
    dims = min(svd_dims, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=max(dims, 2), random_state=seed)
    return vec, X, svd.fit_transform(X)


def fit_topics(df: pd.DataFrame, k: int = K_TOPICS, svd_dims: int = SVD_DIMS,
               seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Corpus -> (paper_id/topic table, report). Deterministic in inputs."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    df = df.sort_values("paper_id").reset_index(drop=True)
    vec, X, Z = _vectorize(df["text"], svd_dims, seed)
    k = min(k, len(df))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(Z)

    # Top terms per topic, from the TF-IDF centroid of its members.
    terms = np.array(vec.get_feature_names_out())
    top_terms: dict[str, list[str]] = {}
    for t in range(k):
        members = X[labels == t]
        if members.shape[0] == 0:
            top_terms[str(t)] = []
            continue
        centroid = np.asarray(members.mean(axis=0)).ravel()
        top_terms[str(t)] = terms[np.argsort(centroid)[::-1][:TOP_TERMS]].tolist()

    # Held-out stability (the source paper's check): cluster two disjoint
    # thirds separately, compare their labelings of the held-out third.
    stability: dict = {}
    if len(df) >= 3 * k:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(df))
        third = len(df) // 3
        a_idx, b_idx, held = order[:third], order[third:2 * third], order[2 * third:]
        km_a = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z[a_idx])
        km_b = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z[b_idx])
        la, lb = km_a.predict(Z[held]), km_b.predict(Z[held])
        stability = {
            "adjusted_rand": round(float(adjusted_rand_score(la, lb)), 4),
            "adjusted_mutual_info": round(float(adjusted_mutual_info_score(la, lb)), 4),
            "held_out_n": int(len(held)),
        }

    out = pd.DataFrame({"paper_id": df["paper_id"], "topic": labels.astype(int)})
    report = {
        "k": k, "svd_dims": int(Z.shape[1]), "seed": seed, "papers": len(df),
        "topic_sizes": {str(t): int((labels == t).sum()) for t in range(k)},
        "top_terms": top_terms,
        "stability": stability,
    }
    return out, report


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=os.path.join(repo, "data", "processed"))
    ap.add_argument("--out-dir",       default=os.path.join(repo, "data", "outcomes"))
    ap.add_argument("--venues", nargs="+", default=["neurips"])
    ap.add_argument("--k", type=int, default=K_TOPICS)
    ap.add_argument("--svd-dims", type=int, default=SVD_DIMS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    corpus = load_corpus(args.processed_dir, args.venues)
    if corpus.empty:
        print("  no processed papers; topics skipped")
        return
    topics, report = fit_topics(corpus, args.k, args.svd_dims, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    pq_path = os.path.join(args.out_dir, "topics.parquet")
    topics.to_parquet(pq_path, index=False)
    with open(os.path.join(args.out_dir, "topics_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"  {len(topics)} papers → {report['k']} topics → {pq_path}"
          + (f" (stability ARI {report['stability']['adjusted_rand']})"
             if report["stability"] else ""))


if __name__ == "__main__":
    main()
