"""Per-paper correlation between writing metrics and the LLM-as-judge
readability score on NeurIPS abstracts.

This is a purely correlational analysis. It reports what the judge score
co-varies with; it makes no causal claim.

The LLM judge was scored on NeurIPS 1987-2024 only (the 2025 cohort was
never scored), so this analysis covers those 24,772 papers, not the full
30,595 used elsewhere.

Outputs (deterministic, consumed by the manuscripts):
  data/pgfplots/judge_metric_corr.csv     metric x {pooled, within-year, per-model} Spearman r
  figs/fig_judge_metric_heatmap.png       metric x model Spearman heatmap

Run from the repo root:  python src/analysis/judge_feature_correlation.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

_D = os.environ.get("AA_DATA", "data")
BASE = os.path.join(_D, "per_paper")
JUDGE = os.environ.get("AA_JUDGE", "judge_scores")
OUT  = os.path.join(_D, "pgfplots")
FIGS = os.environ.get("AA_FIGS")  # optional diagnostic PNG target; unset means skip

MODELS = ["Gemma-3-27B-Instruct", "Gemma-4-31B-Instruct",
          "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct",
          "Mixtral-8x7B-Instruct", "Qwen2.5-32B-Instruct"]

# (key, per_paper_dir, column, pretty label, group)
# group: "penalized" (judge scores lower), "rewarded" (judge scores higher),
# assigned after the fact; here we just list every metric to correlate.
METRICS = [
    ("flesch_ease", "readability", "flesch_ease", "Flesch ease"),
    ("flesch_kincaid", "readability", "flesch_kincaid", "Flesch-Kincaid"),
    ("gunning_fog", "readability", "gunning_fog", "Gunning Fog"),
    ("smog", "readability", "smog", "SMOG"),
    ("dale_chall", "readability", "dale_chall", "Dale-Chall"),
    ("spache", "readability", "spache", "Spache"),
    ("coleman_liau", "readability", "coleman_liau", "Coleman-Liau"),
    ("ari", "readability", "ari", "ARI"),
    ("linsear_write", "readability", "linsear_write", "Linsear Write"),
    ("lix", "readability", "lix", "LIX"),
    ("rix", "readability", "rix", "RIX"),
    ("forcast", "readability", "forcast", "FORCAST"),
    ("powers_sumner_kearl", "readability", "powers_sumner_kearl", "Powers-Sumner-Kearl"),
    ("avg_sentence_length", "readability", "avg_sentence_length", "Sentence length"),
    ("avg_syllables_per_word", "readability", "avg_syllables_per_word", "Syllables/word"),
    ("acronym_density", "acronyms", "abstract_acronym_density", "Acronym density"),
    ("np_density", "np_density", "np_density", "NP density"),
    ("parse_depth", "parse_depth", "parse_depth", "Parse depth"),
    ("passive_rate", "passive_rate", "passive_rate", "Passive rate"),
    ("ttr", "ttr", "ttr", "Type-token ratio"),
    ("clauses_per_sent", "clauses_per_sentence", "clauses_per_sent", "Clauses/sentence"),
    ("signposting_per_100", "signposting", "signposting_per_100", "Signposting"),
    ("hedging_per_100", "hedging", "hedging_per_100", "Hedging"),
    ("has_narrator", "active_narration", "has_narrator", "Active narration"),
    ("hype_total_per_100", "sensational_language", "hype_total_per_100", "Sensational language"),
    ("noun_chunks_per_100", "noun_chunks", "noun_chunks_per_100", "Noun chunks"),
    ("noun_per_100", "nouns", "noun_per_100", "Noun density"),
    ("verb_per_100", "verbs", "verb_per_100", "Verb density"),
    ("num_per_100", "numbers", "num_per_100", "Numbers"),
]


def spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 100:
        return np.nan
    return a[m].rank().corr(b[m].rank())


def load_judge() -> pd.DataFrame:
    frames = {}
    for m in MODELS:
        d = pd.read_csv(f"{JUDGE}/neurips/{m}.csv",
                        usecols=["paper_id", "score"])
        d = d[pd.to_numeric(d["score"], errors="coerce").notna()]
        d["score"] = d["score"].astype(float)
        frames[m] = d.groupby("paper_id")["score"].median()
    J = pd.DataFrame(frames)
    J["judge_avg"] = J[MODELS].mean(axis=1)
    return J


def build_table(J: pd.DataFrame) -> pd.DataFrame:
    yr = pd.read_csv(f"{BASE}/readability/neurips.csv",
                     usecols=["paper_id", "year"]).set_index("paper_id")
    M = J.join(yr, how="inner")
    for key, d, col, _label in METRICS:
        s = pd.read_csv(f"{BASE}/{d}/neurips.csv",
                        usecols=["paper_id", col]).set_index("paper_id")[col]
        M = M.join(s.rename(key), how="left")
    return M


def zwithin(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("year")[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    J = load_judge()
    M = build_table(J)
    keys = [k for k, *_ in METRICS]
    labels = {k: lab for k, _d, _c, lab in METRICS}
    print(f"NeurIPS papers with judge scores: {len(M)}  "
          f"years {int(M.year.min())}-{int(M.year.max())}")

    # --- correlations: pooled, within-year, per model ---
    rows = []
    for k in keys:
        row = {"metric": k, "label": labels[k],
               "r_pooled": spearman(M[k], M["judge_avg"]),
               "r_within": spearman(zwithin(M, k), zwithin(M, "judge_avg"))}
        for m in MODELS:
            row[m] = spearman(M[k], M[m])
        rows.append(row)
    corr = pd.DataFrame(rows).sort_values("r_pooled")
    corr.to_csv(f"{OUT}/judge_metric_corr.csv", index=False)
    print(f"wrote {OUT}/judge_metric_corr.csv  ({len(corr)} metrics)")

    # --- heatmap PNG: metrics (rows, sorted by r_pooled) x models (cols) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        H = corr.set_index("label")[MODELS]
        short = [m.replace("-Instruct", "") for m in MODELS]
        fig, ax = plt.subplots(figsize=(6.8, 7.0))
        im = ax.imshow(H.values, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(H.index)))
        ax.set_yticklabels(H.index, fontsize=7)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                ax.text(j, i, f"{H.values[i, j]:+.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, label="Spearman r", shrink=0.55)
        ax.set_title("Writing metric vs LLM-as-judge readability:\n"
                     "per-paper Spearman r (NeurIPS, n={:,})".format(len(M)),
                     fontsize=9)
        plt.tight_layout()
        if not FIGS:
            print("diagnostic heatmap not written (set AA_FIGS to enable); "
                  "the paper figure comes from paper/analysis/make_heatmap.py")
        else:
            os.makedirs(FIGS, exist_ok=True)
            plt.savefig(f"{FIGS}/fig_judge_metric_heatmap.png", dpi=150)
            print(f"wrote {FIGS}/fig_judge_metric_heatmap.png")
    except Exception as e:  # pragma: no cover
        print(f"heatmap skipped: {e}")


if __name__ == "__main__":
    main()
