#!/usr/bin/env python3
"""Tier-1 rubric features (deterministic, every abstract) and the
rubric_abstract stage.

Features — each documented in docs/rubric.md with the Nanda section it
derives from:

  concrete_number     bool   a number with %, x, pp, "n =", or a unit
  claim_strength      str    {guarantee, systematic, hedged, existence}
  role_sequence_dist  int    edit distance of the sentence-role sequence to
                             the template [situate, gap, contribution,
                             evidence, impact]
  jargon_density      float  share of word tokens absent from a general-
                             English frequency list (acronyms excluded —
                             they are already measured by the acronym rule)

Stage output: data/rubric/rubric_abstract.parquet — one row per paper with
the Tier-1 features plus the Tier-2 judge medians merged from
judge_scores/rubric/*.csv when present (null otherwise, counted in
data/rubric/rubric_abstract_report.json).

The stage refuses to run unless preregistration.md exists at the repo root
(the design spec's pre-registration guard).

Usage:
  python src/rubric/tier1.py
  python src/rubric/tier1.py --venues neurips
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from schema import load_processed
from metrics.acronyms import _is_acronym_token

_LEXICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lexicon")

ROLES = ["situate", "gap", "contribution", "evidence", "impact"]
TEMPLATE = list(ROLES)


def _load_cues(name: str) -> dict[str, re.Pattern]:
    with open(os.path.join(_LEXICON_DIR, name), encoding="utf-8") as fh:
        raw = json.load(fh)
    patterns = {}
    for cls, cues in raw.items():
        if cls.startswith("_"):
            continue
        cues = sorted(cues, key=len, reverse=True)
        alts = [r"\s+".join(re.escape(w) for w in c.split()) for c in cues]
        patterns[cls] = re.compile(
            "|".join(r"\b" + a + r"\b" for a in alts),
            re.IGNORECASE,
        )
    return patterns

_CLAIM_PATTERNS = _load_cues("claim_strength.json")
_ROLE_PATTERNS  = _load_cues("role_cues.json")


def _load_english_words() -> frozenset[str]:
    path = os.path.join(_LEXICON_DIR, "english_top20k.txt")
    words = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(line)
    return frozenset(words)

_ENGLISH = _load_english_words()


# ── concrete_number ───────────────────────────────────────────────────────────

_NUM = r"\d+(?:[.,]\d+)?"
_UNITS = (r"%|percent(?:age)?|pp|×|x|times|points?|pts?|dB|ms|sec(?:onds)?|"
          r"[KMGT]B|[BMK]\b|billion|million|thousand|parameters?|params?|"
          r"examples?|samples?|epochs?|layers?|models?|datasets?|tasks?|"
          r"languages?|seeds?|tokens?|FLOPs?")
_CONCRETE_PATTERNS = [
    re.compile(rf"{_NUM}\s*(?:{_UNITS})(?![a-z])", re.IGNORECASE),
    re.compile(rf"n\s*=\s*{_NUM}", re.IGNORECASE),
    re.compile(rf"{_NUM}\s*[×x]\s"),
    re.compile(rf"[×]\s*{_NUM}"),
]


def concrete_number(text: str) -> bool:
    """True when the abstract states at least one concrete quantity.

    Nanda (Abstract): "Include a concrete metric or result that shows your
    results are real and substantial." Bare years or section numbers do not
    count; a number needs %, x, pp, "n =" or a unit to qualify.
    """
    return any(p.search(text) for p in _CONCRETE_PATTERNS)


# ── claim_strength ────────────────────────────────────────────────────────────

# Cues the literal lexicon cannot express: a count of models/datasets/tasks
# is a systematic-claim marker ("across 13 models", "on 7 benchmarks").
_SYSTEMATIC_NUMERIC = re.compile(
    r"\bacross\s+\d+\b|\bon\s+\d+\s+(?:datasets?|models?|tasks?|benchmarks?|languages?|domains?)\b",
    re.IGNORECASE)


def claim_strength(text: str) -> str:
    """Majority class over matched cue phrases; 'hedged' on ties.

    Zero matches is a tie among nothing and also resolves to 'hedged', the
    weakest reading, so an abstract never gains strength it did not state.
    """
    counts = {cls: len(pat.findall(text)) for cls, pat in _CLAIM_PATTERNS.items()}
    counts["systematic"] += len(_SYSTEMATIC_NUMERIC.findall(text))
    best = max(counts.values())
    winners = [cls for cls, n in counts.items() if n == best and best > 0]
    if len(winners) == 1:
        return winners[0]
    return "hedged"


# ── role_sequence_dist ────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# Guard the common abbreviations that would otherwise split a sentence.
_ABBREV = re.compile(r"\b(e\.g|i\.e|cf|vs|et al|etc|Fig|Eq|Sec|resp)\.$")


def split_sentences(text: str) -> list[str]:
    parts, out = _SENT_SPLIT.split(text.strip()), []
    for part in parts:
        if out and _ABBREV.search(out[-1]):
            out[-1] = out[-1] + " " + part
        elif part:
            out.append(part)
    return [s for s in (p.strip() for p in out) if s]


def label_roles(sentences: list[str]) -> list[str]:
    """One template role per sentence, from cue counts plus position.

    Ties (including zero matched cues) resolve toward template order: among
    the tied roles, the first whose template index is >= the previous
    sentence's index wins; with none, the previous label repeats. The first
    sentence starts from index 0, so it defaults to 'situate'.
    """
    labels: list[str] = []
    prev_idx = 0
    for sent in sentences:
        counts = {role: len(pat.findall(sent)) for role, pat in _ROLE_PATTERNS.items()}
        best = max(counts.values())
        tied = [r for r, n in counts.items() if n == best and best > 0]
        if len(tied) == 1:
            choice = tied[0]
        elif tied:
            forward = [r for r in tied if ROLES.index(r) >= prev_idx]
            choice = min(forward, key=ROLES.index) if forward else max(tied, key=ROLES.index)
        else:
            choice = ROLES[prev_idx] if labels else "situate"
        labels.append(choice)
        prev_idx = ROLES.index(choice)
    return labels


def _collapse(seq: list[str]) -> list[str]:
    out: list[str] = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


def _levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def role_sequence_dist(text: str) -> int:
    """Edit distance of the collapsed sentence-role sequence to the Nanda
    template. Consecutive repeats collapse first — a contribution developed
    over three sentences is still one 'contribution' slot — so the distance
    measures ordering and omissions, not abstract length."""
    sentences = split_sentences(text)
    if not sentences:
        return len(TEMPLATE)
    return _levenshtein(_collapse(label_roles(sentences)), TEMPLATE)


# ── jargon_density ────────────────────────────────────────────────────────────

_WORD = re.compile(r"[A-Za-z][A-Za-z\-]*")


def jargon_density(text: str) -> float:
    """Share of word tokens outside the general-English top-20k list.

    Extends the acronym rule rather than double-counting it: tokens the
    Barnett & Doubleday acronym detector accepts are skipped entirely, and
    hyphenated words count as jargon only when no component is common
    English. Single letters are ignored.
    """
    tokens = [t for t in _WORD.findall(text) if len(t) > 1]
    if not tokens:
        return 0.0
    n_jargon = 0
    for tok in tokens:
        if _is_acronym_token(tok):
            continue
        parts = [p for p in tok.lower().split("-") if p]
        if parts and all(p not in _ENGLISH for p in parts):
            n_jargon += 1
    return round(n_jargon / len(tokens), 4)


# ── per-paper row ─────────────────────────────────────────────────────────────

def compute_row(paper: dict) -> dict:
    abstract = paper.get("abstract", "")
    return {
        "paper_id":           paper["paper_id"],
        "venue":              paper["venue"],
        "year":               paper["year"],
        "concrete_number":    concrete_number(abstract),
        "claim_strength":     claim_strength(abstract),
        "role_sequence_dist": role_sequence_dist(abstract),
        "jargon_density":     jargon_density(abstract),
    }


# ── Tier-2 merge ──────────────────────────────────────────────────────────────

TIER2_PROMPTS = ["n_claims", "motivation", "evidence_standard"]


def load_tier2_medians(judge_dir: str) -> pd.DataFrame | None:
    """judge_scores/rubric/*.csv -> one row per paper.

    Protocol per abstract-audit: per model, the median over the three runs
    of each prompt; across models, the mean of those medians. Columns:
    judge_{prompt} and judge_n_models.
    """
    paths = sorted(glob.glob(os.path.join(judge_dir, "*.csv")))
    if not paths:
        return None
    per_model = []
    for path in paths:
        df = pd.read_csv(path)
        df = df.dropna(subset=["score"])
        med = (df.groupby(["paper_id", "prompt_name"])["score"]
                 .median().unstack("prompt_name"))
        med["__model"] = os.path.splitext(os.path.basename(path))[0]
        per_model.append(med)
    allm = pd.concat(per_model)
    out = allm.groupby("paper_id").agg(
        {p: "mean" for p in TIER2_PROMPTS if p in allm.columns} | {"__model": "nunique"})
    out = out.rename(columns={p: f"judge_{p}" for p in TIER2_PROMPTS} |
                             {"__model": "judge_n_models"})
    return out.reset_index()


# ── stage ─────────────────────────────────────────────────────────────────────

def require_preregistration(repo: str) -> None:
    if not os.path.exists(os.path.join(repo, "preregistration.md")):
        sys.exit("rubric stages refuse to run: preregistration.md is missing "
                 "from the repository root. Commit the preregistered "
                 "hypotheses, feature list and models first (design spec §5).")


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=os.path.join(repo, "data", "processed"))
    ap.add_argument("--judge-dir",     default=os.path.join(repo, "judge_scores", "rubric"))
    ap.add_argument("--out-dir",       default=os.path.join(repo, "data", "rubric"))
    ap.add_argument("--venues", nargs="+", default=["neurips"])
    args = ap.parse_args()

    require_preregistration(repo)

    rows = []
    for venue in args.venues:
        venue_dir = os.path.join(args.processed_dir, venue)
        if not os.path.isdir(venue_dir):
            print(f"  [{venue}] not found")
            continue
        for fname in sorted(os.listdir(venue_dir)):
            if fname.endswith(".json"):
                papers = load_processed(os.path.join(venue_dir, fname))
                rows.extend(compute_row(p) for p in papers)
                print(f"  [{venue}] {fname}: {len(papers)} papers", flush=True)
    if not rows:
        sys.exit("no processed papers found")

    df = pd.DataFrame(rows)
    tier2 = load_tier2_medians(args.judge_dir)
    if tier2 is not None:
        df = df.merge(tier2, on="paper_id", how="left")
        scored = int(df["judge_n_models"].notna().sum())
    else:
        for p in TIER2_PROMPTS:
            df[f"judge_{p}"] = pd.NA
        df["judge_n_models"] = pd.NA
        scored = 0
        print("  no Tier-2 judge scores found; judge columns are null")

    df = df.sort_values("paper_id").reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    pq_path = os.path.join(args.out_dir, "rubric_abstract.parquet")
    df.to_parquet(pq_path, index=False)
    report = {
        "papers":            int(len(df)),
        "tier2_scored":      scored,
        "tier2_unscored":    int(len(df) - scored),
        "claim_strength":    {k: int(v) for k, v in df["claim_strength"].value_counts().items()},
        "concrete_number":   int(df["concrete_number"].sum()),
    }
    with open(os.path.join(args.out_dir, "rubric_abstract_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"  {len(df)} rows → {pq_path}")


if __name__ == "__main__":
    main()
