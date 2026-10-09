#!/usr/bin/env python3
"""Tier-3 rubric: full-text booleans over the sampled set, with evidence.

Each feature is a boolean extracted by section-targeted LLM JSON extraction:
the judge sees only the sections relevant to the feature and must answer

    {"value": true|false, "evidence": "<verbatim quote from the text>"}

The quoted span is verified against the supplied text (whitespace-
normalized containment); an unverifiable span keeps the boolean but flags
evidence_verified=False, so cherry-checked rationales are visible, never
silently trusted.

Features (docs/rubric.md gives definitions and Nanda grounding):
  baseline_present, ablation_present, limitations_present,
  variance_reported, code_link, prepost_disclosure, fig1_is_diagram

Two entry points:

  aggregate (default, run by main.py):
      judge_scores/tier3/{model}.jsonl + data/fulltext/{paper_id}.json
      -> data/rubric/rubric_fulltext.parquet (+ report)
      Majority over runs per model, then majority across models; an
      unparseable judgment after 3 runs is a null, counted in the report.

  score --model …  (GPU, never run by the pipeline):
      scores the sampled papers with one open-weight model, resumable,
      abstract-audit decoding protocol (temperature=0.7, three runs;
      max_new_tokens=128 here because the answer carries a quote).

10% of the sample is hand-labeled (data/hand_labels/tier3.csv);
src/analysis/judge_validity.py reports Cohen's kappa per feature.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import pandas as pd

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from rubric.tier1 import require_preregistration

FEATURES = ["baseline_present", "ablation_present", "limitations_present",
            "variance_reported", "code_link", "prepost_disclosure",
            "fig1_is_diagram"]

# Which normalized sections (sections.py vocabulary) each feature reads.
# "captions" is the figure-caption list; "urls" the collected links.
SECTION_TARGETS: dict[str, list[str]] = {
    "baseline_present":    ["methods", "experiments", "results"],
    "ablation_present":    ["methods", "experiments", "results", "appendix"],
    "limitations_present": ["limitations", "discussion", "conclusion"],
    "variance_reported":   ["experiments", "results", "appendix"],
    "code_link":           ["abstract", "introduction", "conclusion", "urls"],
    "prepost_disclosure":  ["methods", "experiments", "discussion", "limitations"],
    "fig1_is_diagram":     ["captions"],
}

QUESTIONS: dict[str, str] = {
    "baseline_present":
        "Does the paper compare its method against at least one baseline or "
        "alternative method (not just report its own numbers)?",
    "ablation_present":
        "Does the paper report an ablation — removing or varying a component "
        "of its method to measure that component's contribution?",
    "limitations_present":
        "Does the paper explicitly discuss limitations of its own work?",
    "variance_reported":
        "Does the paper report variance for its results — error bars, "
        "standard deviations, confidence intervals, or results over "
        "multiple seeds?",
    "code_link":
        "Does the paper provide a link to its code (e.g. a repository URL "
        "or an explicit code-availability statement)?",
    "prepost_disclosure":
        "Does the paper disclose which analyses were planned before seeing "
        "results versus exploratory/post-hoc (e.g. a preregistration, a "
        "held-out test protocol stated in advance, or an explicit "
        "'exploratory' label)?",
    "fig1_is_diagram":
        "Based on this figure caption, is Figure 1 an explanatory diagram "
        "or schematic (as opposed to a results plot, table or photo)?",
}

PROMPT_TEMPLATE = (
    "You are auditing a machine-learning paper. Answer the question using "
    "ONLY the text below. Reply with a single JSON object and nothing else:\n"
    '{{"value": true or false, "evidence": "a short verbatim quote from the '
    'text that justifies the answer, or \\"\\" if value is false"}}\n\n'
    "Question: {QUESTION}\n\n"
    "Text:\n{TEXT}"
)


# ── JSON parsing (tested: malformed, truncated, extra-text) ──────────────────

def parse_tier3_json(raw: str) -> dict | None:
    """Judge output -> {"value": bool, "evidence": str} or None.

    Tolerates prose around the JSON object (first balanced {...} is taken)
    and boolean spellings True/False; a truncated or otherwise unparseable
    object is None — the protocol counts nulls rather than guessing.
    """
    if not raw or not isinstance(raw, str):
        return None
    start = raw.find("{")
    if start == -1:
        return None
    depth, end = 0, None
    in_str = esc = False
    for i, ch in enumerate(raw[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:                       # truncated object
        return None
    snippet = raw[start:end]
    snippet = re.sub(r"\bTrue\b", "true", snippet)
    snippet = re.sub(r"\bFalse\b", "false", snippet)
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("value"), bool):
        return None
    evidence = obj.get("evidence")
    return {"value": obj["value"],
            "evidence": evidence if isinstance(evidence, str) else ""}


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_evidence(evidence: str, text: str) -> bool:
    """A quoted span counts as verified only if it appears verbatim in the
    text the judge was shown (whitespace-insensitive)."""
    if not evidence:
        return False
    return _norm_ws(evidence) in _norm_ws(text)


# ── feature context from a fulltext JSON ─────────────────────────────────────

def feature_text(doc: dict, feature: str, max_chars: int = 12000) -> str:
    targets = SECTION_TARGETS[feature]
    chunks: list[str] = []
    if "captions" in targets:
        caps = doc.get("figure_captions") or []
        fig1 = [c for c in caps if re.match(r"\s*fig(?:ure)?\.?\s*1\b", c, re.I)]
        chunks.extend(fig1 or caps[:1])
    if "urls" in targets:
        urls = doc.get("urls") or []
        if urls:
            chunks.append("Links found in the paper: " + " ".join(urls))
    wanted = [t for t in targets if t not in ("captions", "urls")]
    for sec in doc.get("sections") or []:
        if sec.get("normalized") in wanted and sec.get("text"):
            chunks.append(f"[{sec.get('heading') or sec['normalized']}]\n{sec['text']}")
    # Fall back to everything when normalization found none of the targets;
    # an empty prompt would make the judge guess.
    if not chunks:
        chunks = [s.get("text", "") for s in doc.get("sections") or []]
    return "\n\n".join(chunks)[:max_chars]


def build_prompt(doc: dict, feature: str) -> str:
    return PROMPT_TEMPLATE.format(QUESTION=QUESTIONS[feature],
                                  TEXT=feature_text(doc, feature))


# ── aggregation: judge jsonl -> rubric_fulltext.parquet ──────────────────────

def _majority(values: list[bool]) -> bool | None:
    if not values:
        return None
    t = sum(values)
    if t * 2 == len(values):
        return None                        # split vote is a counted null
    return t * 2 > len(values)


def load_fulltext(fulltext_dir: str, paper_id: str) -> dict | None:
    path = os.path.join(fulltext_dir, f"{paper_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def aggregate(judge_dir: str, fulltext_dir: str, sample_csv: str,
              out_dir: str) -> None:
    paths = sorted(glob.glob(os.path.join(judge_dir, "*.jsonl")))
    if not os.path.exists(sample_csv):
        print("  no sample.csv; rubric_fulltext skipped")
        return
    sample = pd.read_csv(sample_csv)
    if "is_replacement" in sample.columns:
        repl = sample["is_replacement"].astype(str).str.lower().isin(("true", "1"))
        sampled = list(sample.loc[~repl, "paper_id"])
    else:
        sampled = list(sample["paper_id"])

    # rows[(paper, feature, model)] = [bool judgments]; ev holds candidate spans
    rows: dict[tuple[str, str, str], list[bool]] = {}
    ev: dict[tuple[str, str], list[str]] = {}
    unparseable = 0
    for path in paths:
        model = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                parsed = parse_tier3_json(rec.get("raw_output", ""))
                key = (rec["paper_id"], rec["feature"])
                if parsed is None:
                    unparseable += 1
                    continue
                rows.setdefault((*key, model), []).append(parsed["value"])
                if parsed["evidence"]:
                    ev.setdefault(key, []).append(parsed["evidence"])

    out_rows = []
    null_counts = {f: 0 for f in FEATURES}
    for paper_id in sampled:
        doc = load_fulltext(fulltext_dir, paper_id)
        if doc is not None and doc.get("fulltext_status") != "ok":
            doc = None                     # excluded from Tier 3 per spec
        row: dict = {"paper_id": paper_id,
                     "fulltext_ok": doc is not None}
        for feat in FEATURES:
            per_model = [_majority(v) for (pid, f, _m), v in rows.items()
                         if pid == paper_id and f == feat]
            per_model = [v for v in per_model if v is not None]
            verdict = _majority(per_model) if doc is not None else None
            if verdict is None:
                null_counts[feat] += 1
            evidence, verified = "", False
            if doc is not None:
                text = _norm_ws(feature_text(doc, feat))
                for span in ev.get((paper_id, feat), []):
                    if _norm_ws(span) in text:
                        evidence, verified = span, True
                        break
                if not evidence and ev.get((paper_id, feat)):
                    evidence = ev[(paper_id, feat)][0]
            row[feat] = verdict
            row[f"{feat}_evidence"] = evidence
            row[f"{feat}_evidence_verified"] = verified
        out_rows.append(row)

    df = pd.DataFrame(out_rows).sort_values("paper_id").reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    pq_path = os.path.join(out_dir, "rubric_fulltext.parquet")
    df.to_parquet(pq_path, index=False)
    report = {
        "sampled":        len(sampled),
        "fulltext_ok":    int(df["fulltext_ok"].sum()),
        "judge_models":   [os.path.splitext(os.path.basename(p))[0] for p in paths],
        "unparseable_judgments": unparseable,
        "null_by_feature": null_counts,
    }
    with open(os.path.join(out_dir, "rubric_fulltext_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"  {len(df)} rows → {pq_path} "
          f"({report['fulltext_ok']} with full text, {unparseable} unparseable judgments)")


# ── GPU scoring runner ───────────────────────────────────────────────────────

def score(args: argparse.Namespace) -> None:
    model_name = os.path.basename(args.model.rstrip("/"))
    out_path = os.path.join(args.out_base, f"{model_name}.jsonl")
    os.makedirs(args.out_base, exist_ok=True)

    sample = pd.read_csv(args.sample)
    paper_ids = list(sample["paper_id"])

    done: set[tuple[str, str, int]] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["paper_id"], r["feature"], r["run"]))
    print(f"Sampled papers: {len(paper_ids)}; already scored: {len(done)} tuples")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 device_map="auto")
    model.eval()

    out_fh = open(out_path, "a", encoding="utf-8")
    total = 0
    for paper_id in paper_ids:
        doc = load_fulltext(args.fulltext_dir, paper_id)
        if doc is None or doc.get("fulltext_status") != "ok":
            continue
        for feat in FEATURES:
            for run in range(args.n_runs):
                if (paper_id, feat, run) in done:
                    continue
                prompt = build_prompt(doc, feat)
                msgs = [{"role": "user", "content": prompt}]
                inp = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True)
                enc = tokenizer([inp], return_tensors="pt", truncation=True,
                                max_length=8192).to(model.device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=128,
                                         do_sample=True, temperature=0.7,
                                         pad_token_id=tokenizer.eos_token_id)
                raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip()
                out_fh.write(json.dumps({"paper_id": paper_id, "feature": feat,
                                         "run": run, "raw_output": raw},
                                        ensure_ascii=False) + "\n")
                total += 1
            out_fh.flush()
    out_fh.close()
    print(f"Done. {total} judgments written → {out_path}")


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    agg = sub.add_parser("aggregate", help="judge jsonl -> rubric_fulltext.parquet")
    sc  = sub.add_parser("score", help="GPU: score sampled papers with one model")
    for p in (agg, sc):
        p.add_argument("--judge-dir",    default=os.path.join(repo, "judge_scores", "tier3"))
        p.add_argument("--fulltext-dir", default=os.path.join(repo, "data", "fulltext"))
        p.add_argument("--sample",       default=os.path.join(repo, "data", "sample", "sample.csv"))
        p.add_argument("--out-dir",      default=os.path.join(repo, "data", "rubric"))
    sc.add_argument("--model", required=True)
    sc.add_argument("--out-base", default=os.path.join(repo, "judge_scores", "tier3"))
    sc.add_argument("--n-runs", type=int, default=3)

    args = ap.parse_args()
    require_preregistration(repo)
    if args.cmd == "score":
        score(args)
    else:
        if args.cmd is None:
            args = agg.parse_args([])
        aggregate(args.judge_dir, args.fulltext_dir, args.sample, args.out_dir)


if __name__ == "__main__":
    main()
