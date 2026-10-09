#!/usr/bin/env python3
"""Tier-2 rubric judge: three prompts over every abstract, integer output.

Prompts (docs/rubric.md gives the Nanda grounding):

  n_claims           0-5  number of distinct claims the abstract makes
  motivation         0-2  is the motivation explicit?
  evidence_standard  0-2  does the abstract state its standard of evidence?

Protocol is abstract-audit's (src/metrics/llm_scores.py): open-weight
models, temperature=0.7, do_sample, max_new_tokens=8, three runs per paper
and prompt, the median reported downstream, per-model z-scores against the
1987-2022 baseline at analysis time.

Reads:  data/processed/{venue}/{venue}_{year}.json
Writes: judge_scores/rubric/{model_name}.csv  (resumable, same schema as
        the readability judges: paper_id, venue, year, prompt_name, run,
        score, raw_output)

Scoring needs a GPU; the pipeline never invokes this script. tier1.py
aggregates whatever CSVs exist under judge_scores/rubric/.

Usage (on cluster):
  python src/rubric/judge_prompts.py --venue neurips --model /path/to/Qwen2.5-32B-Instruct
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from metrics.llm_scores import FIELDNAMES, already_scored, load_processed

PROMPTS: dict[str, tuple[str, int, int]] = {
    # name: (template, lo, hi) — integer answer bounded inclusive
    "n_claims": (
        "How many distinct claims does the following research abstract make? "
        "A claim is a specific assertion the paper argues is true, not "
        "background or motivation. Answer with a single integer from 0 to 5; "
        "answer 5 if there are five or more. \n\n"
        "Format the output as follows: \n"
        "Score: <integer> \n"
        "Text: {SUMMARY}",
        0, 5,
    ),
    "motivation": (
        "Does the following research abstract make its motivation explicit — "
        "why the problem matters and why the reader should care? "
        "Answer with a single integer: 0 if no motivation is given, 1 if the "
        "motivation is implied but never stated, 2 if the motivation is "
        "explicitly stated. \n\n"
        "Format the output as follows: \n"
        "Score: <integer> \n"
        "Text: {SUMMARY}",
        0, 2,
    ),
    "evidence_standard": (
        "Does the following research abstract state what standard of evidence "
        "supports its claims — for example a proof, a systematic evaluation, "
        "concrete measured results, or a worked example? "
        "Answer with a single integer: 0 if the abstract gives no indication "
        "of its evidence, 1 if the kind of evidence is implied but not "
        "specific, 2 if the abstract states concretely what evidence backs "
        "its claims. \n\n"
        "Format the output as follows: \n"
        "Score: <integer> \n"
        "Text: {SUMMARY}",
        0, 2,
    ),
}


def parse_int_score(text: str, lo: int, hi: int) -> int | None:
    """First in-range integer in the judge output; None when there is none.

    Accepts 'Score: 3', a bare '3', and '3/5'; rejects out-of-range
    integers rather than clamping them, so a confabulated '7' is a counted
    null, not a silent 5.
    """
    m = re.search(r"Score:\s*(-?\d+)", text)
    if not m:
        m = re.search(r"-?\d+", text)
    if not m:
        return None
    val = int(m.group(0) if m.lastindex is None else m.group(1))
    return val if lo <= val <= hi else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue",         required=True)
    ap.add_argument("--model",         required=True, help="path to model directory")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out-base",      default=os.path.join("judge_scores", "rubric"))
    ap.add_argument("--batch-size",    type=int,   default=32)
    ap.add_argument("--n-runs",        type=int,   default=3)
    ap.add_argument("--temperature",   type=float, default=0.7)
    ap.add_argument("--load-in-4bit",  action="store_true")
    args = ap.parse_args()

    model_name = os.path.basename(args.model.rstrip("/"))
    out_path = os.path.join(args.out_base, f"{model_name}.csv")
    os.makedirs(args.out_base, exist_ok=True)

    papers = load_processed(args.processed_dir, args.venue)
    if not papers:
        sys.exit(f"No papers found in {args.processed_dir}/{args.venue}/")
    done = already_scored(out_path)
    print(f"Papers: {len(papers):,}; already scored: {len(done)} tuples")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_cfg = BitsAndBytesConfig(load_in_4bit=True) if args.load_in_4bit else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant_cfg,
        torch_dtype="auto" if not args.load_in_4bit else None, device_map="auto")
    model.eval()

    is_new = not os.path.exists(out_path)
    out_fh = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=FIELDNAMES)
    if is_new:
        writer.writeheader()
    total = 0

    for prompt_name, (template, lo, hi) in PROMPTS.items():
        to_score = [p for p in papers
                    if not all((p["paper_id"], prompt_name, run) in done
                               for run in range(args.n_runs))]
        if not to_score:
            print(f"  [{prompt_name}] all done, skipping", flush=True)
            continue
        print(f"  [{prompt_name}] scoring {len(to_score):,} papers × {args.n_runs} runs …",
              flush=True)
        for i in range(0, len(to_score), args.batch_size):
            batch = to_score[i:i + args.batch_size]
            for run in range(args.n_runs):
                msgs = [[{"role": "user",
                          "content": template.replace("{SUMMARY}", p.get("abstract", ""))}]
                        for p in batch]
                inputs = [tokenizer.apply_chat_template(m, tokenize=False,
                                                        add_generation_prompt=True)
                          for m in msgs]
                enc = tokenizer(inputs, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024).to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=8,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature if args.temperature > 0 else None,
                        pad_token_id=tokenizer.eos_token_id)
                for p, inp_ids, gen_ids in zip(batch, enc["input_ids"], out):
                    if (p["paper_id"], prompt_name, run) in done:
                        continue
                    raw = tokenizer.decode(gen_ids[len(inp_ids):],
                                           skip_special_tokens=True).strip()
                    score = parse_int_score(raw, lo, hi)
                    writer.writerow({
                        "paper_id": p["paper_id"], "venue": args.venue,
                        "year": p["year"], "prompt_name": prompt_name,
                        "run": run, "score": score if score is not None else "",
                        "raw_output": raw[:512]})
                    total += 1
            out_fh.flush()
    out_fh.close()
    print(f"\nDone. {total} rows written → {out_path}", flush=True)


if __name__ == "__main__":
    main()
