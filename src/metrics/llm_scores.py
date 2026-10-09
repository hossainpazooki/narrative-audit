#!/usr/bin/env python3
"""
LLM-as-judge readability scoring for a single venue with a single model.

Reads:  data/processed/{venue}/{venue}_{year}.json
Writes: judge_scores/{venue}/{model_name}.csv

One row per paper × prompt × run. Fully resumable — already-scored paper_ids
are skipped on re-run.

Three prompt templates (from Hohmann et al. / original analyze_llm_scores.py):
  simple        — bare 1–5 scale
  ascb          — with readability guidance criteria
  own_reasoning — asking model to use its own judgment

Usage (on cluster):
  python src/metrics/llm_scores.py --venue neurips --model /path/to/Mistral-7B-Instruct
  python src/metrics/llm_scores.py --venue neurips --model /path/to/Llama-3.1-70B-Instruct --load-in-4bit
  python src/metrics/llm_scores.py --venue neurips --model /path/to/model --batch-size 16 --n-runs 3
"""

from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import sys

# Exact prompts from Cachola et al. (2025) "Evaluating the Evaluators", Table 10.
# Text is placed at the end via {SUMMARY} substitution, matching their setup exactly.
PROMPTS = {
    "simple": (
        "On a scale of 1 to 5, what is the reading ease of the following text? "
        "1 indicates the text requires expert background knowledge and 5 indicates "
        "the text is readable to the general population. Assume the reader is an adult. \n\n"
        "Format the output as follows: \n"
        "Score: <score> \n"
        "Text: {SUMMARY}"
    ),
    "ascb": (
        "On a scale of 1 to 5, what is the reading ease of the following text? "
        "1 indicates the text requires expert background knowledge and 5 indicates "
        "the text is readable to the general population. "
        "Characteristics of a highly readable text include: \n"
        "- Know your audience, and focus and organize your information for that particular audience. \n"
        "- Focus on the big picture. What larger problem is your work a part of? "
        "What major ideas or issues does your work address? "
        "How will your work help global understanding of some issue? \n"
        "- Avoid jargon. If you must use a technical term, make sure to explain it, but simplify the language. \n"
        "- Try to use metaphors or analogies to everyday experiences that people can relate to. \n"
        "- Underscore the importance of public support for exploratory research and scientific information, "
        "and the role of this information in providing the context for effective policy making. \n\n"
        "Assume the reader is an adult. Do not use Flesch-Kincaid or other readability formulas. "
        "Use your own judgment to rate the text. \n\n"
        "Format the output as follows: \n"
        "Score: <score> \n"
        "Text: {SUMMARY}"
    ),
    "own_reasoning": (
        "On a scale of 1 to 5, what is the reading ease of the following text? "
        "1 indicates the text requires expert background knowledge and 5 indicates "
        "the text is readable to the general population. \n"
        "Assume the reader is an adult. Do not use Flesch-Kincaid or other readability formulas. "
        "Use your own judgment to rate the text. \n\n"
        "Format the output as follows: \n"
        "Score: <score> \n"
        "Text: {SUMMARY}"
    ),
}

FIELDNAMES = ["paper_id", "venue", "year", "prompt_name", "run", "score", "raw_output"]


def parse_score(text: str) -> float | None:
    m = re.search(r"Score:\s*([1-5](?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\b([1-5](?:\.\d+)?)\b", text)
    if m:
        return float(m.group(1))
    return None


def load_processed(processed_dir: str, venue: str) -> list[dict]:
    papers = []
    for path in sorted(glob.glob(os.path.join(processed_dir, venue, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            papers.extend(json.load(fh))
    return papers


def already_scored(out_path: str) -> set[tuple[str, str, int]]:
    """Return set of (paper_id, prompt_name, run) already written."""
    done: set[tuple[str, str, int]] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            done.add((row["paper_id"], row["prompt_name"], int(row["run"])))
    return done


# NOTE: a dead `score_batch` helper used to live here with max_new_tokens=32;
# the live generation path below uses max_new_tokens=8. It was removed because
# its 32-token setting leaked into the paper's methods text by mistake.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue",          required=True,
                        help="e.g. neurips, iclr, icml")
    parser.add_argument("--model",          required=True,
                        help="Path to model directory")
    parser.add_argument("--processed-dir",  default="data/processed")
    parser.add_argument("--out-base",       default="judge_scores")
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--n-runs",         type=int,   default=3)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--load-in-4bit",   action="store_true")
    args = parser.parse_args()

    model_name = os.path.basename(args.model.rstrip("/"))
    out_dir    = os.path.join(args.out_base, args.venue)
    out_path   = os.path.join(out_dir, f"{model_name}.csv")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Venue:  {args.venue}")
    print(f"Model:  {model_name}")
    print(f"Output: {out_path}")

    papers = load_processed(args.processed_dir, args.venue)
    if not papers:
        sys.exit(f"No papers found in {args.processed_dir}/{args.venue}/")
    print(f"Papers: {len(papers):,}")

    done = already_scored(out_path)
    print(f"Already scored: {len(done)} (paper×prompt×run tuples)")

    # Load model
    print("Loading model …", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_cfg = BitsAndBytesConfig(load_in_4bit=True) if args.load_in_4bit else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_cfg,
        torch_dtype="auto" if not args.load_in_4bit else None,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.", flush=True)

    # Open output CSV (append if resuming)
    is_new = not os.path.exists(out_path)
    out_fh = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=FIELDNAMES)
    if is_new:
        writer.writeheader()

    total_written = 0

    for prompt_name, prompt_text in PROMPTS.items():
        # Which papers still need scoring for this prompt?
        to_score = [
            p for p in papers
            if not all(
                (p["paper_id"], prompt_name, run) in done
                for run in range(args.n_runs)
            )
        ]
        if not to_score:
            print(f"  [{prompt_name}] all done, skipping", flush=True)
            continue

        print(f"  [{prompt_name}] scoring {len(to_score):,} papers × {args.n_runs} runs …",
              flush=True)

        texts = [p.get("abstract", "") for p in to_score]

        for i in range(0, len(to_score), args.batch_size):
            batch_papers = to_score[i : i + args.batch_size]
            batch_texts  = texts[i : i + args.batch_size]

            # Score this batch for all runs at once
            for run in range(args.n_runs):
                import torch
                messages_batch = [
                    [{"role": "user", "content": prompt_text.replace("{SUMMARY}", t)}]
                    for t in batch_texts
                ]
                inputs_batch = [
                    tokenizer.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=True
                    )
                    for m in messages_batch
                ]
                enc = tokenizer(
                    inputs_batch, return_tensors="pt",
                    padding=True, truncation=True, max_length=1024,
                ).to(model.device)

                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=8,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature if args.temperature > 0 else None,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                for j, (inp_ids, gen_ids) in enumerate(
                    zip(enc["input_ids"], out)
                ):
                    p = batch_papers[j]
                    if (p["paper_id"], prompt_name, run) in done:
                        continue
                    new_tokens = gen_ids[len(inp_ids):]
                    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    score = parse_score(raw)
                    writer.writerow({
                        "paper_id":   p["paper_id"],
                        "venue":      args.venue,
                        "year":       p["year"],
                        "prompt_name": prompt_name,
                        "run":        run,
                        "score":      score if score is not None else "",
                        "raw_output": raw[:512],
                    })
                    total_written += 1

            out_fh.flush()
            done_count = i + len(batch_papers)
            if done_count % 500 == 0 or done_count == len(to_score):
                print(f"    {done_count}/{len(to_score)}", flush=True)

    out_fh.close()
    print(f"\nDone. {total_written} rows written → {out_path}", flush=True)


if __name__ == "__main__":
    main()
