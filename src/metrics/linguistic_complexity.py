#!/usr/bin/env python3
"""
Linguistic complexity metrics derived from spaCy dependency parsing.

One spaCy pass per abstract (parser + lemmatizer enabled); writes five CSVs:
  data/per_paper/parse_depth/{venue}.csv
  data/per_paper/clauses_per_sentence/{venue}.csv
  data/per_paper/np_density/{venue}.csv
  data/per_paper/passive_rate/{venue}.csv
  data/per_paper/ttr/{venue}.csv

Metrics:
  parse_depth      — avg max dependency-tree depth per sentence
  clauses_per_sent — avg subordinate clauses per sentence
  np_density       — spaCy noun phrases / total non-punct tokens
  passive_rate     — proportion of sentences containing passive construction
  ttr              — Type-Token Ratio: unique content lemmas / content tokens

Source: Hohmann et al. (2025); original R code in agbarnett/narrator.

Run:
  python src/metrics/linguistic_complexity.py
  python src/metrics/linguistic_complexity.py --venues iclr icml
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import sys

import numpy as np
import spacy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from schema import load_processed as _load_one


def _load_venue(venue: str, processed_dir: str) -> list[dict]:
    import glob, json
    pattern = os.path.join(processed_dir, venue, "*.json")
    papers = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            papers.extend(json.load(fh))
    return papers

VENUES = ["neurips", "iclr", "icml", "arxiv"]
OUT_BASE = "data/per_paper"
BATCH_SIZE = 128

CLAUSE_DEPS  = {"advcl", "relcl", "ccomp", "xcomp", "acl"}
# OntoNotes (spaCy default) uses nsubjpass/auxpass; UD uses nsubj:pass/aux:pass
PASSIVE_DEPS = {"nsubjpass", "auxpass", "nsubj:pass", "aux:pass"}
CONTENT_POS  = {"NOUN", "VERB", "ADJ", "ADV"}

# (dir_name, csv_column_name, key_in_metrics_dict)
OUTPUTS = [
    ("parse_depth",          "parse_depth",      "parse_depth"),
    ("clauses_per_sentence", "clauses_per_sent",  "clauses_per_sent"),
    ("np_density",           "np_density",        "np_density"),
    ("passive_rate",         "passive_rate",       "passive_rate"),
    ("ttr",                  "ttr",               "ttr"),
]


def _tree_depth(token) -> int:
    depth = 0
    while token.head != token:
        token = token.head
        depth += 1
    return depth


def _compute(doc) -> dict | None:
    sentences = list(doc.sents)
    if not sentences:
        return None

    depths, clause_counts, passive_flags = [], [], []
    for sent in sentences:
        toks = list(sent)
        depths.append(max((_tree_depth(t) for t in toks), default=0))
        clause_counts.append(sum(1 for t in toks if t.dep_ in CLAUSE_DEPS))
        passive_flags.append(int(any(t.dep_ in PASSIVE_DEPS for t in toks)))

    content = [t for t in doc if t.pos_ in CONTENT_POS]
    ttr = len({t.lemma_.lower() for t in content}) / max(len(content), 1)

    non_ws = [t for t in doc if not t.is_punct and not t.is_space]
    np_density = len(list(doc.noun_chunks)) / max(len(non_ws), 1)

    return {
        "parse_depth":     float(np.mean(depths)),
        "clauses_per_sent": float(np.mean(clause_counts)),
        "np_density":      np_density,
        "passive_rate":    float(np.mean(passive_flags)),
        "ttr":             ttr,
    }


def process_venue(venue: str, nlp, out_base: str, processed_dir: str):
    papers = _load_venue(venue, processed_dir)
    if not papers:
        print(f"  [{venue}] no data", file=sys.stderr)
        return

    # Open one writer per output metric
    file_handles, writers = {}, {}
    for dir_name, col, _ in OUTPUTS:
        d = os.path.join(out_base, dir_name)
        os.makedirs(d, exist_ok=True)
        fh = open(os.path.join(d, f"{venue}.csv"), "w", newline="", encoding="utf-8")
        w = csv.writer(fh)
        w.writerow(["paper_id", "venue", "year", "n_words", col])
        file_handles[dir_name] = fh
        writers[dir_name] = w

    ids, texts, years, nwords = [], [], [], []
    for p in papers:
        text = (p.get("abstract") or "").strip()
        if len(text.split()) < 5:
            continue
        ids.append(p["paper_id"])
        texts.append(text[:8000])
        years.append(p["year"])
        nwords.append(len(re.findall(r"\w+", text)))

    done = 0
    for i in range(0, len(texts), BATCH_SIZE):
        bt = texts[i : i + BATCH_SIZE]
        bi = ids[i : i + BATCH_SIZE]
        by = years[i : i + BATCH_SIZE]
        bn = nwords[i : i + BATCH_SIZE]

        for doc, pid, yr, nw in zip(nlp.pipe(bt, batch_size=BATCH_SIZE), bi, by, bn):
            m = _compute(doc)
            if m is None:
                continue
            for dir_name, col, key in OUTPUTS:
                writers[dir_name].writerow([pid, venue, yr, nw, m[key]])
        done += len(bt)
        if done % 5000 == 0:
            print(f"    {venue}: {done}/{len(texts)} abstracts")

    for fh in file_handles.values():
        fh.close()

    metric_names = ", ".join(d for d, _, _ in OUTPUTS)
    print(f"  [{venue}] {done} papers → {metric_names}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venues",        nargs="+", default=VENUES)
    parser.add_argument("--out-base",      default=OUT_BASE)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model",         default="en_core_web_sm")
    args = parser.parse_args()

    print(f"Loading spaCy model '{args.model}' (parser + lemmatizer enabled) …")
    nlp = spacy.load(args.model, disable=["ner"])
    nlp.max_length = 10000

    total = 0
    for venue in args.venues:
        process_venue(venue, nlp, args.out_base, args.processed_dir)
        total += len(_load_venue(venue, args.processed_dir))
    print(f"\nTotal: {total} papers")


if __name__ == "__main__":
    main()
