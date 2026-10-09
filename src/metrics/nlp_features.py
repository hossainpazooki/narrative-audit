#!/usr/bin/env python3
"""
Per-paper NLP features: noun chunks, verbs, nouns, and numbers.

Based on Hohmann, Barnett, King & Connell (2025), Scientometrics 130:3349-3366.
DOI: https://doi.org/10.1007/s11192-025-05353-8
Source R code: https://github.com/agbarnett/narrator (1_process_pubmed.R)

The original R code uses UDpipe with the English EWT model for POS tagging.
This Python port uses spaCy (en_core_web_sm) which uses the same Universal
Dependencies UPOS tag set (NOUN, VERB, NUM, etc.), making the results directly
comparable.

FOUR METRICS computed in one NLP pass (for efficiency):

1. Noun chunks   — count of 3+ consecutive NOUN tokens per 100 words
   From R: pasted = paste(upos, collapse='/'); str_count(pasted, 'NOUN/NOUN/NOUN')
   Counts non-overlapping occurrences of NOUN/NOUN/NOUN in the POS sequence
   (punctuation and hyphens removed before counting, per lines 53–55 of R code).

2. Verbs         — VERB token count per 100 words
   From R: universal counts of UPOS == 'VERB'

3. Nouns         — NOUN token count per 100 words
   From R: universal counts of UPOS == 'NOUN'

4. Numbers       — NUM token count per 100 words
   From R: universal counts of UPOS == 'NUM'

Setup:
  pip install spacy
  python -m spacy download en_core_web_sm

Output: four CSVs, one per metric, all under data/per_paper/{metric}/{venue}.csv

  data/per_paper/noun_chunks/{venue}.csv
    Columns: paper_id, venue, year, n_words, noun_chunks_count, noun_chunks_per_100

  data/per_paper/verbs/{venue}.csv
    Columns: paper_id, venue, year, n_words, verb_count, verb_per_100

  data/per_paper/nouns/{venue}.csv
    Columns: paper_id, venue, year, n_words, noun_count, noun_per_100

  data/per_paper/numbers/{venue}.csv
    Columns: paper_id, venue, year, n_words, num_count, num_per_100

Usage:
  python src/metrics/nlp_features.py
  python src/metrics/nlp_features.py --venues neurips iclr --batch-size 64
"""

from __future__ import annotations
import argparse
import os
import re
import sys

import pandas as pd

try:
    import spacy
except ImportError:
    sys.exit("pip install spacy && python -m spacy download en_core_web_sm")

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)
from metrics._narrator_utils import preprocess, word_count, iter_venue_papers

VENUES = ["neurips", "iclr", "icml", "arxiv"]

# Remove PUNCT and SPACE before building POS sequence (matches R lines 53–55)
_SKIP_POS = {"PUNCT", "SPACE"}


def _noun_chunks(doc) -> int:
    """Count non-overlapping NOUN/NOUN/NOUN patterns in POS sequence."""
    pos_seq = "/".join(
        t.pos_ for t in doc if t.pos_ not in _SKIP_POS
    )
    return len(re.findall(r"NOUN/NOUN/NOUN", pos_seq))


def process_venue(
    venue: str,
    processed_dir: str,
    out_base: str,
    nlp,
    batch_size: int = 64,
) -> int:
    papers = []
    texts = []
    for _, paper in iter_venue_papers(processed_dir, [venue]):
        abstract = preprocess(paper.get("abstract", ""))
        papers.append({
            "paper_id": paper["paper_id"],
            "venue":    paper["venue"],
            "year":     paper["year"],
            "n_words":  word_count(abstract),
        })
        texts.append(abstract)

    rows_nc = []
    rows_vb = []
    rows_nn = []
    rows_nm = []

    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        meta = papers[i]
        nw = meta["n_words"]
        denom = nw if nw > 0 else 1

        nc = _noun_chunks(doc)
        vb = sum(1 for t in doc if t.pos_ == "VERB")
        nn = sum(1 for t in doc if t.pos_ == "NOUN")
        nm = sum(1 for t in doc if t.pos_ == "NUM")

        rows_nc.append({**meta, "noun_chunks_count": nc,
                        "noun_chunks_per_100": round(nc / denom * 100, 4)})
        rows_vb.append({**meta, "verb_count": vb,
                        "verb_per_100": round(vb / denom * 100, 4)})
        rows_nn.append({**meta, "noun_count": nn,
                        "noun_per_100": round(nn / denom * 100, 4)})
        rows_nm.append({**meta, "num_count": nm,
                        "num_per_100": round(nm / denom * 100, 4)})

        if (i + 1) % 1000 == 0:
            print(f"    {venue}: {i + 1}/{len(papers)} abstracts", flush=True)

    for metric, rows in [
        ("noun_chunks", rows_nc),
        ("verbs",       rows_vb),
        ("nouns",       rows_nn),
        ("numbers",     rows_nm),
    ]:
        out_dir = os.path.join(out_base, metric)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{venue}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"  [{venue}] {len(papers)} papers → noun_chunks, verbs, nouns, numbers")
    return len(papers)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=os.path.join(repo, "data", "processed"))
    parser.add_argument("--out-base",      default=os.path.join(repo, "data", "per_paper"))
    parser.add_argument("--venues", nargs="+", default=VENUES)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="en_core_web_sm",
                        help="spaCy model (en_core_web_sm or en_core_web_lg)")
    args = parser.parse_args()

    print(f"Loading spaCy model: {args.model}")
    nlp = spacy.load(args.model, disable=["ner", "lemmatizer"])

    total = 0
    for venue in args.venues:
        total += process_venue(
            venue, args.processed_dir, args.out_base, nlp, args.batch_size
        )
    print(f"\nTotal: {total} papers")


if __name__ == "__main__":
    main()
