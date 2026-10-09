# Narrative Audit

## About

Does narrative quality, as defined by Neel Nanda's *Highly Opinionated Advice
on How to Write ML Papers*, predict citation outcomes among accepted NeurIPS
papers (1987–2025)?

This repository is a fork of
[`armandyam/abstract-audit`](https://github.com/armandyam/abstract-audit). It
keeps that pipeline whole — the stage runner, the Pydantic schema, the
classical writing metrics, the LLM-judge protocol, `reference/` and
`verify.py` — and adds a rubric-measurement layer and an outcomes layer on
top:

- **Outcomes.** Citations per paper from OpenAlex, age-normalized, with
  subfield, author count and preprint-before-conference confounds.
- **Rubric, Tier 1.** Deterministic abstract features derived from the Nanda
  post: concrete numbers, claim strength, role-sequence distance to his
  abstract template, jargon density.
- **Rubric, Tier 2.** Three LLM-judge prompts over every abstract (number of
  distinct claims, explicit motivation, stated standard of evidence), run
  under the abstract-audit judge protocol.
- **Rubric, Tier 3.** Full-text booleans (baselines, ablations, limitations,
  variance reporting, code link, pre/post-hoc disclosure, Figure-1 type) over
  a stratified 1,500-paper sample, extracted by section-targeted LLM JSON
  extraction with quoted evidence spans.
- **Analysis.** Per-feature Spearman correlations by year band, and
  mixed-effects regressions of `log1p(citations)` on rubric features with
  year random intercepts.

Research question: *Among accepted NeurIPS papers, do abstract- and
full-text-level rubric features predict age-normalized citations, after
controlling for year, length, subfield, and author count?*

Stated limitation: the corpus contains only accepted papers, so results are
conditional on acceptance and say nothing about acceptance itself.

Every rubric feature is documented in `docs/rubric.md` with its definition,
the Nanda section it derives from, its tier, and how it is checked.
`preregistration.md` states the hypotheses, feature list and models, and is
committed before any `rubric_*` stage runs; the rubric stages refuse to run
without it.

## Corpus

NeurIPS abstracts are not redistributed here. `src/scraping/scrape_neurips.py`
retrieves them from `papers.nips.cc`. Every record carries a `paper_id` of the
form `{year}_{hash}`, where the hash is the one in the source URL, so each
record resolves to its origin.

## Contents

```
src/            pipeline (abstract-audit stages + outcomes, rubric, sample, fulltext)
docs/rubric.md  every rubric feature: definition, source section, tier, check
preregistration.md
tests/          tests, documented in tests/README.md
paper/          figure sources, their generators, and rendered figures
judge_scores/   shipped judge scores (readability judges from abstract-audit;
                rubric and tier3 judges are written here by the GPU runners)
reference/      expected pipeline outputs for verify.py
data/           generated, absent from a fresh clone
```

## Execution

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python src/scraping/scrape_neurips.py --start 1987 --end 2025
python main.py
```

`main.py --list` enumerates the stages. `--stages`, `--from` and `--skip`
select subsets. The new stages sit between `metrics_nlp` and `aggregate`:

```
outcomes         OpenAlex match -> data/outcomes/outcomes.parquet
rubric_abstract  Tier-1 features + Tier-2 judge medians -> data/rubric/rubric_abstract.parquet
sample           stratified year-band x citation-tercile sample -> data/sample/sample.csv
fulltext         arXiv PDF -> GROBID (primary) / pymupdf fallback -> data/fulltext/{paper_id}.json
rubric_fulltext  Tier-3 booleans + rationale -> data/rubric/rubric_fulltext.parquet
```

Network- and GPU-dependent inputs degrade explicitly, never silently: a paper
that cannot be matched, fetched or parsed is counted in the stage's report
file and flagged in its output.

Single-paper debugging:

```bash
PYTHONPATH=src python -m rubric score 2024_f545448535dfde4f9786555403ab7c49
```

### Full text

The `fulltext` stage needs a running [GROBID](https://github.com/kermitt2/grobid)
server (`GROBID_URL`, default `http://localhost:8070`); papers GROBID cannot
parse fall back to a pymupdf heading heuristic. Papers with
`fulltext_status != ok` are excluded from Tier 3 and counted in
`data/fulltext/fulltext_report.json`; the sampler pre-draws same-stratum
replacements up to 10% overdraw.

### Judges

Tier-2 and Tier-3 judge scoring follows the abstract-audit protocol: open-weight
models, `temperature=0.7`, three runs per paper and prompt, median reported,
per-model z-scores against the 1987–2022 baseline. Scoring needs a GPU
(`src/rubric/judge_prompts.py --model …`, `src/rubric/tier3.py score --model …`);
the pipeline itself only aggregates whatever judge outputs are present under
`judge_scores/`.

## Analysis

1. Spearman per feature vs `cites_per_year`, per year band (heatmap).
2. Mixed-effects: `log1p(citations) ~ features + confounds + (1 | year)`,
   partial effects with confidence intervals.
3. The Tier-3 subset: model 2 plus the full-text booleans.

Guards, per the design spec: abstract word count as a negative control;
judge–judge vs judge–human vs judge–Tier-1 agreement reported regardless of
direction; arXiv-v1 abstracts scored where available as a reverse-causation
check; effect sizes reported, with anything at `.005 < p < .05` flagged weak.

## Verification

`verify.py` compares regenerated artefacts against `reference/`: per-paper
metric CSVs and the new per-paper parquet/CSV files byte for byte, plot-ready
CSVs and regression tables numerically at `atol=0`, figures byte for byte.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

`tests/README.md` states the coverage criterion inherited from
abstract-audit: every branch that handles malformed or edge-case input has a
test.

## Licence

Copyright 2026 Hossain Pazooki. Portions copyright 2026 Ajay Mandyam
Rangarajan and Jeyashree Krishnan (abstract-audit).

Code under `src/` and `paper/`: MIT (`LICENSE`). Derived data under
`judge_scores/` and `reference/`: CC BY 4.0 (`LICENSE-DATA`). Neither covers
the NeurIPS abstracts, which are not distributed here.
