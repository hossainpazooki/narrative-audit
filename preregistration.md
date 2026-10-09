# Preregistration

Date: 2026-10-09
Status: committed before any `rubric_*` stage has produced results. The
rubric stages (`src/rubric/tier1.py`, `src/rubric/tier3.py`) refuse to run
when this file is absent. Features or models added after this file are
labeled exploratory in the paper.

## Research question

Among accepted NeurIPS papers (1987–2025), do abstract- and
full-text-level rubric features — narrative quality as operationalized
from Neel Nanda's *Highly Opinionated Advice on How to Write ML Papers* —
predict age-normalized citations, after controlling for year, abstract
length, subfield, and author count?

Scope limitation, stated in advance: the corpus contains only accepted
papers. Results are conditional on acceptance and say nothing about
acceptance itself.

## Hypotheses

- **H1 (concreteness).** `concrete_number` is positively associated with
  `cites_per_year`.
- **H2 (structure).** `role_sequence_dist` is negatively associated with
  `cites_per_year` (closer to the template → more citations).
- **H3 (jargon).** `jargon_density` is negatively associated with
  `cites_per_year`.
- **H4 (claims).** Claim scope (`existence < hedged < systematic <
  guarantee`) is positively associated with `cites_per_year`.
- **H5 (judge features).** `judge_motivation` and
  `judge_evidence_standard` are positively associated with
  `cites_per_year`; `judge_n_claims` is tested two-sided with no
  directional prediction.
- **H6 (full text).** Each Tier-3 boolean (`baseline_present`,
  `ablation_present`, `limitations_present`, `variance_reported`,
  `code_link`, `prepost_disclosure`, `fig1_is_diagram`) is tested
  two-sided; we predict positive signs for `baseline_present`,
  `variance_reported` and `code_link` and make no directional prediction
  for the rest.
- **Negative control.** `abstract_word_count` enters every model; a
  material "effect" of raw length at the scale of the rubric effects is
  evidence of residual confounding, and is reported either way.

## Feature list

Exactly the features defined in `docs/rubric.md` at this commit:

- Tier 1: `concrete_number`, `claim_strength`, `role_sequence_dist`,
  `jargon_density`.
- Tier 2: `judge_n_claims`, `judge_motivation`, `judge_evidence_standard`.
- Tier 3: `baseline_present`, `ablation_present`, `limitations_present`,
  `variance_reported`, `code_link`, `prepost_disclosure`,
  `fig1_is_diagram`.

Confounds: year (random intercept), `abstract_word_count`,
`author_count`, `subfield` (ten largest OpenAlex concepts, rest pooled),
`preprint_before_conference`.

Outcomes: `log1p(citations)` (primary, regressions) and `cites_per_year`
(correlations); both reported.

## Models

1. Spearman correlation of each feature with `cites_per_year`, per year
   band (`src/analysis/rubric_spearman.py`; bands defined in
   `src/sample/stratify.py`).
2. Mixed-effects linear model over all scored abstracts:
   `log1p(citations) ~ Tier-1 + Tier-2 features + confounds + (1 | year)`
   (`src/analysis/mixed_effects.py`), REML, partial effects with 95% CIs.
3. The Tier-3 subset: model 2's terms plus the Tier-3 booleans, over the
   sampled papers with usable full text.

## Inference criteria

Effect sizes with confidence intervals are the primary report. Following
the source post's statistical-rigor advice, any result with
`.005 < p < .05` is flagged weak (`weak_p` in every output table) and not
claimed as a finding on its own.

## Measurement validity checks (reported regardless of direction)

- Judge–judge, judge–human (hand labels over 10% of the Tier-3 sample,
  Cohen's κ per feature), and judge–Tier-1 agreement
  (`src/analysis/judge_validity.py`).
- Reverse-causation check: Tier-1 features scored on the arXiv v1
  abstract where one exists, deltas vs camera-ready reported
  (`src/analysis/reverse_causation.py`).

## Sampling

Stratified by year band × within-band citation tercile, seed `20261008`,
target 1,500 primary papers, same-stratum replacement queue capped at 10%
overdraw (`src/sample/stratify.py`). Papers whose full text cannot be
fetched or parsed are excluded from Tier 3 and counted, never silently
dropped.

## Exploratory labeling rule

Anything not listed above — additional features, alternative encodings
(beyond the preregistered ordinal claim-scope encoding and its
unordered-classes robustness check), additional model terms, subgroup
analyses — is exploratory and will be labeled as such.
