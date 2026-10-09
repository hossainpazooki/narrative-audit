# Preregistration

Date: 2026-10-09
Status: committed before any `rubric_*` stage has produced results. The
rubric stages (`src/rubric/tier1.py`, `src/rubric/tier3.py`) refuse to run
when this file is absent. Features or models added after this file are
labeled exploratory in the paper. Amendment 1 (below, same date) was also
committed before any rubric stage ran on real data; its changes are part
of the preregistered design, not exploratory.

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
the source post's statistical-rigor advice (which cites a 74% replication
rate for psychology findings at `p ≤ .005` against 28% at
`.005 < p < .05`, Gordon et al. 2021), any result with `.005 < p < .05`
is flagged weak (`weak_p` in every output table) and not claimed as a
finding on its own.

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

## Amendment 1 (2026-10-09, before any rubric run on real data)

Motivated by grounding the design on the published precedent and the
upstream judge-validity study (both summarized in
`docs/related_work.md`), the model specification is amended as follows:

1. **Subfield control.** The primary subfield control is the
   corpus-derived topic (`src/outcomes/topics.py`: TF-IDF over cleaned
   title+abstract, truncated SVD, k-means, k = 15, seed 20261008,
   held-out stability reported), entering as fixed effects. The OpenAlex
   concept (ten largest, rest pooled) is the robustness alternative.
   Reason: the precedent shows a Simpson's reversal in the
   readability-citation association across vs within topics, so the
   subfield variable is load-bearing.
2. **Confound functional form.** `abstract_word_count` and
   `author_count` enter as logs (the negative-control role of abstract
   length is unchanged).
3. **Standardization.** All continuous predictors are z-scored on the
   estimation sample, so reported coefficients are standardized and
   directly comparable to the precedent's Flesch anchor (0.026–0.049).
   Binary predictors stay 0/1.
4. **Citation maturity.** The most recent year in the data is excluded
   from all regressions as citation-immature.
5. **Robustness specs** (reported alongside the primary random-intercept
   models, in `mixed_effects_robustness.csv`): (a) OLS with year fixed
   effects and standard errors clustered by year — the precedent's exact
   design; (b) the primary spec restricted to papers before 2023, since
   the judge-familiarity confound identified by the upstream study and
   citation immaturity both concentrate after 2022.
6. **Tier-2 human anchoring.** In addition to the Tier-3 hand labels, a
   seeded validation set of 100 abstracts is hand-labeled on the three
   Tier-2 prompts (`src/rubric/hand_label.py`, seed 20261008), and
   judge–human agreement for Tier 2 is reported in
   `judge_validity.csv`. Per the upstream study, inter-judge agreement
   is reported but never treated as evidence of validity, and Tier-2
   score distributions are reported so scale collapse is visible.

## Exploratory labeling rule

Anything not listed above — additional features, alternative encodings
(beyond the preregistered ordinal claim-scope encoding and its
unordered-classes robustness check), additional model terms, subgroup
analyses — is exploratory and will be labeled as such.
