# Related work grounding this design

Three papers sit directly under this study. Copies are in
[`docs/papers/`](papers/); this note records what each one is and which
design decisions here it drives.

## 1. The upstream study: LLM judges vs readability metrics

Rangarajan & Krishnan, *LLM Judges Agree With Each Other and Disagree
With Human-Grounded Readability Metrics* (UncertaiNLP 2026, non-archival).
[PDF](papers/rangarajan-krishnan-2026-llm-judges-vs-readability-metrics.pdf)
— the paper whose pipeline this repository forks
([`armandyam/abstract-audit`](https://github.com/armandyam/abstract-audit)).

Over 30,595 NeurIPS abstracts (1987–2025), all 15 classical readability
formulas agree readability declined, while six open-weight judge models
under three prompts all report it flat before 2022 and rising after. The
panel passes every standard reliability check — repeatability (44–98%
identical scores across sampled runs), inter-judge Spearman 0.37–0.41,
Cronbach's α 0.75–0.79, robustness to the standardization baseline — and
is still directionally wrong against the human-grounded reference.
**Agreement is not validity.** Two further measured facts: the judges use
roughly two levels of the five-point scale (97–99% of abstracts round to
a six-model mean of 2 or 3), and the post-2022 rise cannot be separated
from a familiarity account (judges favoring increasingly LLM-assisted
text) without provenance data.

Consequences here:

- Judge–judge agreement in `src/analysis/judge_validity.py` is reported
  but never treated as evidence of validity; the judge–human and
  judge–Tier-1 legs carry the weight.
- Tier-2 judge scores get their own hand-labeled validation set
  (`src/rubric/hand_label.py`), not only Tier 3.
- `rubric_abstract_report.json` records the Tier-2 score distributions,
  so scale collapse is visible before anyone interprets judge variance.
- A pre-2023 restriction is a preregistered robustness spec for every
  model using Tier-2 features (the familiarity confound, and citation
  immaturity, both concentrate after 2022).

## 2. The citation precedent: measurable writing standards

Anonymous, *Measurable Writing Standards for AI-Native Venues* (NeurIPS
2026 submission).
[PDF](papers/anon-2026-measurable-writing-standards.pdf) — note the
draft-template "do not distribute" footer; the copy here is kept at the
repository owner's direction.

Scales the same measurement program to ~2.8M arXiv, 30,595 NeurIPS and
24.5M PubMed papers. Its finding F3 and Appendix E are the closest
published precedent to this study's research question: more readable
NeurIPS papers are more cited. Their citation model —
`log(1+citations)` on z-scored Flesch with log abstract length and log
author count, year fixed effects by within-transformation, standard
errors clustered by year, most recent year excluded as citation-immature
— gives a standardized Flesch coefficient of 0.049, falling to
0.026–0.037 under topic controls with the confidence interval always
excluding zero. Crucially, there is a Simpson's reversal: across topics
the least readable topics are the *most* cited (r = −0.53 over topic
means), while within topics the association is positive. Their topics
are corpus-derived — TF-IDF over cleaned titles+abstracts, truncated
SVD, k-means (k = 10–30), held-out cluster stability checked — not an
external taxonomy.

Consequences here:

- Subfield control determines the sign of the answer, so the primary
  subfield control is corpus-derived topics (`src/outcomes/topics.py`,
  same recipe, seeded), with the OpenAlex concept retained as a
  robustness alternative.
- The regression spec is reconciled to be comparable
  (`src/analysis/mixed_effects.py`): log confounds, z-scored continuous
  predictors so coefficients land on the same scale as their
  0.026–0.049 anchor, the most recent year excluded, and a year-FE +
  year-clustered-SE specification run alongside the preregistered
  random-intercept model.
- Their Semantic Scholar title-matching (99.8% NeurIPS coverage) is the
  fallback citation source if the OpenAlex match rate disappoints.

## 3. The template fixture: refusal is mediated by a single direction

Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee & Nanda, *Refusal in
Language Models Is Mediated by a Single Direction* (NeurIPS 2024;
arXiv:2406.11717v3).
[PDF](papers/arditi-etal-2024-refusal-single-direction.pdf)

The worked example of the source post's abstract template: the post's
"Annotated Abstract" figure labels this paper's abstract sentence by
sentence, and that labeling is the test fixture for
`role_sequence_dist` (`tests/test_role_sequence.py`; the label-to-role
projection is in [`rubric.md`](rubric.md)). The abstract text in the
fixture is verbatim identical to this PDF. The paper is also a useful
positive control for Tier 3: it carries baselines (HarmBench
comparisons), variance (error bars), coherence ablations, limitations,
and a code link — the last in a title-page footnote, which is why the
Tier-3 `code_link` feature reads the globally collected URL list, not
only named sections.
