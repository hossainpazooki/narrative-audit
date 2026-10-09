# The rubric

Every feature the pipeline measures, with its definition, the section of
Neel Nanda's *Highly Opinionated Advice on How to Write ML Papers* (the
source post) it derives from, its tier, and how it is checked. The tiers
follow the design spec:

- **Tier 1** — deterministic, computed over every abstract (`src/rubric/tier1.py`)
- **Tier 2** — LLM judge, every abstract (`src/rubric/judge_prompts.py`)
- **Tier 3** — full text, sampled set (`src/rubric/tier3.py`)

Provenance note: definitions below were checked against the post's full
text (11 May 2025). The abstract template is the structure its Abstract
section prescribes: (1) a first sentence that is "something
uncontroversially true that clearly states which part of ML you're
focused on"; (2) a second sentence "that makes clear there's a need,
something unknown, or a problem for your paper to solve", conveying the
motivation; (3) "the crucial contribution of this paper and why it is
exciting", with key definitions for any necessary jargon; (4) the next
few sentences each carrying "key experimental evidence or additional
important claims", one sentence per idea, with "a concrete metric or
result" folded in wherever possible; (5) one or two closing sentences on
why the paper matters, its implications, and how it fits the broader
context — "also a good place to clearly state your standard of
evidence".

## Tier 1 — deterministic, all abstracts

### `concrete_number` (bool)

**Definition.** The abstract states at least one concrete quantity: a
number attached to `%`, `×`/`x`, `pp`, an `n =`, or a unit
(parameters, models, datasets, epochs, B/M/K, dB, ms, …).

**Source section.** The Abstract section: "If possible, include a
concrete metric or result in any of the above that gives readers a sense
that your results are real and substantial."

**Check.** Regex (`_CONCRETE_PATTERNS` in `tier1.py`). Bare integers
without a unit do not count, so years and section numbers cannot satisfy
the feature. Tested on hand-written positive and negative abstracts.

### `claim_strength` ({guarantee, systematic, hedged, existence})

**Definition.** The calibration class of the abstract's claims, by
majority over matched cue phrases; ties — including zero matches —
resolve to `hedged`, the weakest reading.

**Source section.** "Crafting a Narrative": "Depending on the strength of
the evidence, you can adjust the confidence of a claim" — existence-proof
claims, systematic claims, hedged claims, narrow claims, guarantees;
"stronger statements make for more interesting papers, but require higher
standards of evidence". The post's fifth class, narrow claims ("X is the
best method for specific situations V & W"), is a scope qualifier that
composes with the other four rather than competing with them; the design
spec's four-class feature omits it, and narrow-scope cues are left
unmeasured rather than folded into a neighbouring class.

**Check.** Cue-phrase lexicon (`src/rubric/lexicon/claim_strength.json`),
case-insensitive, longest-first. One code-level extension the literal
lexicon cannot express: "across N …" / "on N datasets|models|…" counts as
a systematic cue. One hand-written abstract per class is a test fixture.

For correlation and regression the classes are encoded as ordinal claim
*scope* (`existence=0 < hedged=1 < systematic=2 < guarantee=3`,
`src/analysis/rubric_features.py`); the unordered-classes robustness check
belongs to the paper, not the pipeline.

### `role_sequence_dist` (int)

**Definition.** Label every sentence of the abstract with one template
role — `situate`, `gap`, `contribution`, `evidence`, `impact` — collapse
consecutive repeats, and take the Levenshtein distance to the template
sequence `[situate, gap, contribution, evidence, impact]`. 0 means the
abstract follows the post's structure exactly; collapsing first means the
distance measures ordering and omissions, not abstract length.

**Source section.** The abstract template (five slots above).

**Check.** Sentence labels come from cue counts
(`src/rubric/lexicon/role_cues.json`) plus position priors; ties resolve
toward template order and cue-less sentences continue the previous role
(first sentence: `situate`). The post's own case study — its "Annotated
Abstract" figure breaks the abstract of *Refusal in Language Models Is
Mediated by a Single Direction* (Arditi et al., NeurIPS 2024) down into
the purpose of each sentence — is the test fixture. The figure's eight
annotation labels project onto the five template roles as follows, and
the labeler reproduces that projection exactly
(`situate, gap, contribution ×3, evidence, impact ×2`, distance 0):

| Post annotation | Template role |
|---|---|
| Topic | `situate` |
| Motivation | `gap` |
| Contribution; Detail/Nuance; Contribution 2 | `contribution` |
| Evidence; Weaker result | `evidence` |
| Narrow impact; Broad impact | `impact` |

### `jargon_density` (float)

**Definition.** The share of word tokens absent from a general-English
frequency list (top 20,000 word forms,
`src/rubric/lexicon/english_top20k.txt`, generated once from wordfreq and
committed). Extends the existing acronym rule rather than double-counting
it: tokens the Barnett & Doubleday acronym detector accepts are skipped
(acronym density already measures them), and a hyphenated word counts as
jargon only when no component is common English.

**Source section.** The Abstract section ("Include key definitions for
any necessary jargon, though jargon should be avoided if possible") and
"Unnecessary Complexity and Verbosity": "use plain language and minimize
jargon except where the jargon is needed to precisely convey your
meaning".

**Check.** Deterministic set membership; tested on text with known jargon
shares.

### Inherited Tier-1 features

The abstract-audit metrics continue to run unchanged and are analysis
covariates rather than rubric features: the readability set, hedging,
hype (sensational language), signposting, acronym density, passive rate,
and TTR. Their definitions are documented in the abstract-audit sources
(`src/metrics/`).

## Tier 2 — LLM judge, all abstracts

Protocol: open-weight models, `temperature=0.7`, `max_new_tokens=8`, three
runs per paper × prompt × model; the per-model median is taken, means
across models are stored in `rubric_abstract.parquet`
(`judge_*` columns), and analyses z-score per model against the 1987–2022
baseline, exactly as abstract-audit does for its readability judges.
Integer answers outside the stated range are nulls, counted — never
clamped. Prompts live in `src/rubric/judge_prompts.py`.

### `judge_n_claims` (0–5)

**Definition.** The number of distinct claims the abstract makes (5 =
five or more). **Source section.** "The Essence of a Paper": "a paper
should present a narrative of one to three specific concrete claims that
you believe to be true". **Check.** LLM judge; parser tested on malformed
output; validity triangulated in `judge_validity.csv`.

### `judge_motivation` (0/1/2)

**Definition.** Is the motivation explicit? 0 none, 1 implied, 2 explicit.
**Source section.** The Abstract section's second-sentence slot ("makes
clear there's a need … this should convey (some of) the motivation") and
"The Essence of a Paper": "Motivate why someone should care about them."
**Check.** As above.

### `judge_evidence_standard` (0/1/2)

**Definition.** Does the abstract state what standard of evidence supports
its claims (proof, systematic evaluation, measured result, worked
example)? 0 no indication, 1 implied, 2 concrete. **Source section.**
The Abstract section's closing slot: "a good place to clearly state your
standard of evidence", with the post's own examples ("A preliminary step
towards…", "Provides compelling evidence that…"); backed by "Crafting a
Narrative": stronger statements require higher standards of evidence.
**Check.** As above.

## Tier 3 — full text, sampled set

Each feature is a boolean extracted by section-targeted LLM JSON
extraction (`{"value": …, "evidence": "verbatim quote"}`); the quoted span
is verified against the text the judge saw, and unverifiable spans are
flagged (`*_evidence_verified = false`), never silently trusted. Majority
over three runs per model, then across models; split votes are counted
nulls. 10% of the sample is hand-labeled (`data/hand_labels/tier3.csv`)
and Cohen's κ per feature is reported in `judge_validity.csv`.

| Feature | Definition | Source section |
|---|---|---|
| `baseline_present` | compares against at least one baseline/alternative, not only its own numbers | "Baselines are Crucial": show better results "than plausible alternatives", not just "decent" ones |
| `ablation_present` | removes or varies a component to measure its contribution | "Ablation studies": "remove one change at a time, observe its effect" |
| `limitations_present` | explicitly discusses limitations of its own work | "Discussion": "Explaining the limitations of your work is a crucial part of scientific good practice" |
| `variance_reported` | error bars, standard deviations, CIs, or multi-seed results | "Can you trust your results?": "What is your sample size? What is your standard deviation? Are your results clearly distinguishable from noise?" |
| `code_link` | a repository link or explicit code-availability statement | "Reproducibility & Publishing code": sharing code "enables others to build on your work" |
| `prepost_disclosure` | discloses planned-vs-exploratory analyses (preregistration, stated held-out protocol, explicit post-hoc labels) | "Avoiding Misleading Evidence (Cherry-Picking and Post-Hoc Analysis)": "clearly track which experimental results were obtained before versus after you formulated your claim" |
| `fig1_is_diagram` | Figure 1 is an explanatory diagram/schematic rather than a results plot (judged from the Figure 1 caption) | "Figures": "an explanatory diagram rather than a graph … a high-effort but very effective figure one" |

Sections fed to the judge per feature are listed in
`SECTION_TARGETS` (`src/rubric/tier3.py`); section names come from the
fulltext stage's heading normalization (`src/fulltext/sections.py`).

## Outcome and confounds (not rubric features)

From the `outcomes` stage (`src/outcomes/`): `citations`,
`cites_per_year` (age-normalized against a fixed census year),
`field_pct` (percentile within subfield × year), `author_count`,
`subfield` (top OpenAlex concept), `arxiv_id`, `arxiv_v1_date`, and
`preprint_before_conference`. `abstract_word_count` (from the hedging
metric) is the preregistered negative control.

The primary subfield control in the regressions is not the OpenAlex
concept but the corpus-derived topic (`src/outcomes/topics.py`,
`data/outcomes/topics.parquet`); see `docs/related_work.md` for why the
subfield variable is load-bearing, and preregistration Amendment 1 for
the spec.
