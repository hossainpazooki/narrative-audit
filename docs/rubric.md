# The rubric

Every feature the pipeline measures, with its definition, the section of
Neel Nanda's *Highly Opinionated Advice on How to Write ML Papers* (the
source post) it derives from, its tier, and how it is checked. The tiers
follow the design spec:

- **Tier 1** — deterministic, computed over every abstract (`src/rubric/tier1.py`)
- **Tier 2** — LLM judge, every abstract (`src/rubric/judge_prompts.py`)
- **Tier 3** — full text, sampled set (`src/rubric/tier3.py`)

Provenance note: the source post is paraphrased here from its published
text and widely mirrored summaries. The abstract template below is the
post's five-slot structure: (1) an uncontroversially true first sentence
that situates the reader in the right sub-field; (2) a sentence making
clear there is a need, something unknown, or a problem to solve; (3) the
crucial contribution, with key definitions for necessary jargon; (4) a
concrete metric or result showing the results are real and substantial;
(5) one or two closing sentences on why the paper matters and how it fits
the broader context.

## Tier 1 — deterministic, all abstracts

### `concrete_number` (bool)

**Definition.** The abstract states at least one concrete quantity: a
number attached to `%`, `×`/`x`, `pp`, an `n =`, or a unit
(parameters, models, datasets, epochs, B/M/K, dB, ms, …).

**Source section.** The abstract template, slot 4: include a concrete
metric or result that shows the results are real and substantial.

**Check.** Regex (`_CONCRETE_PATTERNS` in `tier1.py`). Bare integers
without a unit do not count, so years and section numbers cannot satisfy
the feature. Tested on hand-written positive and negative abstracts.

### `claim_strength` ({guarantee, systematic, hedged, existence})

**Definition.** The calibration class of the abstract's claims, by
majority over matched cue phrases; ties — including zero matches —
resolve to `hedged`, the weakest reading.

**Source section.** "Crafting claims": confidence should be adjusted to
evidence strength — existence-proof claims, systematic claims, hedged
claims, or guarantees; stronger statements need higher standards of
evidence.

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
(first sentence: `situate`). The post's own worked example — the abstract
of *Refusal in Language Models Is Mediated by a Single Direction* (Arditi
et al., NeurIPS 2024) — is the test fixture: it labels
`situate, gap, contribution ×3, evidence, impact ×2` and scores distance 0.

### `jargon_density` (float)

**Definition.** The share of word tokens absent from a general-English
frequency list (top 20,000 word forms,
`src/rubric/lexicon/english_top20k.txt`, generated once from wordfreq and
committed). Extends the existing acronym rule rather than double-counting
it: tokens the Barnett & Doubleday acronym detector accepts are skipped
(acronym density already measures them), and a hyphenated word counts as
jargon only when no component is common English.

**Source section.** The abstract template, slot 3 ("key definitions for
necessary jargon") and the writing-style advice to avoid jargon and rare
words.

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
five or more). **Source section.** "The Narrative": a paper should present
one to three specific concrete claims. **Check.** LLM judge; parser tested
on malformed output; validity triangulated in `judge_validity.csv`.

### `judge_motivation` (0/1/2)

**Definition.** Is the motivation explicit? 0 none, 1 implied, 2 explicit.
**Source section.** The abstract template slot 2, and "What makes a good
narrative": motivate why someone should care. **Check.** As above.

### `judge_evidence_standard` (0/1/2)

**Definition.** Does the abstract state what standard of evidence supports
its claims (proof, systematic evaluation, measured result, worked
example)? 0 no indication, 1 implied, 2 concrete. **Source section.**
"Rigorous supporting evidence" / "Crafting claims": stronger claims need
higher standards of evidence, and the abstract should carry a concrete
result. **Check.** As above.

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
| `baseline_present` | compares against at least one baseline/alternative, not only its own numbers | "Baselines are crucial" |
| `ablation_present` | removes or varies a component to measure its contribution | "Rigorous supporting evidence" (experiments that distinguish hypotheses) |
| `limitations_present` | explicitly discusses limitations of its own work | "Discussion/Conclusion": explaining limitations is crucial scientific practice |
| `variance_reported` | error bars, standard deviations, CIs, or multi-seed results | "Statistical rigor": how noisy is the experiment, is the result distinguishable from noise |
| `code_link` | a repository link or explicit code-availability statement | "Reproducibility": share your code |
| `prepost_disclosure` | discloses planned-vs-exploratory analyses (preregistration, stated held-out protocol, explicit post-hoc labels) | "Avoiding misleading evidence": track pre/post-hoc analysis |
| `fig1_is_diagram` | Figure 1 is an explanatory diagram/schematic rather than a results plot (judged from the Figure 1 caption) | "Figure placement": an eye-catching, explanatory Figure 1 |

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
