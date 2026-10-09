# tests

356 tests covering what the pipeline run cannot: the 269 inherited from
abstract-audit, and 87 for the narrative-audit layer.

## What these are for

`verify.py` already compares 40 artefacts against `reference/`, 25 of them
byte-for-byte, over 30,595 real papers. As a regression test that is stronger
than any unit test could be, and tests that re-walk the same happy path would
add lines without adding information.

So these tests target what an end-to-end run structurally cannot reach:

- **Inputs absent from the corpus.** Empty abstracts, single sentences, text
  with no terminator, heavy LaTeX, non-ASCII. The premise of this repository
  is that other people measure their own abstracts, and every one of those is
  a live path for them and a dead path for NeurIPS.
- **Stateful resume logic.** A partially written output CSV. The docstring in
  `schema.py` records the bug this replaced: a year with any rows at all was
  treated as finished, so an interrupted run dropped the rest of that year
  permanently and looked complete.
- **Error paths.** A missing venue file, a missing column, a malformed record.
- **Domain logic that is not self-evident.** The acronym rule, and the judge
  score parser.
- **Third-party markup.** The papers.nips.cc parsers are the only thing here
  that can break with no change to this code.

The narrative-audit layer holds the same line, per the design spec's test
list:

- **Claim-strength lexicon** (`test_claim_strength.py`): one hand-written
  abstract per class, plus the tie and zero-cue defaults.
- **Role sequencer** (`test_role_sequence.py`): the source post's worked
  example — the refusal abstract (Arditi et al., NeurIPS 2024) — as
  fixture, pinned sentence by sentence to distance 0.
- **OpenAlex matcher** (`test_openalex_match.py`): title collisions
  (resolvable and not), the missing-DOI path, no match at all, and the
  age-normalization/report math. No network: the HTTP side is exercised by
  real runs, the decisions here.
- **Section splitter** (`test_sections.py`): three NeurIPS-era page
  layouts, one per stretch of the corpus, for the pymupdf fallback —
  built with pymupdf at test time because real NeurIPS PDFs are not
  redistributable — and recorded TEI fixtures for the GROBID path, which
  pins the part of that path this repository owns.
- **Judge parsers** (`test_judge_parsers.py`): malformed, truncated and
  extra-text outputs for both the Tier-2 integer parser and the Tier-3
  JSON parser, plus evidence-span verification.
- **Sampler** (`test_stratify.py`): same seed and input produce
  byte-identical `sample.csv` even under shuffled input rows; overdraw
  cap and replacement ordering.
- **Verify** (`test_verify_diff.py`): an injected one-byte diff fails,
  on both the byte-for-byte and the atol=0 numeric path.

## What is deliberately not tested

The fifteen readability formulas from `textstat`, and the spaCy-derived
features. Testing those tests someone else's library. What is tested is the
two formulas implemented here by hand, and the guard that decides whether an
abstract is scored at all.

## Coverage

About 54% of `src/` by statement. That number is low on purpose and the shape
matters more: the pure logic sits between 74% and 97%, and nearly all of the
remainder is `main()` functions, file-reading loops, the GPU scoring path and
network fetching. Those are exercised by a full pipeline run, which checks
them against 40 reference artefacts rather than against a mock.

The criterion held here is not a percentage but this: every branch that
handles malformed or edge-case input has a test, because those are exactly
the branches 30,595 well-formed papers never enter.

## Running

    pip install -r requirements-dev.txt
    pytest

    pytest --cov=src --cov-report=term-missing     # with coverage
    pytest tests/test_acronyms.py -v               # one module

The tests do not gate the pipeline. `python main.py` never invokes them.
