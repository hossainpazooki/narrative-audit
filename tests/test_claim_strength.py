"""claim_strength: one hand-written abstract per class, plus the tie and
no-cue defaults the lexicon cannot express by itself."""

from rubric.tier1 import claim_strength

GUARANTEE = (
    "We prove that gradient descent on overparameterized linear networks "
    "converges to the minimum-norm solution. Our theorem holds for any "
    "initialization scale, and we derive a bound on the convergence rate "
    "that is provably tight."
)

SYSTEMATIC = (
    "We evaluate instruction-tuned models across 12 benchmarks and find "
    "that chain-of-thought prompting consistently improves accuracy. The "
    "effect holds across multiple model families and across diverse task "
    "formats, from arithmetic to commonsense reasoning."
)

HEDGED = (
    "Our results suggest that sharpness-aware minimization may improve "
    "generalization on vision tasks. The improvement appears to depend on "
    "batch size, and we believe the mechanism is likely related to "
    "implicit regularization, though further work is needed."
)

EXISTENCE = (
    "We exhibit a single transformer attention head that implements "
    "modular addition. There exists a choice of embedding for which the "
    "circuit is exactly recoverable, giving a proof of concept that "
    "mechanistic reverse-engineering can succeed end to end."
)


def test_guarantee():
    assert claim_strength(GUARANTEE) == "guarantee"


def test_systematic():
    assert claim_strength(SYSTEMATIC) == "systematic"


def test_hedged():
    assert claim_strength(HEDGED) == "hedged"


def test_existence():
    assert claim_strength(EXISTENCE) == "existence"


def test_no_cues_defaults_to_hedged():
    assert claim_strength("A short abstract with no calibration cues at all.") == "hedged"


def test_tie_defaults_to_hedged():
    # one guarantee cue, one existence cue
    text = "We prove convergence. There exists an edge case too."
    assert claim_strength(text) == "hedged"


def test_numeric_systematic_cue():
    # "across 13 ..." is the code-level systematic cue the literal lexicon
    # cannot express
    text = "Refusal is mediated by a single direction across 13 chat models."
    assert claim_strength(text) == "systematic"


def test_empty_abstract():
    assert claim_strength("") == "hedged"
