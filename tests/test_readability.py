"""Tests for the readability stage.

Most of the fifteen formulas come from textstat, and testing those would
test textstat rather than this repository. What is tested here is the code
this project actually owns:

  - the two formulas implemented by hand (FORCAST, Powers-Sumner-Kearl)
  - the short-abstract guard, which decides whether a paper gets scores at all
  - the shape of the returned row, which downstream aggregation depends on

The short-abstract guard matters because the NeurIPS corpus contains no
abstract under ten words, so that branch never executes in the pipeline run
and would otherwise be wholly untested.
"""
import pytest

from metrics.readability import _COLUMNS, _MIN_WORDS, _forcast, _powers_sumner_kearl, compute

# A real-shaped abstract, comfortably over the ten-word threshold.
ABSTRACT = (
    "We introduce a method for training deep neural networks that improves "
    "convergence on large datasets. The approach requires no additional "
    "supervision and generalises across several benchmark tasks."
)


# ── the short-abstract guard ─────────────────────────────────────────────

@pytest.mark.parametrize("text", ["", "   ", "Too short to score."])
def test_short_input_yields_all_none(text):
    row = compute(text)
    assert set(row) == set(_COLUMNS)
    assert all(v is None for v in row.values())


def test_guard_boundary_is_exactly_min_words():
    """Below the threshold every value is None; at it, values are produced."""
    just_under = " ".join(["word"] * (_MIN_WORDS - 1)) + "."
    just_at = " ".join(["word"] * _MIN_WORDS) + "."

    assert all(v is None for v in compute(just_under).values())
    assert compute(just_at)["word_count"] is not None


def test_row_shape_is_identical_whether_scored_or_not():
    """Aggregation reads fixed columns, so both branches must agree on keys."""
    assert set(compute("").keys()) == set(compute(ABSTRACT).keys()) == set(_COLUMNS)


# ── a normal abstract ────────────────────────────────────────────────────

def test_normal_abstract_produces_numbers():
    row = compute(ABSTRACT)
    for key in ("word_count", "sentence_count", "flesch_ease",
                "flesch_kincaid", "coleman_liau", "forcast",
                "powers_sumner_kearl"):
        assert isinstance(row[key], (int, float)), key


def test_counts_are_positive_and_consistent():
    row = compute(ABSTRACT)
    assert row["word_count"] > 0
    assert row["sentence_count"] > 0
    assert row["syllable_count"] >= row["word_count"]      # >= 1 syllable per word
    assert row["monosyllable_count"] <= row["word_count"]


def test_values_are_rounded_to_three_places():
    row = compute(ABSTRACT)
    assert row["flesch_ease"] == round(row["flesch_ease"], 3)


# ── FORCAST ──────────────────────────────────────────────────────────────

def test_forcast_returns_none_for_empty_text():
    assert _forcast("") is None
    assert _forcast("   ") is None


def test_forcast_all_monosyllabic_hits_the_formula_floor():
    """Grade = 20 - rate * 15, so a rate of 1.0 gives exactly 5."""
    assert _forcast("the cat sat on the mat") == 5.0


def test_forcast_decreases_as_monosyllabic_share_rises():
    """More one-syllable words means an easier text, so a lower grade."""
    easy = _forcast("the cat sat on the mat and the dog ran")
    hard = _forcast("sophisticated methodologies facilitate experimental verification")
    assert easy < hard


# ── Powers-Sumner-Kearl ──────────────────────────────────────────────────

def test_psk_returns_none_when_there_are_no_words():
    assert _powers_sumner_kearl("") is None


def test_psk_returns_a_number_for_ordinary_prose():
    assert isinstance(_powers_sumner_kearl(ABSTRACT), float)


def test_psk_never_raises():
    """It wraps its body in try/except and must return None, not propagate."""
    for text in ["", "   ", ".", "!!!", "\n\n", "a", "123 456"]:
        result = _powers_sumner_kearl(text)
        assert result is None or isinstance(result, float)


# ── inputs the NeurIPS corpus does not contain ───────────────────────────

@pytest.mark.parametrize("text", [
    "Sin punto final este resumen no tiene puntuacion alguna en absoluto",
    "This abstract has no sentence terminator at all and just keeps running on",
    "Math heavy: $\\alpha = \\beta^2$ and $\\sum_{i=1}^n x_i$ appear throughout here.",
    "Ünïcödé chäräctérs appéar thróughóut thís partícular tést abstráct hére.",
    "ALL CAPS ABSTRACT TEXT THAT SOMEONE PASTED IN WITHOUT ANY LOWERCASE.",
])
def test_unusual_but_valid_input_does_not_raise(text):
    """Other corpora will contain these; ours does not. Must not crash."""
    row = compute(text)
    assert set(row) == set(_COLUMNS)
