"""Tests for parsing a judge model's output into a score.

The parser tries a labelled "Score: N" pattern first and falls back to any
standalone digit in 1-5. Both paths are pinned here, along with the case
where neither matches and the function returns None, because a model that
emits no digit within its token budget scores nothing and the resulting
null is easy to miss downstream.
"""
import pytest

from metrics.llm_scores import parse_score


# ── the intended path ────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Score: 1", 1.0),
    ("Score: 3", 3.0),
    ("Score: 5", 5.0),
    ("Score: 3.5", 3.5),
    ("Score:4", 4.0),          # no space
    ("Score:   4", 4.0),       # extra space
    ("Reason: dense prose. Score: 2", 2.0),
])
def test_labelled_score(text, expected):
    assert parse_score(text) == expected


def test_labelled_score_wins_over_a_stray_digit():
    """The Score: pattern is tried first, so an earlier digit does not win."""
    assert parse_score("There are 5 sentences. Score: 2") == 2.0


# ── the bare-digit fallback ──────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("4", 4.0),
    ("I would say 3", 3.0),
    ("2.5", 2.5),
])
def test_bare_digit_fallback(text, expected):
    assert parse_score(text) == expected


def test_fallback_is_loose_and_can_capture_an_unrelated_number():
    """Known limitation, pinned deliberately.

    The fallback matches any standalone 1-5 anywhere in the text. With
    max_new_tokens=8 the output is usually truncated before "Score:" appears,
    so a number that is not a score can be picked up. This test documents the
    behaviour rather than endorsing it; if the parser is tightened, it should
    fail and be updated.
    """
    assert parse_score("The abstract has 4 sentences") == 4.0


# ── the failure that mattered ────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "",
    "Okay, so I'm trying to",          # reasoning-model preamble, verbatim shape
    "Okay, so I need to figure",
    "no digits at all here",
    "   ",
])
def test_returns_none_when_no_score_present(text):
    assert parse_score(text) is None


# ── out-of-range and boundary handling ───────────────────────────────────

@pytest.mark.parametrize("text", [
    "Score: 7",      # outside 1-5, so the labelled pattern does not match
    "Score: 0",
    "10",            # no word boundary after the 1
    "1987",          # a year must not parse as a score
    "42",
])
def test_out_of_range_values_are_not_scores(text):
    assert parse_score(text) is None


def test_score_is_always_within_the_rating_scale():
    """Whatever comes back is either None or inside 1-5."""
    samples = ["Score: 1", "Score: 5", "3", "Score: 9", "", "banana", "2.5"]
    for s in samples:
        v = parse_score(s)
        assert v is None or 1.0 <= v <= 5.0
