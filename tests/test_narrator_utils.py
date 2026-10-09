"""Tests for the shared writing-metric counters.

These back the hedging, signposting, active-narration and hype metrics.
They are regex counters over free text, so the risks are boundary handling,
case sensitivity, and division-by-zero in the callers. The NeurIPS corpus
never contains an empty abstract, so those paths are exercised only here.
"""
import pytest

from metrics._narrator_utils import (
    count_hedging,
    count_hype,
    count_sentences,
    count_signposting,
    has_narrator,
    preprocess,
    word_count,
)


# ── word_count ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("two words", 2),
    ("", 0),
    ("   ", 0),
    ("hyphen-separated", 2),      # \w+ splits on the hyphen
    ("don't", 2),                 # and on the apostrophe
    ("a1 b2", 2),
    ("...", 0),                   # punctuation only
])
def test_word_count(text, expected):
    assert word_count(text) == expected


# ── count_sentences ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("One. Two.", 2),
    ("One sentence.", 1),
    ("No terminator", 1),          # floor of 1
    ("", 1),                       # floor of 1, never zero
    ("A. B. C.", 3),
    ("Dr. Smith wrote it.", 2),    # known limitation: abbreviation counts
])
def test_count_sentences(text, expected):
    assert count_sentences(text) == expected


def test_count_sentences_never_returns_zero():
    """Callers divide by this, so it must never be 0."""
    for text in ["", "   ", "no punctuation", "!!!", "\n"]:
        assert count_sentences(text) >= 1


# ── has_narrator: case sensitivity is load-bearing ───────────────────────

@pytest.mark.parametrize("text", ["We show that", "we show that",
                                  "shown to us", "Us and them"])
def test_detects_first_person(text):
    assert has_narrator(text) is True


@pytest.mark.parametrize("text,why", [
    ("US government data", "US is the country abbreviation, not first person"),
    ("the WE protocol",    "all-caps WE is an acronym, not first person"),
    ("website usage",      "us inside a word must not match"),
    ("bonus features",     "us at a word end must not match"),
    ("",                   "empty text"),
])
def test_ignores_non_first_person(text, why):
    assert has_narrator(text) is False, why


# ── count_hedging ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("The result may be useful", 1),
    ("This could possibly work", 2),
    ("", 0),
    ("A definite, proven result.", 0),
])
def test_count_hedging(text, expected):
    assert count_hedging(text) == expected


def test_hedging_is_case_insensitive():
    assert count_hedging("MAY be") == count_hedging("may be") == 1


def test_hedging_respects_word_boundaries():
    """'maybe' must not also fire the 'may' pattern inside it."""
    assert count_hedging("mayhem and dismay") == 0


# ── count_signposting ────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Firstly, we note. However, it fails.", 2),
    ("", 0),
    ("A plain sentence with no markers.", 0),
])
def test_count_signposting(text, expected):
    assert count_signposting(text) == expected


# ── count_hype ───────────────────────────────────────────────────────────

def test_count_hype_returns_every_category():
    result = count_hype("")
    assert set(result) == {"importance", "novelty", "rigor", "scale", "utility",
                           "quality", "attitude", "problem", "additional"}
    assert all(v == 0 for v in result.values())


def test_count_hype_assigns_to_the_right_category():
    r = count_hype("A novel and groundbreaking method")
    assert r["novelty"] == 2
    assert r["rigor"] == 0


def test_count_hype_handles_regex_alternates():
    """Several entries are patterns, not literals (inter.?disciplinary)."""
    assert count_hype("interdisciplinary work")["scale"] == 1
    assert count_hype("inter-disciplinary work")["scale"] == 1


# ── preprocess ───────────────────────────────────────────────────────────

def test_preprocess_removes_structured_headings():
    assert "BACKGROUND" not in preprocess("BACKGROUND: we study this")


def test_preprocess_keeps_ordinary_words_containing_abstract():
    """A bare ABSTRACT heading is stripped, but 'abstraction' must survive.

    An earlier case-insensitive pattern turned 'abstraction' into 'ion'.
    """
    assert "abstraction" in preprocess("The abstraction is useful")


@pytest.mark.parametrize("text,fragment", [
    ("x = y",   "equal to"),
    ("a < b",   "less than"),
    ("a > b",   "greater than"),
    ("a &lt; b", "less than"),
    ("p &amp; q", "and"),
    ("5 ± 2",   "plus or minus"),
])
def test_preprocess_expands_symbols(text, fragment):
    assert fragment in preprocess(text)


def test_preprocess_collapses_whitespace():
    assert preprocess("a   b\n\nc") == "a b c"


def test_preprocess_of_empty_text_is_empty():
    assert preprocess("") == ""
