"""Tests for NeurIPS record processing.

The pre-2008 path is the fiddly one. Those conference pages bundle OCR'd
full text into a single field, and 67 records carry "Abstract Unavailable"
with the real text elsewhere. Dropping them would distort the per-year
counts, so there is a fallback and a header-stripping cleaner, both driven
by heuristics that only a handful of real records exercise.
"""
import pytest

from processing.process_neurips import (
    _PRE2008_WORD_CAP,
    _looks_like_prose,
    best_abstract,
    clean_ocr_abstract,
    parse_authors,
)


# ── parse_authors ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    (["Hinton", "Sejnowski"],      ["Hinton", "Sejnowski"]),
    (["  Hinton  "],               ["Hinton"]),
    (["Hinton", "", "  "],         ["Hinton"]),
    ("Hinton, Sejnowski",          ["Hinton", "Sejnowski"]),
    ("Hinton,  Sejnowski ,",       ["Hinton", "Sejnowski"]),
    ("Solo",                       ["Solo"]),
    (None,                         []),
    (42,                           []),
    ([],                           []),
    ("",                           []),
])
def test_parse_authors(raw, expected):
    assert parse_authors(raw) == expected


def test_parse_authors_drops_non_string_list_entries():
    """Malformed scrapes have produced nested nulls in the author list."""
    assert parse_authors(["Hinton", None, 7, "Sejnowski"]) == ["Hinton", "Sejnowski"]


# ── _looks_like_prose ────────────────────────────────────────────────────

def test_prose_needs_at_least_six_words():
    assert _looks_like_prose("one two three four five") is False
    assert _looks_like_prose("one two three four five six") is True


def test_all_caps_header_is_not_prose():
    assert _looks_like_prose("DEEP LEARNING FOR VISION AND SPEECH") is False


def test_mostly_lowercase_sentence_is_prose():
    assert _looks_like_prose("we present a method for training networks") is True


def test_empty_string_is_not_prose():
    assert _looks_like_prose("") is False


# ── clean_ocr_abstract ───────────────────────────────────────────────────

def test_modern_abstracts_are_not_truncated():
    """The word cap applies only before 2008."""
    long_text = " ".join(["word"] * (_PRE2008_WORD_CAP + 200)) + "."
    assert len(clean_ocr_abstract(long_text, 2020).split()) > _PRE2008_WORD_CAP


def test_pre2008_abstracts_are_capped():
    long_text = " ".join(["word"] * (_PRE2008_WORD_CAP + 200)) + "."
    assert len(clean_ocr_abstract(long_text, 1995).split()) <= _PRE2008_WORD_CAP


def test_cid_artifacts_are_removed():
    """OCR of these PDFs emits (cid:NNN) glyph references."""
    assert "cid" not in clean_ocr_abstract("Text (cid:12) with artifacts.", 2020)


def test_whitespace_is_collapsed():
    assert clean_ocr_abstract("a   b\n\nc", 2020) == "a b c"


def test_empty_input_stays_empty():
    assert clean_ocr_abstract("", 1995) == ""
    assert clean_ocr_abstract("", 2020) == ""


# ── best_abstract ────────────────────────────────────────────────────────

GOOD = ("We present a learning procedure for networks of neuron-like units "
        "that discovers useful internal representations of the input domain.")


def test_uses_the_abstract_when_it_is_usable():
    assert best_abstract({"abstract": GOOD}, 2020).startswith("We present")


def test_falls_back_to_full_text_when_abstract_unavailable():
    """The literal placeholder in 67 pre-2008 records."""
    full = " ".join(["realword"] * 100) + "."
    result = best_abstract({"abstract": "Abstract Unavailable", "full_text": full}, 1995)
    assert "Unavailable" not in result
    assert len(result.split()) > 20


def test_falls_back_when_the_abstract_is_too_short():
    full = " ".join(["realword"] * 100) + "."
    result = best_abstract({"abstract": "Too short.", "full_text": full}, 1995)
    assert len(result.split()) > 20


def test_returns_empty_when_neither_field_is_usable():
    assert best_abstract({"abstract": "", "full_text": ""}, 1995) == ""
    assert best_abstract({}, 1995) == ""


def test_handles_none_valued_fields():
    """Scraped JSON carries nulls, not missing keys."""
    assert best_abstract({"abstract": None, "full_text": None}, 1995) == ""


def test_full_text_fallback_is_capped_before_cleaning():
    """A 600-word prefix is taken, then the pre-2008 cap applies on top."""
    full = " ".join(["realword"] * 5000) + "."
    result = best_abstract({"abstract": "", "full_text": full}, 1995)
    assert len(result.split()) <= _PRE2008_WORD_CAP
