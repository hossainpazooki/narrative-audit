"""Tests for text sanitisation and record construction.

sanitize_text runs over every title and abstract during processing, so a
bug here would corrupt the corpus before any metric sees it. The unicode
and control-character paths are the ones the NeurIPS data exercises least.
"""
import pytest

from schema import make_paper, sanitize_text


# ── sanitize_text ────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, 42, [], {}, ""])
def test_non_string_input_yields_empty_string(bad):
    """Scrapers occasionally hand back None; this must not raise."""
    assert sanitize_text(bad) == ""


def test_collapses_all_whitespace_runs():
    assert sanitize_text("a   b\t\tc\n\n\nd") == "a b c d"


def test_strips_leading_and_trailing_whitespace():
    assert sanitize_text("   padded   ") == "padded"


def test_removes_control_characters():
    assert sanitize_text("clean\x00text\x07here") == "cleantexthere"


def test_keeps_tabs_and_newlines_as_spaces():
    """Control characters are removed, but the whitespace ones collapse."""
    assert sanitize_text("line1\nline2\tline3") == "line1 line2 line3"


def test_normalises_unicode_to_composed_form():
    """Decomposed and composed accents must compare equal afterwards."""
    decomposed = "école"    # e + combining acute
    composed = "école"       # e-acute
    assert sanitize_text(decomposed) == sanitize_text(composed) == composed


def test_preserves_ordinary_unicode():
    assert sanitize_text("Schrödinger's café") == "Schrödinger's café"


def test_is_idempotent():
    """Running it twice must not change the result."""
    for text in ["  a \n b ", "école", "x\x00y", "plain"]:
        once = sanitize_text(text)
        assert sanitize_text(once) == once


# ── make_paper ───────────────────────────────────────────────────────────

BASE = dict(paper_id="2020_abc", venue="neurips", year=2020,
            title="A title", abstract="An abstract.", authors=["Someone"])


def test_includes_every_required_field():
    rec = make_paper(**BASE)
    assert set(rec) == {"paper_id", "venue", "year", "title", "abstract", "authors"}


@pytest.mark.parametrize("field", ["arxiv_categories", "arxiv_primary_category"])
def test_omits_optional_fields_when_absent(field):
    """The schema forbids arxiv fields on non-arxiv records, so they must
    be left out entirely rather than written as None."""
    assert field not in make_paper(**BASE)


def test_includes_optional_fields_when_provided():
    rec = make_paper(**BASE, arxiv_categories=["cs.LG"],
                     arxiv_primary_category="cs.LG")
    assert rec["arxiv_categories"] == ["cs.LG"]
    assert rec["arxiv_primary_category"] == "cs.LG"


@pytest.mark.parametrize("empty", [None, [], ""])
def test_falsy_optional_values_are_omitted_not_stored(empty):
    rec = make_paper(**BASE, arxiv_categories=empty)
    assert "arxiv_categories" not in rec
