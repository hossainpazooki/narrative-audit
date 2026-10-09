"""Tests for the processed-paper schema.

Every test that matters here asserts a REJECTION. A validator that silently
accepts everything passes a naive test suite while providing no protection,
so each field validator is checked against input it must refuse, and the
accept cases only establish the baseline.
"""
import pytest
from pydantic import ValidationError

from validate import (
    MIN_ABSTRACT_WORDS,
    VALID_VENUES,
    YEAR_MAX,
    YEAR_MIN,
    ProcessedPaper,
)

VALID = {
    "paper_id": "1987_0a5209deb79dc584d8ddb41a792d8549",
    "venue": "neurips",
    "year": 1987,
    "title": "A learning algorithm for Boltzmann machines",
    "abstract": "We describe a learning procedure for networks of units.",
    "authors": ["Ackley", "Hinton", "Sejnowski"],
}


def record(**overrides):
    return {**VALID, **overrides}


def test_a_valid_record_is_accepted():
    p = ProcessedPaper(**VALID)
    assert p.venue == "neurips"
    assert p.year == 1987


# ── paper_id ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["", "   ", "\t"])
def test_rejects_blank_paper_id(bad):
    with pytest.raises(ValidationError, match="paper_id is empty"):
        ProcessedPaper(**record(paper_id=bad))


# ── venue ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["pubmed", "NeurIPS", "nips", "", "acl"])
def test_rejects_unknown_venue(bad):
    """Case matters, and pubmed is deliberately not a processed venue here."""
    with pytest.raises(ValidationError, match="unknown venue"):
        ProcessedPaper(**record(venue=bad))


@pytest.mark.parametrize("venue", sorted(VALID_VENUES))
def test_accepts_every_declared_venue(venue):
    extra = {}
    if venue == "arxiv":
        extra = {"arxiv_categories": ["cs.LG"], "arxiv_primary_category": "cs.LG"}
    ProcessedPaper(**record(venue=venue, **extra))


# ── year ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [YEAR_MIN - 1, YEAR_MAX + 1, 0, -1, 9999])
def test_rejects_year_outside_range(bad):
    with pytest.raises(ValidationError, match="outside expected range"):
        ProcessedPaper(**record(year=bad))


@pytest.mark.parametrize("year", [YEAR_MIN, YEAR_MAX])
def test_accepts_the_range_boundaries(year):
    """The bounds are inclusive."""
    assert ProcessedPaper(**record(year=year)).year == year


# ── title and abstract ───────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["", "   "])
def test_rejects_empty_title(bad):
    with pytest.raises(ValidationError, match="title has only"):
        ProcessedPaper(**record(title=bad))


def test_rejects_abstract_below_the_word_minimum():
    short = " ".join(["word"] * (MIN_ABSTRACT_WORDS - 1))
    with pytest.raises(ValidationError, match="abstract has only"):
        ProcessedPaper(**record(abstract=short))


def test_accepts_abstract_at_the_word_minimum():
    exact = " ".join(["word"] * MIN_ABSTRACT_WORDS)
    assert ProcessedPaper(**record(abstract=exact)).abstract == exact


# ── authors ──────────────────────────────────────────────────────────────

def test_rejects_empty_author_list():
    with pytest.raises(ValidationError, match="authors list is empty"):
        ProcessedPaper(**record(authors=[]))


@pytest.mark.parametrize("bad", [[""], ["Hinton", ""], ["  "]])
def test_rejects_blank_author_names(bad):
    """A blank string in the list is a scraping failure, not a real author."""
    with pytest.raises(ValidationError, match="empty strings"):
        ProcessedPaper(**record(authors=bad))


# ── cross-field consistency ──────────────────────────────────────────────

def test_arxiv_requires_categories():
    with pytest.raises(ValidationError, match="arxiv_categories must be present"):
        ProcessedPaper(**record(venue="arxiv", arxiv_primary_category="cs.LG"))


def test_arxiv_requires_a_primary_category():
    with pytest.raises(ValidationError, match="arxiv_primary_category must be present"):
        ProcessedPaper(**record(venue="arxiv", arxiv_categories=["cs.LG"]))


def test_non_arxiv_must_not_carry_arxiv_fields():
    """Guards against a venue mix-up during processing."""
    with pytest.raises(ValidationError, match="should not be set"):
        ProcessedPaper(**record(venue="neurips", arxiv_categories=["cs.LG"]))


def test_arxiv_record_with_both_fields_is_accepted():
    p = ProcessedPaper(**record(venue="arxiv",
                                arxiv_categories=["cs.LG", "stat.ML"],
                                arxiv_primary_category="cs.LG"))
    assert p.arxiv_primary_category == "cs.LG"


# ── type enforcement ─────────────────────────────────────────────────────

@pytest.mark.parametrize("field,bad", [
    ("year", "nineteen eighty seven"),
    ("authors", "Hinton"),           # a bare string, not a list
    ("title", 42),
])
def test_rejects_wrong_types(field, bad):
    with pytest.raises(ValidationError):
        ProcessedPaper(**record(**{field: bad}))


def test_missing_required_field_is_rejected():
    incomplete = {k: v for k, v in VALID.items() if k != "abstract"}
    with pytest.raises(ValidationError):
        ProcessedPaper(**incomplete)
