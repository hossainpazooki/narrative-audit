"""Tests for resume bookkeeping.

Every metric script can be re-run and will skip work already done. That
makes it stateful, and the state lives in a partially written CSV. The
docstring in schema.py records the bug this logic replaced: the older
convention treated a year as complete if ANY row existed, so an interrupted
run silently dropped the remaining papers of that year forever, and the
output looked finished.

A clean pipeline run never exercises these paths, so they are tested here.
"""
import csv

import pytest

from schema import resume_year_counts, resume_year_ids

ROWS = [
    {"paper_id": "1987_a", "venue": "neurips", "year": "1987", "value": "1"},
    {"paper_id": "1987_b", "venue": "neurips", "year": "1987", "value": "2"},
    {"paper_id": "1988_a", "venue": "neurips", "year": "1988", "value": "3"},
]


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "neurips.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["paper_id", "venue", "year", "value"])
        w.writeheader()
        w.writerows(ROWS)
    return str(p)


# ── absent output ────────────────────────────────────────────────────────

def test_counts_of_a_missing_file_are_empty(tmp_path):
    """A first run has no output yet and must not raise."""
    assert resume_year_counts(str(tmp_path / "nope.csv")) == {}


def test_ids_of_a_missing_file_are_empty(tmp_path):
    assert resume_year_ids(str(tmp_path / "nope.csv"), 1987) == set()


def test_header_only_file_is_empty(tmp_path):
    """An interrupted run can leave a file with only a header row."""
    p = tmp_path / "empty.csv"
    p.write_text("paper_id,venue,year,value\n", encoding="utf-8")
    assert resume_year_counts(str(p)) == {}
    assert resume_year_ids(str(p), 1987) == set()


# ── counting ─────────────────────────────────────────────────────────────

def test_counts_are_per_year(csv_path):
    assert resume_year_counts(csv_path) == {1987: 2, 1988: 1}


def test_counts_use_integer_keys(csv_path):
    """Callers compare against len(papers) keyed by int, not str."""
    counts = resume_year_counts(csv_path)
    assert all(isinstance(k, int) for k in counts)


def test_a_partial_year_reports_its_partial_count(csv_path):
    """This is the property that prevents the old data-loss bug.

    The caller decides completeness by comparing this count against the
    number of papers in the processed file. If the count were reported as
    "done" rather than as a number, the rest of the year would be skipped.
    """
    counts = resume_year_counts(csv_path)
    assert counts[1987] == 2          # a real 1987 has far more papers
    assert counts[1987] != 0          # present, but not asserted complete


# ── id lookup ────────────────────────────────────────────────────────────

def test_ids_are_scoped_to_the_requested_year(csv_path):
    assert resume_year_ids(csv_path, 1987) == {"1987_a", "1987_b"}
    assert resume_year_ids(csv_path, 1988) == {"1988_a"}


def test_ids_of_an_absent_year_are_empty(csv_path):
    assert resume_year_ids(csv_path, 2024) == set()


# ── malformed state ──────────────────────────────────────────────────────

def test_rows_without_a_year_are_ignored(tmp_path):
    """A truncated final line can leave a row with an empty year field."""
    p = tmp_path / "partial.csv"
    p.write_text(
        "paper_id,venue,year,value\n"
        "1987_a,neurips,1987,1\n"
        "1987_b,neurips,,\n",
        encoding="utf-8")
    assert resume_year_counts(str(p)) == {1987: 1}
    assert resume_year_ids(str(p), 1987) == {"1987_a"}


def test_duplicate_ids_are_deduplicated_by_the_id_set(tmp_path):
    """Counts see two rows; the id set sees one paper."""
    p = tmp_path / "dupes.csv"
    p.write_text(
        "paper_id,venue,year,value\n"
        "1987_a,neurips,1987,1\n"
        "1987_a,neurips,1987,2\n",
        encoding="utf-8")
    assert resume_year_counts(str(p)) == {1987: 2}
    assert resume_year_ids(str(p), 1987) == {"1987_a"}
