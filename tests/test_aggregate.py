"""Tests for the per-year aggregation helper.

The maths is pandas groupby, which is not this project's to test. What is
tested here is the contract around it: the column naming downstream code
depends on, behaviour when a venue file is missing, and the single-row
group where standard deviation is undefined.
"""
import os

import pandas as pd
import pytest

from aggregate._utils import aggregate

COLS = ["flesch_ease", "word_count"]


def write(tmp_path, venue, rows):
    d = tmp_path / "per_paper" / "readability"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{venue}.csv", index=False)
    return str(tmp_path / "per_paper")


def test_column_naming_is_venue_metric_stat(tmp_path):
    """Downstream code reads neurips_<col>_mean by name."""
    pp = write(tmp_path, "neurips", [
        {"paper_id": "a", "year": 2020, "flesch_ease": 10.0, "word_count": 100},
        {"paper_id": "b", "year": 2020, "flesch_ease": 20.0, "word_count": 200},
    ])
    out = aggregate("readability", COLS, venues=["neurips"],
                    per_paper_dir=pp, out_dir=str(tmp_path / "agg"))
    df = pd.read_csv(out)
    for col in COLS:
        for stat in ("mean", "std", "count"):
            assert f"neurips_{col}_{stat}" in df.columns


def test_mean_and_count_are_correct(tmp_path):
    pp = write(tmp_path, "neurips", [
        {"paper_id": "a", "year": 2020, "flesch_ease": 10.0, "word_count": 100},
        {"paper_id": "b", "year": 2020, "flesch_ease": 20.0, "word_count": 200},
        {"paper_id": "c", "year": 2021, "flesch_ease": 30.0, "word_count": 300},
    ])
    out = aggregate("readability", COLS, venues=["neurips"],
                    per_paper_dir=pp, out_dir=str(tmp_path / "agg"))
    df = pd.read_csv(out).set_index("year")
    assert df.loc[2020, "neurips_flesch_ease_mean"] == 15.0
    assert df.loc[2020, "neurips_flesch_ease_count"] == 2
    assert df.loc[2021, "neurips_flesch_ease_mean"] == 30.0


def test_single_paper_year_has_undefined_std(tmp_path):
    """A year with one paper yields NaN std, which must not become 0."""
    pp = write(tmp_path, "neurips", [
        {"paper_id": "a", "year": 1987, "flesch_ease": 25.0, "word_count": 100},
    ])
    out = aggregate("readability", COLS, venues=["neurips"],
                    per_paper_dir=pp, out_dir=str(tmp_path / "agg"))
    df = pd.read_csv(out).set_index("year")
    assert df.loc[1987, "neurips_flesch_ease_count"] == 1
    assert pd.isna(df.loc[1987, "neurips_flesch_ease_std"])


def test_years_are_sorted_ascending(tmp_path):
    pp = write(tmp_path, "neurips", [
        {"paper_id": "a", "year": 2021, "flesch_ease": 1.0, "word_count": 10},
        {"paper_id": "b", "year": 1995, "flesch_ease": 2.0, "word_count": 20},
        {"paper_id": "c", "year": 2008, "flesch_ease": 3.0, "word_count": 30},
    ])
    out = aggregate("readability", COLS, venues=["neurips"],
                    per_paper_dir=pp, out_dir=str(tmp_path / "agg"))
    years = pd.read_csv(out)["year"].tolist()
    assert years == sorted(years)


def test_missing_venue_file_raises_rather_than_writing_an_empty_file(tmp_path):
    """Silently producing an empty aggregate would corrupt every figure."""
    (tmp_path / "per_paper" / "readability").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="No data found"):
        aggregate("readability", COLS, venues=["neurips"],
                  per_paper_dir=str(tmp_path / "per_paper"),
                  out_dir=str(tmp_path / "agg"))


def test_missing_column_is_skipped_not_fatal(tmp_path):
    """A metric file lacking one requested column still aggregates the rest."""
    pp = write(tmp_path, "neurips", [
        {"paper_id": "a", "year": 2020, "flesch_ease": 10.0},
    ])
    out = aggregate("readability", COLS, venues=["neurips"],
                    per_paper_dir=pp, out_dir=str(tmp_path / "agg"))
    df = pd.read_csv(out)
    assert "neurips_flesch_ease_mean" in df.columns
    assert "neurips_word_count_mean" not in df.columns


def test_output_directory_is_created(tmp_path):
    pp = write(tmp_path, "neurips", [
        {"paper_id": "a", "year": 2020, "flesch_ease": 1.0, "word_count": 10},
    ])
    out_dir = tmp_path / "does" / "not" / "exist"
    out = aggregate("readability", COLS, venues=["neurips"],
                    per_paper_dir=pp, out_dir=str(out_dir))
    assert os.path.exists(out)
