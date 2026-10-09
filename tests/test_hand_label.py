"""Hand-label workflow: seeded draws must be preregistered-reproducible,
and validation must reject exactly the malformed labels a spreadsheet
round-trip produces."""

import json
import os

import pandas as pd
import pytest

from rubric.hand_label import (
    _seeded_draw, draw_tier2, validate_tier2, validate_tier3,
)


def _processed(tmp_path, n=20):
    venue_dir = tmp_path / "processed" / "neurips"
    venue_dir.mkdir(parents=True)
    papers = [{"paper_id": f"2020_{i:03d}", "venue": "neurips", "year": 2020,
               "title": f"Paper {i}", "abstract": f"Abstract number {i}.",
               "authors": ["A"]} for i in range(n)]
    (venue_dir / "neurips_2020.json").write_text(json.dumps(papers))
    return str(tmp_path / "processed")


def test_seeded_draw_deterministic_and_order_free():
    ids = [f"id_{i}" for i in range(50)]
    a = _seeded_draw(ids, 10, seed=7)
    b = _seeded_draw(list(reversed(ids)), 10, seed=7)
    assert a == b
    assert len(set(a)) == 10
    assert _seeded_draw(ids, 10, seed=8) != a


def test_draw_tier2_sheet(tmp_path):
    processed = _processed(tmp_path)
    out_dir = str(tmp_path / "hand_labels")
    sheet = draw_tier2(processed, ["neurips"], out_dir, n=5, seed=1)
    assert len(sheet) == 5
    assert list(sheet.columns) == ["paper_id", "abstract", "n_claims",
                                   "motivation", "evidence_standard"]
    again = draw_tier2(processed, ["neurips"], out_dir, n=5, seed=1)
    assert list(sheet["paper_id"]) == list(again["paper_id"])


def _filled_tier2(tmp_path, **overrides):
    sheet = pd.DataFrame({
        "paper_id": ["2020_000", "2020_001"],
        "abstract": ["a", "b"],
        "n_claims": [2, 3],
        "motivation": [1, 2],
        "evidence_standard": [0, 2],
    })
    for col, vals in overrides.items():
        sheet[col] = vals
    path = tmp_path / "tier2_sheet.csv"
    sheet.to_csv(path, index=False)
    return str(path)


def test_validate_tier2_good_sheet(tmp_path):
    path = _filled_tier2(tmp_path)
    out = validate_tier2(path, str(tmp_path / "out"))
    assert len(out) == 6                     # 2 papers x 3 prompts
    assert set(out.columns) == {"paper_id", "prompt", "value"}
    assert os.path.exists(tmp_path / "out" / "tier2.csv")


def test_validate_tier2_rejects_out_of_range(tmp_path):
    path = _filled_tier2(tmp_path, motivation=[1, 7])
    with pytest.raises(SystemExit, match="outside"):
        validate_tier2(path, str(tmp_path / "out"))


def test_validate_tier2_rejects_empty_and_non_integer(tmp_path):
    path = _filled_tier2(tmp_path, n_claims=["", "many"])
    with pytest.raises(SystemExit):
        validate_tier2(path, str(tmp_path / "out"))


def _filled_tier3(tmp_path, values=("true", "FALSE")):
    sheet = pd.DataFrame({
        "paper_id": ["2020_000", "2020_001"],
        "feature": ["baseline_present", "code_link"],
        "question": ["q", "q"],
        "section_text": ["t", "t"],
        "fulltext_ok": [True, True],
        "value": list(values),
    })
    path = tmp_path / "tier3_sheet.csv"
    sheet.to_csv(path, index=False)
    return str(path)


def test_validate_tier3_good_sheet(tmp_path):
    out = validate_tier3(_filled_tier3(tmp_path), str(tmp_path / "out"))
    assert list(out["value"]) == [True, False]


def test_validate_tier3_rejects_non_boolean(tmp_path):
    with pytest.raises(SystemExit, match="not a boolean"):
        validate_tier3(_filled_tier3(tmp_path, values=("true", "maybe")),
                       str(tmp_path / "out"))


def test_validate_tier3_rejects_unknown_feature(tmp_path):
    sheet = pd.read_csv(_filled_tier3(tmp_path))
    sheet.loc[0, "feature"] = "novelty_present"
    path = tmp_path / "bad_sheet.csv"
    sheet.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="unknown feature"):
        validate_tier3(str(path), str(tmp_path / "out"))
