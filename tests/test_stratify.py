"""The sampler: same seed, same input -> byte-identical sample.csv; and
the allocation/overdraw properties the fulltext stage depends on."""

import numpy as np
import pandas as pd

from sample.stratify import OVERDRAW, assign_strata, draw, year_band


def _outcomes(n_per_year=40, years=range(1990, 2024, 3), seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for i in range(n_per_year):
            rows.append({
                "paper_id": f"{year}_{i:03d}",
                "year": year,
                "citations": float(rng.integers(0, 500)),
                "cites_per_year": float(rng.random() * 30),
                "arxiv_id": f"{year % 100:02d}01.{i:05d}",
            })
    return pd.DataFrame(rows)


def test_year_band_edges():
    assert year_band(1987) == "1987-1999"
    assert year_band(1999) == "1987-1999"
    assert year_band(2000) == "2000-2009"
    assert year_band(2025) == "2022-2025"
    assert year_band(1980) == "out_of_range"


def test_same_seed_identical_csv_bytes(tmp_path):
    outcomes = _outcomes()
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    draw(outcomes, target=200, seed=20261008).to_csv(a, index=False)
    draw(outcomes.sample(frac=1, random_state=3),  # shuffled input rows
         target=200, seed=20261008).to_csv(b, index=False)
    assert a.read_bytes() == b.read_bytes()


def test_different_seed_differs():
    outcomes = _outcomes()
    s1 = draw(outcomes, target=200, seed=1)
    s2 = draw(outcomes, target=200, seed=2)
    assert not s1["paper_id"].equals(s2["paper_id"])


def test_primary_count_hits_target():
    sample = draw(_outcomes(), target=200, seed=5)
    assert (~sample["is_replacement"]).sum() == 200


def test_overdraw_capped_at_ten_percent_per_stratum():
    sample = draw(_outcomes(), target=200, seed=5)
    for (_, _), grp in sample.groupby(["year_band", "tercile"]):
        n_primary = int((~grp["is_replacement"]).sum())
        n_repl = int(grp["is_replacement"].sum())
        assert n_repl <= int(np.ceil(n_primary * OVERDRAW))


def test_replacements_ordered_after_primaries():
    sample = draw(_outcomes(), target=200, seed=5)
    for (_, _), grp in sample.groupby(["year_band", "tercile"]):
        n_primary = int((~grp["is_replacement"]).sum())
        ranks = grp.sort_values("stratum_rank")
        assert list(ranks["is_replacement"])[:n_primary] == [False] * n_primary


def test_no_duplicate_papers():
    sample = draw(_outcomes(), target=200, seed=5)
    assert sample["paper_id"].is_unique


def test_small_corpus_takes_everything():
    outcomes = _outcomes(n_per_year=3, years=[2020, 2021])
    sample = draw(outcomes, target=200, seed=5)
    assert (~sample["is_replacement"]).sum() == len(outcomes)


def test_terciles_balanced_within_band():
    strata = assign_strata(_outcomes())
    for _, grp in strata.groupby("year_band"):
        counts = grp["tercile"].value_counts()
        assert set(counts.index) == {0, 1, 2}
        assert counts.max() - counts.min() <= 2
