"""The OpenAlex matcher's pure logic: title collisions, the missing-DOI
path, no match at all, and the finalization math. No network anywhere —
openalex.py's HTTP side is exercised by real runs, the decisions are
exercised here."""

import pandas as pd

from outcomes.normalize import (
    extract_arxiv_id, finalize, normalize_title, pick_match, stage_report,
    work_to_row,
)

PAPER = {"paper_id": "2020_abc", "year": 2020,
         "title": "Learning to Learn: A Meta-Study"}


def work(title, year, cited=10, **kw):
    return {"title": title, "publication_year": year, "cited_by_count": cited,
            "id": "https://openalex.org/W1", **kw}


def test_normalize_title_strips_latex_and_punctuation():
    assert normalize_title(r"Attention \emph{Is} All--You Need!") == \
        "attention is all you need"


def test_exact_match():
    w, method = pick_match(PAPER, [work("Learning to Learn: A Meta-Study", 2020)])
    assert w is not None and method == "exact_title"


def test_no_match_at_all():
    w, method = pick_match(PAPER, [work("Completely Different Topic", 2020)])
    assert w is None and method == "none"


def test_year_filter_rejects_distant_years():
    w, method = pick_match(PAPER, [work("Learning to Learn: A Meta-Study", 2015)])
    assert w is None and method == "none"


def test_title_collision_resolved_by_year_then_citations():
    a = work("Learning to Learn: A Meta-Study", 2021, cited=5)
    b = work("Learning to Learn: A Meta-Study", 2020, cited=3)
    w, method = pick_match(PAPER, [a, b])
    assert w is b and method == "exact_title"


def test_title_collision_unresolvable_is_no_match():
    # same title, same year, same citations: guessing would be silent error
    a = work("Learning to Learn: A Meta-Study", 2020, cited=5)
    b = dict(work("Learning to Learn: A Meta-Study", 2020, cited=5),
             id="https://openalex.org/W2")
    w, method = pick_match(PAPER, [a, b])
    assert w is None and method == "none"


LONG_PAPER = {"paper_id": "2020_long", "year": 2020,
              "title": "Deep Residual Learning Methods for Large Scale "
                       "Image Recognition Tasks"}


def test_fuzzy_match_accepted_with_clear_margin():
    # one extra token: Jaccard 10/11 ~ 0.909, no competitor
    cand = work("Deep Residual Learning Methods for Large Scale Image "
                "Recognition Tasks Benchmark", 2020)
    w, method = pick_match(LONG_PAPER, [cand])
    assert w is cand and method == "fuzzy_title"


def test_fuzzy_match_needs_margin():
    # two near-identical candidates (0.909 vs 0.900): ambiguous collision,
    # so no match rather than a silent guess
    a = work("Deep Residual Learning Methods for Large Scale Image "
             "Recognition Tasks Benchmark", 2020)
    b = work("Deep Residual Learning Methods for Large Scale Image "
             "Recognition", 2020)
    w, method = pick_match(LONG_PAPER, [a, b])
    assert w is None and method == "none"


def test_missing_doi_leaves_row_unmatched():
    # The record has no DOI, so after a failed title match there is no
    # fallback: the row keeps nulls and the method says none.
    row = work_to_row({**PAPER}, None, "none")
    assert row["citations"] is None
    assert row["match_method"] == "none"
    assert row["paper_id"] == "2020_abc"


def test_extract_arxiv_id_from_ids_and_locations():
    w = work("t", 2020, ids={"arxiv": "https://arxiv.org/abs/2406.11717v3"})
    assert extract_arxiv_id(w) == "2406.11717"
    w2 = work("t", 2020, locations=[{
        "source": {"display_name": "arXiv (Cornell University)"},
        "landing_page_url": "https://arxiv.org/abs/1706.03762"}])
    assert extract_arxiv_id(w2) == "1706.03762"
    assert extract_arxiv_id(work("t", 2020)) is None


def _raw():
    return pd.DataFrame([
        {"paper_id": "2020_a", "year": 2020, "match_method": "exact_title",
         "openalex_id": "W1", "citations": 60, "author_count": 3,
         "subfield": "NLP", "arxiv_id": "2001.00001",
         "arxiv_v1_date": "2019-12-01"},
        {"paper_id": "2020_b", "year": 2020, "match_method": "none",
         "openalex_id": None, "citations": None, "author_count": None,
         "subfield": None, "arxiv_id": None, "arxiv_v1_date": None},
        {"paper_id": "2020_c", "year": 2020, "match_method": "doi",
         "openalex_id": "W3", "citations": 6, "author_count": 2,
         "subfield": "NLP", "arxiv_id": None, "arxiv_v1_date": "2021-05-01"},
    ])


def test_finalize_age_normalization_and_flags():
    df = finalize(_raw()).set_index("paper_id")
    # census year 2026, floor 1 year -> age 6 for 2020 papers
    assert df.loc["2020_a", "cites_per_year"] == 10.0
    assert bool(df.loc["2020_a", "preprint_before_conference"]) is True
    assert bool(df.loc["2020_c", "preprint_before_conference"]) is False
    assert pd.isna(df.loc["2020_b", "cites_per_year"])
    assert pd.isna(df.loc["2020_b", "field_pct"])
    # within (NLP, 2020): a outranks c
    assert df.loc["2020_a", "field_pct"] > df.loc["2020_c", "field_pct"]


def test_stage_report_counts_unmatched():
    report = stage_report(finalize(_raw()))
    assert report["papers"] == 3
    assert report["matched"] == 2
    assert report["unmatched"] == 1
    assert report["unmatched_by_year"] == {2020: 1}
    assert report["by_method"]["none"] == 1
