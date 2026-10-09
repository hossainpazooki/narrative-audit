"""Pure matching and normalization logic for the outcomes stage.

Everything here is deterministic and network-free so the matcher's edge
cases (title collisions, missing DOIs, no match at all) are unit-testable
without touching OpenAlex. openalex.py owns the HTTP side and calls in.

Age normalization: cites_per_year divides by the paper's age in years with a
floor of one year, so current-year papers are not divided by a fraction.
field_pct is the percentile of cites_per_year within (subfield, year),
computed over the corpus itself rather than taken from OpenAlex, so it is
reproducible from outcomes_raw.csv alone.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# The census year citations are counted against. Fixed, not datetime.now():
# the outcome variable must not change between a run today and a rerun next
# month, or verify.py's byte-for-byte comparison is meaningless.
CENSUS_YEAR = 2026

ARXIV_ID_PAT = re.compile(r"(?:arxiv[.:]|abs/|pdf/)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)


def normalize_title(title: str) -> str:
    """Canonical form used for title matching against OpenAlex.

    Lowercase, unicode-normalized, LaTeX commands and all non-alphanumerics
    dropped, whitespace collapsed. Deliberately aggressive: OpenAlex titles
    differ from papers.nips.cc titles in casing, punctuation, and math markup,
    and the year filter carries the rest of the disambiguation burden.
    """
    if not title:
        return ""
    t = unicodedata.normalize("NFKD", title)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"\\[a-z]+", " ", t)          # latex commands
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def title_tokens(title: str) -> set[str]:
    return set(normalize_title(title).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def pick_match(paper: dict, candidates: list[dict]) -> tuple[dict | None, str]:
    """Choose the OpenAlex work for one paper from a candidate list.

    Returns (work, method) where method is one of:
      exact_title   normalized titles equal, publication year within 1
      fuzzy_title   best Jaccard >= 0.9 over title tokens, year within 1,
                    and the best candidate leads the runner-up by >= 0.05
                    (otherwise the collision is ambiguous -> no match)
      doi           caller matched via the record's DOI
      none          no candidate survives

    A collision — two candidates with the same normalized title and
    acceptable year — resolves to the one whose year is nearest, then the
    more-cited one; if still tied, no match, because guessing silently is
    worse than a counted null.
    """
    year = paper.get("year")
    want = normalize_title(paper.get("title", ""))
    if not want or not candidates:
        return None, "none"

    def year_ok(w: dict) -> bool:
        wy = w.get("publication_year")
        return wy is not None and year is not None and abs(wy - year) <= 1

    exact = [w for w in candidates if normalize_title(w.get("title") or w.get("display_name") or "") == want and year_ok(w)]
    if len(exact) == 1:
        return exact[0], "exact_title"
    if len(exact) > 1:
        exact.sort(key=lambda w: (abs(w.get("publication_year", 0) - year),
                                  -(w.get("cited_by_count") or 0)))
        a, b = exact[0], exact[1]
        if (abs(a.get("publication_year", 0) - year), -(a.get("cited_by_count") or 0)) == \
           (abs(b.get("publication_year", 0) - year), -(b.get("cited_by_count") or 0)):
            return None, "none"
        return exact[0], "exact_title"

    want_toks = set(want.split())
    scored = sorted(
        ((jaccard(want_toks, title_tokens(w.get("title") or w.get("display_name") or "")), w)
         for w in candidates if year_ok(w)),
        key=lambda sw: -sw[0],
    )
    if scored and scored[0][0] >= 0.9:
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05:
            return None, "none"
        return scored[0][1], "fuzzy_title"
    return None, "none"


def extract_arxiv_id(work: dict) -> str | None:
    """arXiv id from an OpenAlex work's ids/locations, else None."""
    ids = work.get("ids") or {}
    for key in ("arxiv", "arxiv_id"):
        if ids.get(key):
            m = ARXIV_ID_PAT.search(ids[key])
            if m:
                return m.group(1)
    for loc in (work.get("locations") or []):
        src = (loc.get("source") or {})
        url = loc.get("landing_page_url") or loc.get("pdf_url") or ""
        if "arxiv" in (src.get("display_name") or "").lower() or "arxiv.org" in url:
            m = ARXIV_ID_PAT.search(url)
            if m:
                return m.group(1)
    return None


def work_to_row(paper: dict, work: dict | None, method: str) -> dict:
    """Flatten one matched (or unmatched) OpenAlex work to an output row.

    Unmatched papers keep their row with nulls so the stage report can count
    them and downstream joins stay total over the corpus.
    """
    row = {
        "paper_id":     paper["paper_id"],
        "year":         paper["year"],
        "match_method": method,
        "openalex_id":  None,
        "citations":    None,
        "author_count": None,
        "subfield":     None,
        "arxiv_id":     None,
        "arxiv_v1_date": None,
    }
    if work is None:
        return row
    concepts = work.get("concepts") or []
    top = max(concepts, key=lambda c: c.get("score", 0.0), default=None)
    row.update({
        "openalex_id":  (work.get("id") or "").rsplit("/", 1)[-1] or None,
        "citations":    work.get("cited_by_count"),
        "author_count": len(work.get("authorships") or []) or None,
        "subfield":     (top or {}).get("display_name"),
        "arxiv_id":     extract_arxiv_id(work),
        "arxiv_v1_date": work.get("first_online_date") or None,
    })
    return row


def finalize(raw: pd.DataFrame) -> pd.DataFrame:
    """outcomes_raw rows -> the outcomes table the spec names.

    Adds cites_per_year (age-normalized, census year fixed), field_pct
    (percentile of cites_per_year within subfield x year; within year alone
    when the subfield is null), and preprint_before_conference.
    """
    df = raw.copy()
    age = (CENSUS_YEAR - df["year"]).clip(lower=1)
    df["cites_per_year"] = df["citations"] / age

    v1_year = pd.to_datetime(df["arxiv_v1_date"], errors="coerce").dt.year
    df["preprint_before_conference"] = (v1_year <= df["year"]).where(v1_year.notna())

    group = df["subfield"].fillna("__none__")
    df["field_pct"] = (
        df.groupby([group, df["year"]])["cites_per_year"]
          .rank(pct=True, method="average")
    )
    df.loc[df["citations"].isna(), ["cites_per_year", "field_pct"]] = None
    return df


def stage_report(df: pd.DataFrame) -> dict:
    """Counts for the unmatched-paper report the spec requires."""
    return {
        "papers":            int(len(df)),
        "matched":           int(df["citations"].notna().sum()),
        "unmatched":         int(df["citations"].isna().sum()),
        "by_method":         {k: int(v) for k, v in df["match_method"].value_counts().items()},
        "with_arxiv_id":     int(df["arxiv_id"].notna().sum()),
        "with_v1_date":      int(df["arxiv_v1_date"].notna().sum()),
        "unmatched_by_year": {int(k): int(v)
                              for k, v in df.loc[df["citations"].isna(), "year"]
                                            .value_counts().sort_index().items()},
    }
