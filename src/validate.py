"""
Pydantic model for validated processed-paper records.

Every record in data/processed/{venue}/{venue}_{year}.json must conform
to ProcessedPaper. Use validate_file() or validate_all() to check.
"""

from __future__ import annotations
import json
import os
import sys
from typing import Optional

try:
    from pydantic import BaseModel, field_validator, model_validator
except ImportError:
    sys.exit("pip install pydantic")

VALID_VENUES    = {"neurips", "iclr", "icml", "arxiv"}
YEAR_MIN        = 1987
YEAR_MAX        = 2030
MIN_TITLE_WORDS = 1
MIN_ABSTRACT_WORDS = 5


class ProcessedPaper(BaseModel):
    paper_id:               str
    venue:                  str
    year:                   int
    title:                  str
    abstract:               str
    authors:                list[str]
    arxiv_categories:       Optional[list[str]] = None
    arxiv_primary_category: Optional[str] = None

    # ── field validators ────────────────────────────────────────────────

    @field_validator("paper_id")
    @classmethod
    def paper_id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("paper_id is empty")
        return v

    @field_validator("venue")
    @classmethod
    def venue_known(cls, v: str) -> str:
        if v not in VALID_VENUES:
            raise ValueError(f"unknown venue '{v}' — expected one of {VALID_VENUES}")
        return v

    @field_validator("year")
    @classmethod
    def year_in_range(cls, v: int) -> int:
        if not (YEAR_MIN <= v <= YEAR_MAX):
            raise ValueError(f"year {v} outside expected range {YEAR_MIN}–{YEAR_MAX}")
        return v

    @field_validator("title")
    @classmethod
    def title_min_words(cls, v: str) -> str:
        words = v.strip().split()
        if len(words) < MIN_TITLE_WORDS:
            raise ValueError(f"title has only {len(words)} word(s): {repr(v)}")
        return v

    @field_validator("abstract")
    @classmethod
    def abstract_min_words(cls, v: str) -> str:
        words = v.strip().split()
        if len(words) < MIN_ABSTRACT_WORDS:
            raise ValueError(
                f"abstract has only {len(words)} word(s) (min {MIN_ABSTRACT_WORDS}): "
                f"{repr(v[:80])}"
            )
        return v

    @field_validator("authors")
    @classmethod
    def authors_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("authors list is empty")
        blank = [a for a in v if not a.strip()]
        if blank:
            raise ValueError(f"{len(blank)} author name(s) are empty strings")
        return v

    # ── cross-field validators ───────────────────────────────────────────

    @model_validator(mode="after")
    def arxiv_fields_consistent(self) -> "ProcessedPaper":
        if self.venue == "arxiv":
            if not self.arxiv_categories:
                raise ValueError("arxiv_categories must be present for arXiv papers")
            if not self.arxiv_primary_category:
                raise ValueError("arxiv_primary_category must be present for arXiv papers")
        else:
            if self.arxiv_categories is not None:
                raise ValueError(
                    f"arxiv_categories should not be set for venue='{self.venue}'"
                )
        return self


# ── Validation helpers ────────────────────────────────────────────────────────

class ValidationResult:
    def __init__(self, path: str):
        self.path       = path
        self.total      = 0
        self.failures: list[tuple[int, dict, str]] = []  # (index, record, error_msg)

    @property
    def n_ok(self) -> int:
        return self.total - len(self.failures)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        s = f"[{status}] {self.path}  —  {self.n_ok}/{self.total} valid"
        if self.failures:
            shown = self.failures[:5]
            for idx, rec, msg in shown:
                pid = rec.get("paper_id", "?")
                s += f"\n      record[{idx}] paper_id={pid!r}: {msg}"
            if len(self.failures) > 5:
                s += f"\n      … and {len(self.failures) - 5} more failures"
        return s


def validate_file(path: str) -> ValidationResult:
    result = ValidationResult(path)
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    result.total = len(records)
    for i, rec in enumerate(records):
        try:
            ProcessedPaper.model_validate(rec)
        except Exception as e:
            result.failures.append((i, rec, str(e)))
    return result


def validate_all(
    processed_dir: str = "data/processed",
    venues: list[str] | None = None,
) -> dict[str, list[ValidationResult]]:
    """Validate all processed JSON files. Returns {venue: [ValidationResult]}."""
    if venues is None:
        venues = sorted(VALID_VENUES)

    all_results: dict[str, list[ValidationResult]] = {}
    for venue in venues:
        venue_dir = os.path.join(processed_dir, venue)
        if not os.path.isdir(venue_dir):
            continue
        files = sorted(f for f in os.listdir(venue_dir) if f.endswith(".json"))
        all_results[venue] = [
            validate_file(os.path.join(venue_dir, f)) for f in files
        ]
    return all_results


def main() -> None:
    """Validate every processed JSON against the schema.

        python src/validate.py --venues neurips
    """
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--processed-dir",
                    default=os.path.join(os.environ.get("AA_DATA", "data"), "processed"))
    ap.add_argument("--venues", nargs="+", default=["neurips"])
    a = ap.parse_args()

    results = validate_all(a.processed_dir, a.venues)
    if not results:
        sys.exit(f"no processed venues found under {a.processed_dir}")

    total = failed = 0
    for venue, files in results.items():
        n_ok = sum(r.n_ok for r in files)
        n_all = sum(r.total for r in files)
        bad = [r for r in files if not r.passed]
        total += n_all
        failed += n_all - n_ok
        print(f"  [{venue}] {n_ok:,}/{n_all:,} records valid across {len(files)} files")
        for r in bad:
            print("  " + r.summary())

    if failed:
        sys.exit(f"{failed:,} of {total:,} records failed schema validation")
    print(f"  all {total:,} records valid")


if __name__ == "__main__":
    main()
