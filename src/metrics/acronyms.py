#!/usr/bin/env python3
"""
Per-paper acronym metrics — exact port of Barnett & Doubleday (2020) algorithm.

Reference:
  Barnett A, Doubleday Z. (2020). The growth of acronyms in the scientific
  literature. eLife 9:e60080. https://doi.org/10.7554/eLife.60080
  Source R code: https://github.com/agbarnett/acronyms
  Algorithm detail: docs/metrics/acronym.md

The implementation follows 99_main_function_title.R and
99_main_function_abstract.R line-by-line, translating R to Python.

Output — one CSV per venue in data/per_paper/acronyms/{venue}.csv:
  paper_id, venue, year,
  title_acronym_count, title_word_count, title_acronym_density,
  abstract_acronym_count, abstract_word_count, abstract_acronym_density,
  title_acronyms, abstract_acronyms,
  arxiv_primary_category

Acronyms are stored in their original case after plural normalization.

Run:
  python src/metrics/acronyms.py
  python src/metrics/acronyms.py --venues neurips iclr
"""

from __future__ import annotations
import argparse
import os
import re
import sys

import pandas as pd

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)  # allows direct invocation
from schema import load_processed, resume_year_counts, resume_year_ids

VENUES = ["neurips", "iclr", "icml", "arxiv"]


# ── B&D exclusion/replacement sets (from 99_main_function_title.R lines 8–84) ─

# Roman numerals: only 2+ character ones (single-char ones excluded per B&D line 10)
# "roman.start = roman.start[nchar(roman.start)>1]"
_ROMAN_2PLUS = {
    'II', 'III', 'IV', 'VI', 'VII', 'VIII', 'IX',
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
    'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX',
}
# Roman numerals suffixed with 'th' or lowercase letters (up to roman 2–9)
_ROMAN_SUFFIXED = set()
for _r in ['II', 'III', 'IV', 'VI', 'VII', 'VIII', 'IX']:
    _ROMAN_SUFFIXED.add(_r + 'th')
    for _lc in 'abcde':
        _ROMAN_SUFFIXED.add(_r + _lc)
_ALL_ROMAN = _ROMAN_2PLUS | _ROMAN_SUFFIXED

# Chromosomes — exact list from B&D line 29 (YYYYY = 5 Y's)
_CHROMOSOMES = {'XX', 'XY', 'XO', 'ZO', 'XXYY', 'ZW', 'ZWW', 'XXX', 'XXXX', 'XXXXX', 'YYYYY'}

# Title-only boilerplate to replace with 'DUMMYDUMMY' (line 77)
_BOGUS_TITLE = {'WITHDRAWN', 'CORRIGENDUM', 'EDITORIAL', 'MEDICAL', 'TRANSACTIONS'}

# Abstract subheadings to replace with 'DUMMYDUMMY' (lines 12–37; adapted for ML)
_BOGUS_ABSTRACT = {
    'ABSTRACT', 'KEYWORDS', 'KEY WORDS',
    'WITHDRAWN', 'CORRIGENDUM', 'BACKGROUND', 'INTRODUCTION', 'OBJECTIVE',
    'METHODS', 'METHOD', 'MATERIALS AND METHODS', 'MATERIALS & METHODS',
    'AIM', 'AIMS', 'DESIGN', 'SETTING', 'SUBJECTS', 'PARTICIPANTS',
    'INTERVENTION', 'OUTCOMES', 'RESULTS', 'DISCUSSION', 'CONCLUSION',
    'LIMITATIONS', 'SUMMARY', 'PURPOSE', 'HYPOTHESIS', 'SIGNIFICANCE',
    'SIGNIFICANCE STATEMENT', 'IMPORTANCE', 'FUNDING', 'DATA AVAILABILITY',
    'RECOMMENDATIONS', 'PERSPECTIVE', 'LAY ABSTRACT',
}

# Composite units — replace with 'units' in abstract (line 43–44)
_UNITS_PAT = re.compile(
    r'\b(?:mmol/[lL]|MMOL/L|kg/m2|KG/M2|pg/m[lL]|PG/ML|ng/m[lL]|NG/ML'
    r'|mg/d[lL]|MG/DL|g/m[lL]|G/ML|g2/m|G2/M)\b'
)

# Number patterns to replace with ' number ' in abstract (line 47–48)
_NUM_PAT = re.compile(r'-\d+\.\d+| \d+\.\d+| \d+|-\d+')

# HTML-like tags to remove (lines 69–73 shared)
_TAG_PAT = re.compile(
    r'</?(?:sub|sup|super|i|b|exp|fraction)>', re.IGNORECASE
)


# ── str_remove_dots equivalent (99_remove_dots.R) ────────────────────────────

def _remove_dots(text: str) -> str:
    """Remove full-stops within acronyms: W.H.O. → WHO, U.S. → US."""
    # Match single uppercase letters separated by dots (with optional trailing dot)
    return re.sub(r'(?<=[A-Za-z])\.(?=[A-Z])', '', text)


# ── Core word classification (lines 178–182, shared by title and abstract) ────

def _word_stats(word: str) -> tuple[int, int, int]:
    """Return (uppercase_count, lowercase_count, digit_count) for a word."""
    u = sum(1 for c in word if c.isupper())
    l = sum(1 for c in word if c.islower())
    n = sum(1 for c in word if c.isdigit())
    return u, l, n


def _is_acronym_token(word: str) -> bool:
    """
    B&D acronym test (line 182):
        (uwords >= (lwords + nwords)) & uwords >= 2

    Naturally excludes sentence-initial words ('We', u=1), chemical symbols
    ('Na', u=1), units ('pH', u=1), and alphanumeric tokens like '2D'
    (u=1, l=0, n=1 → 1 >= 1 but 1 < 2 → NOT an acronym). This matches B&D
    despite the intuition that '2D' should count.
    """
    u, l, n = _word_stats(word)
    return u >= (l + n) and u >= 2


def _is_all_caps_word(word: str) -> bool:
    """A word is 'all caps' if after removing non-uppercase, length is unchanged and len > 1."""
    stripped = re.sub(r'[^A-Z]', '', word)
    return len(word) == len(stripped) and len(word) > 1


# ── Title preprocessing (99_main_function_title.R) ───────────────────────────

def _preprocess_title(text: str) -> tuple[list[str], bool]:
    """
    Preprocess title following B&D lines 94–167.
    Returns (word_list, excluded).
    Words include 'dummy' placeholders (counted in denominator).
    """
    if not text:
        return [], True

    # Line 94: remove HTML tags
    text = _TAG_PAT.sub(' ', text)

    # Line 95: replace boilerplate with 'DUMMYDUMMY'
    for phrase in sorted(_BOGUS_TITLE, key=len, reverse=True):
        text = re.sub(rf'\b{re.escape(phrase)}\b', 'DUMMYDUMMY', text, flags=re.IGNORECASE)

    # Line 96: remove dots in acronyms
    text = _remove_dots(text)

    # Lines 97–98: remove numbers adjacent to hyphens
    text = re.sub(r'-\d+(?:\s|$)', ' ', text)
    text = re.sub(r'(?:^|\s)\d+-', ' ', text)

    # Line 99: remove punctuation (keeping & per line 80 comment "want to keep &")
    text = re.sub(r'[^\w\s&]', ' ', text)

    # Line 100: replace '...' with '~'
    text = text.replace('...', '~')

    # B&D line 101: crude plural normalisation — strips trailing 's' from all words
    text = text.replace('s ', ' ')

    # Lines 103–104: specific replacements
    text = text.replace('MicroRNA', 'MiRNA')
    text = text.replace('ATPase', 'ATP')

    # Collapse whitespace, split into words
    words = text.split()
    words = [w for w in words if w]
    if not words:
        return [], True

    # Lines 112–131: all-caps proportion check
    all_caps = [_is_all_caps_word(w) for w in words]
    words_gt1 = sum(1 for w in words if len(w) > 1)
    if words_gt1 == 0:
        return [], True
    prop_cap = sum(all_caps) / words_gt1
    if prop_cap >= 0.6:
        if len(words) > 2 or prop_cap == 1.0:
            return [], True  # excluded

    # Lines 134–148: cluster of caps at start or end (only for longer titles)
    n = len(words)
    if n > 4:
        # First 4 words all caps AND at least one word >= 5 chars AND sum == 4
        first4_caps = all_caps[:4]
        if all(first4_caps) and any(len(words[i]) >= 5 for i in range(4)) and sum(all_caps[:4]) == 4:
            return [], True
        # Last 4 words all caps AND at least one word >= 5 chars AND total all-caps == 4
        last4_caps = all_caps[n-4:]
        if all(last4_caps) and any(len(words[n-4+i]) >= 5 for i in range(4)) and sum(all_caps) == 4:
            return [], True

    # Lines 151–156: replace long caps words at start with 'dummy'
    if len(words) >= 1 and all_caps[0] and len(words[0]) > 6:
        words[0] = 'dummy'
        if len(words) >= 2 and all_caps[1] and len(words[1]) > 6:
            words[1] = 'dummy'
    if len(words) >= 2 and all_caps[0] and all_caps[1] and max(len(words[0]), len(words[1])) > 6:
        words[0] = words[1] = 'dummy'

    # Line 159: DUMMYDUMMY → dummy
    words = ['dummy' if w == 'DUMMYDUMMY' else w for w in words]

    # Lines 160–162: Roman numerals → dummy
    words = ['dummy' if w in _ALL_ROMAN else w for w in words]

    # Line 163: chromosomes → dummy
    words = ['dummy' if w in _CHROMOSOMES else w for w in words]

    words = [w for w in words if w]

    # Lines 165–167: gene sequences → dummy
    words = ['dummy' if (re.fullmatch(r'[ATCGUp]+', w) and len(w) >= 6) else w for w in words]

    # Line 169–174: skip if <= 1 word
    if len(words) <= 1:
        return [], True

    return words, False


# ── Abstract preprocessing (99_main_function_abstract.R) ─────────────────────

def _preprocess_abstract(text: str) -> tuple[list[str], bool]:
    """
    Preprocess abstract following B&D lines 56–147.
    Returns (word_list, excluded).
    """
    if not text:
        return [], True

    # Line 58: remove HTML tags
    text = _TAG_PAT.sub(' ', text)

    # Line 59: replace abstract subheadings with 'DUMMYDUMMY'
    for phrase in sorted(_BOGUS_ABSTRACT, key=len, reverse=True):
        text = re.sub(rf'\b{re.escape(phrase)}\b:?\s*', 'DUMMYDUMMY ', text, flags=re.IGNORECASE)

    # Line 60: replace numbers with ' number '
    text = _NUM_PAT.sub(' number ', text)

    # Line 61: remove dots in acronyms
    text = _remove_dots(text)

    # Line 62: replace composite units with 'units'
    text = _UNITS_PAT.sub('units', text)

    # Lines 63–64: remove numbers adjacent to hyphens
    text = re.sub(r'-\d+(?:\s|$)', ' ', text)
    text = re.sub(r'(?:^|\s)\d+-', ' ', text)

    # Line 65: remove punctuation
    text = re.sub(r'[^\w\s&]', ' ', text)

    # Line 66: replace '...' with '~'
    text = text.replace('...', '~')

    # B&D line 101: crude plural normalisation — strips trailing 's' from all words
    text = text.replace('s ', ' ')

    # Lines 69–70: specific replacements
    text = text.replace('MicroRNA', 'MiRNA')
    text = text.replace('ATPase', 'ATP')

    # Collapse and split
    words = text.split()
    words = [w for w in words if w]
    if not words:
        return [], True

    # Lines 102–121: all-caps proportion check
    all_caps = [_is_all_caps_word(w) for w in words]
    words_gt1 = sum(1 for w in words if len(w) > 1)
    if words_gt1 == 0:
        return [], True
    prop_cap = sum(all_caps) / words_gt1
    if prop_cap >= 0.6:
        if len(words) > 2 or prop_cap == 1.0:
            return [], True

    # Lines 123–131: 4+ consecutive all-caps words → exclude
    run = 0
    for cap in all_caps:
        if cap:
            run += 1
            if run >= 4:
                return [], True
        else:
            run = 0

    # Line 135: if first word is all-caps and long → dummy
    if all_caps[0] and len(words[0]) > 6:
        words[0] = 'dummy'

    # Line 138: DUMMYDUMMY → dummy
    words = ['dummy' if w == 'DUMMYDUMMY' else w for w in words]

    # Lines 139–142: Roman numerals → dummy
    words = ['dummy' if w in _ALL_ROMAN else w for w in words]

    # Line 142: chromosomes → dummy
    words = ['dummy' if w in _CHROMOSOMES else w for w in words]

    words = [w for w in words if w]

    # Lines 144–146: gene sequences → dummy
    words = ['dummy' if (re.fullmatch(r'[ATCGUp]+', w) and len(w) >= 6) else w for w in words]

    # Line 147–152: skip if <= 1 word
    n_words = len(words)
    if n_words <= 1:
        return [], True

    return words, False


# ── Acronym extraction (lines 154–168, shared logic) ─────────────────────────

def _extract_acronyms(words: list[str]) -> list[str]:
    """
    B&D acronym extraction (lines 176–193):
      acronym if (uppercase >= lowercase + digits) AND uppercase >= 2
    Post-detection: strip trailing 's' (str_remove pattern='s$').
    B&D Appendix 1 Step 25: exclude acronyms longer than 15 characters.
    """
    result = []
    for w in words:
        if w == 'dummy':
            continue
        u, l, n = _word_stats(w)
        if u >= (l + n) and u >= 2:
            # Post-detection plural strip (line 167/193)
            token = w[:-1] if w.endswith('s') and len(w) > 2 else w
            # Appendix 1 Step 25: max 15 chars
            if len(token) <= 15:
                result.append(token)
    return result


# ── Per-paper row ─────────────────────────────────────────────────────────────

def compute_row(paper: dict) -> dict:
    title_words, title_excluded    = _preprocess_title(paper.get("title", ""))
    abstract_words, abst_excluded  = _preprocess_abstract(paper.get("abstract", ""))

    title_acr    = _extract_acronyms(title_words)    if not title_excluded    else []
    abstract_acr = _extract_acronyms(abstract_words) if not abst_excluded     else []

    t_words = len(title_words)
    a_words = len(abstract_words)

    t_density = (len(title_acr) / t_words * 100) if t_words > 0 else 0.0
    a_density = (len(abstract_acr) / a_words * 100) if a_words > 0 else 0.0

    return {
        "paper_id":                 paper["paper_id"],
        "venue":                    paper["venue"],
        "year":                     paper["year"],
        "title_acronym_count":      len(title_acr),
        "title_word_count":         t_words,
        "title_acronym_density":    round(t_density, 4),
        "abstract_acronym_count":   len(abstract_acr),
        "abstract_word_count":      a_words,
        "abstract_acronym_density": round(a_density, 4),
        "title_acronyms":           "|".join(title_acr),
        "abstract_acronyms":        "|".join(abstract_acr),
        "arxiv_primary_category":   paper.get("arxiv_primary_category", ""),
    }


# ── Venue processing ──────────────────────────────────────────────────────────

def process_venue(venue: str, processed_dir: str, out_dir: str) -> None:
    venue_dir = os.path.join(processed_dir, venue)
    if not os.path.isdir(venue_dir):
        print(f"  [{venue}] not found: {venue_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{venue}.csv")

    done_counts = resume_year_counts(out_path)
    if done_counts:
        print(f"  [{venue}] resuming — {len(done_counts)} years present in output")

    header_written = os.path.exists(out_path)
    total = 0
    files = sorted(f for f in os.listdir(venue_dir) if f.endswith(".json"))
    for fname in files:
        m = re.search(r'(\d{4})', fname)
        year = int(m.group(1)) if m else -1
        papers = load_processed(os.path.join(venue_dir, fname))
        if done_counts.get(year, 0) >= len(papers):
            print(f"  [{venue}] {fname}: skip (complete)")
            continue
        if done_counts.get(year, 0) > 0:
            done = resume_year_ids(out_path, year)
            papers = [p for p in papers if p["paper_id"] not in done]
            print(f"  [{venue}] {fname}: partial year — {len(papers)} papers to top up")
        rows = []
        for p in papers:
            try:
                rows.append(compute_row(p))
            except Exception as e:
                print(f"    skip {p.get('paper_id', '?')}: {e}")
        if rows:
            pd.DataFrame(rows).to_csv(out_path, mode='a', header=not header_written, index=False)
            header_written = True
            total += len(rows)
        print(f"  [{venue}] {fname}: {len(papers)} papers → {len(rows)} rows (total {total})")

    print(f"  [{venue}] done — {total} new rows → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir",       default="data/per_paper/acronyms")
    parser.add_argument("--venues", nargs="+", default=VENUES)
    args = parser.parse_args()

    for venue in args.venues:
        print(f"\nProcessing {venue} …")
        process_venue(venue, args.processed_dir, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
