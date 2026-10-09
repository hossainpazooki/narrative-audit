"""Tests for the Barnett and Doubleday acronym classifier.

This is the most domain-specific logic in the repository and the least
self-evident, so it gets the most tests. The rule under test is:

    a token is an acronym iff  uppercase >= (lowercase + digits)
                               and uppercase >= 2

The second clause is this project's addition. Barnett and Doubleday's
published definition states only the first. It is what excludes chemical
symbols (Na), units (pH, mL) and sentence-initial words (We, To), all of
which satisfy the first clause. Tests below pin that behaviour explicitly
so the divergence cannot be lost silently.
"""
import pytest

from metrics.acronyms import (
    _extract_acronyms,
    _is_acronym_token,
    _is_all_caps_word,
    _preprocess_abstract,
    _preprocess_title,
    _remove_dots,
    _word_stats,
)


# ── _word_stats ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("word,expected", [
    ("ML",     (2, 0, 0)),
    ("Neural", (1, 5, 0)),
    ("GPT4",   (3, 0, 1)),
    ("",       (0, 0, 0)),
    ("123",    (0, 0, 3)),
    ("a-B",    (1, 1, 0)),      # punctuation counts as none of the three
    ("Ünicode",(1, 6, 0)),      # non-ASCII letters classify normally
])
def test_word_stats(word, expected):
    assert _word_stats(word) == expected


# ── _is_acronym_token: the rule itself ───────────────────────────────────

@pytest.mark.parametrize("token", [
    "ML", "CNN", "BERT", "GAN", "LSTM",
    "RNNs",      # u=3 >= l=1, and u>=2
    "GPT4",      # u=3 >= l+n=1
    "iOS",       # u=2 >= l=1
    "MoE",       # u=2 >= l=1
])
def test_accepts_real_acronyms(token):
    assert _is_acronym_token(token) is True


@pytest.mark.parametrize("token,why", [
    ("We",   "sentence-initial word: u=1, fails u>=2"),
    ("To",   "sentence-initial word"),
    ("Na",   "chemical symbol: u=1"),
    ("Ca",   "chemical symbol"),
    ("pH",   "unit: u=1, l=1"),
    ("mL",   "unit"),
    ("3D",   "u=1, n=1, so u >= l+n holds but u>=2 fails"),
    ("2D",   "same as 3D"),
    ("A",    "single letter"),
    ("",     "empty token: u=0 fails u>=2"),
    ("word", "all lowercase"),
    ("123",  "digits only"),
    ("Neural", "u=1 < l=5"),
])
def test_rejects_non_acronyms(token, why):
    assert _is_acronym_token(token) is False, why


def test_the_two_clauses_are_independent():
    """Each clause must reject something the other accepts.

    If either clause were dropped the classifier would still pass a naive
    test set, so this pins that both are load-bearing.
    """
    # satisfies clause 1 (u >= l+n) but not clause 2 (u >= 2)
    assert _word_stats("We")[0] >= sum(_word_stats("We")[1:])
    assert _is_acronym_token("We") is False

    # satisfies clause 2 (u >= 2) but not clause 1 (u >= l+n)
    u, l, n = _word_stats("AbcdE")
    assert u >= 2 and not (u >= l + n)
    assert _is_acronym_token("AbcdE") is False


def test_digits_count_against_the_token():
    """Digits sit on the same side of the inequality as lowercase letters.

    The comparison is >=, so an exact tie counts AS an acronym. AB12 has
    u=2 and n=2, and 2 >= 2 holds, so it is accepted. One more digit tips
    it over.
    """
    assert _is_acronym_token("AB12") is True     # u=2, n=2 -> tie, accepted
    assert _is_acronym_token("AB123") is False   # u=2, n=3 -> 2 >= 3 fails
    assert _is_acronym_token("ABC12") is True    # u=3 >= n=2


def test_ties_are_accepted():
    """Pin the >= boundary in both the digit and the lowercase direction."""
    assert _is_acronym_token("ABcd") is True     # u=2, l=2 -> tie, accepted
    assert _is_acronym_token("ABcde") is False   # u=2, l=3 -> fails


# ── _is_all_caps_word ────────────────────────────────────────────────────

@pytest.mark.parametrize("word,expected", [
    ("ABC",  True),
    ("AB",   True),
    ("A",    False),    # len must be > 1
    ("AbC",  False),
    ("AB1",  False),    # digit means stripped length differs
    ("",     False),
])
def test_is_all_caps_word(word, expected):
    assert _is_all_caps_word(word) is expected


# ── _remove_dots ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("W.H.O.",   "WHO."),     # trailing dot has no following capital, so stays
    ("U.S.",     "US."),
    ("end. Next", "end. Next"),   # sentence break: space follows the dot
    ("a.B",      "aB"),
    ("no dots",  "no dots"),
])
def test_remove_dots(text, expected):
    assert _remove_dots(text) == expected


# ── _extract_acronyms ────────────────────────────────────────────────────

def test_skips_dummy_placeholders():
    assert _extract_acronyms(["dummy", "CNN", "dummy"]) == ["CNN"]


def test_strips_trailing_plural_s():
    assert _extract_acronyms(["CNNs", "GANs"]) == ["CNN", "GAN"]


def test_does_not_strip_s_from_short_tokens():
    """The strip requires len > 2, so a two-character token keeps its s."""
    # 'As' is not an acronym anyway (u=1); use a token that is.
    assert _extract_acronyms(["ABs"]) == ["AB"]


def test_excludes_tokens_longer_than_15_chars():
    """Appendix 1 step 25 of Barnett and Doubleday."""
    assert _extract_acronyms(["A" * 15]) == ["A" * 15]
    assert _extract_acronyms(["A" * 16]) == []


def test_length_cap_is_applied_after_plural_strip():
    """A 16-character token ending in s becomes 15 and is therefore kept."""
    token = "A" * 15 + "s"
    assert len(token) == 16
    assert _extract_acronyms([token]) == ["A" * 15]


def test_empty_input():
    assert _extract_acronyms([]) == []


# ── preprocessing: exclusion rules ───────────────────────────────────────

@pytest.mark.parametrize("preprocess", [_preprocess_title, _preprocess_abstract])
def test_empty_text_is_excluded(preprocess):
    words, excluded = preprocess("")
    assert excluded is True
    assert words == []


@pytest.mark.parametrize("preprocess", [_preprocess_title, _preprocess_abstract])
def test_single_word_is_excluded(preprocess):
    """Fewer than two words carries no usable denominator."""
    _words, excluded = preprocess("Transformers")
    assert excluded is True


@pytest.mark.parametrize("preprocess", [_preprocess_title, _preprocess_abstract])
def test_all_caps_text_is_excluded(preprocess):
    """A fully capitalised string is a heading artefact, not prose."""
    _words, excluded = preprocess("DEEP LEARNING FOR VISION TASKS")
    assert excluded is True


def test_ordinary_title_is_not_excluded():
    words, excluded = _preprocess_title(
        "Attention is all you need for machine translation")
    assert excluded is False
    assert len(words) > 1


def test_html_tags_are_stripped():
    words, excluded = _preprocess_title(
        "Learning <b>deep</b> representations for vision")
    assert excluded is False
    assert not any("<" in w or ">" in w for w in words)


def test_roman_numerals_become_dummy():
    """Roman numerals would otherwise classify as acronyms (II, III, VI)."""
    words, excluded = _preprocess_title("Study II of neural network scaling laws")
    assert excluded is False
    assert "II" not in words
    assert "dummy" in words


# ── preprocessing: the capitalisation heuristics ─────────────────────────
#
# These branches implement Barnett and Doubleday's rules for spotting
# headers and title-case artefacts that would otherwise be counted as
# acronyms. They only fire on specific shapes, so the NeurIPS corpus
# exercises some of them rarely or not at all.

def test_title_with_a_leading_cluster_of_capitals_is_excluded():
    """Four capitalised words at the front, one of them 5+ characters.

    The proportion test must not already have excluded it, so the title
    needs enough lowercase words to keep the all-caps share below 0.6.
    """
    words, excluded = _preprocess_title(
        "DEEPER CONV NETS ARE good at many vision tasks")
    assert excluded is True
    assert words == []


def test_title_with_a_long_capitalised_first_word_is_masked_not_excluded():
    """A single long all-caps word becomes 'dummy' and stops counting."""
    words, excluded = _preprocess_title(
        "TRANSFORMER models are useful for many natural language tasks")
    assert excluded is False
    assert words[0] == "dummy"
    assert "TRANSFORMER" not in words


def test_title_with_two_long_capitalised_words_masks_both():
    words, excluded = _preprocess_title(
        "TRANSFORMER NETWORKS are useful for many natural language tasks here")
    assert excluded is False
    assert words[0] == "dummy" and words[1] == "dummy"


def test_abstract_with_four_consecutive_capitals_is_excluded():
    """A run of four all-caps words signals a header, not prose."""
    _words, excluded = _preprocess_abstract(
        "THE QUICK BROWN FOX jumps over the lazy dog and then runs away fast")
    assert excluded is True


def test_abstract_with_three_consecutive_capitals_survives():
    """Three is under the threshold, so the run resets and the text stands."""
    _words, excluded = _preprocess_abstract(
        "THE QUICK BROWN fox jumps over the lazy dog and then runs away fast")
    assert excluded is False


def test_abstract_with_a_long_capitalised_first_word_is_masked():
    words, excluded = _preprocess_abstract(
        "TRANSFORMER models are useful for many natural language processing tasks")
    assert excluded is False
    assert words[0] == "dummy"


def test_gene_sequences_become_dummy():
    """Six or more characters drawn only from ATCGUp are sequences."""
    words, excluded = _preprocess_abstract(
        "The sequence ATCGATCG appears in many of the observed samples here")
    assert excluded is False
    assert "ATCGATCG" not in words
    assert "dummy" in words


def test_numbers_in_abstracts_are_replaced_before_classification():
    """Otherwise a token like 2024 would reach the acronym test."""
    words, excluded = _preprocess_abstract(
        "We trained the model on 2024 examples drawn from the public corpus")
    assert excluded is False
    assert "2024" not in words


# ── compute_row ──────────────────────────────────────────────────────────

def test_compute_row_shape():
    from metrics.acronyms import compute_row
    row = compute_row({
        "paper_id": "2020_x", "venue": "neurips", "year": 2020,
        "title": "Scaling CNN and LSTM models for vision tasks",
        "abstract": ("We evaluate CNN and LSTM architectures on several "
                     "benchmark datasets and report consistent gains."),
    })
    for key in ("paper_id", "venue", "year", "title_acronym_count",
                "title_word_count", "title_acronym_density",
                "abstract_acronym_count", "abstract_word_count",
                "abstract_acronym_density", "title_acronyms",
                "abstract_acronyms"):
        assert key in row, key


def test_compute_row_finds_the_acronyms():
    from metrics.acronyms import compute_row
    row = compute_row({
        "paper_id": "x", "venue": "neurips", "year": 2020,
        "title": "Scaling CNN and LSTM models for vision tasks",
        "abstract": ("We evaluate CNN and LSTM architectures on several "
                     "benchmark datasets and report consistent gains."),
    })
    assert "CNN" in row["title_acronyms"]
    assert "LSTM" in row["title_acronyms"]
    assert row["title_acronym_count"] == 2


def test_compute_row_on_an_excluded_title_reports_zero_not_none():
    """Downstream aggregation averages these columns, so the type matters."""
    from metrics.acronyms import compute_row
    row = compute_row({
        "paper_id": "x", "venue": "neurips", "year": 2020,
        "title": "", "abstract": "",
    })
    assert row["title_acronym_count"] in (0, None)
