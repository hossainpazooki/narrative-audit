"""concrete_number and jargon_density edge cases: the guard branches the
well-formed corpus never enters."""

from rubric.tier1 import concrete_number, jargon_density


def test_concrete_number_percent():
    assert concrete_number("improves accuracy by 4.2%")


def test_concrete_number_n_equals():
    assert concrete_number("a user study (n = 50)")


def test_concrete_number_times():
    assert concrete_number("a 3.5x speedup") or concrete_number("3.5× faster")


def test_concrete_number_unit():
    assert concrete_number("across 13 models up to 72B parameters")


def test_bare_number_does_not_count():
    # years and bare integers are not concrete results
    assert not concrete_number("since 2016 the field has grown")
    assert not concrete_number("we address this in section 3")


def test_concrete_number_empty():
    assert not concrete_number("")


def test_jargon_density_plain_english_is_low():
    text = "We study how people learn from each other in large groups."
    assert jargon_density(text) < 0.1


def test_jargon_density_counts_rare_terms():
    text = "backpropagation hyperparameters minibatch regularizer"
    assert jargon_density(text) > 0.5


def test_jargon_density_skips_acronyms():
    # acronyms are measured by the acronym rule, not double-counted here
    assert jargon_density("LSTM GRU CNN") == 0.0


def test_jargon_density_hyphenated_common_parts():
    # a hyphenated word with a common-English component is not jargon
    assert jargon_density("state-of-the-art fine-tuning") == 0.0


def test_jargon_density_empty():
    assert jargon_density("") == 0.0
