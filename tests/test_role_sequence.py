"""role_sequence_dist: the source post's worked example as fixture.

The Nanda post's "Annotated Abstract" figure breaks the abstract of
*Refusal in Language Models Is Mediated by a Single Direction* (Arditi et
al., NeurIPS 2024) down sentence by sentence: Topic, Motivation,
Contribution, Detail/Nuance, Evidence / Contribution 2, Weaker result,
Narrow impact, Broad impact. Projected onto the five template roles
(docs/rubric.md gives the mapping) that is situate, gap, contribution x3,
evidence, impact x2 — which is what the labeler must produce, and which
collapses to the template exactly (distance 0). The edge cases cover what
the corpus never provides: empty text, one sentence, abbreviations at
sentence boundaries.
"""

from rubric.tier1 import (
    TEMPLATE, _collapse, _levenshtein, label_roles, role_sequence_dist,
    split_sentences,
)

REFUSAL = (
    "Conversational large language models are fine-tuned for both "
    "instruction-following and safety, resulting in models that obey benign "
    "requests but refuse harmful ones. While this refusal behavior is "
    "widespread across chat models, its underlying mechanisms remain poorly "
    "understood. In this work, we show that refusal is mediated by a "
    "one-dimensional subspace, across 13 popular open-source chat models up "
    "to 72B parameters in size. Specifically, for each model, we find a "
    "single direction such that erasing this direction from the model's "
    "residual stream activations prevents it from refusing harmful "
    "instructions, while adding this direction elicits refusal on even "
    "harmless instructions. Leveraging this insight, we propose a novel "
    "white-box jailbreak method that surgically disables refusal with "
    "minimal effect on other capabilities. Finally, we mechanistically "
    "analyze how adversarial suffixes suppress propagation of the "
    "refusal-mediating direction. Our findings underscore the brittleness "
    "of current safety fine-tuning methods. More broadly, our work "
    "showcases how an understanding of model internals can be leveraged to "
    "develop practical methods for controlling model behavior."
)


def test_refusal_sentence_count():
    assert len(split_sentences(REFUSAL)) == 8


def test_refusal_roles():
    # the post's figure, projected: Topic, Motivation, Contribution,
    # Detail/Nuance, Evidence / Contribution 2, Weaker result,
    # Narrow impact, Broad impact
    labels = label_roles(split_sentences(REFUSAL))
    assert labels == ["situate", "gap", "contribution", "contribution",
                      "contribution", "evidence", "impact", "impact"]


def test_refusal_distance_zero():
    assert role_sequence_dist(REFUSAL) == 0


def test_empty_abstract_is_max_distance():
    assert role_sequence_dist("") == len(TEMPLATE)


def test_single_sentence():
    # One situating sentence: four template slots missing -> distance 4.
    assert role_sequence_dist("Deep learning is a subfield of machine learning.") == 4


def test_abbreviations_do_not_split():
    text = "We compare methods, e.g. SGD and Adam. Results improve."
    assert len(split_sentences(text)) == 2


def test_collapse():
    assert _collapse(["a", "a", "b", "a"]) == ["a", "b", "a"]
    assert _collapse([]) == []


def test_levenshtein():
    assert _levenshtein(["a", "b"], ["a", "b"]) == 0
    assert _levenshtein(["a"], ["a", "b"]) == 1
    assert _levenshtein(["x", "b"], ["a", "b"]) == 1
    assert _levenshtein([], ["a", "b"]) == 2


def test_reversed_template_distance():
    # An abstract told backwards should be far from the template.
    sents = ["These findings have broad implications for the field.",
             "Experiments show a 40% improvement over baselines.",
             "We propose a new method for this problem.",
             "However, existing approaches remain poorly understood.",
             "Machine learning is widely used in recent years."]
    text = " ".join(sents)
    assert role_sequence_dist(text) >= 3
