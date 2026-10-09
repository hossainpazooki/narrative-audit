"""The Tier-2 integer parser and the Tier-3 JSON parser: malformed,
truncated, and extra-text outputs — exactly the shapes a sampling judge
produces and the corpus never will."""

from rubric.judge_prompts import parse_int_score
from rubric.tier3 import parse_tier3_json, verify_evidence


# ── Tier 2: integer scores ───────────────────────────────────────────────────

def test_int_score_formatted():
    assert parse_int_score("Score: 3", 0, 5) == 3


def test_int_score_bare():
    assert parse_int_score("4", 0, 5) == 4


def test_int_score_with_prose():
    assert parse_int_score("I think the answer is Score: 2 because", 0, 5) == 2


def test_int_score_out_of_range_is_null_not_clamped():
    assert parse_int_score("Score: 7", 0, 5) is None
    assert parse_int_score("Score: -1", 0, 5) is None


def test_int_score_malformed():
    assert parse_int_score("three", 0, 5) is None
    assert parse_int_score("", 0, 5) is None


def test_int_score_fraction_takes_numerator():
    assert parse_int_score("3/5", 0, 5) == 3


# ── Tier 3: JSON extraction ──────────────────────────────────────────────────

def test_json_clean():
    out = parse_tier3_json('{"value": true, "evidence": "we ran ablations"}')
    assert out == {"value": True, "evidence": "we ran ablations"}


def test_json_extra_text_around_object():
    raw = ('Sure! Here is the JSON you asked for:\n'
           '{"value": false, "evidence": ""}\n'
           'Let me know if you need anything else.')
    assert parse_tier3_json(raw) == {"value": False, "evidence": ""}


def test_json_truncated_is_null():
    assert parse_tier3_json('{"value": true, "evidence": "we ran abl') is None


def test_json_malformed_is_null():
    assert parse_tier3_json('{"value": maybe, "evidence": }') is None
    assert parse_tier3_json("no json here at all") is None
    assert parse_tier3_json("") is None
    assert parse_tier3_json(None) is None


def test_json_python_booleans_accepted():
    assert parse_tier3_json('{"value": True, "evidence": "x"}') == \
        {"value": True, "evidence": "x"}


def test_json_value_must_be_boolean():
    assert parse_tier3_json('{"value": "yes", "evidence": "x"}') is None
    assert parse_tier3_json('{"evidence": "x"}') is None


def test_json_braces_inside_evidence_string():
    raw = '{"value": true, "evidence": "loss {train} dropped"} trailing'
    assert parse_tier3_json(raw) == {"value": True,
                                     "evidence": "loss {train} dropped"}


def test_json_nested_object_taken_whole():
    raw = 'note {"value": true, "evidence": "x", "extra": {"a": 1}} after'
    out = parse_tier3_json(raw)
    assert out is not None and out["value"] is True


# ── evidence verification ────────────────────────────────────────────────────

def test_evidence_verified_whitespace_insensitive():
    text = "We  report results\nover five seeds."
    assert verify_evidence("results over five seeds", text)


def test_evidence_not_in_text_fails():
    assert not verify_evidence("we ran ablations", "no such sentence here")


def test_empty_evidence_fails():
    assert not verify_evidence("", "anything")
