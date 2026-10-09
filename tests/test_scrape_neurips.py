"""Tests for the papers.nips.cc HTML parsers.

These are the only part of the pipeline that depends on a third party's
markup, so they are the most likely thing to break without any code change
here. They are also the one stage a reproduction run cannot check, because
it starts from already-scraped JSON.

Fixtures below are reduced to the structure each parser keys on. They are
not full pages; the point is to pin which structural features matter.
"""
import pytest

from scraping.scrape_neurips import (
    _parse_abstract_html,
    _parse_authors_html,
    _parse_hashes_new,
    _parse_hashes_old,
    _parse_title_html,
)

# Pre-2020 index: the parser keys on the SECOND container-fluid div.
OLD_INDEX = """
<html><body>
  <div class="container-fluid"><p>navigation, ignored</p></div>
  <div class="container-fluid">
    <ul>
      <li><a href="/paper/1987/hash/aaa111-Abstract.html">First paper</a></li>
      <li><a href="/paper/1987/hash/bbb222-Abstract.html">Second paper</a></li>
    </ul>
  </div>
</body></html>
"""

# 2020+ index: keys on ul.paper-list and a[title="paper title"].
NEW_INDEX = """
<html><body>
  <ul class="paper-list">
    <li><a title="paper title" href="/paper_files/paper/2020/hash/ccc333-Abstract.html">A</a></li>
    <li><a title="paper title" href="/paper_files/paper/2020/hash/ddd444-Abstract.html">B</a></li>
    <li><a title="author" href="/author/somebody">Not a paper</a></li>
  </ul>
</body></html>
"""

ABSTRACT_PAGE = """
<html><body>
  <h4>Abstract</h4>
  <p class="paper-abstract">We present a method for learning representations.</p>
</body></html>
"""


# ── index parsing ────────────────────────────────────────────────────────

def test_parses_pre2020_index():
    assert _parse_hashes_old(OLD_INDEX) == ["aaa111", "bbb222"]


def test_parses_post2020_index():
    assert _parse_hashes_new(NEW_INDEX) == ["ccc333", "ddd444"]


def test_post2020_ignores_links_that_are_not_papers():
    """Author links live in the same list and must not be collected."""
    assert "somebody" not in _parse_hashes_new(NEW_INDEX)


def test_pre2020_ignores_the_first_container():
    """The first container-fluid div is site navigation."""
    html = OLD_INDEX.replace(
        '<div class="container-fluid"><p>navigation, ignored</p></div>',
        '<div class="container-fluid"><li><a href="/paper/nav/zzz999">nav</a></li></div>')
    assert "zzz999" not in _parse_hashes_old(html)


@pytest.mark.parametrize("parser", [_parse_hashes_old, _parse_hashes_new])
def test_repeated_calls_do_not_accumulate(parser):
    """The inner class declares `hashes` as a class attribute.

    That is a shared mutable default, and it is safe only because the class
    is redefined on every call. If the class were ever hoisted out of the
    function, results would accumulate across years and every paper count
    would inflate. This test fails loudly if that happens.
    """
    html = OLD_INDEX if parser is _parse_hashes_old else NEW_INDEX
    first = parser(html)
    second = parser(html)
    assert first == second
    assert len(second) == len(first)


@pytest.mark.parametrize("parser", [_parse_hashes_old, _parse_hashes_new])
def test_empty_or_unrecognised_html_yields_no_hashes(parser):
    assert parser("") == []
    assert parser("<html><body><p>nothing here</p></body></html>") == []


@pytest.mark.parametrize("parser", [_parse_hashes_old, _parse_hashes_new])
def test_malformed_html_does_not_raise(parser):
    """The site has served unclosed tags before; the parser must be tolerant."""
    parser("<div class='container-fluid'><ul class='paper-list'><li><a href=")


# ── abstract page parsing ────────────────────────────────────────────────

def test_parses_abstract():
    assert _parse_abstract_html(ABSTRACT_PAGE) == \
        "We present a method for learning representations."


def test_abstract_absent_yields_empty_string():
    assert _parse_abstract_html("<html><body><p>other</p></body></html>") == ""
    assert _parse_abstract_html("") == ""


def test_abstract_keeps_text_from_nested_tags():
    """Abstracts contain <i> and <sub>; their text belongs to the abstract."""
    html = ('<p class="paper-abstract">We study <i>deep</i> nets on '
            'CIFAR<sub>10</sub>.</p>')
    result = _parse_abstract_html(html)
    assert "deep" in result and "10" in result


def test_abstract_is_stripped():
    html = '<p class="paper-abstract">   padded text   </p>'
    assert _parse_abstract_html(html) == "padded text"


# ── title and authors ────────────────────────────────────────────────────

def test_title_and_authors_return_strings_on_unrecognised_input():
    """Both must degrade to a falsy value rather than raise."""
    for fn in (_parse_title_html, _parse_authors_html):
        result = fn("<html><body></body></html>")
        assert result in ("", [], None) or isinstance(result, (str, list))
