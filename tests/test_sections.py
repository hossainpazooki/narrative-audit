"""The section splitter on three NeurIPS-era page layouts, for both
parsers.

Real NeurIPS PDFs are not redistributable, so the fixtures are minimal
reconstructions of the three layouts the corpus spans, built with pymupdf
at test time: the single-column all-caps-heading style of the early
proceedings (1987–1999), the numbered-heading style of the 2000s, and the
modern style with an explicit Abstract block and figure captions. GROBID
itself is a server and is not run here; its TEI output format is pinned
with recorded-fixture XML instead, which is the part of the GROBID path
this repository owns.
"""

import pymupdf
import pytest

from fulltext.fallback import fallback_sections
from fulltext.grobid import parse_tei
from fulltext.sections import build_doc, normalize_heading

BODY = ("This paragraph stands in for body text long enough to be body. " * 3).strip()


def _make_pdf(path, blocks, width=612, height=792):
    """blocks: list of (text, fontsize) laid out top to bottom."""
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    y = 60.0
    for text, size in blocks:
        for chunk in [text[i:i + 80] for i in range(0, len(text), 80)]:
            page.insert_text((50, y), chunk, fontsize=size)
            y += size * 1.4
            if y > height - 60:
                page = doc.new_page(width=width, height=height)
                y = 60.0
        y += size * 0.8
    doc.save(path)
    doc.close()


def test_fallback_1990s_single_column_allcaps(tmp_path):
    pdf = str(tmp_path / "1992.pdf")
    _make_pdf(pdf, [
        ("A Network Model of Something", 16),
        ("INTRODUCTION", 12), (BODY, 10),
        ("METHODS", 12), (BODY, 10),
        ("RESULTS", 12), (BODY, 10),
        ("CONCLUSIONS", 12), (BODY, 10),
    ])
    parsed = fallback_sections(pdf)
    assert parsed is not None
    headings = [s["heading"].lower() for s in parsed["sections"] if s["heading"]]
    assert "introduction" in headings
    assert "methods" in headings
    assert any(s["text"] for s in parsed["sections"])


def test_fallback_2000s_numbered_headings(tmp_path):
    pdf = str(tmp_path / "2005.pdf")
    _make_pdf(pdf, [
        ("Kernel Methods for Structured Data", 16),
        ("1 Introduction", 13), (BODY, 10),
        ("2 Related Work", 13), (BODY, 10),
        ("3 Experiments", 13), (BODY, 10),
        ("4 Conclusion", 13), (BODY, 10),
    ])
    parsed = fallback_sections(pdf)
    assert parsed is not None
    normalized = {normalize_heading(s["heading"]) for s in parsed["sections"]}
    assert {"introduction", "related_work", "experiments", "conclusion"} <= normalized


def test_fallback_2020s_with_abstract_caption_and_url(tmp_path):
    pdf = str(tmp_path / "2021.pdf")
    _make_pdf(pdf, [
        ("Scaling Laws for Everything", 17),
        ("Abstract", 12),
        ("We study scaling. Code at https://github.com/example/scaling.", 10),
        ("1 Introduction", 13), (BODY, 10),
        ("Figure 1: Overview diagram of the proposed architecture.", 9),
        ("2 Method", 13), (BODY, 10),
        ("3 Limitations", 13), (BODY, 10),
    ])
    parsed = fallback_sections(pdf)
    assert parsed is not None
    normalized = {normalize_heading(s["heading"]) for s in parsed["sections"]}
    assert {"abstract", "introduction", "methods", "limitations"} <= normalized
    assert any(c.startswith("Figure 1") for c in parsed["figure_captions"])
    assert "https://github.com/example/scaling" in parsed["urls"]


def test_fallback_unreadable_pdf(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 not actually a pdf")
    assert fallback_sections(str(bad)) is None


# ── GROBID TEI parsing ───────────────────────────────────────────────────────

TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <profileDesc>
      <abstract><div><p>We study refusal directions.</p></div></abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head>1 Introduction</head><p>Chat models refuse things.</p></div>
      <div><head>4 Experiments</head><p>We evaluate on 13 models over five
        seeds.</p><p>Error bars everywhere.</p></div>
      <div><head>Limitations</head><p>Only open models.</p></div>
      <figure><head>Figure 1</head><figDesc>Overview of the refusal
        direction.</figDesc></figure>
      <div><p>An untitled trailing div.</p>
        <ptr target="https://github.com/example/refusal"/></div>
    </body>
  </text>
</TEI>"""


def test_parse_tei_sections_captions_urls():
    parsed = parse_tei(TEI.encode())
    assert parsed is not None
    heads = [s["heading"] for s in parsed["sections"]]
    assert heads[0] == "Abstract"
    assert "1 Introduction" in heads
    assert "" in heads                       # the untitled div is kept
    exp = next(s for s in parsed["sections"] if s["heading"] == "4 Experiments")
    assert "five seeds" in exp["text"] and "Error bars" in exp["text"]
    assert any("refusal direction" in c.lower() for c in parsed["figure_captions"])
    assert "https://github.com/example/refusal" in parsed["urls"]


def test_parse_tei_malformed_xml():
    assert parse_tei(b"<TEI><unclosed") is None


def test_parse_tei_no_body():
    tei = ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><text></text></TEI>')
    assert parse_tei(tei.encode()) is None


# ── heading normalization and doc assembly ───────────────────────────────────

@pytest.mark.parametrize("heading,expected", [
    ("1 Introduction", "introduction"),
    ("IV. EXPERIMENTAL RESULTS", "experiments"),
    ("Related Work", "related_work"),
    ("Broader Impact", "discussion"),
    ("Limitations and Future Work", "limitations"),
    ("A Appendix", "appendix"),
    ("Acknowledgements", "other"),
    ("", "other"),
])
def test_normalize_heading(heading, expected):
    assert normalize_heading(heading) == expected


def test_build_doc_statuses():
    ok = build_doc("2020_a", "2001.1", {"sections": [{"heading": "1 Methods",
                                                      "text": "t"}],
                                        "figure_captions": [], "urls": []},
                   "grobid", "ok")
    assert ok["fulltext_status"] == "ok"
    assert ok["sections"][0]["normalized"] == "methods"
    failed = build_doc("2020_b", "2001.2", None, None, "fetch_failed")
    assert failed["fulltext_status"] == "fetch_failed"
    assert failed["sections"] == []
