"""GROBID client and TEI parsing — the primary section splitter.

parse_tei() is pure (bytes in, sections out) so it is testable against
recorded TEI fixtures without a server; grobid_sections() owns the HTTP
call against a running GROBID instance (GROBID_URL, default
http://localhost:8070).
"""

from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
DEFAULT_URL = os.environ.get("GROBID_URL", "http://localhost:8070")


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def parse_tei(tei_bytes: bytes) -> dict | None:
    """TEI XML -> {"sections": [...], "figure_captions": [...], "urls": [...]}.

    Returns None when the XML does not parse or holds no body — the caller
    treats that as parse_failed and falls back. Section headings come from
    <head>; a <div> without one becomes an untitled section so no text is
    dropped.
    """
    try:
        root = ET.fromstring(tei_bytes)
    except ET.ParseError:
        return None

    sections: list[dict] = []
    abstract = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    if abstract is not None and _text(abstract):
        sections.append({"heading": "Abstract", "text": _text(abstract)})

    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is None:
        return None
    for div in body.findall("tei:div", TEI_NS):
        head = div.find("tei:head", TEI_NS)
        heading = _text(head) if head is not None else ""
        paras = [_text(p) for p in div.findall("tei:p", TEI_NS)]
        text = " ".join(p for p in paras if p)
        if heading or text:
            sections.append({"heading": heading, "text": text})
    if not sections:
        return None

    captions = []
    for fig in root.findall(".//tei:figure", TEI_NS):
        desc = fig.find("tei:figDesc", TEI_NS)
        if desc is not None and _text(desc):
            head = fig.find("tei:head", TEI_NS)
            label = _text(head) if head is not None else ""
            captions.append((label + ": " if label and not _text(desc).lower()
                             .startswith(label.lower()) else "") + _text(desc))

    urls = sorted({el.get("target") for el in root.findall(".//tei:ptr", TEI_NS)
                   if el.get("target")})
    raw = " ".join(s["text"] for s in sections)
    in_text = {u.rstrip(".,;:")
               for u in re.findall(r"https?://[^\s)\]}>\"']+", raw)}
    urls = sorted(set(urls) | in_text)

    return {"sections": sections, "figure_captions": captions, "urls": urls}


def grobid_sections(pdf_path: str, grobid_url: str = DEFAULT_URL) -> dict | None:
    """POST a PDF to GROBID and parse the TEI. None on any failure."""
    import requests
    try:
        with open(pdf_path, "rb") as fh:
            r = requests.post(
                f"{grobid_url.rstrip('/')}/api/processFulltextDocument",
                files={"input": fh},
                data={"segmentSentences": "0"},
                timeout=120,
            )
    except Exception as exc:
        print(f"  grobid error: {exc}", file=sys.stderr)
        return None
    if r.status_code != 200:
        return None
    return parse_tei(r.content)
