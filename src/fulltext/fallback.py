"""pymupdf heading-heuristic section splitter — the fallback when GROBID is
unavailable or returns nothing usable.

The heuristic: a line is a heading when it is short, not a caption, and
either (a) its dominant font size clearly exceeds the body size, or (b) it
matches a numbered- or canonical-heading pattern ("3 Experiments",
"IV. METHODS", "Conclusion"). Text between headings becomes that section's
body. This is deliberately simple; GROBID is the primary parser and this
path exists so a GROBID-less run degrades to flagged-but-present sections
rather than nothing.
"""

from __future__ import annotations

import re
import statistics
import sys

CANONICAL = ("abstract|introduction|background|related work|method(?:s|ology)?|"
             "approach|model|experiments?|experimental (?:setup|results)|"
             "evaluation|results?(?: and discussion)?|discussion|analysis|"
             "limitations?|conclusions?|future work|broader impacts?|"
             "acknowledg(?:e)?ments?|references|appendix|ethics statement")
_NUMBERED = re.compile(rf"^(?:\d+|[IVX]+)[.\s]+\S")
_CANON = re.compile(rf"^(?:(?:\d+|[IVX]+)[.\s]+)?(?:{CANONICAL})\s*$", re.IGNORECASE)
_CAPTION = re.compile(r"^(?:fig(?:ure)?|table)\.?\s*\d+", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s)\]}>\"']+")


def _line_spans(page) -> list[tuple[str, float]]:
    """(text, max_font_size) per line of a page, reading order."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line.get("spans", [])).strip()
            if text:
                size = max(s["size"] for s in line["spans"])
                out.append((text, size))
    return out


def _is_heading(text: str, size: float, body_size: float) -> bool:
    if len(text) > 80 or _CAPTION.match(text):
        return False
    if _CANON.match(text):
        return True
    return size > body_size * 1.15 and (_NUMBERED.match(text) is not None
                                        or text[:1].isupper())


def fallback_sections(pdf_path: str) -> dict | None:
    """PDF -> same shape as grobid.parse_tei, or None when unreadable."""
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        print(f"  pymupdf error: {exc}", file=sys.stderr)
        return None

    lines: list[tuple[str, float]] = []
    try:
        for page in doc:
            lines.extend(_line_spans(page))
    finally:
        doc.close()
    if not lines:
        return None

    body_size = statistics.median(size for _, size in lines)
    sections: list[dict] = []
    captions: list[str] = []
    current = {"heading": "", "parts": []}

    def flush():
        text = re.sub(r"\s+", " ", " ".join(current["parts"])).strip()
        if current["heading"] or text:
            sections.append({"heading": current["heading"], "text": text})

    for text, size in lines:
        if _CAPTION.match(text):
            captions.append(text)
            continue
        if _is_heading(text, size, body_size):
            flush()
            current = {"heading": text, "parts": []}
        else:
            current["parts"].append(text)
    flush()

    if not any(s["text"] for s in sections):
        return None
    urls = sorted({m.group(0).rstrip(".,;:") for t, _ in lines
                   for m in _URL.finditer(t)})
    return {"sections": sections, "figure_captions": captions, "urls": urls}
