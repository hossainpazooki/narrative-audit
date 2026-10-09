"""arXiv retrieval for the fulltext stage: PDFs, and v1 abstracts for the
reverse-causation guard.

Both functions back off exponentially on rate limits and transient errors
and return None on final failure — the caller records the failure, it is
never an exception. arXiv asks for a few seconds between requests; the
stage runner sleeps between papers, this module only handles retries.
"""

from __future__ import annotations

import os
import re
import sys
import time

PDF_URL = "https://export.arxiv.org/pdf/{id}"
API_URL = "https://export.arxiv.org/api/query"
MAX_RETRIES = 5


def _session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "narrative-audit fulltext stage"
    return s


def _get(session, url: str, params: dict | None = None):
    delay = 2.0
    for _ in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code not in (429, 500, 502, 503, 504):
                print(f"  HTTP {r.status_code} for {url}", file=sys.stderr)
                return None
        except Exception as exc:
            print(f"  request error: {exc}", file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    return None


def fetch_pdf(arxiv_id: str, dest_dir: str, session=None) -> str | None:
    """Download the arXiv PDF; returns the local path or None.

    An existing non-empty file short-circuits, so the stage resumes without
    refetching. The id may carry a version suffix; it is kept as given.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, arxiv_id.replace("/", "_") + ".pdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    session = session or _session()
    r = _get(session, PDF_URL.format(id=arxiv_id))
    if r is None or not r.content.startswith(b"%PDF"):
        return None
    with open(dest, "wb") as fh:
        fh.write(r.content)
    return dest


def fetch_v1_abstract(arxiv_id: str, session=None) -> str | None:
    """The version-1 abstract of a paper, for the reverse-causation check.

    Queries the arXiv API for {id}v1 and returns the <summary> text, or
    None when the API has no v1 record.
    """
    session = session or _session()
    base = re.sub(r"v\d+$", "", arxiv_id)
    r = _get(session, API_URL, {"id_list": f"{base}v1", "max_results": 1})
    if r is None:
        return None
    m = re.search(r"<summary>(.*?)</summary>", r.text, re.S)
    if not m:
        return None
    text = re.sub(r"\s+", " ", m.group(1)).strip()
    return text or None
