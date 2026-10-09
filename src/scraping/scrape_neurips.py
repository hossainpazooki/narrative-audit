#!/usr/bin/env python3
"""
Scrape NeurIPS papers from papers.nips.cc into data/raw/neurips/.

Two-step process run automatically for all years:

  Step 1 — Download paper metadata (title, authors, full_text/abstract):
    Pre-2020:  papers.nips.cc/paper/{year}/file/{hash}-Metadata.json
               → title, authors in one JSON request; abstract is null in
                 old Metadata.json files, so Step 2 fills it in.
    2020+:     papers.nips.cc/paper_files/paper/{year}/hash/{hash}-Abstract.html
               → title, abstract, authors from HTML / citation meta-tags.

  Step 2 — Patch missing/short abstracts from proceedings.neurips.cc:
    For any paper with fewer than 30 abstract words, fetches the clean
    abstract from proceedings.neurips.cc. Results are cached locally so
    re-runs are instant. This is a no-op for 2020+ papers (already have
    abstracts) and does real work for pre-2008 papers.

Output schema per paper:
  {year, id, title, abstract, full_text, authors (list[str]), source_id}

Usage:
  python src/scraping/scrape_neurips.py --start 1987 --end 2025
  python src/scraping/scrape_neurips.py --start 2020 --end 2025 --workers 20
"""

import argparse
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from urllib.request import Request, urlopen

NIPS_BASE  = "https://papers.nips.cc"
PROC_BASE  = "https://proceedings.neurips.cc"
HEADERS    = {"User-Agent": "Mozilla/5.0 (research; contact: a.rangarajan@proton.me)"}
RETRIES    = 4
RETRY_DELAY = 1.5


# ── HTTP ──────────────────────────────────────────────────────────────────────

def fetch(url: str) -> str | None:
    for attempt in range(RETRIES):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=20) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            if attempt < RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"    FAILED {url}: {e}")
    return None


def fetch_json(url: str) -> dict | None:
    text = fetch(url)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ── Step 1a: get paper hashes for a year ─────────────────────────────────────

def _parse_hashes_old(html: str) -> list[str]:
    """Pre-2020 HTML: <div class="container-fluid"> → <li> → <a href="/paper/...">"""
    from html.parser import HTMLParser as _HP
    class P(_HP):
        hashes: list[str] = []
        _in_container = _in_li = False
        _container_count = 0
        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            if tag == "div" and "container-fluid" in (d.get("class") or ""):
                self._container_count += 1
                if self._container_count == 2:
                    self._in_container = True
            if self._in_container and tag == "li":
                self._in_li = True
            if self._in_li and tag == "a":
                href = d.get("href", "")
                if "/paper/" in href:
                    h = href.split("/")[-1].split("-")[0]
                    if h:
                        self.hashes.append(h)
        def handle_endtag(self, tag):
            if tag == "li":
                self._in_li = False
    p = P(); p.feed(html)
    return p.hashes


def _parse_hashes_new(html: str) -> list[str]:
    """2020+ HTML: <ul class="paper-list"> → <a title="paper title">"""
    from html.parser import HTMLParser as _HP
    class P(_HP):
        hashes: list[str] = []
        _in_list = False
        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            if tag == "ul" and "paper-list" in (d.get("class") or ""):
                self._in_list = True
            if self._in_list and tag == "a" and d.get("title") == "paper title":
                href = d.get("href", "")
                h = href.split("/")[-1].split("-")[0]
                if h:
                    self.hashes.append(h)
        def handle_endtag(self, tag):
            if tag == "ul":
                self._in_list = False
    p = P(); p.feed(html)
    return p.hashes


def get_hashes(year: int) -> list[str]:
    for url in [f"{NIPS_BASE}/paper_files/paper/{year}",
                f"{NIPS_BASE}/paper/{year}"]:
        html = fetch(url)
        if not html:
            continue
        hashes = _parse_hashes_new(html) if year >= 2020 else _parse_hashes_old(html)
        if hashes:
            return hashes
    return []


# ── Step 1b: download metadata for a single paper ────────────────────────────

def _parse_authors_metadata(author_list: list[dict]) -> list[str]:
    """Parse authors from Metadata.json format: [{given_name, family_name}]."""
    names = []
    for a in author_list:
        first = a.get("given_name", "").strip()
        last  = a.get("family_name", "").strip()
        name  = f"{first} {last}".strip() if first else last
        if name:
            names.append(name)
    return names


def _parse_authors_html(html: str) -> list[str]:
    """Parse authors from citation_author meta tags in Abstract HTML."""
    class P(HTMLParser):
        authors: list[str] = []
        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            if tag == "meta" and d.get("name") == "citation_author":
                raw = d.get("content", "").strip()
                if not raw:
                    return
                if "," in raw:
                    parts = raw.split(",", 1)
                    name = f"{parts[1].strip()} {parts[0].strip()}"
                else:
                    name = raw
                if name:
                    self.authors.append(name)
    p = P(); p.feed(html)
    return p.authors


def _parse_abstract_html(html: str) -> str:
    """Parse abstract from Abstract HTML page (2020+ format).
    Tag: <p class="paper-abstract">...</p>
    """
    class P(HTMLParser):
        abstract = ""
        _depth = 0
        _buf = ""
        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            cls = d.get("class") or ""
            if tag == "p" and "paper-abstract" in cls:
                self._depth = 1; self._buf = ""
            elif self._depth > 0:
                self._depth += 1
        def handle_endtag(self, tag):
            if self._depth > 0:
                self._depth -= 1
                if self._depth == 0:
                    self.abstract = self._buf.strip()
        def handle_data(self, data):
            if self._depth > 0:
                self._buf += data
    p = P(); p.feed(html)
    return p.abstract


def _parse_title_html(html: str) -> str:
    class P(HTMLParser):
        title = ""
        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            if tag == "meta" and d.get("name") == "citation_title":
                self.title = d.get("content", "").strip()
    p = P(); p.feed(html)
    return p.title


def scrape_paper_pre2020(paper_hash: str, year: int) -> dict | None:
    """Fetch Metadata.json → title, abstract, full_text, authors."""
    url  = f"{NIPS_BASE}/paper/{year}/file/{paper_hash}-Metadata.json"
    doc  = fetch_json(url)
    if not doc:
        return None
    authors = _parse_authors_metadata(doc.get("authors", []))
    return {
        "year":      year,
        "id":        paper_hash,
        "title":     doc.get("title", ""),
        "abstract":  doc.get("abstract", ""),
        "full_text": doc.get("full_text", ""),
        "authors":   authors,
        "source_id": doc.get("sourceid", paper_hash),
    }


def scrape_paper_post2020(paper_hash: str, year: int) -> dict | None:
    """Parse Abstract HTML → title, abstract, authors (no full_text available)."""
    url  = f"{NIPS_BASE}/paper_files/paper/{year}/hash/{paper_hash}-Abstract.html"
    html = fetch(url)
    if not html:
        return None
    return {
        "year":      year,
        "id":        paper_hash,
        "title":     _parse_title_html(html),
        "abstract":  _parse_abstract_html(html),
        "full_text": "",
        "authors":   _parse_authors_html(html),
        "source_id": paper_hash,
    }


def scrape_paper(paper_hash: str, year: int) -> dict | None:
    if year < 2020:
        return scrape_paper_pre2020(paper_hash, year)
    else:
        return scrape_paper_post2020(paper_hash, year)


# ── Step 2 (optional): improve abstracts from proceedings.neurips.cc ─────────

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


class _AbstractParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.abstract = ""
        self._depth = 0
        self._buf = ""

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        cls = d.get("class") or ""
        if tag == "p" and "paper-abstract" in cls:
            self._depth = 1; self._buf = ""
        elif self._depth > 0:
            self._depth += 1

    def handle_endtag(self, tag):
        if self._depth > 0:
            self._depth -= 1
            if self._depth == 0:
                text = re.sub(r'\(cid:\d+\)', '', self._buf)
                text = re.sub(r'[ \t]+', ' ', text)
                self.abstract = text.strip()
                self._buf = ""

    def handle_data(self, data):
        if self._depth > 0:
            self._buf += data


def _get_proc_abstract_links(year: int) -> list[str]:
    for url in [f"{PROC_BASE}/paper_files/paper/{year}",
                f"{PROC_BASE}/paper/{year}"]:
        html = fetch(url)
        if not html:
            continue
        class P(HTMLParser):
            links: list[str] = []
            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    for k, v in attrs:
                        if k == "href" and v and "Abstract" in v and str(year) in v:
                            self.links.append(v)
        p = P(); p.feed(html)
        if p.links:
            return p.links
    return []


def improve_abstracts(year: int, raw_dir: str, min_words: int = 30) -> None:
    """
    For papers whose abstract is missing or short (< min_words words),
    re-fetch from proceedings.neurips.cc and patch in place.
    Uses a local cache so re-runs are fast.
    """
    raw_path = os.path.join(raw_dir, f"neurips_{year}_data.json")
    if not os.path.exists(raw_path):
        print(f"  [{year}] raw file not found, skipping abstract improvement")
        return

    with open(raw_path, encoding="utf-8") as fh:
        papers = json.load(fh)

    needs_patch = [p for p in papers
                   if len((p.get("abstract") or "").split()) < min_words]
    if not needs_patch:
        print(f"  [{year}] all abstracts already adequate — skip")
        return

    print(f"  [{year}] {len(needs_patch)} papers need abstract improvement …")

    cache_dir  = os.path.join(raw_dir, "abstract_cache")
    cache_path = os.path.join(cache_dir, f"neurips_{year}_abstracts.json")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        print(f"  [{year}] loading proceedings cache …")
        with open(cache_path, encoding="utf-8") as fh:
            scraped = json.load(fh)
    else:
        links = _get_proc_abstract_links(year)
        print(f"  [{year}] fetching {len(links)} abstracts from proceedings.neurips.cc …")
        scraped: dict[str, dict] = {}
        for i, path in enumerate(links):
            m   = re.search(r'/hash/([a-f0-9]+)-Abstract', path)
            key = m.group(1) if m else None
            url = PROC_BASE + path if path.startswith("/") else path
            html = fetch(url)
            if html:
                p = _AbstractParser(); p.feed(html)
                if p.abstract and key:
                    scraped[key] = p.abstract
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(links)} …")
            time.sleep(0.15)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(scraped, fh, ensure_ascii=False)

    patched = 0
    for p in papers:
        paper_hash = (p.get("id") or "").strip()
        if paper_hash in scraped:
            new_abs = scraped[paper_hash]
            if len(new_abs.split()) >= 5:
                p["abstract"] = new_abs
                patched += 1

    # Fallback: for papers still empty, try alternative URL suffixes directly.
    # Some NeurIPS papers use -Abstract.html or -Abstract-Datasets_and_Benchmarks.html
    # instead of -Abstract-Conference.html, and are not listed on the proceedings index.
    _FALLBACK_SUFFIXES = [
        "-Abstract-Datasets_and_Benchmarks.html",
        "-Abstract-Competition.html",
        "-Abstract.html",
    ]
    still_empty = [p for p in papers if len((p.get("abstract") or "").split()) < min_words]
    if still_empty:
        print(f"  [{year}] {len(still_empty)} still empty — trying fallback URL suffixes …")
        for p in still_empty:
            paper_hash = (p.get("id") or "").strip()
            if not paper_hash:
                continue
            for suffix in _FALLBACK_SUFFIXES:
                url = f"{PROC_BASE}/paper_files/paper/{year}/hash/{paper_hash}{suffix}"
                html = fetch(url)
                if html:
                    parser = _AbstractParser(); parser.feed(html)
                    if parser.abstract and len(parser.abstract.split()) >= min_words:
                        p["abstract"] = parser.abstract
                        patched += 1
                        break
                time.sleep(0.15)

    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(papers, fh, ensure_ascii=False)
    print(f"  [{year}] patched {patched}/{len(needs_patch)} abstracts")


# ── Main ──────────────────────────────────────────────────────────────────────

def scrape_year(year: int, out_dir: str, workers: int, delay: float) -> int:
    hashes = get_hashes(year)
    if not hashes:
        print(f"  [NeurIPS {year}] no papers found")
        return 0

    print(f"  [NeurIPS {year}] {len(hashes)} papers found, downloading …")
    papers = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(scrape_paper, h, year): h for h in hashes}
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result:
                papers.append(result)
            if delay > 0:
                time.sleep(delay)
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(hashes)} done …")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"neurips_{year}_data.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(papers, fh, ensure_ascii=False)
    print(f"  [NeurIPS {year}] saved {len(papers)} papers → {path}")
    return len(papers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   type=int, default=1987)
    parser.add_argument("--end",     type=int, default=2025)
    parser.add_argument("--raw-dir", default="data/raw/neurips")
    parser.add_argument("--workers", type=int, default=20,
                        help="Concurrent download workers per year")
    parser.add_argument("--delay",   type=float, default=0.1,
                        help="Delay between requests (seconds)")
    args = parser.parse_args()

    total = 0
    for year in range(args.start, args.end + 1):
        total += scrape_year(year, args.raw_dir, args.workers, args.delay)

    # Always patch abstracts from proceedings.neurips.cc.
    # For years where abstracts were already scraped correctly (2020+) this
    # is a no-op since improve_abstracts skips papers already above min_words.
    # For older years (especially pre-2008) the Metadata.json returns null
    # abstracts so this step fetches them from proceedings.neurips.cc.
    print("\nPatching short/missing abstracts from proceedings.neurips.cc …")
    for year in range(args.start, args.end + 1):
        improve_abstracts(year, args.raw_dir)

    print(f"\nDone. {total} papers scraped across {args.end - args.start + 1} years.")


if __name__ == "__main__":
    main()
