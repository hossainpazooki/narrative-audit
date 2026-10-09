#!/usr/bin/env python3
"""Render every figure source to paper/figures.pdf.

Proves the figure sources in this repository compile against the data the
pipeline just produced. Skips with a clear message if pdflatex is absent,
since the rest of the pipeline does not need LaTeX.

    python paper/build_figures.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    if shutil.which("pdflatex") is None:
        print("pdflatex not found, skipping figure rendering "
              "(the pipeline and verify.py do not require it)")
        return

    # Fix the timestamp pdflatex embeds, so the same sources and the same TeX
    # Live produce a byte-identical PDF instead of one that differs on every
    # build. TeX Live's own version still affects the bytes and cannot be
    # pinned from requirements.txt, so verify.py does not compare this file.
    env = dict(os.environ, SOURCE_DATE_EPOCH="0", FORCE_SOURCE_DATE="1")

    for i in range(2):  # second pass resolves references
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figures.tex"],
            cwd=HERE, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            tail = "\n".join(r.stdout.strip().split("\n")[-25:])
            sys.exit(f"pdflatex failed on pass {i + 1}:\n{tail}")

    pdf = os.path.join(HERE, "figures.pdf")
    if not os.path.exists(pdf):
        sys.exit("pdflatex reported success but figures.pdf is missing")
    print(f"wrote {pdf} ({os.path.getsize(pdf):,} bytes)")


if __name__ == "__main__":
    main()
