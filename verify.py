#!/usr/bin/env python3
"""Check every regenerated artefact against reference/.

reference/ holds the files that produced the figures in the paper. This
script compares what the pipeline just built against them and exits
non-zero on any mismatch, so a clean-room reproduction either passes or
fails loudly.

Three comparisons:

  per-paper metrics   byte-for-byte
  plot-ready CSVs     numeric, atol=0, NaN treated as equal
  figure PNG          byte-for-byte

    python verify.py
"""
from __future__ import annotations

import filecmp
import glob
import importlib.metadata as md
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("AA_DATA", "data")
# NA_REF lets the tests point verification at a fixture tree; a normal run
# always compares against the shipped reference/.
REF  = os.environ.get("NA_REF", os.path.join(HERE, "reference"))

ok = fail = missing = 0


def report(name: str, verdict: str, detail: str = "") -> None:
    global ok, fail, missing
    if verdict == "ok":
        ok += 1
    elif verdict == "missing":
        missing += 1
        print(f"  MISSING   {name}  {detail}")
    else:
        fail += 1
        print(f"  MISMATCH  {name}  {detail}")


def check_bytes(name: str, built: str, ref: str) -> None:
    if not os.path.exists(built):
        return report(name, "missing", "(not built)")
    report(name, "ok" if filecmp.cmp(built, ref, shallow=False) else "bad",
           "" if filecmp.cmp(built, ref, shallow=False) else "bytes differ")


def check_csv(name: str, built: str, ref: str) -> None:
    if not os.path.exists(built):
        return report(name, "missing", "(not built)")
    a, b = pd.read_csv(ref), pd.read_csv(built)
    # reference holds exactly the columns the paper's figures read; the run may
    # carry extra ones (an all-NaN pubmed column, for instance) without harm.
    absent = [c for c in a.columns if c not in b.columns]
    if absent:
        return report(name, "bad", f"missing columns {absent}")
    b = b[list(a.columns)]
    if len(a) != len(b):
        return report(name, "bad", f"rows {len(a)} vs {len(b)}")
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    same = np.isclose(a[num].astype(float), b[num].astype(float),
                      rtol=0, atol=0, equal_nan=True).all()
    if same:
        return report(name, "ok")
    mx = np.nanmax(np.abs(a[num].astype(float).values - b[num].astype(float).values))
    report(name, "bad", f"max|delta|={mx:.3e}")


# The versions reference/ was built with. A mismatch here is the usual cause
# of small numeric differences: textstat has changed its formula
# implementations between releases, and the spaCy model determines parse
# depth, NP density, passive rate and TTR.
EXPECTED = {
    "pandas": "3.0.2", "numpy": "2.4.4", "matplotlib": "3.10.8",
    "textstat": "0.7.13", "spacy": "3.8.13",
}


def environment_report() -> None:
    print("\nenvironment (reference/MANIFEST.md records what these were built with)")
    drift = False
    for pkg, want in EXPECTED.items():
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            got = "not installed"
        flag = "" if got == want else "   <- differs"
        drift = drift or bool(flag)
        print(f"  {pkg:<12} expected {want:<9} installed {got}{flag}")
    try:
        import spacy
        got = spacy.load("en_core_web_sm").meta["version"]
    except Exception:
        got = "not loadable"
    flag = "" if got == "3.8.0" else "   <- differs"
    drift = drift or bool(flag)
    print(f"  {'en_core_web_sm':<12} expected {'3.8.0':<9} installed {got}{flag}")
    if drift:
        print("\n  A version difference above is the likely cause. This is a report "
              "about your environment,\n  not evidence that the pipeline is wrong. "
              "Install the pinned versions in requirements.txt\n  to reproduce the "
              "published numbers exactly.")


def main() -> None:
    print("verifying against reference/\n")

    print("per-paper metrics")
    for ref_path in sorted(glob.glob(os.path.join(REF, "per_paper", "*", "neurips.csv"))):
        family = os.path.basename(os.path.dirname(ref_path))
        check_bytes(f"per_paper/{family}/neurips.csv",
                    os.path.join(DATA, "per_paper", family, "neurips.csv"),
                    ref_path)

    print("\nplot-ready data")
    for ref_path in sorted(glob.glob(os.path.join(REF, "pgfplots", "*.csv"))):
        base = os.path.basename(ref_path)
        check_csv(f"pgfplots/{base}", os.path.join(DATA, "pgfplots", base), ref_path)

    # The narrative-audit per-paper files: byte-for-byte, like the metric
    # CSVs. reference/ carries them once a reference run exists; globbing
    # means a fork without them verifies the rest unchanged.
    print("\noutcomes / rubric / sample (byte-for-byte)")
    for sub, pattern in (("outcomes", "*.parquet"), ("rubric", "*.parquet"),
                         ("sample", "*.csv")):
        for ref_path in sorted(glob.glob(os.path.join(REF, sub, pattern))):
            base = os.path.basename(ref_path)
            check_bytes(f"{sub}/{base}", os.path.join(DATA, sub, base), ref_path)

    # Regression and correlation tables: numeric at atol=0 (design spec §5).
    print("\nanalysis tables")
    for ref_path in sorted(glob.glob(os.path.join(REF, "analysis", "*.csv"))):
        base = os.path.basename(ref_path)
        check_csv(f"analysis/{base}", os.path.join(DATA, "analysis", base), ref_path)

    print("\nfigures")
    check_bytes("paper/figs/fig_judge_metric_heatmap.png",
                os.path.join(HERE, "paper", "figs", "fig_judge_metric_heatmap.png"),
                os.path.join(REF, "fig_judge_metric_heatmap.png"))

    total = ok + fail + missing
    print(f"\n{ok}/{total} artefacts match reference")
    environment_report()
    if fail or missing:
        sys.exit(f"FAILED: {fail} mismatched, {missing} missing")
    print("reproduction verified")


if __name__ == "__main__":
    main()
