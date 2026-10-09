"""verify.py must fail loudly on a one-byte difference. The byte-for-byte
and numeric comparison paths are both exercised against a fixture
reference tree via the NA_REF/AA_DATA overrides."""

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_verify(data_dir, ref_dir):
    env = dict(os.environ, AA_DATA=str(data_dir), NA_REF=str(ref_dir))
    return subprocess.run([sys.executable, os.path.join(ROOT, "verify.py")],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def _fixture(tmp_path):
    data, ref = tmp_path / "data", tmp_path / "ref"
    csv = "paper_id,year,score\n1990_a,1990,1.5\n1990_b,1990,2.0\n"
    for base in (data, ref):
        (base / "per_paper" / "readability").mkdir(parents=True)
        (base / "per_paper" / "readability" / "neurips.csv").write_text(csv)
        (base / "sample").mkdir()
        (base / "sample" / "sample.csv").write_text(
            "paper_id,year\n1990_a,1990\n")
        (base / "analysis").mkdir()
        (base / "analysis" / "mixed_effects_abstract.csv").write_text(
            "term,coef\nconst,0.123456\n")
    # the figure check compares a repo path against the reference; give the
    # fixture reference the same bytes the repo carries
    shutil.copy(os.path.join(ROOT, "paper", "figs", "fig_judge_metric_heatmap.png"),
                ref / "fig_judge_metric_heatmap.png")
    # verify's per_paper walk expects the built file under AA_DATA; ref
    # drives the glob, so only files present in ref/ are compared
    return data, ref


def test_identical_trees_pass(tmp_path):
    data, ref = _fixture(tmp_path)
    r = _run_verify(data, ref)
    assert r.returncode == 0, r.stdout + r.stderr


def test_one_byte_diff_in_bytes_file_fails(tmp_path):
    data, ref = _fixture(tmp_path)
    path = data / "sample" / "sample.csv"
    raw = bytearray(path.read_bytes())
    raw[-2] ^= 0x01                       # flip one byte
    path.write_bytes(bytes(raw))
    r = _run_verify(data, ref)
    assert r.returncode != 0
    assert "MISMATCH" in r.stdout


def test_one_byte_diff_in_metric_csv_fails(tmp_path):
    data, ref = _fixture(tmp_path)
    path = data / "per_paper" / "readability" / "neurips.csv"
    path.write_text(path.read_text().replace("1.5", "1.6", 1))
    r = _run_verify(data, ref)
    assert r.returncode != 0


def test_numeric_tolerance_is_zero(tmp_path):
    # the analysis tables are compared numerically at atol=0: the last
    # decimal moving by one must fail
    data, ref = _fixture(tmp_path)
    path = data / "analysis" / "mixed_effects_abstract.csv"
    path.write_text("term,coef\nconst,0.123457\n")
    r = _run_verify(data, ref)
    assert r.returncode != 0


def test_missing_built_file_fails(tmp_path):
    data, ref = _fixture(tmp_path)
    (data / "sample" / "sample.csv").unlink()
    r = _run_verify(data, ref)
    assert r.returncode != 0
    assert "MISSING" in r.stdout
