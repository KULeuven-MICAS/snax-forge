"""paper/: the sets load, and the numbers paper/README.md states come out of the estimate."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "paper_run", Path(__file__).resolve().parents[2] / "paper" / "run.py"
)
paper = importlib.util.module_from_spec(_SPEC)
sys.modules["paper_run"] = paper  # its dataclasses look their module up
_SPEC.loader.exec_module(paper)

pytestmark = pytest.mark.slow  # every point imports its kernel through DaCe


def point(name: str, pid: str):
    """The point and the set's platform, as ``row_of`` takes them."""
    platform, points = paper.load_set(name)
    return next(p for p in points if p.id == pid), platform


def test_the_sets_load_with_unique_ids_and_full_range_data():
    for name, n in (("forms", 21), ("limits", 36)):
        _, points = paper.load_set(name)
        assert len(points) == n
        assert all("design.data=full" in p.sets for p in points)


def test_the_estimate_gives_the_readmes_numbers():
    row = paper.row_of(*point("forms", "mvt_column"), run=False)
    assert row["status"] == "ok" and row["est_cycles"] == 1028
    assert row["static_nj"] == pytest.approx(26.44, abs=0.01)
    assert row["area_um2"] == pytest.approx(561514, rel=1e-3)
    copy = paper.row_of(*point("forms", "mvt_dma_copy"), run=False)
    assert (copy["est_cycles"], copy["est_copy_cycles"]) == (11951, 10923)
    assert copy["static_nj"] == pytest.approx(373.35, abs=0.01)


def test_the_estimate_does_not_see_the_layout_and_flags_the_bank_bound():
    a = paper.row_of(*point("limits", "mvt_contiguous_w8"), run=False)
    b = paper.row_of(*point("limits", "mvt_padded_w8"), run=False)
    assert a["est_cycles"] == b["est_cycles"] == 1028
    assert a["static_nj"] == pytest.approx(b["static_nj"]) == pytest.approx(243.60, abs=0.01)
    w32 = paper.row_of(*point("limits", "mvt_contiguous_w32"), run=False)
    assert (w32["est_cycles"], w32["est_bank_bound"]) == (260, 520)


def test_a_refused_point_keeps_its_row_with_the_reason():
    row = paper.row_of(*point("limits", "bicg_pipeline_padded_w32"), run=False)
    assert row["status"].startswith("refused: [connect.xbar_inputs]") and "132" in row["status"]
    assert "est_cycles" not in row


def test_only_replaces_those_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER", tmp_path)
    (tmp_path / "forms.json").write_text((paper.ROOT / "paper" / "forms.json").read_text())
    data = tmp_path / "data"
    data.mkdir()
    keep = {c: "" for c in paper.COLUMNS} | {"set": "forms", "id": "mvt_column", "status": "x"}
    other = keep | {"id": "atax_column"}
    paper.write_csv(data / "forms.csv", [keep, other])
    assert paper.main(["forms", "--only", "atax_column"]) == 0
    rows = list(csv.DictReader((data / "forms.csv").open()))
    assert [r["id"] for r in rows] == ["mvt_column", "atax_column"]
    assert rows[0]["status"] == "x" and rows[1]["status"] == "ok"
