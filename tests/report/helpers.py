"""Shared helpers for the report tests (REP1)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCEN = REPO / "scenarios"
SCENARIOS = ("dma", "fmul", "reduce", "vecadd", "vecadd_conflict", "vecadd_tiled")
