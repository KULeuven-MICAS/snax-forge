"""Shared helpers for the flow tests and the tests that run the flow (viz, report).

The flow runs vecadd from the checked-in recipe on ``platforms/small16.json``,
starting from the import fixture (``PLAIN``) rather than the kernel, so these
tests need no DaCe. ``PINS`` puts B and C where ``scenarios/vecadd`` has them
(77 cycles); ``B_PIN`` moves B alone, as LOOP1 does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from snax_forge.flow import Flow, run_flow
from tests.design.helpers import FIXTURES, RECIPE, REPO, SMALL16

SCEN = REPO / "scenarios"
PLAIN = FIXTURES / "vecadd.snaxdfg"  # the import of the vecadd kernel (IMP1)
B_PIN = [("B.l1.base", 576)]
PINS = [*B_PIN, ("C.l1.base", 1152)]


def run_flows(base: Path, flows: dict[str, dict[str, Any]], level: str) -> dict[str, Flow]:
    """vecadd from ``PLAIN`` once per entry of ``flows`` (``run_flow`` keyword arguments),
    each into ``base / name``, traced at ``level``."""
    return {
        name: run_flow(RECIPE, SMALL16, graph_path=PLAIN, out=base / name, trace_level=level, **kw)
        for name, kw in flows.items()
    }
