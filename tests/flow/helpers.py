"""Shared helpers for the flow tests and the tests that run the flow (viz, report).

The flow runs vecadd from the checked-in recipe on ``platforms/small16.json``,
starting from the import fixture (``PLAIN``) rather than the kernel, so these
tests need no DaCe. ``PINS`` puts B and C where ``scenarios/vecadd`` has them
(77 cycles); ``B_PIN`` moves B alone, as LOOP1 does.

``excerpts_hold`` is the check the walk-throughs in ``examples/`` share: a
table quoted in a README under ``<!-- excerpt: FOLDER/report/FILE -->`` must
be, line for line, part of that report.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from snax_forge.flow import Flow, run_flow
from tests.design.helpers import FIXTURES, RECIPE, REPO, SMALL16

SCEN = REPO / "scenarios"
PLAIN = FIXTURES / "vecadd.snaxdfg"  # the import of the vecadd kernel (IMP1)
B_PIN = [("B.l1.base", 576)]
PINS = [*B_PIN, ("C.l1.base", 1152)]
EXCERPT = re.compile(r"<!-- excerpt: (\S+) -->")


def run_flows(base: Path, flows: dict[str, dict[str, Any]], level: str) -> dict[str, Flow]:
    """vecadd from ``PLAIN`` once per entry of ``flows`` (``run_flow`` keyword arguments),
    each into ``base / name``, traced at ``level``."""
    return {
        name: run_flow(RECIPE, SMALL16, graph_path=PLAIN, out=base / name, trace_level=level, **kw)
        for name, kw in flows.items()
    }


def excerpts_hold(readme: str, root: Path) -> int:
    """Every excerpt of ``readme`` (the lines after its marker, up to the first empty one) is
    made of lines of the report it names, a path under ``root``; returns how many there are."""
    lines = readme.splitlines()
    seen = 0
    for i, line in enumerate(lines):
        m = EXCERPT.fullmatch(line.strip())
        if not m:
            continue
        report = set((root / m.group(1)).read_text().splitlines())
        block = []
        for x in lines[i + 1 :]:
            if not x.strip():
                break
            block.append(x)
        assert block, f"excerpt at line {i + 1} is empty"
        missing = [x for x in block if x not in report]
        assert not missing, f"README line {i + 1}: not in {m.group(1)}: {missing}"
        seen += 1
    return seen
