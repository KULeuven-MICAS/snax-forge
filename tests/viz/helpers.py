"""Shared helpers for the visualiser tests.

Runs are flow folders made the way the flow makes them, SNAX-GVSoC stubbed
(tests/flow/helpers.py: ``stubbed_flow``, its synthetic events), so the
viewer reads exactly what the flow writes. ``SCEN`` is for the memory and
movement tests, skipped until FLOW3 round 2.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

from snax_forge.viz.server import make_server
from tests.flow.helpers import stubbed_flow

REPO = Path(__file__).resolve().parents[2]
SCEN = REPO / "scenarios"


def run_dir(out: Path, name: str = "vecadd", level: str = "task") -> Path:
    """The flow of recipe ``name`` at trace ``level`` into the folder ``out``; returns it."""
    return stubbed_flow(out.parent, name, level, out=out)


@contextmanager
def serving(dirs):
    """A viewer server on a free port, serving in a thread; yields it."""
    srv = make_server(list(dirs), port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)
