"""Shared helpers for the visualiser tests (VIS1).

Run directories are made the way a user makes them: a checked-in scenario
run through ``snax_model.scenario.run`` and written with ``write_outputs``,
so the viewer reads exactly what the CLI writes (D44, D54).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

from snax_forge.snax_model.scenario import Scenario, run, write_outputs
from snax_forge.viz.server import make_server

REPO = Path(__file__).resolve().parents[2]
SCEN = REPO / "scenarios"


def run_dir(out: Path, name: str = "vecadd", level: str = "beat", **kw) -> Path:
    """Run ``scenarios/<name>`` at trace ``level`` into ``out``; returns ``out``."""
    sc = Scenario.load(SCEN / name / "scenario.json")
    return write_outputs(run(sc, trace_level=level, **kw), out)


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
