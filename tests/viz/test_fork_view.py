"""The viewer draws a forked read (fork_read, D189) as its own edge style into each unit."""

from __future__ import annotations

from pathlib import Path

from snax_forge.viz.dfg.api import Entry, graph_view
from tests.helpers import steps


def test_the_forked_reads_of_a_are_marked():
    g = steps("bicg_v3", M=8, N=8, W=8)[-1]
    v = graph_view(Entry(g.name, Path(g.name), g, None), expanded=("matvec", "matvec_1"))
    into = {e["to"]: e["style"] for e in v["drawing"]["edges"] if e.get("data") == "A"}
    assert into["unit:mul"] == into["unit:mac"] == "fork"
    assert {e["label"] for e in v["drawing"]["edges"] if e["style"] == "fork"} == {"fork"}
