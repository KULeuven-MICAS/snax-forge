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
    # one read of A: from q's innermost loop, which s's loops are drawn on (one box, D189)
    reads = [(e["from"], e["to"]) for e in v["drawing"]["edges"] if e.get("data") == "A"]
    assert reads == [("A@0", "q_mult_map"), ("q_mult_map", "q_mult_map_1"),
                     ("q_mult_map_1", "unit:mul"), ("q_mult_map_1", "unit:mac")]  # fmt: skip
    # a click on s's edge lights q's hops to A as well: its edge names q's port (``via``)
    (to_mac,) = [e for e in v["drawing"]["edges"] if e.get("data") == "A" and e["to"] == "unit:mac"]
    assert to_mac["via"] == {"node": "q_mult", "port": "a"}
    (o,) = v["drawing"]["outlines"]
    assert (o["id"], o["text"], o["block"]) == ("grp:bicg_types",
                                                "group bicg_types · from matvec, matvec_1", "matvec")  # fmt: skip
    loops = {b["id"]: (b["parent"], b.get("chips")) for b in v["drawing"]["nodes"]
             if b["kind"] == "loop"}  # fmt: skip
    assert loops == {
        "q_mult_map": ("grp:bicg_types", [["sum_map", None], ["s_mult_map_1", None]]),
        "q_mult_map_1": ("grp:bicg_types", [["sum_map_1", None], ["s_mult_map", None]]),
    }


def test_folded_the_group_holds_both_blocks():
    """The default view folds the lifted blocks: the group spanning them is drawn around both."""
    g = steps("bicg_v3", M=8, N=8, W=8)[-1]
    v = graph_view(Entry(g.name, Path(g.name), g, None))
    (o,) = v["drawing"]["outlines"]
    assert o["id"] == "grp:bicg_types"
    assert {b["id"] for b in v["drawing"]["nodes"] if b.get("parent") == "grp:bicg_types"} == {
        "matvec", "matvec_1"}  # fmt: skip
