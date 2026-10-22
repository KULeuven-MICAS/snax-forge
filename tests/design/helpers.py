"""Shared helpers for the SNAX-DESIGN tests.

The platform is the checked-in ``platforms/small16.json`` (``snax_forge.json``
for a platform read from snax_cluster's hjson, the snax-gvsoc submodule's
cluster files, which only CI's gvsoc job fetches, D128); the graphs are the SNAX-DFG
fixtures of tests/dfg/fixtures (vecadd bound to ``elementwise_add`` with
W = 4, split, as imported), and vecadd bound with W = 8 through the
checked-in recipe. ``design`` pairs them in memory, with optional edits to
either side as plain dicts, the way a person edits the files.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from snax_forge.design import Design, Platform, run_checks
from snax_forge.dfg import Graph
from snax_forge.sandbox import Recipe, apply_recipe

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "dfg" / "fixtures"
SMALL16 = REPO / "platforms" / "small16.json"
SNAX_FORGE = REPO / "platforms" / "snax_forge.json"
CFG = REPO / "gvsoc" / "snax-gvsoc" / "sw" / "cfg"  # empty unless the submodule is checked out
RECIPE = REPO / "recipes" / "vecadd.json"
BOUND = FIXTURES / "vecadd_accelerated.snaxdfg"

Edit = Callable[[dict[str, Any]], Any]


def platform_dict() -> dict[str, Any]:
    return json.loads(SMALL16.read_text())


def graph_dict(name: str = "vecadd_accelerated") -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.snaxdfg").read_text())


def bound_vecadd(w: int) -> Graph:
    """vecadd bound with W = ``w`` by the checked-in recipe."""
    recipe = Recipe.load(RECIPE).with_params({"W": w})
    return apply_recipe(recipe, Graph.load(FIXTURES / "vecadd.snaxdfg"))[-1].graph


def bound_w8() -> Graph:
    """vecadd bound with W = 8 by the checked-in recipe."""
    return bound_vecadd(8)


def design(
    graph: str | Graph = "vecadd_accelerated",
    edit_platform: Edit | None = None,
    edit_graph: Edit | None = None,
    sets: list[tuple[str, Any]] | None = None,
) -> Design:
    """A Design from small16 and a fixture graph, each optionally edited as a dict."""
    pd = platform_dict()
    if edit_platform:
        edit_platform(pd)
    platform = Platform.from_dict(pd).with_changes(sets or [])
    if isinstance(graph, Graph):
        g = graph
    else:
        gd = copy.deepcopy(graph_dict(graph))
        if edit_graph:
            edit_graph(gd)
        g = Graph.from_dict(gd)
    return Design(platform, g)


def problems(d: Design) -> list:
    return run_checks(d)


def codes(d: Design) -> list[str]:
    return [p.code for p in run_checks(d)]


def _streamer(port: str, write: bool) -> dict[str, Any]:
    return {"instance": "acc", "port": port, "write": write, "n_ports": 4, "spatial_bounds": [4],
            "temporal_dims": 1, "fifo_depth": 2}  # fmt: skip


# vecadd at W = 4 on small16: its streamers as the design point holds them, in port order
VECADD_STREAMERS = {"acc_a": _streamer("a", False), "acc_b": _streamer("b", False),
                    "acc_out": _streamer("out", True)}  # fmt: skip


def accel_node(gd: dict[str, Any]) -> dict[str, Any]:
    """The accelerated node of a vecadd fixture dict."""
    return gd["body"][0]["body"][0]


# --- graph edits for the lowering cases ---


def two_loops(g: dict[str, Any]) -> None:
    """vecadd's temporal map split in two temporal maps, i_o over 0:2 around i_t over 0:N // 8."""
    inner = g["body"][0]
    inner["attrs"]["range"] = "0:N // 8"
    del inner["attrs"]["loop.split"]
    node = inner["body"][0]
    for m in [*node["inputs"].values(), *node["outputs"].values()]:
        m["subset"] = ["32 * i_o + 4 * i_t:32 * i_o + 4 * i_t + 4"]
    tasklet = node["attrs"]["replaced"]["body"][0]
    for m in [*tasklet["inputs"].values(), *tasklet["outputs"].values()]:
        m["subset"] = ["32 * i_o + 4 * i_t + i_s"]
    outer = {"id": "add_map_o", "kind": "map", "inputs": {}, "outputs": {},
             "attrs": {"var": "i_o", "range": "0:2", "loop.kind": "temporal"}, "body": [inner]}  # fmt: skip
    g["body"] = [outer]


def chained(g: dict[str, Any]) -> None:
    """vecadd followed by D = C + B on a second instance, acc2."""
    second = copy.deepcopy(g["body"][0])
    second["id"] = "add_map2"
    node = second["body"][0]
    node["id"], node["attrs"]["instance"] = "add2", "acc2"
    node["inputs"]["a"]["data"], node["outputs"]["out"]["data"] = "C", "D"
    rep = node["attrs"]["replaced"]
    rep["id"], rep["body"][0]["id"] = "add_map2_s", "add2"
    rep["body"][0]["inputs"]["in1"]["data"], rep["body"][0]["outputs"]["out"]["data"] = "C", "D"
    g["containers"]["D"] = copy.deepcopy(g["containers"]["C"])
    g["body"].append(second)
