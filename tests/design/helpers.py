"""Shared helpers for the SNAX-DESIGN tests (DP1a).

The platform is the checked-in ``platforms/small16.json``; the graphs are
the SNAX-DFG fixtures of tests/dfg/fixtures (vecadd bound to
``elementwise_add`` with W = 4, split, as imported), and vecadd bound with
W = 8 through the checked-in recipe. ``design`` pairs them in memory, with
optional edits to either side as plain dicts, the way a person edits the
files.
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
ALU4 = REPO / "scenarios" / "clusters" / "alu4.json"
RECIPE = REPO / "recipes" / "vecadd.json"
BOUND = FIXTURES / "vecadd_accelerated.snaxdfg"

Edit = Callable[[dict[str, Any]], Any]


def platform_dict() -> dict[str, Any]:
    return json.loads(SMALL16.read_text())


def graph_dict(name: str = "vecadd_accelerated") -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.snaxdfg").read_text())


def bound_w8() -> Graph:
    """vecadd bound with W = 8 by the checked-in recipe."""
    recipe = Recipe.load(RECIPE).with_params({"W": 8})
    return apply_recipe(recipe, Graph.load(FIXTURES / "vecadd.snaxdfg"))[-1].graph


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


def alu4_streamers() -> dict[str, dict[str, Any]]:
    """alu4's streamer entries by name, in component order."""
    comps = json.loads(ALU4.read_text())["components"]
    return {c["name"]: c["config"] for c in comps if c["kind"] == "streamer"}


def accel_node(gd: dict[str, Any]) -> dict[str, Any]:
    """The accelerated node of a vecadd fixture dict."""
    return gd["body"][0]["body"][0]
