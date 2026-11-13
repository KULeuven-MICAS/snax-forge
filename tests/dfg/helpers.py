"""Shared helpers for the SNAX-DFG tests.

The fixtures (``fixture``) are graphs in the form ``Graph.to_json`` writes, among them vecadd as
imported, after ``split_map`` (W = 4, N = 64) and after ``bind`` to simd
(``vecadd_accelerated.snaxdfg``); dot as imported (a multiply map into ``tmp0``, then a sum map
folding into ``out[0]``) and after ``recipes/dot/dot_1.json``'s binds; vecmul and vsum as
imported. Graphs in the error tests are a fixture as a dict changed by one edit (``edited``),
the way a person or an LLM edits the file.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.helpers import FIXTURES

NAMES = ("vecadd", "vecadd_split", "vecadd_accelerated")  # the vecadd steps
ALL = (*NAMES, "dot", "dot_accelerated", "vecmul", "vsum")


def fixture(name: str) -> Path:
    return FIXTURES / f"{name}.snaxdfg"


def as_dict(name: str) -> dict[str, Any]:
    return json.loads(fixture(name).read_text())


def edited(name: str, edit: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    """The fixture as a dict with ``edit`` applied to a copy."""
    d = copy.deepcopy(as_dict(name))
    edit(d)
    return d


def inner(d: dict[str, Any]) -> dict[str, Any]:
    """The node inside the first top-level map (the tasklet or the accelerated node)."""
    node = d["body"][0]["body"][0]
    while node["kind"] == "map":
        node = node["body"][0]
    return node
