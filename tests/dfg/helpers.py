"""Shared helpers for the SNAX-DFG tests (DFG1).

The fixtures are vecadd at three steps of the flow, kept in the form
``Graph.to_json`` writes:

    vecadd.snaxdfg              as imported: one map over 0:N, a tasklet (IMP1's target)
    vecadd_split.snaxdfg        after split_map with W = 4, N bound to 64
    vecadd_accelerated.snaxdfg  after bind to elementwise_add, instance acc

Graphs in the error tests are the plain fixture as a dict, changed by one
edit (``edited``), the way a person or an LLM edits the file.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NAMES = ("vecadd", "vecadd_split", "vecadd_accelerated")


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
