"""Node kinds are registered, never subclassed (D19, D77): a kind added from outside the core."""

from __future__ import annotations

import pytest

from snax_forge.dfg import KINDS, DfgError, Graph, register_kind

from .helpers import as_dict


@pytest.fixture
def barrier():
    """``test_barrier``: a kind with one required attr, a default and its own check."""

    def check(node, scope, what):
        if node.attrs["count"] < 1:
            raise DfgError(f"{what}.count: must be positive")

    register_kind("test_barrier", required=("count",), defaults={"label": ""}, check=check)
    yield
    del KINDS["test_barrier"]


def test_registered_kind_is_usable(barrier):
    d = as_dict("vecadd")
    d["body"].append({"id": "sync", "kind": "test_barrier", "attrs": {"count": 2, "user.a": 1}})
    g = Graph.from_dict(d)
    assert g.node("sync").to_dict() == {
        "id": "sync",
        "kind": "test_barrier",
        "inputs": {},
        "outputs": {},
        "attrs": {"count": 2, "label": "", "user.a": 1},
    }
    assert Graph.from_dict(g.to_dict()) == g
    d["body"][-1]["attrs"]["count"] = 0
    with pytest.raises(DfgError, match="node 'sync'.count: must be positive"):
        Graph.from_dict(d)


def test_a_kind_is_registered_once():
    with pytest.raises(ValueError, match="already registered"):
        register_kind("map")


def test_the_built_in_kinds():
    assert {"map", "tasklet", "accelerated"} <= set(KINDS)
    assert KINDS["map"].body and KINDS["accelerated"].body and not KINDS["tasklet"].body
