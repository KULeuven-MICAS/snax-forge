"""Node kinds are registered, never subclassed: a kind added from outside the core."""

from __future__ import annotations

import pytest

from snax_forge.dfg import KINDS, DfgError, register_kind


@pytest.fixture
def barrier():
    """``test_barrier``: a kind with one required attr, a default and its own check."""

    def check(node, scope, what):
        if node.attrs["count"] < 1:
            raise DfgError(f"{what}.count: must be positive")

    register_kind("test_barrier", required=("count",), defaults={"label": ""}, check=check)
    yield
    del KINDS["test_barrier"]


def test_the_built_in_kinds():
    assert {"map", "tasklet", "accelerated"} <= set(KINDS)
    assert KINDS["map"].body and KINDS["accelerated"].body and not KINDS["tasklet"].body
