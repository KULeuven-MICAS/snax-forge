"""encapsulate and unencapsulate: groups into one accelerator and back.

dot's two instances, ``mul`` and ``sum``, each a group of its own, become one
accelerator ``dot``: every accelerated node of each carries ``accelerator``,
which round-trips through the file; ``unencapsulate`` gives the graph back
byte for byte. The graph check: one instance's nodes carry one accelerator
(or none), and an accelerator's name is no instance's.
"""

from __future__ import annotations

import json

import pytest

from snax_forge.dfg import Graph
from snax_forge.sandbox import SandboxError, encapsulate, unencapsulate

from .helpers import graph


def dot() -> Graph:
    return graph("dot_accelerated")


def accelerators(g: Graph) -> dict[str, str | None]:
    return {n.attrs["instance"]: n.attrs.get("accelerator") for n, _ in g.walk()
            if n.kind == "accelerated"}  # fmt: skip


def test_the_attr_round_trips_and_is_written_after_the_kinds_own():
    g = encapsulate(dot(), "dot", ["mul", "sum"])
    assert Graph.from_dict(json.loads(g.to_json())) == g
    attrs = list(g.node("mult").attrs)
    assert attrs[: attrs.index("accelerator") + 1] == [
        "instance", "brm", "implementation", "code", "params", "replaced", "accelerator"
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("accelerator", "units", "message"),
    [
        ("mul", ["mul", "sum"], "an instance is already named 'mul'"),
        ("dot", ["mul", "add"], "no instance 'add'"),
        ("dot", ["sum", "mul"], "in another order than the graph runs them"),
        ("dot", ["mul", "mul"], "listed twice"),
        ("dot", [], "non-empty list"),
        ("2dot", ["mul"], "must be a name"),
    ],
)
def test_encapsulate_refuses(accelerator, units, message):
    with pytest.raises(SandboxError, match=message):
        encapsulate(dot(), accelerator, units)


def test_a_unit_in_an_accelerator_is_refused_until_unencapsulated():
    g = encapsulate(dot(), "mults", ["mul"])
    with pytest.raises(SandboxError, match="'mul' is in accelerator 'mults' already"):
        encapsulate(g, "dot", ["mul", "sum"])
    with pytest.raises(SandboxError, match="an accelerator is already named 'mults'"):
        encapsulate(g, "mults", ["sum"])
    assert accelerators(encapsulate(unencapsulate(g, "mults"), "dot", ["mul", "sum"])) == {
        "mul": "dot", "sum": "dot"
    }  # fmt: skip


def _set(d, instance, value):
    for n in d["body"]:
        for m in n.get("body", []):
            if m["kind"] == "accelerated" and m["attrs"]["instance"] == instance:
                m["attrs"]["accelerator"] = value
