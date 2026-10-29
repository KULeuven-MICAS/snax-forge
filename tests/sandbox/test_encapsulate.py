"""encapsulate and unencapsulate (D148): groups into one accelerator and back.

dot's two instances, ``mul`` and ``sum``, each a group of its own, become one
accelerator ``dot``: every accelerated node of each carries ``accelerator``,
which round-trips through the file; ``unencapsulate`` gives the graph back
byte for byte. The graph check: one instance's nodes carry one accelerator
(or none), and an accelerator's name is no instance's.
"""

from __future__ import annotations

import copy
import json

import pytest

from snax_forge.dfg import DfgError, Graph
from snax_forge.sandbox import SandboxError, encapsulate, unencapsulate
from tests.helpers import graph_dict

from .helpers import graph


def dot() -> Graph:
    return graph("dot_accelerated")


def accelerators(g: Graph) -> dict[str, str | None]:
    return {n.attrs["instance"]: n.attrs.get("accelerator") for n, _ in g.walk()
            if n.kind == "accelerated"}  # fmt: skip


def test_encapsulate_puts_both_groups_into_one_accelerator():
    g = encapsulate(dot(), "dot", ["mul", "sum"])
    assert accelerators(g) == {"mul": "dot", "sum": "dot"}
    assert accelerators(dot()) == {"mul": None, "sum": None}  # the input is not changed


def test_the_attr_round_trips_and_is_written_after_the_kinds_own():
    g = encapsulate(dot(), "dot", ["mul", "sum"])
    assert Graph.from_dict(json.loads(g.to_json())) == g
    attrs = list(g.node("mult").attrs)
    assert attrs[: attrs.index("accelerator") + 1] == [
        "instance", "brm", "implementation", "code", "params", "replaced", "accelerator"
    ]  # fmt: skip


def test_a_graph_without_it_is_written_as_before():
    """An optional attr, written only when set: no existing file changes."""
    assert "accelerator" not in dot().to_json()


def test_unencapsulate_gives_the_graph_back():
    g = unencapsulate(encapsulate(dot(), "dot", ["mul", "sum"]), "dot")
    assert g.to_json() == dot().to_json()


def test_one_unit_names_its_accelerator():
    g = encapsulate(dot(), "mults", ["mul"])
    assert accelerators(g) == {"mul": "mults", "sum": None}


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


def test_unencapsulate_refuses_an_unknown_accelerator():
    with pytest.raises(SandboxError, match="no accelerator of that name"):
        unencapsulate(dot(), "dot")


def _set(d, instance, value):
    for n in d["body"]:
        for m in n.get("body", []):
            if m["kind"] == "accelerated" and m["attrs"]["instance"] == instance:
                m["attrs"]["accelerator"] = value


def test_the_graph_check_refuses_an_instance_in_two_accelerators():
    d = graph_dict("dot_accelerated")
    _set(d, "mul", "dot")
    second = copy.deepcopy(d["body"][0])  # a second node on mul, in another accelerator
    second["id"] = "mult_map2"
    second["body"][0]["id"] = "mult2"
    second["body"][0]["attrs"]["accelerator"] = "other"
    d["body"].append(second)
    with pytest.raises(DfgError, match="instance 'mul' is in accelerator 'dot' elsewhere"):
        Graph.from_dict(d)


def test_the_graph_check_refuses_an_accelerator_named_as_an_instance():
    d = graph_dict("dot_accelerated")
    _set(d, "mul", "sum")
    with pytest.raises(DfgError, match="accelerator 'sum' is also an instance's name"):
        Graph.from_dict(d)


def test_the_graph_check_refuses_an_accelerator_that_is_no_name():
    d = graph_dict("dot_accelerated")
    _set(d, "mul", "a-b")
    with pytest.raises(DfgError, match="not a valid name"):
        Graph.from_dict(d)
