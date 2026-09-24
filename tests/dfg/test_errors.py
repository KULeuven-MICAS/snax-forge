"""Malformed graphs are rejected, and the error names what is wrong (DFG1)."""

from __future__ import annotations

import pytest

from snax_forge.dfg import DfgError, Graph

from .helpers import as_dict, edited, inner


def _set(path, value):
    def edit(d):
        *head, last = path
        for k in head:
            d = d[k]
        d[last] = value

    return edit


def _node(edit):
    """An edit of the innermost node of the fixture."""
    return lambda d: edit(inner(d))


def _map(edit):
    return lambda d: edit(d["body"][0])


CASES = [
    # kinds and keys
    ("vecadd", _node(lambda n: n.update(kind="gemm")), "unknown kind 'gemm'"),
    ("vecadd", _node(lambda n: n.update(colour="red")), "unknown keys \\['colour'\\]"),
    ("vecadd", _set(("extra",), 1), "unknown keys \\['extra'\\]"),
    ("vecadd", _node(lambda n: n.pop("id")), "missing key 'id'"),
    ("vecadd", _node(lambda n: n["attrs"].pop("code")), "node 'add'.attrs: missing 'code'"),
    ("vecadd", _node(lambda n: n["attrs"].update(op="add")), "'op' is not an attr of a 'tasklet'"),
    ("vecadd", _node(lambda n: n.update(body=[])), "a 'tasklet' has no body"),
    # dangling references
    ("vecadd", _node(lambda n: n["inputs"]["in1"].update(data="X")), "in1: unknown container 'X'"),
    ("vecadd", _node(lambda n: n["inputs"]["in1"].update(subset=["j"])), "'j' is not a symbol"),
    ("vecadd", _node(lambda n: n["inputs"]["in1"].update(subset=["i", 0])), "2 dimensions for 'A'"),
    ("vecadd", _node(lambda n: n["attrs"].update(code="out = in1 + x")), "'x' is not an input"),
    (
        "vecadd",
        _node(lambda n: n["attrs"].update(code="y = in1 + in2")),
        "'y', which is not an out",
    ),
    ("vecadd", _node(lambda n: n["attrs"].update(code="out = in1\nout = in2")), "exactly once"),
    ("vecadd", _node(lambda n: n["attrs"].update(code="out = in1 / in2")), "Div is not allowed"),
    ("vecadd", _node(lambda n: n["attrs"].update(code="print(in1)")), "is not 'output = exp"),
    ("vecadd", _map(lambda m: m["attrs"].update(range="0:M")), "range: 'M' is not a symbol"),
    ("vecadd", _set(("containers", "A", "shape"), ["M"]), "container 'A'.shape: 'M' is not a sym"),
    # names
    ("vecadd", _node(lambda n: n.update(id="add_map")), "node 'add_map': the id is used twice"),
    ("vecadd", _node(lambda n: n.update(id="add.x")), "'add.x' is not a valid name"),
    ("vecadd", _map(lambda m: m["attrs"].update(var="N")), "'N' is already a symbol"),
    ("vecadd", _map(lambda m: m["attrs"].update(var="A")), "'A' is already a symbol, container"),
    ("vecadd_split", lambda d: d["body"][0]["body"][0]["attrs"].update(var="i_t"), "variable"),
    ("vecadd", _set(("symbols", "A"), None), "'A' is also a symbol"),
    ("vecadd", _set(("symbols", "N"), 6.5), "symbols.N: must be an int or null"),
    # maps
    ("vecadd", _map(lambda m: m["attrs"].update(range="0")), "range: 0 is not a range"),
    ("vecadd", _map(lambda m: m["attrs"].update(range="0:N:0")), "step of '0:N:0'"),
    ("vecadd", _map(lambda m: m["attrs"].update({"loop.kind": "vector"})), "'vector' is not one"),
    ("vecadd", _map(lambda m: m["inputs"].update(A={"data": "A", "subset": ["0:N"]})), "no conn"),
    # tasklets, containers
    ("vecadd", _node(lambda n: n["inputs"]["in1"].update(subset=["0:4"])), "one element"),
    ("vecadd", _node(lambda n: n["outputs"].update(in1=n["outputs"]["out"])), "both an input"),
    ("vecadd", _set(("containers", "A", "dtype"), "int"), "'int' is not a NumPy dtype name"),
    ("vecadd", _set(("containers", "A", "dtype"), "vector"), "'vector' is not a NumPy dtype"),
    ("vecadd", _set(("containers", "A", "shape"), [0]), "0 is not a positive size"),
    ("vecadd", _set(("containers", "A", "attrs"), {"home": "l2"}), "'home' must be namespaced"),
    ("vecadd", _set(("containers", "A", "transient"), 1), "transient: must be true or false"),
    # accelerated
    ("vecadd_accelerated", _node(lambda n: n["attrs"].pop("brm")), "missing 'brm'"),
    ("vecadd_accelerated", _node(lambda n: n["attrs"].update(params=[4])), "params: must be an"),
    ("vecadd_accelerated", _node(lambda n: n["attrs"].update(instance="a-b")), "not a valid name"),
]


@pytest.mark.parametrize(("name", "edit", "message"), CASES)
def test_rejected_by_name(name, edit, message):
    with pytest.raises(DfgError, match=message):
        Graph.from_dict(edited(name, edit))


def test_one_instance_bound_two_ways():
    d = as_dict("vecadd_accelerated")
    a = d["body"][0]["body"][0]
    b = {**a, "id": "add2", "attrs": {**a["attrs"], "params": {"W": 8, "op": "add"}}}
    d["body"][0]["body"].append(b)
    with pytest.raises(DfgError, match="instance 'acc' is bound elsewhere"):
        Graph.from_dict(d)
    b["attrs"]["params"] = {"W": 4, "op": "add"}
    Graph.from_dict(d)  # the same binding twice is fine


def test_a_variable_is_bound_only_inside_its_map():
    d = as_dict("vecadd")
    d["body"].append(
        {
            "id": "late",
            "kind": "tasklet",
            "inputs": {"x": {"data": "A", "subset": ["i"]}},
            "outputs": {"y": {"data": "C", "subset": [0]}},
            "attrs": {"code": "y = x"},
        }
    )
    with pytest.raises(DfgError, match="node 'late'.inputs.x: 'i' is not a symbol"):
        Graph.from_dict(d)
