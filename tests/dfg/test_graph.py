"""The .snaxdfg format (DFG1, D71, D77): fixtures, round trips, stored form, access."""

from __future__ import annotations

import pytest

from snax_forge.dfg import Container, Graph, Memlet, Node

from .helpers import NAMES, as_dict, fixture


@pytest.mark.parametrize("name", NAMES)
def test_fixture_is_in_written_form(name):
    """A fixture reads back to exactly its own text: every field written, canonical."""
    text = fixture(name).read_text()
    assert Graph.load(fixture(name)).to_json() == text


@pytest.mark.parametrize("name", NAMES)
def test_round_trip(name):
    g = Graph.load(fixture(name))
    assert Graph.from_dict(g.to_dict()) == g


def test_hand_built_vecadd_equals_the_fixture():
    """The plain vecadd built in Python is the file."""
    add = Node(
        "add",
        "tasklet",
        inputs={"in1": Memlet("A", ["i"]), "in2": Memlet("B", ["i"])},
        outputs={"out": Memlet("C", ["i"])},
        attrs={"code": "out = in1 + in2"},
    )
    g = Graph(
        "vecadd",
        {"N": None},
        {k: Container(["N"], "int64") for k in "ABC"},
        [Node("add_map", "map", attrs={"var": "i", "range": "0:N"}, body=[add])],
    )
    assert g.to_json() == fixture("vecadd").read_text()


def test_hand_built_accelerated_equals_the_fixture():
    lanes = "4*i_t:4*i_t+4"
    acc = Node(
        "add",
        "accelerated",
        inputs={"a": Memlet("A", [lanes]), "b": Memlet("B", [lanes])},
        outputs={"out": Memlet("C", [lanes])},
        attrs={
            "instance": "acc",
            "brm": "elementwise_add",
            "implementation": "chisel_tiled_spatial",
            "params": {"W": 4, "op": "add"},
        },
    )
    tmap = Node(
        "add_map",
        "map",
        attrs={"var": "i_t", "range": "0:N//4", "loop.kind": "temporal"},
        body=[acc],
    )
    g = Graph("vecadd", {"N": 64}, {k: Container(["N"], "int64") for k in "ABC"}, [tmap])
    assert g.to_json() == fixture("vecadd_accelerated").read_text()


def test_missing_keys_take_defaults_and_every_field_is_written():
    d = {
        "name": "k",
        "containers": {"A": {"shape": [8], "dtype": "int32"}},
        "body": [{"id": "m", "kind": "map", "attrs": {"var": "i", "range": "0:8"}}],
    }
    out = Graph.from_dict(d).to_dict()
    assert out["symbols"] == {}
    assert out["containers"]["A"] == {
        "shape": [8],
        "dtype": "int32",
        "transient": False,
        "attrs": {},
    }
    assert out["body"][0] == {
        "id": "m",
        "kind": "map",
        "inputs": {},
        "outputs": {},
        "attrs": {"var": "i", "range": "0:8"},
        "body": [],
    }


def test_body_is_written_only_for_kinds_that_have_one():
    g = Graph.load(fixture("vecadd"))
    assert "body" in g.to_dict()["body"][0]
    assert "body" not in g.to_dict()["body"][0]["body"][0]  # the tasklet
    acc = Graph.load(fixture("vecadd_accelerated")).node("add")
    assert acc.to_dict()["body"] == []  # accelerated: a body, empty for a leaf block


def test_expressions_are_stored_canonical():
    d = as_dict("vecadd_split")
    tasklet = d["body"][0]["body"][0]["body"][0]
    tasklet["inputs"]["in1"]["subset"] = ["(4*i_t) + i_s + 0*1"]
    tasklet["attrs"]["code"] = "out=(in1+in2)"
    d["body"][0]["attrs"]["range"] = "0:N//4:1"
    d["containers"]["A"]["shape"] = ["N+0"]
    g = Graph.from_dict(d)
    t = g.node("add")
    assert t.inputs["in1"].subset == ["4 * i_t + i_s + 0 * 1"]  # canonical, not simplified
    assert t.attrs["code"] == "out = in1 + in2"
    assert g.node("add_map").attrs["range"] == "0:N // 4"  # a step of 1 is left out
    assert g.containers["A"].shape == ["N + 0"]
    d["containers"]["A"]["shape"] = ["2*32"]
    assert Graph.from_dict(d).containers["A"].shape == [64]  # no names: an int


def test_namespaced_attrs_pass_through_untouched():
    d = as_dict("vecadd")
    note = {"why": ["kept", 1], "n": None}
    d["body"][0]["attrs"]["user.note"] = note
    d["body"][0]["attrs"]["hw.whatever"] = 3
    d["containers"]["A"]["attrs"]["mem.home"] = "l2"
    g = Graph.from_dict(d)
    assert g.node("add_map").attrs == {
        "var": "i",
        "range": "0:N",
        "user.note": note,
        "hw.whatever": 3,
    }
    assert g.to_dict()["containers"]["A"]["attrs"] == {"mem.home": "l2"}
    assert Graph.from_dict(g.to_dict()) == g


def test_kind_owned_attrs_come_first_with_defaults():
    d = as_dict("vecadd_accelerated")
    acc = d["body"][0]["body"][0]
    acc["attrs"] = {"user.x": 1, "brm": "elementwise_add", "instance": "acc", "implementation": "i"}
    attrs = Graph.from_dict(d).node("add").attrs
    assert list(attrs) == ["instance", "brm", "implementation", "params", "user.x"]
    assert attrs["params"] == {}


def test_walk_is_execution_order_with_enclosing_nodes():
    g = Graph.load(fixture("vecadd_split"))
    assert [(n.id, [o.id for o in outer]) for n, outer in g.walk()] == [
        ("add_map", []),
        ("add_map_s", ["add_map"]),
        ("add", ["add_map", "add_map_s"]),
    ]
    assert g.node("add_map_s").attrs["loop.kind"] == "spatial"


def test_save_and_load(tmp_path):
    g = Graph.load(fixture("vecadd_accelerated"))
    g.save(tmp_path / "v.snaxdfg")
    assert (tmp_path / "v.snaxdfg").read_text() == fixture("vecadd_accelerated").read_text()


def test_nested_accelerated_node():
    """An accelerated node may hold a body (a nested block, D71); names bind as usual."""
    d = as_dict("vecadd_accelerated")
    outer = d["body"][0]["body"][0]
    inner = {**outer, "id": "add_inner", "attrs": {**outer["attrs"], "instance": "acc2"}}
    outer["body"] = [inner]
    g = Graph.from_dict(d)
    assert [n.id for n, _ in g.walk()] == ["add_map", "add", "add_inner"]
