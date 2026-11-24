"""The .snaxdfg format: fixtures, round trips, stored form, access."""

from __future__ import annotations

import pytest

from snax_forge.dfg import Container, Graph, Memlet, Node

from .helpers import ALL, fixture


@pytest.mark.parametrize("name", ALL)
def test_fixture_is_in_written_form(name):
    """A fixture reads back to exactly its own text: every field written, canonical."""
    text = fixture(name).read_text()
    assert Graph.load(fixture(name)).to_json() == text


@pytest.mark.parametrize("name", ALL)
def test_round_trip(name):
    g = Graph.load(fixture(name))
    assert Graph.from_dict(g.to_dict()) == g


def test_hand_built_vecadd_equals_the_fixture():
    """The plain vecadd built in Python is the file."""
    add = Node(
        "add",
        "tasklet",
        inputs={"in1": Memlet("A", ["n"]), "in2": Memlet("B", ["n"])},
        outputs={"out": Memlet("C", ["n"])},
        attrs={"code": "out = in1 + in2"},
    )
    g = Graph(
        "vecadd",
        {"N": None},
        {k: Container(["N"], "int64") for k in "ABC"},
        [
            Node(
                "add_map",
                "map",
                attrs={"var": "n", "range": "0:N", "loop.dim": "n", "loop.piece": 1},
                body=[add],
            )
        ],
    )
    assert g.to_json() == fixture("vecadd").read_text()


def test_hand_built_accelerated_equals_the_fixture():
    lanes = "4*n_1_t:4*n_1_t+4"
    split = Graph.load(fixture("vecadd_split"))
    acc = Node(
        "add",
        "accelerated",
        inputs={"a": Memlet("A", [lanes]), "b": Memlet("B", [lanes])},
        outputs={"out": Memlet("C", [lanes])},
        attrs={
            "instance": "acc",
            "brm": "simd",
            "implementation": "chisel_tiled_spatial",
            "code": "out = a + b",
            "params": {"W": 4, "op": "add", "streams": 2},
            "replaced": split.node("add_map_s").to_dict(),
        },
    )
    tmap = Node(
        "add_map",
        "map",
        attrs={
            "var": "n_1_t",
            "range": "0:N//4",
            "loop.dim": "n",
            "loop.piece": 1,
            "loop.kind": "temporal",
            "loop.split": {"var": "n", "range": "0:N"},
        },
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
