"""Packing narrow elements into words (D142): the streamers' dtype, num_slc and num_ch, and
the checks connect.pack and connect.num_ch, on small hand-built graphs."""

from __future__ import annotations

import copy
import json

from snax_forge.design import Platform, instances, load, resolve
from snax_forge.dfg import Graph
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe

from .helpers import BOUND, REPO, SMALL16, accel_node, graph_dict, platform_dict, problems
from .helpers import design as paired


def _vecadd(dtypes: dict[str, str], brm: str = "elementwise_add") -> Graph:
    """vecadd at W = 4, its containers of ``dtypes``, bound to ``brm``."""
    d = graph_dict()
    for c, dt in dtypes.items():
        d["containers"][c]["dtype"] = dt
    if brm == "elementwise_mul":
        node = accel_node(d)
        node["attrs"].update(brm=brm, code="out = a * b", params={"W": 4, "op": "mul"})
    return Graph.from_dict(d)


def _grid(shape: list[int], loops: list[tuple[str, str]], subset: list[str]) -> Graph:
    """vecadd's node on int32 matrices of ``shape``: temporal maps ``loops`` (var, range),
    outermost first, around it, every port's memlet ``subset``."""
    d = graph_dict()
    for c in d["containers"].values():
        c.update(shape=list(shape), dtype="int32")
    node = copy.deepcopy(accel_node(d))
    for m in (*node["inputs"].values(), *node["outputs"].values()):
        m["subset"] = list(subset)
    tasklet = node["attrs"]["replaced"]["body"][0]
    for m in (*tasklet["inputs"].values(), *tasklet["outputs"].values()):
        m["subset"] = [dim.split(":")[0] for dim in subset]
    body = [node]
    for var, rng in reversed(loops):
        body = [{"id": f"map_{var}", "kind": "map", "inputs": {}, "outputs": {},
                 "attrs": {"var": var, "range": rng, "loop.kind": "temporal"}, "body": body}]  # fmt: skip
    d["body"] = body
    return Graph.from_dict(d)


def _vsum(dtypes: dict[str, str]) -> Graph:
    """vsum bound by recipes/vsum_tree.json (accumulate, adder tree, W = 4), of ``dtypes``."""
    recipe = Recipe.load(REPO / "recipes" / "vsum_tree.json")
    d = apply_recipe(recipe, import_kernel("vsum"))[-1].graph.to_dict()
    for c, dt in dtypes.items():
        d["containers"][c]["dtype"] = dt
    return Graph.from_dict(d)


def _shell(g: Graph, platform: Platform | None = None) -> dict[str, tuple]:
    pf = platform or Platform.load(SMALL16)
    return {k: (s.dtype, s.num_slc, s.num_ch) for k, s in resolve(pf, g, instances(g)).items()}


def _pack(g: Graph) -> list:
    return [p for p in problems(paired(g)) if p.code == "connect.pack"]


# =============================================================================
# Accepted
# =============================================================================


def test_vecadd_int32_takes_two_channels_per_port():
    g = _vecadd({"A": "int32", "B": "int32", "C": "int32"})
    assert problems(paired(g)) == []
    assert _shell(g) == {p: ("int32", 2, 2) for p in ("acc_a", "acc_b", "acc_out")}


def test_vmul_int32_into_int64():
    g = _vecadd({"A": "int32", "B": "int32", "C": "int64"}, brm="elementwise_mul")
    assert problems(paired(g)) == []
    assert _shell(g) == {
        "acc_a": ("int32", 2, 2), "acc_b": ("int32", 2, 2), "acc_out": ("int64", 1, 4)
    }  # fmt: skip


def test_vsum_int32_into_int64():
    g = _vsum({"A": "int32", "out": "int64"})
    assert problems(paired(g)) == []
    assert _shell(g) == {"acc_a": ("int32", 2, 2), "acc_out": ("int64", 1, 1)}


def test_matadd_int32_4x6_in_2x2_tiles():
    g = _grid([4, 6], [("i", "0:2"), ("j", "0:3")], ["2 * i:2 * i + 2", "2 * j:2 * j + 2"])
    assert problems(paired(g)) == []
    assert _shell(g) == {p: ("int32", 2, 2) for p in ("acc_a", "acc_b", "acc_out")}


# =============================================================================
# Refused
# =============================================================================


def test_matadd_int32_3x5_in_2x2_tiles_pads_the_row():
    g = _grid([3, 5], [("i", "0:1"), ("j", "0:2")], ["2 * i:2 * i + 2", "2 * j:2 * j + 2"])
    got = _pack(g)
    assert [(p.code, p.where) for p in got] == [
        ("connect.pack", "node add.a"), ("connect.pack", "node add.b"),
        ("connect.pack", "node add.out"),
    ]  # fmt: skip
    assert got[0].message == (
        "row stride 5 of A[3, 5] is not a multiple of num_slc (int32, num_slc = 2 slices per word): the "
        "spatial loop over dimension 0 steps 5 elements, so its beats would start inside a word"
    )
    assert got[0].fix == "pad the leading dimension to 6"


def test_s0_not_a_multiple_of_num_slc_is_one_problem_per_port():
    """int16 in 4 x 2 tiles: S_0 = 2 < num_slc = 4. Loop j steps by S_0 and fails only because of
    it, so it is not reported again."""
    d = _grid([8, 8], [("i", "0:2"), ("j", "0:4")], ["4 * i:4 * i + 4", "2 * j:2 * j + 2"])
    d = d.to_dict()
    for c in d["containers"].values():
        c["dtype"] = "int16"
    got = _pack(Graph.from_dict(d))
    assert [p.where for p in got] == ["node add.a", "node add.b", "node add.out"]
    assert got[0].message == (
        "the fastest spatial bound S_0 = 2 is not a multiple of num_slc (int16, num_slc = 4 slices "
        "per word): a beat's fastest lanes must fill whole words (D142)"
    )
    assert got[0].fix == (
        "make the fastest spatial bound a multiple of 4 (the split factor in the recipe)"
    )


def test_a_column_does_not_run_along_contiguous_elements():
    g = _grid([4, 6], [("i", "0:2"), ("j", "0:6")], ["2 * i:2 * i + 2", "j"])
    got = _pack(g)
    assert [p.where for p in got] == ["node add.a", "node add.b", "node add.out"]
    assert got[0].message == (
        "the lanes do not run along contiguous elements: the fastest spatial loop steps 6 "
        "elements of A[4, 6], not 1 (int32, num_slc = 2 slices per word)"
    )
    assert got[0].fix == (
        "make A a container of 64-bit elements (int64 or uint64, one per word), or change the "
        "loop order so the lanes run along its last dimension"
    )


def test_a_one_lane_narrow_output_needs_a_word():
    got = _pack(_vsum({"A": "int32", "out": "int32"}))
    assert [(p.code, p.where) for p in got] == [("connect.pack", "node sum.out")]
    assert got[0].message == (
        "W = 1 lane < num_slc = 2 (int32, num_slc = 2 slices per word): a beat would fill part of a word"
    )
    assert got[0].fix == (
        "make the output container out int64 or uint64 (one element per word), or give the "
        "port num_slc = 2 lanes"
    )


def test_num_ch_other_than_the_wire_default_is_refused():
    g = _vecadd({"A": "int32", "B": "int32", "C": "int32"})
    got = problems(paired(g, sets=[("streamers.acc_out.num_ch", 4)]))
    assert [(p.code, p.where) for p in got] == [
        ("connect.num_ch", "platform.streamers.acc_out.num_ch")
    ]
    assert got[0].message == (
        "num_ch = 4 for 4 lanes of int32 (num_slc = 2): rho = num_ch * num_slc / W = 2. Only the "
        "wire default W / num_slc = 2 is placed for now: rho != 1 needs the FIFO slicer, which "
        "the flow does not place yet (D142). With rho <= 1, num_ch may be 1, 2 (the wire "
        "default); a value with rho > 1 never helps a unit firing once per cycle"
    )
    assert got[0].fix == ("remove num_ch, or --set platform.streamers.acc_out.num_ch=2")


def test_a_platform_with_l1_dtype_names_its_removal(tmp_path):
    d = platform_dict()
    d["l1"]["dtype"] = "int64"
    path = tmp_path / "p.json"
    path.write_text(json.dumps(d))
    (p,) = problems(load(BOUND, path))
    assert (p.code, p.where, p.message) == (
        "platform.keys", "l1.dtype", "is no longer a platform field"
    )  # fmt: skip
    assert p.fix == "remove it; each container's dtype decides the packing (D142)"
