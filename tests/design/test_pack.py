"""Packing narrow elements into words: the streamers' dtype, num_slc, num_ch, d_out and d_in,
and the checks connect.pack and connect.num_ch, on small hand-built graphs and recipes."""

from __future__ import annotations

import copy

from snax_forge.design import Platform, instances, resolve
from snax_forge.dfg import Graph
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe
from tests.helpers import graph_dict, imported, recipe_path

from .helpers import SMALL16, accel_node, problems
from .helpers import design as paired


def _vecadd(dtypes: dict[str, str], brm: str = "simd") -> Graph:
    """vecadd at W = 4, its containers of ``dtypes``, bound to ``brm``."""
    d = graph_dict()
    for c, dt in dtypes.items():
        d["containers"][c]["dtype"] = dt
    if brm == "simd":
        node = accel_node(d)
        node["attrs"].update(brm=brm, code="out = a * b", params={"W": 4, "op": "mul"})
    return Graph.from_dict(d)


def _grid(shape: list[int], loops: list[tuple[str, str]], subset: list[str],
          out: str = "int32") -> Graph:  # fmt: skip
    """vecadd's node on matrices of ``shape``, A and B int32 or (with a narrower ``out``)
    int64, C ``out``: temporal maps ``loops`` (var, range), outermost first, around it, every
    port's memlet ``subset``."""
    d = graph_dict()
    for c, cont in d["containers"].items():
        cont.update(shape=list(shape), dtype=out if c == "C" or out == "int32" else "int64")
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


def _narrow_out(d: dict) -> dict:
    """``_grid``'s graph with its inputs int64 (one element a word) and its output int32."""
    for c, cont in d["containers"].items():
        cont["dtype"] = "int32" if c == "C" else "int64"
    return d


def _vsum(dtypes: dict[str, str]) -> Graph:
    """vsum bound by recipes/reduce/vsum_tree.json (accumulate, adder tree, W = 4), of ``dtypes``."""
    recipe = Recipe.load(recipe_path("vsum_tree"))
    d = apply_recipe(recipe, import_kernel("vsum"))[-1].graph.to_dict()
    for c, dt in dtypes.items():
        d["containers"][c]["dtype"] = dt
    return Graph.from_dict(d)


def _shell(g: Graph, platform: Platform | None = None) -> dict[str, tuple]:
    pf = platform or Platform.load(SMALL16)
    return {k: (s.dtype, s.num_slc, s.num_ch) for k, s in resolve(pf, g, instances(g)).items()}


def _d_out(g: Graph) -> dict[str, int]:
    return {k: s.d_out for k, s in resolve(Platform.load(SMALL16), g, instances(g)).items()}


def _d_in(g: Graph) -> dict[str, int]:
    return {k: s.d_in for k, s in resolve(Platform.load(SMALL16), g, instances(g)).items()}


def _recipe(name: str, **params: int | str) -> Graph:
    """Recipe ``name`` at ``params`` (``types`` a type preset), its last graph."""
    r = Recipe.load(recipe_path(name)).with_params(params)
    return apply_recipe(r, imported(r.kernel, r.dtypes, r.types))[-1].graph


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
    g = _vecadd({"A": "int32", "B": "int32", "C": "int64"}, brm="simd")
    assert problems(paired(g)) == []
    assert _shell(g) == {
        "acc_a": ("int32", 2, 2), "acc_b": ("int32", 2, 2), "acc_out": ("int64", 1, 4)
    }  # fmt: skip


def test_vsum_int32_into_int64():
    g = _vsum({"A": "int32", "out": "int64"})
    assert problems(paired(g)) == []
    assert _shell(g) == {"acc_a": ("int32", 2, 2), "acc_out": ("int64", 1, 1)}


def test_vsum_int32_out_is_packed_two_to_a_word():
    g = _vsum({"A": "int32", "out": "int32"})
    assert problems(paired(g)) == []
    assert _shell(g) == {"acc_a": ("int32", 2, 2), "acc_out": ("int32", 2, 1)}
    assert _d_out(g) == {"acc_a": 1, "acc_out": 2}


def test_a_4x6_int32_output_at_w_1_is_packed():
    """Rows of 6 int32 at W = 1: the pushes step one element, a row is three whole words."""
    g = _grid([4, 6], [("i", "0:4"), ("j", "0:6")], ["i", "j:j + 1"], out="int32")
    g = Graph.from_dict(_narrow_out(g.to_dict()))
    assert _pack(g) == []  # connect.dtype refuses an output narrower than its inputs: a fold's
    assert _d_out(g)["acc_out"] == 2


def test_vecadd_i32_at_w_1_slices_its_inputs_and_packs_its_output():
    """One int32 lane a beat: a and b sliced 2 from a word, out packed 2 to one, num_ch 1."""
    g = _recipe("vecadd_i32", W=1)
    assert problems(paired(g)) == []
    assert _shell(g) == {p: ("int32", 2, 1) for p in ("acc_a", "acc_b", "acc_out")}
    assert _d_in(g) == {"acc_a": 2, "acc_b": 2, "acc_out": 1}
    assert _d_out(g) == {"acc_a": 1, "acc_b": 1, "acc_out": 2}


def test_a_narrow_input_is_sliced():
    """vsum_tree at int8, W = 4: its input sliced 2 from a word, its int32 sum packed 2."""
    g = _vsum({"A": "int8", "out": "int32"})
    assert _pack(g) == []
    assert _d_in(g) == {"acc_a": 2, "acc_out": 1}
    assert _d_out(g) == {"acc_a": 1, "acc_out": 2}


def test_an_input_re_read_by_an_outer_loop_is_sliced():
    """A 6-element int32 row read again on every pass of i (a step of 0): whole words."""
    g = _grid([6], [("i", "0:4"), ("j", "0:6")], ["j:j + 1"])
    assert _pack(g) == []
    assert _d_in(g) == {"acc_a": 2, "acc_b": 2, "acc_out": 1}


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
        "per word): a beat's fastest lanes must fill whole words"
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


def test_vecadd_i8_at_w_3_is_neither_sliced_nor_packed():
    """W = 3 neither divides nor is a multiple of num_slc: int8 inputs (8 a word) and the int32
    output (2 a word) are refused, with the fix."""
    got = _pack(_recipe("vecadd", types="i8", W=3))
    assert [p.where for p in got] == ["node add.a", "node add.b", "node add.out"]
    assert got[0].message == (
        "W = 3 lanes < num_slc = 8 (int8, num_slc = 8 slices per word) and does not divide it: a "
        "beat would fill part of a word, and only a W dividing num_slc is sliced"
    )
    assert got[0].fix == (
        "give the port W = 1, 2, 4 (sliced) or a multiple of 8 (wired), or make the container A "
        "int64 or uint64 (one element per word)"
    )
    # out is int32 (num_slc = 2): W = 3 is above it, not a multiple, so S_0's rule refuses it
    assert got[2].message.startswith(
        "the fastest spatial bound S_0 = 3 is not a multiple of num_slc"
    )


def test_a_packed_row_ending_inside_a_word_pads_the_row():
    """4 x 5 int32 at W = 1: a row of 5 pushes ends inside a word."""
    g = _grid([4, 5], [("i", "0:4"), ("j", "0:5")], ["i", "j:j + 1"], out="int32")
    got = _pack(Graph.from_dict(_narrow_out(g.to_dict())))
    assert [p.where for p in got] == ["node add.out"]
    assert got[0].message == (
        "loop j pushes 5 results per pass, not a multiple of d_out = 2: a word would end inside "
        "the pass, and only the run's last word may be short (int32, num_slc = 2, W = 1: packed "
        "d_out = 2 results to a word)"
    )
    assert got[0].fix == "pad the row to 6 elements"


def test_packed_pushes_down_a_column_are_not_contiguous():
    g = _grid([4, 6], [("j", "0:6"), ("i", "0:4")], ["i", "j:j + 1"], out="int32")
    got = _pack(Graph.from_dict(_narrow_out(g.to_dict())))
    assert [p.where for p in got] == ["node add.out"]
    assert got[0].message == (
        "loop i steps 6 elements of C, not W = 1: the pushes are not contiguous (int32, num_slc = "
        "2, W = 1: packed d_out = 2 results to a word)"
    )


def test_a_packed_output_has_one_channel():
    g = _vsum({"A": "int32", "out": "int32"})
    got = problems(paired(g, sets=[("streamers.acc_out.num_ch", 2)]))
    assert [(p.code, p.where) for p in got] == [
        ("connect.num_ch", "platform.streamers.acc_out.num_ch")
    ]
    assert got[0].fix == "remove num_ch, or --set platform.streamers.acc_out.num_ch=1"


def test_num_ch_other_than_the_wire_default_is_refused():
    g = _vecadd({"A": "int32", "B": "int32", "C": "int32"})
    got = problems(paired(g, sets=[("streamers.acc_out.num_ch", 4)]))
    assert [(p.code, p.where) for p in got] == [
        ("connect.num_ch", "platform.streamers.acc_out.num_ch")
    ]
    assert got[0].message == (
        "num_ch = 4 for 4 lanes of int32 (num_slc = 2): rho = num_ch * num_slc / W = 2. Only the "
        "wire default W / num_slc = 2 is placed for now: rho != 1 needs the FIFO slicer, which "
        "the flow does not place yet. With rho <= 1, num_ch may be 1, 2 (the wire "
        "default); a value with rho > 1 never helps a unit firing once per cycle"
    )
    assert got[0].fix == ("remove num_ch, or --set platform.streamers.acc_out.num_ch=2")
