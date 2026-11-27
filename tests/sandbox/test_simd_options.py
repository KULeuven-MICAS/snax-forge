"""The simd options the workloads need, bound on the imported, fused bodies.

Accepted when each program (``simd_programs.py``) imports as NumPy writes it, op by op, fuses
into one nest by ``fuse_map`` alone (gray's three products a tree, the rest chains), splits by
4 and binds to ``simd`` at its option: requant at one stream (d 1024, lo 0, hi 127, int32 into
int8), gray and both blurs at axpbypcz over three streams (k = 77, 150, 29 and d 256; k2 = 2,
d 4, k1 and k3 unset), threshold at one stream with k from the scalar t (hi 255); the bound
graph's reference run equals NumPy on full-range data; ``unbind`` and ``unfuse_map`` give the
import back. Refused: a divisor that is no power of two, and the fuse of two nests no value
joins. The tree fuse itself (``fused_tree``, a local in each step) is checked on gray.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg import Graph, execute
from snax_forge.dfg.import_sdfg import import_sdfg
from snax_forge.sandbox import SandboxError, bind, fuse_map, split_map, unbind, unfuse_map
from snax_forge.sandbox.compose import fuse_pairs
from snax_forge.sandbox.patterns import fused_tree, fused_view
from snax_forge.sandbox.transforms import tag_map

from . import simd_programs as progs

BIND = {"brm": "simd", "implementation": "chisel_tiled_spatial", "instance": "acc"}
RNG = np.random.default_rng(7)


def imported(prog, **symbols: int) -> Graph:
    g = import_sdfg(prog.to_sdfg(simplify=True), prog.name.rsplit("_programs_", 1)[1])
    g.symbols.update(symbols)
    return g


def fused(g: Graph) -> tuple[Graph, list[tuple[str, str]]]:
    """``g`` fused into one nest by ``fuse_map`` alone, each time the first pair it takes; the
    pairs fused."""
    done = []
    while pairs := fuse_pairs(g):
        g = fuse_map(g, *pairs[0])
        done.append(pairs[0])
    assert len(g.body) == 1, [n.id for n in g.body]
    return g, done


def split(g: Graph) -> tuple[Graph, str]:
    """The fused nest's innermost map split by 4, the maps around it temporal; the spatial
    map's id."""
    inner = g.body[0]
    while inner.body[0].kind == "map":
        g = tag_map(g, inner.id, "temporal")
        inner = inner.body[0]
    return split_map(g, inner.id, 4), f"{inner.id}_s"


def bound(prog, params: dict, **symbols: int) -> Graph:
    """The bound graph: fused, split by 4, bound."""
    g, node = split(fused(imported(prog, **symbols))[0])
    return bind(g, **(BIND | {"node": node, "params": params}))


def accelerated(g: Graph):
    (n,) = [n for n, _ in g.walk() if n.kind == "accelerated"]
    return n


def full(dtype: str, *shape: int) -> np.ndarray:
    info = np.iinfo(dtype)
    return RNG.integers(info.min, info.max, size=shape, endpoint=True, dtype=dtype)


def ran(g: Graph, inputs: dict[str, np.ndarray], out: str, shape: tuple[int, ...],
        dtype: str) -> np.ndarray:  # fmt: skip
    return execute(g, {**inputs, out: np.zeros(shape, dtype)})[out]


# =============================================================================
# Each option on its body
# =============================================================================


def test_requant_binds_its_clamp_into_int8():
    g = bound(progs.requant, {}, N=64)
    n = accelerated(g)
    assert n.attrs["code"] == "out = min(max(a // d, lo), hi)"
    assert n.attrs["params"] == {"W": 4, "op": "requant", "streams": 1, "d": 1024, "lo": 0,
                                 "hi": 127}  # fmt: skip
    z = full("int32", 64)
    want = np.minimum(np.maximum(z // np.int32(1024), 0), 127).astype(np.int8)
    assert np.array_equal(ran(g, {"z": z}, "a", (64,), "int8"), want)


def test_gray_binds_axpbypcz_with_three_baked_weights():
    g = bound(progs.gray, {}, N=64)
    n = accelerated(g)
    assert n.attrs["code"] == "out = (k1 * a + k2 * b + k3 * c) // d"
    assert n.attrs["params"] == {"W": 4, "op": "axpbypcz", "streams": 3, "k1": 77, "k2": 150,
                                 "k3": 29, "d": 256}  # fmt: skip
    assert n.attrs["opType"] == "uint16"
    R, G, B = (full("uint8", 64) for _ in range(3))
    u = np.uint16
    want = (u(77) * R.astype(u) + u(150) * G.astype(u) + u(29) * B.astype(u)) // u(256)
    assert np.array_equal(ran(g, {"R": R, "G": G, "B": B}, "Y", (64,), "uint8"),
                          want.astype(np.uint8))  # fmt: skip


@pytest.mark.parametrize(
    ("prog", "shape", "src", "out"),
    [(progs.blur_rows, (6, 10), "Y", "X"), (progs.blur_cols, (6, 8), "X", "V")],
    ids=["rows", "cols"],
)
def test_a_blur_binds_axpbypcz_with_its_middle_weight_only(prog, shape, src, out):
    """``Y[j] + 2 Y[j + 1] + Y[j + 2]``: three streams of one container, k1 and k3 unset (a
    weight of 1 has no multiplier). Along a row the neighbours start mid-word, which the
    design's streamers refuse (the halo link's place); bind takes the code either way."""
    H, W = shape
    g = bound(prog, {}, H=H, W=W)
    n = accelerated(g)
    assert n.attrs["code"] == "out = (a + k2 * b + c) // d"
    assert n.attrs["params"] == {"W": 4, "op": "axpbypcz", "streams": 3, "k2": 2, "d": 4}
    x = full("uint8", H, W)
    u = np.uint16
    y = x.astype(u)  # NumPy < 2 keeps uint16(2) * a uint8 array at uint8; DaCe's is uint16
    if prog is progs.blur_rows:
        want, oshape = (y[:, :-2] + u(2) * y[:, 1:-1] + y[:, 2:]) // u(4), (H, W - 2)
    else:
        want, oshape = (y[:-2] + u(2) * y[1:-1] + y[2:]) // u(4), (H - 2, W)
    assert np.array_equal(ran(g, {src: x}, out, oshape, "uint8"), want.astype(np.uint8))


def test_threshold_binds_its_step_with_k_from_the_scalar():
    g = bound(progs.threshold, {}, N=64)
    n = accelerated(g)
    assert n.attrs["code"] == "out = hi * min(max(a - k + 1, 0), 1)"
    assert n.attrs["params"] == {"W": 4, "op": "threshold", "streams": 1, "hi": 255}
    assert n.attrs["opType"] == "int16"
    V = full("uint8", 64)
    for t in (np.int16(-32768), np.int16(0), np.int16(128), np.int16(255), np.int16(32767)):
        got = ran(g, {"V": V, "t": np.array([t])}, "T", (64,), "uint8")
        want = (np.int16(255) * np.minimum(np.maximum(V.astype(np.int16) - t + np.int16(1), 0),
                                           1)).astype(np.uint8)  # fmt: skip
        assert np.array_equal(got, want), t


@pytest.mark.parametrize(
    ("prog", "symbols"),
    [(progs.requant, {"N": 8}), (progs.gray, {"N": 8}), (progs.blur_cols, {"H": 6, "W": 8}),
     (progs.threshold, {"N": 8})],
    ids=["requant", "gray", "blur_cols", "threshold"],
)  # fmt: skip
def test_unbind_and_unfuse_give_the_import_back(prog, symbols):
    g0 = imported(prog, **symbols)
    g, done = fused(g0)
    apart, node = split(g)
    b = bind(apart, **(BIND | {"node": node}))
    assert unbind(b, accelerated(b).id).to_dict() == apart.to_dict()
    for first, _ in reversed(done):
        g = unfuse_map(g, first)
    assert g.to_dict() == g0.to_dict()


# =============================================================================
# The tree fuse and the refusals
# =============================================================================


def test_gray_fuses_as_a_tree_each_step_joined_by_one_local():
    g, done = fused(imported(progs.gray, N=8))
    assert done == [("mult_map_1", "add_map"), ("mult_map", "mult_map_1"),
                    ("mult_map_2", "add_map_1"), ("mult_map", "mult_map_2"),
                    ("mult_map", "floordiv_map"), ("mult_map", "convert_to_uint8_map")]  # fmt: skip
    locals_ = {c for c, x in g.containers.items() if "loop.local" in x.attrs}
    assert len(locals_) == 6  # each tasklet's output but the last's
    g = split_map(g, "mult_map", 4)
    tasklets, view = fused_tree(g.node("mult_map_s"))
    assert [t.id for t in tasklets] == ["mult", "mult_1", "add", "mult_2", "add_1", "floordiv",
                                        "convert_to_uint8"]  # fmt: skip
    assert view.attrs["code"] == (
        "out = uint8((77 * uint16(in1) + 150 * uint16(in3) + 29 * uint16(in2)) // 256)"
    )
    assert [view.inputs[c].data for c in ("in1", "in3", "in2")] == ["R", "G", "B"]
    assert fused_view(g.node("mult_map_s")) is None  # two tasklets only


def test_a_divisor_that_is_no_power_of_two_is_refused():
    with pytest.raises(SandboxError, match="d = 1000 is not a power of two: the unit divides by "
                       "shifting"):  # fmt: skip
        bound(progs.requant_by_1000, {}, N=8)


def test_two_nests_no_value_joins_are_not_fused():
    g = imported(progs.gray, N=8)
    with pytest.raises(SandboxError, match="mult and mult_1 are not joined by a transient"):
        fuse_map(g, "mult_map", "mult_map_1")
