"""The design checks: one case per problem code, each naming its fix."""

from __future__ import annotations

import copy

import pytest

from snax_forge.design import (
    CHECKS,
    Design,
    DesignError,
    Platform,
    load,
    register_check,
)
from snax_forge.design.check import RUNTIME_MARGIN, RUNTIME_STACK
from snax_forge.design.memory import MEMORY_PASSES, MemorySpec
from snax_forge.design.problems import Problem
from snax_forge.dfg import Graph
from snax_forge.lower.layout import Layout
from tests.helpers import recipe_path

from .helpers import (
    BOUND,
    SMALL16,
    SNAX_FORGE,
    accel_node,
    bound_vecadd,
    codes,
    design,
    problems,
)

# =============================================================================
# 1. What passes
# =============================================================================


# =============================================================================
# 2. Stage platform
# =============================================================================


def test_a_missing_platform_file_is_a_problem(tmp_path):
    got = problems(load(BOUND, tmp_path / "nope.json"))
    assert [p.code for p in got] == ["platform.keys"]


@pytest.mark.parametrize(
    "field, value, fix",
    [
        ("beat_bits", 256, "--set platform.l2.beat_bits=512"),
        ("width_bits", 32, "--set platform.l2.width_bits=64"),
    ],
)
def test_l1_and_l2_must_agree(field, value, fix):
    def edit(d):
        d["l2"][field] = value

    got = [p for p in problems(design(edit_platform=edit)) if p.where == f"l2.{field}"]
    assert [(p.code, p.fix) for p in got] == [("platform.l2", fix)]


# =============================================================================
# 3. Stage graph
# =============================================================================


def test_a_graph_that_does_not_load_is_a_problem(tmp_path):
    (tmp_path / "g.snaxdfg").write_text("{")
    got = problems(load(tmp_path / "g.snaxdfg", SMALL16))
    assert [p.code for p in got] == ["graph.load"]
    got = problems(load(tmp_path / "none.snaxdfg", SMALL16))
    assert [p.code for p in got] == ["graph.load"]


def test_an_empty_map_and_an_empty_graph_are_unbound():
    def empty_map(g):
        g["body"][0]["body"] = []

    (p,) = problems(design(edit_graph=empty_map))
    assert (p.code, p.where, p.message) == ("graph.unbound", "map add_map", "holds nothing to run")

    def empty(g):
        g["body"] = []

    (p,) = problems(design(edit_graph=empty))
    assert (p.code, p.message) == ("graph.unbound", "has no accelerated node")


@pytest.mark.parametrize(
    "attr, value, text",
    [
        ("brm", "elementwise_div", "no BRM 'elementwise_div'"),
        ("implementation", "verilog", "no implementation 'verilog'"),
        ("params", {"W": 4, "op": "and"}, "not in the param's values"),
        ("code", "out = a - b", "computes 'out = a - b', but simd computes"),
    ],
)
def test_a_bad_accelerated_node_is_named(attr, value, text):
    def edit(g):
        accel_node(g)["attrs"][attr] = value

    (p,) = problems(design(edit_graph=edit))
    assert (p.code, p.where) == ("graph.brm", "node add") and text in p.message


def test_connectors_must_be_the_ports():
    def edit(g):
        n = accel_node(g)
        n["inputs"]["x"] = n["inputs"].pop("b")
        n["attrs"]["code"] = "out = a + x"

    got = [p for p in problems(design(edit_graph=edit)) if "connectors" in p.message]
    assert [p.code for p in got] == ["graph.brm"]


def test_an_instance_may_not_take_a_fixed_name():
    def edit(g):
        accel_node(g)["attrs"]["instance"] = "dma"

    (p,) = problems(design(edit_graph=edit))
    assert (p.code, p.where) == ("graph.instance", "instance dma") and "rename" in p.fix


# =============================================================================
# 4. Stage connect
# =============================================================================


def test_write_and_n_ports_come_from_the_port():
    def edit(d):
        d["streamers"]["acc_out"] = {"write": False, "n_ports": 8}

    got = problems(design(edit_platform=edit))
    assert [(p.code, p.where) for p in got] == [
        ("connect.derived", "platform.streamers.acc_out.write"),
        ("connect.derived", "platform.streamers.acc_out.n_ports"),
    ]
    assert "direction 'out'" in got[0].message and "4 lanes" in got[1].message
    assert got[0].fix == "remove 'write' from the entry"


def test_spatial_bounds_must_give_the_lanes():
    (p,) = problems(design(sets=[("streamers.acc_a.spatial_bounds", [8])]))
    assert (p.code, p.where) == ("connect.lanes", "platform.streamers.acc_a.spatial_bounds")
    assert "[8] gives 8 lanes, but port a of instance acc (simd, W=4" in p.message
    assert "change the lanes in the recipe" in p.fix


def test_spatial_bounds_must_be_the_nests():
    (p,) = problems(design(sets=[("streamers.acc_b.spatial_bounds", [2, 2])]))
    assert p.code == "connect.lanes" and "[4] (fastest first)" in p.message
    assert p.fix == "remove spatial_bounds, or --set platform.streamers.acc_b.spatial_bounds=[4]"


def _dtypes(**by):
    """A graph edit: containers to the given dtypes."""
    return lambda g: [g["containers"][c].update(dtype=dt) for c, dt in by.items()]


def _second_node_of_acc(g):
    """A second node on instance acc, on int32 containers E, F -> G."""
    second = copy.deepcopy(g["body"][0])
    second["id"] = "add_map2"
    n = second["body"][0]
    n["id"] = "add2"
    n["attrs"]["replaced"]["id"] = "add_map2_s"
    for conn, c in (("a", "E"), ("b", "F"), ("out", "G")):
        (n["inputs"] | n["outputs"])[conn]["data"] = c
        g["containers"][c] = dict(g["containers"]["A"], dtype="int32")
    g["body"].append(second)


def test_one_instance_moves_one_dtype_per_port():
    got = [p for p in problems(design(edit_graph=_second_node_of_acc)) if p.code == "connect.dtype"]
    assert [p.where for p in got] == ["streamer acc_a", "streamer acc_b", "streamer acc_out"]
    assert got[0].message == (
        "port a of instance acc (simd, W=4, op=add, streams=2) gets int32 from add2, int64 from "
        "add: one streamer moves one dtype"
    )
    assert got[0].fix == "bind the nodes of each dtype to an instance of their own in the recipe"


def _on_snax_forge(graph: Graph) -> Design:
    return Design(Platform.load(SNAX_FORGE), graph)


def _vecmul(w: int) -> Graph:
    """``recipes/elementwise/vecmul_v1.json`` at N = 2048 and W = ``w``."""
    from snax_forge.sandbox import Recipe, apply_recipe
    from tests.helpers import imported

    r = Recipe.load(recipe_path("vecmul_v1")).with_params({"N": 2048, "W": w})
    return apply_recipe(r, imported("vecmul"))[-1].graph


@pytest.mark.slow
def test_more_crossbar_inputs_than_the_platform_takes_are_refused():
    """connect.xbar_inputs: vecmul at W = 32 has 96 streamer channels, with snax_forge's
    one core and AXI 98 crossbar inputs, over a platform taking 64; W = 16 has 50 and
    passes;
    the platforms take 128 (SNAX's crossbar as checked), which lets W = 32 through (with the
    note design.xbar_inputs, tests/design/test_share_ports.py)."""
    from snax_forge.design.check import xbar_inputs

    narrow = Platform.load(SNAX_FORGE).with_changes([("l1.max_xbar_inputs", 64)])
    wide = Design(narrow, _vecmul(32))
    (p,) = [p for p in problems(wide) if p.code == "connect.xbar_inputs"]
    assert (p.where, p.message) == ("acc", (
        "98 TCDM crossbar inputs, the platform takes at most 64 (96 streamer channels, 2 for "
        "the core and AXI; l1.max_xbar_inputs, 128 as checked against SNAX's crossbar RTL)"))  # fmt: skip
    assert p.fix == "fewer lanes, or share_ports on a container between two accelerators"
    half = _on_snax_forge(_vecmul(16))
    assert problems(half) == [] and xbar_inputs(half) == (48, 2)
    assert problems(_on_snax_forge(_vecmul(32))) == []
    assert (
        Platform.load(SNAX_FORGE).l1.max_xbar_inputs
        == Platform.load(SMALL16).l1.max_xbar_inputs
        == 128
    )


# =============================================================================
# 5. Registration and the error
# =============================================================================


def test_a_registered_check_runs_in_its_stage():
    def no_acc(d: Design):
        if "acc" in d.instances:
            yield Problem("connect.test", "instance acc", "not allowed here")

    register_check("connect.test", "connect", no_acc)
    try:
        assert codes(design()) == ["connect.test"]
    finally:
        del CHECKS["connect.test"]
    with pytest.raises(ValueError, match="unknown stage"):
        register_check("x.y", "later", no_acc)
    with pytest.raises(ValueError, match="already registered"):
        register_check("graph.brm", "graph", no_acc)


def test_the_error_has_every_problem():
    found = problems(
        design(sets=[("l1.n_banks", 12)], edit_graph=lambda g: g["symbols"].update(N=None))
    )
    e = DesignError(found)
    assert e.codes == ["platform.banks", "graph.symbols"]
    text = str(e)
    assert text.startswith("design check: 2 problems\n [platform.banks] l1.n_banks: ")
    assert "\n     fix: --set platform.l1.n_banks=16" in text


# =============================================================================
# 6. Stage memory
# =============================================================================


def with_memory(memory=(), **kw) -> Design:
    d = design(**kw)
    d.memory_spec = MemorySpec().with_changes(list(memory))
    return d


def test_a_pin_must_name_a_container_and_one_of_its_memories():
    (p,) = problems(with_memory([("Cc.l1.base", 0)]))
    assert (p.code, p.where) == ("memory.pin", "memory.Cc.l1.base")
    assert p.fix == "did you mean memory.C.l1.base?"

    def no_l2(d):
        d["l2"] = None

    (p,) = problems(with_memory([("C.l2.base", 0)], edit_platform=no_l2))
    assert p.code == "memory.pin" and "does not live in 'l2'" in p.message
    (p,) = problems(with_memory([("C.l2.base", 0)]))
    assert (p.code, p.where) == ("memory.pin", "memory.C.l2.base")
    assert "the program's link places C's l2 array, not the plan" in p.message


def test_an_unknown_pass_is_named():
    (p,) = problems(with_memory([("passes.layout", "tiled")]))
    assert (p.code, p.where) == ("memory.pin", "memory.passes.layout")
    assert "registered: ['contiguous', 'padded_rows']" in p.message


def test_memory_load_problems_are_reported(tmp_path):
    (tmp_path / "m.json").write_text('{"passes": {"x": "y"}}')
    got = problems(load(BOUND, SMALL16, memory_path=tmp_path / "m.json"))
    assert [p.code for p in got] == ["memory.pin"]


def _with_pass(kind, name, fn, test):
    MEMORY_PASSES[kind][name] = fn
    try:
        test()
    finally:
        del MEMORY_PASSES[kind][name]


def test_an_accelerator_operand_needs_an_l1_layout():
    def l2_only(c):
        return {k: ["l2"] for k in c.containers}

    def test():
        got = problems(with_memory([("passes.residency", "l2_only")]))
        assert {p.code for p in got} == {"memory.residency"}
        assert "has no L1 layout: the streamers reach only the L1" in got[0].message

    _with_pass("residency", "l2_only", l2_only, test)


def test_a_layout_must_have_the_containers_shape():
    def short(c, residency):
        return {k: {m: Layout.contiguous(0, (32,), 8) for m in ms} for k, ms in residency.items()}

    def test():
        got = [
            p
            for p in problems(with_memory([("passes.layout", "short")]))
            if p.code == "memory.layout"
        ]
        assert len(got) == 6 and "shape [32] is not the container's [64]" in got[0].message

    _with_pass("layout", "short", short, test)


def test_a_moved_layout_must_start_on_a_beat():
    (p,) = problems(with_memory([("B.l1.base", 600)]))
    assert (p.code, p.where, p.fix) == ("memory.align", "B in l1", "--set memory.B.l1.base=640")
    (p,) = problems(with_memory([("B.l1.base", 604)]))
    assert "must be a multiple of the 8-byte word" in p.message
    assert p.fix == "--set memory.B.l1.base=608"


def test_a_pin_below_a_nonzero_base_gets_the_pin_at_the_memorys_first_byte():
    """snax_forge's L1 starts at 0x1000_0000: a pin below it is outside, and the fix pins it
    at the first byte, which then fits."""

    def pinned(a: int) -> Design:
        d = Design(Platform.load(SNAX_FORGE), bound_vecadd(4))
        d.memory_spec = MemorySpec().with_changes([("A.l1.base", a)])
        return d

    (a,) = problems(pinned(576))
    assert (a.code, a.where) == ("memory.fit", "A in l1")
    assert "L1 is 131072 B (32 banks x 512 rows x 8 B) at [268435456, 268566528)" in a.message
    assert a.fix == "pin it at or above 268435456 (--set memory.A.l1.base=268435456)"
    assert problems(pinned(268435456)) == []


def test_an_l2_layout_must_fit_the_l2():
    """An L2 layout has base 0, the link placing its array: only its size is checked."""
    got = problems(with_memory([], edit_platform=lambda d: d["l2"].update(size_bytes=256)))
    assert [(p.code, p.where) for p in got] == [("memory.fit", f"{c} in l2") for c in "ABC"]
    assert got[0].message == "spans bytes [0, 512); L2 is 256 B"
    assert got[0].fix == "--set platform.l2.size_bytes=512"


def test_layouts_stay_below_snruntimes_top_of_l1():
    """snax_forge's hjson has 1 core: snRuntime keeps 1 KiB of stack and 256 B at the top
    of L1 (start.S), so C's L1 layout (512 bytes, moved) may end there but not one beat past."""
    l1_end = 268435456 + 131072
    top = 1 * RUNTIME_STACK + RUNTIME_MARGIN
    assert top == 1280

    def pinned(c: int) -> Design:
        d = Design(Platform.load(SNAX_FORGE), bound_vecadd(4))
        d.memory_spec = MemorySpec().with_changes([("C.l1.base", c)])
        return d

    assert problems(pinned(l1_end - top - 512)) == []
    (p,) = problems(pinned(l1_end - top - 512 + 64))
    assert (p.code, p.where) == ("memory.runtime", "C in l1")
    assert f"snRuntime keeps [{l1_end - top}, {l1_end}) at the top of L1" in p.message
    assert "1 core of platforms/snax_forge_cluster.hjson, 256 B for CLS and TLS" in p.message
    assert p.fix == f"pin it lower (--set memory.C.l1.base={l1_end - top - 512})"
    (p,) = problems(pinned(l1_end - 256))  # past the end: memory.fit's, not this check's
    assert p.code == "memory.fit"


def test_pins_may_not_overlap():
    (p,) = problems(with_memory([("A.l1.base", 0), ("B.l1.base", 256)]))
    assert (p.code, p.where) == ("memory.overlap", "A and B in l1")
    assert "share 256 bytes" in p.message and "pinned: A, B" in p.message
    assert p.fix == "move the pin past everything: --set memory.B.l1.base=1280, or remove it"


def test_strided_layouts_that_interleave_do_not_overlap():
    def interleave(c, residency):
        return {k: {m: Layout(0, (64,), (16,)) for m in ms} for k, ms in residency.items()}

    def place(c, layouts, pins):
        return {k: {m: 8 * i for m in ms} for i, (k, ms) in enumerate(layouts.items())}

    def test():
        sets = [("passes.layout", "interleave"), ("passes.placement", "gaps")]
        got = problems(with_memory(sets, edit_platform=lambda d: d.update(l2=None)))
        assert [p.code for p in got if p.code == "memory.overlap"] == ["memory.overlap"]  # A, C
        assert got[-1].where == "A and C in l1"

    MEMORY_PASSES["placement"]["gaps"] = place
    try:
        _with_pass("layout", "interleave", interleave, test)
    finally:
        del MEMORY_PASSES["placement"]["gaps"]


def test_memory_waits_for_connect():
    got = codes(with_memory([("Q.l1.base", 0)], sets=[("streamers.acc_q.fifo_depth", 1)]))
    assert got == ["connect.streamer_key"]


# =============================================================================
# 7. Firing loops
# =============================================================================


def test_a_tile_map_is_not_lowered_yet():
    def tile(g):
        g["body"][0]["attrs"]["loop.kind"] = "tile"

    (p,) = problems(design(edit_graph=tile))
    assert (p.code, p.where) == ("connect.temporal", "node add")
    assert "map add_map around add is tile" in p.message


# =============================================================================
# The note char.missing
# =============================================================================


# =============================================================================
# The note cost.extrapolated
# =============================================================================


def test_without_a_platform_nothing_is_noted():
    from snax_forge.design.check import NOTES, Design
    from tests.helpers import steps

    assert list(NOTES["cost.extrapolated"](Design(None, steps("vecmul_v1", W=32)[-1]))) == []
