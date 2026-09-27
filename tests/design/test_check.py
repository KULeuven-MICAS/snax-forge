"""The design checks (DP1a, D85): one case per problem code, each naming its fix."""

from __future__ import annotations

import copy
import json

import pytest

from snax_forge.design import CHECKS, Design, DesignError, Platform, check, load, register_check
from snax_forge.design.memory import MEMORY_PASSES, MemorySpec
from snax_forge.design.problems import Problem
from snax_forge.lower.layout import Layout

from .helpers import (
    BOUND,
    SMALL16,
    accel_node,
    bound_w8,
    codes,
    design,
    platform_dict,
    problems,
)

# =============================================================================
# 1. What passes
# =============================================================================


def test_vecadd_on_small16_passes():
    assert problems(design()) == []
    check(design())  # does not raise


def test_vecadd_w8_passes_on_the_same_platform():
    assert problems(design(bound_w8())) == []


def test_changes_that_still_fit_pass():
    sets = [("l1.n_banks", 32), ("streamers.acc_out.fifo_depth", 4),
            ("streamers.acc_a.temporal_dims", 2), ("register_window", 64)]  # fmt: skip
    assert problems(design(sets=sets)) == []


# =============================================================================
# 2. Stage platform
# =============================================================================


def test_platform_load_problems_come_from_load(tmp_path):
    d = platform_dict()
    d["l1"]["n_banks"] = "16"
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps(d))
    got = problems(load(BOUND, bad))
    assert [(p.code, p.where) for p in got] == [("platform.keys", "l1.n_banks")]


def test_a_missing_platform_file_is_a_problem(tmp_path):
    got = problems(load(BOUND, tmp_path / "nope.json"))
    assert [p.code for p in got] == ["platform.keys"]


@pytest.mark.parametrize(
    "field, value, fix",
    [
        ("beat_bits", 256, "--set platform.l2.beat_bits=512"),
        ("width_bits", 32, "--set platform.l2.width_bits=64"),
        ("dtype", "int32", '--set platform.l2.dtype="int64"'),
    ],
)
def test_l1_and_l2_must_agree(field, value, fix):
    def edit(d):
        d["l2"][field] = value
        if field == "width_bits":
            d["l2"]["dtype"] = "int32"

    got = [p for p in problems(design(edit_platform=edit)) if p.where == f"l2.{field}"]
    assert [(p.code, p.fix) for p in got] == [("platform.l2", fix)]


def test_banks_must_hold_the_dma_port():
    got = problems(design(sets=[("l1.n_banks", 12)]))
    assert [(p.code, p.fix) for p in got] == [("platform.banks", "--set platform.l1.n_banks=16")]
    assert "port group of 8 banks" in got[0].message


def test_banks_do_not_matter_without_an_l2():
    def edit(d):
        d["l2"] = None
        d["l1"]["n_banks"] = 12

    assert "platform.banks" not in codes(design(edit_platform=edit))


# =============================================================================
# 3. Stage graph
# =============================================================================


def test_a_graph_that_does_not_load_is_a_problem(tmp_path):
    (tmp_path / "g.snaxdfg").write_text("{")
    got = problems(load(tmp_path / "g.snaxdfg", SMALL16))
    assert [p.code for p in got] == ["graph.load"]
    got = problems(load(tmp_path / "none.snaxdfg", SMALL16))
    assert [p.code for p in got] == ["graph.load"]


def test_an_unbound_symbol_names_the_recipe():
    def edit(g):
        g["symbols"]["N"] = None

    (p,) = problems(design(edit_graph=edit))
    assert (p.code, p.where) == ("graph.symbols", "symbol N") and '"symbols"' in p.fix


@pytest.mark.parametrize("name", ["vecadd", "vecadd_split"])
def test_a_tasklet_left_is_unbound(name):
    """The imported graph also leaves N unbound (graph.symbols); the split one binds it."""
    (p,) = [p for p in problems(design(name)) if p.code != "graph.symbols"]
    assert p.code == "graph.unbound" and p.where.startswith("tasklet add")
    assert "no core" in p.message and '"bind"' in p.fix


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
        ("brm", "elementwise_mul", "no BRM 'elementwise_mul'"),
        ("implementation", "verilog", "no implementation 'verilog'"),
        ("params", {"W": 4, "op": "sub"}, "not in the param's values"),
        ("code", "out = a - b", "computes 'out = a - b', but elementwise_add computes"),
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


def test_an_instance_may_not_take_a_streamers_name():
    """A second map bound to instance acc_a: its name is acc's streamer for port a."""

    def edit(g):
        second = copy.deepcopy(g["body"][0])
        second["id"] = "add_map2"
        n = second["body"][0]
        n["id"] = "add2"
        n["attrs"]["instance"] = "acc_a"
        n["attrs"]["replaced"]["id"] = "add_map2_s"
        g["body"].append(second)

    got = problems(design(edit_graph=edit))
    assert [(p.code, p.where) for p in got] == [("graph.instance", "instance acc_a")]
    assert "port a of instance acc" in got[0].message


def test_platform_and_graph_problems_come_together_and_stop_connect():
    def graph(g):
        g["symbols"]["N"] = None

    got = codes(design(edit_graph=graph, sets=[("l1.n_banks", 12), ("streamers.acc_c.prio", 1)]))
    assert got == ["platform.banks", "graph.symbols"]  # connect.streamer_key waits


# =============================================================================
# 4. Stage connect
# =============================================================================


def test_an_unknown_streamer_entry_suggests_the_closest():
    (p,) = problems(design(sets=[("streamers.acc_ou.fifo_depth", 4)]))
    assert (p.code, p.where, p.fix) == (
        "connect.streamer_key", "platform.streamers.acc_ou", "did you mean acc_out?"
    )  # fmt: skip
    assert "['acc_a', 'acc_b', 'acc_out']" in p.message
    (p,) = problems(design(sets=[("streamers.zzz.fifo_depth", 4)]))
    assert p.fix == "remove the entry"


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
    assert "[8] gives 8 lanes, but port a of instance acc (elementwise_add, W=4" in p.message
    assert "change the lanes in the recipe" in p.fix


def test_spatial_bounds_must_be_the_nests():
    (p,) = problems(design(sets=[("streamers.acc_b.spatial_bounds", [2, 2])]))
    assert p.code == "connect.lanes" and "[4] (fastest first)" in p.message
    assert p.fix == "remove spatial_bounds, or --set platform.streamers.acc_b.spatial_bounds=[4]"


def test_the_l1_must_hold_the_containers_dtype():
    got = problems(design(sets=[("l1.dtype", "int32"), ("l2.dtype", "int32")]))
    assert [(p.code, p.where) for p in got] == [
        ("connect.dtype", "container A"), ("connect.dtype", "container B"),
        ("connect.dtype", "container C"),
    ]  # fmt: skip
    assert got[0].fix == '--set platform.l1.dtype="int64" --set platform.l2.dtype="int64"'


def test_one_element_per_word_only():
    sets = [("l1.dtype", "int32"), ("l2.dtype", "int32"),
            ("l1.elems_per_word", 2), ("l2.elems_per_word", 2)]  # fmt: skip
    got = [p for p in problems(design(sets=sets)) if p.where == "platform.l1.elems_per_word"]
    assert [p.code for p in got] == ["connect.dtype"] and "open item 21" in got[0].message


def test_registers_must_fit_the_window():
    got = problems(design(sets=[("register_window", 8)]))
    assert [(p.code, p.where) for p in got] == [("connect.regmap", "dma")]
    assert got[0].fix == "--set platform.register_window=16, or lower platform.dma.dims"
    got = problems(design(sets=[("register_window", 16), ("streamers.acc_b.temporal_dims", 6)]))
    assert [(p.code, p.where) for p in got] == [("connect.regmap", "streamer acc_b")]
    assert "17 registers" in got[0].message and "temporal_dims" in got[0].fix


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


def test_check_raises_with_every_problem():
    with pytest.raises(DesignError) as e:
        check(design(sets=[("l1.n_banks", 12)], edit_graph=lambda g: g["symbols"].update(N=None)))
    assert e.value.codes == ["platform.banks", "graph.symbols"]
    text = str(e.value)
    assert text.startswith("design check: 2 problems\n [platform.banks] l1.n_banks: ")
    assert "\n     fix: --set platform.l1.n_banks=16" in text


def test_the_working_copy_is_what_the_checks_read():
    d = load(BOUND, SMALL16, [("streamers.acc_out.fifo_depth", 4)])
    assert d.platform.base == "small16" and d.platform.changes == {
        "streamers.acc_out.fifo_depth": 4
    }
    assert isinstance(d.platform, Platform) and problems(d) == []


# =============================================================================
# 6. Stage memory (DP1b)
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


def test_an_unknown_pass_is_named():
    (p,) = problems(with_memory([("passes.layout", "tiled")]))
    assert (p.code, p.where) == ("memory.pin", "memory.passes.layout")
    assert "registered: ['contiguous']" in p.message


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
    assert "multiples of the 8-byte word" in p.message and p.fix == "--set memory.B.l1.base=608"


def test_a_moved_container_must_be_whole_beats():
    """N = 60: 480 bytes, not whole 64-byte beats (open item 38)."""

    def n60(g):
        g["symbols"]["N"] = 60

    got = [p for p in problems(with_memory(edit_graph=n60)) if p.code == "memory.align"]
    assert [p.where for p in got] == [
        "A in l2",
        "A in l1",
        "B in l2",
        "B in l1",
        "C in l2",
        "C in l1",
    ]
    assert "480 bytes, not whole 64-byte beats" in got[0].message and "8 elements" in got[0].fix


def test_a_layout_must_fit_its_memory():
    got = problems(with_memory([], edit_platform=lambda d: d["l1"].update(rows=8)))
    assert [(p.code, p.where) for p in got] == [("memory.fit", "C in l1")]
    assert "L1 is 1024 B (16 banks x 8 rows x 8 B)" in got[0].message
    assert got[0].fix.startswith("--set platform.l1.rows=12")
    (p,) = problems(with_memory([("C.l1.base", 8000)]))
    assert p.code == "memory.fit" and p.fix.startswith("pin it lower")


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
    got = codes(with_memory([("Q.l1.base", 0)], sets=[("streamers.acc_q.prio", 1)]))
    assert got == ["connect.streamer_key"]
