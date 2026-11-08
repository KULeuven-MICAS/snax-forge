"""The memory plan (D86): passes, pins, the context later policies read."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snax_forge.design import DesignError
from snax_forge.design.memory import (
    DEFAULT_PASSES,
    MEMORY_PASSES,
    MemoryContext,
    MemoryPlan,
    MemorySpec,
    plan,
    register_memory_pass,
)
from snax_forge.lower.layout import Layout

from .helpers import bound_w8, design


def ctx(**kw) -> MemoryContext:
    return design(**kw).context()


def bases(p: MemoryPlan) -> dict[str, dict[str, int]]:
    return {c: {m: lay.base for m, lay in mems.items()} for c, mems in p.layouts.items()}


# =============================================================================
# 1. The default plan
# =============================================================================


def test_vecadd_default_is_contiguous_in_l1():
    """The link places L2's arrays: an L2 layout keeps base 0."""
    p = plan(ctx(), MemorySpec())
    assert p.passes == DEFAULT_PASSES and p.changes == {}
    assert bases(p) == {c: {"l2": 0, "l1": b} for c, b in (("A", 0), ("B", 512), ("C", 1024))}
    assert all(
        lay.shape == (64,) and lay.strides == (8,)
        for mems in p.layouts.values()
        for lay in mems.values()
    )
    assert list(p.layouts["A"]) == ["l2", "l1"]


def test_a_narrow_container_is_laid_out_by_its_itemsize():
    """int32 vecadd (D142): 4-byte strides; each moved container 64 x 4 = 256 bytes, four whole
    beats, placed beat after beat."""

    def int32(g):
        for c in g["containers"].values():
            c["dtype"] = "int32"

    c = ctx(edit_graph=int32)
    assert c.elem_bytes("A") == 4 and c.word_bytes("l1") == 8
    p = plan(c, MemorySpec())
    assert {k: (lay["l1"].strides, lay["l1"].base) for k, lay in p.layouts.items()} == {
        "A": ((4,), 0), "B": ((4,), 256), "C": ((4,), 512)
    }  # fmt: skip


def test_the_layout_follows_the_bound_symbols():
    p = plan(bound_w8_ctx(), MemorySpec())
    assert p.layouts["C"]["l1"] == Layout(1024, (64,), (8,))


def bound_w8_ctx() -> MemoryContext:
    return design(bound_w8()).context()


def test_without_an_l2_everything_is_in_l1_word_aligned():
    def edit(d):
        d["l2"] = None

    p = plan(ctx(edit_platform=edit), MemorySpec())
    assert bases(p) == {"A": {"l1": 0}, "B": {"l1": 512}, "C": {"l1": 1024}}


def test_a_transient_lives_in_l1_only():
    def edit(g):
        g["containers"]["C"]["transient"] = True

    p = plan(ctx(edit_graph=edit), MemorySpec())
    assert list(p.layouts["C"]) == ["l1"] and list(p.layouts["A"]) == ["l2", "l1"]


# =============================================================================
# 2. Pins
# =============================================================================


def test_pins_place_b_and_c():
    spec = MemorySpec().with_changes([("B.l1.base", 576), ("C.l1.base", 1152)])
    assert bases(plan(ctx(), spec)) == {
        "A": {"l2": 0, "l1": 0}, "B": {"l2": 0, "l1": 576}, "C": {"l2": 0, "l1": 1152}
    }  # fmt: skip


def test_the_rest_is_placed_around_a_pin():
    """B pinned at 256 in L1: A no longer fits below it and goes after it, then C."""
    spec = MemorySpec().with_changes([("B.l1.base", 256)])
    assert {c: m["l1"] for c, m in bases(plan(ctx(), spec)).items()} == {
        "A": 768, "B": 256, "C": 1280
    }  # fmt: skip


def test_spec_changes_accumulate_and_round_trip(tmp_path):
    s = MemorySpec().with_changes([("B.l1.base", 576), ("passes.placement", "contiguous")])
    s = s.with_changes([("B.l1.base", 640)])
    assert list(s.changes.items()) == [("passes.placement", "contiguous"), ("B.l1.base", 640)]
    assert s.pins() == {("B", "l1"): 640}
    p = plan(ctx(), s)
    (tmp_path / "m.json").write_text(p.to_json())
    assert MemorySpec.load(tmp_path / "m.json") == s
    assert MemoryPlan.from_dict(json.loads(p.to_json())) == p


@pytest.mark.parametrize(
    "path, value, text",
    [
        ("B.l1", 0, "<container>.<memory>.base"),
        ("B.l1.strides", [8], "only a base can be pinned"),
        ("B.l1.base", "576", "must be an int byte address"),
        ("B.l1.base", True, "must be an int byte address"),
        ("passes.order", "x", "no pass 'order'"),
        ("passes.placement", 3, "must be a pass name"),
    ],
)
def test_a_bad_memory_set_is_named(path, value, text):
    with pytest.raises(DesignError) as e:
        MemorySpec().with_changes([(path, value)])
    (p,) = e.value.problems
    assert (p.code, p.where) == ("memory.pin", f"memory.{path}") and text in p.message


# =============================================================================
# 3. Registered passes and the context they read
# =============================================================================


def test_a_registered_placement_is_picked_by_name():
    def reverse(c, layouts, pins):
        out, at = {k: {} for k in layouts}, {m: c.span(m)[0] for m in c.memories}
        for name in reversed(c.containers):
            for m, lay in layouts[name].items():
                out[name][m] = at[m]
                at[m] += int(np.prod(lay.shape)) * c.word_bytes(m)
        return out

    register_memory_pass("placement", "reverse", reverse)
    try:
        spec = MemorySpec().with_changes([("passes.placement", "reverse")])
        p = plan(ctx(), spec)
        assert {c: m["l1"] for c, m in bases(p).items()} == {"A": 1024, "B": 512, "C": 0}
        assert p.passes["placement"] == "reverse"
    finally:
        del MEMORY_PASSES["placement"]["reverse"]
    with pytest.raises(ValueError, match="unknown pass kind"):
        register_memory_pass("banks", "x", reverse)
    with pytest.raises(ValueError, match="already registered"):
        register_memory_pass("placement", "contiguous", reverse)


def test_accesses_are_each_ports_elements_beat_by_beat():
    c = ctx()
    acc = c.accesses
    assert [(a.node, a.connector, a.container, a.direction) for a in acc] == [
        ("add", "a", "A", "in"), ("add", "b", "B", "in"), ("add", "out", "C", "out")
    ]  # fmt: skip
    a = acc[0].indices
    assert a.shape == (16, 4, 1)
    assert np.array_equal(a[..., 0], np.arange(64).reshape(16, 4))
    assert list(c.groups()) == ["add"] and len(c.groups()["add"]) == 3


def test_accesses_give_banks_under_a_layout():
    """What a conflict-aware policy would compute: A and B contiguous share banks every beat."""
    c = ctx()
    p = plan(c, MemorySpec())
    word, nb = c.word_bytes("l1"), c.platform.l1.n_banks

    def banks(access):
        lay = p.layouts[access.container]["l1"]
        return (lay.address(access.indices) // word) % nb

    a, b, _ = c.accesses
    assert np.array_equal(banks(a), banks(b))  # the vecadd_conflict case, 85 cycles
    shifted = MemorySpec().with_changes([("B.l1.base", 576)])
    p = plan(c, shifted)
    assert not np.intersect1d(banks(a)[0], banks(b)[0]).size


def test_context_sizes():
    c = ctx()
    assert c.memories == ["l2", "l1"] and c.shape("A") == (64,)
    assert c.span("l1") == (0, 8192) and c.span("l2") == (0, 32768) and c.beat_bytes == 64


# =============================================================================
# 4. padded_rows: rows an odd number of words apart (rowsum 2's lane accumulator)
# =============================================================================

PADDED = MemorySpec().with_changes([("passes.layout", "padded_rows")])


def rowsum_2(spec: MemorySpec | None = None, **params: int):
    """rowsum_2 bound by its recipe (W 4, M 8, N 64 unless given) as a Design under ``spec``,
    its checks run (they make the plan)."""
    from snax_forge.design import Design, Platform, run_checks
    from snax_forge.sandbox import Recipe, apply_recipe
    from tests.helpers import imported, recipe_path

    from .helpers import SNAX_FORGE

    r = Recipe.load(recipe_path("rowsum_2")).with_params(params)
    g = apply_recipe(r, imported("rowsum"))[-1].graph
    d = Design(Platform.load(SNAX_FORGE), g, memory_spec=spec or MemorySpec())
    assert run_checks(d) == []
    return d


@pytest.mark.parametrize(
    "spec, stride, words",
    [(MemorySpec(), 512, 64), (PADDED, 520, 65)],
    ids=["contiguous", "padded_rows"],
)  # fmt: skip
def test_rowsum_2s_layout_with_and_without_padded_rows(spec, stride, words):
    """A, 8 x 64 int64: rows 64 words apart, or 65 under padded_rows, in L2 and L1 alike; its
    block 8 x 64 or 8 x 65 words, 4096 or 4160 B, 64 or 65 beats, all moved; y, 1-D, as it was,
    placed after A's block."""
    from snax_forge.lower.layout import moved_bytes

    d = rowsum_2(spec)
    ctx, lays = d.context(), d.memory.layouts
    for m in ("l2", "l1"):
        assert (lays["A"][m].shape, lays["A"][m].strides) == ((8, 64), (stride, 8))
        assert (lays["y"][m].shape, lays["y"][m].strides) == ((8,), (8,))
    block = ctx.block("A", lays["A"]["l1"])
    assert block.shape == (8, words) and block.extent(8)[1] - block.base == 8 * words * 8
    assert moved_bytes(block, 8, ctx.beat_bytes) == 8 * words * 8 == 64 * words
    assert lays["y"]["l1"].base - lays["A"]["l1"].base == 64 * words
    assert d.memory.passes["layout"] == ("padded_rows" if spec is PADDED else "contiguous")


@pytest.mark.parametrize(
    "spec, banks", [(MemorySpec(), [0, 0, 0, 0]), (PADDED, [0, 1, 2, 3])],
    ids=["R=64", "R=65"],
)  # fmt: skip
def test_each_lanes_first_read_one_bank_at_64_distinct_at_65(spec, banks):
    """The lanes read down a column, A[4 i_t + w, j]: rows R words apart put lane w in bank
    (w R + j) mod 32, all one bank at R = 64, w + j at R = 65 (L1's 32 banks)."""
    d = rowsum_2(spec)
    ctx = d.context()
    (a,) = [x for x in ctx.accesses if x.connector == "a"]
    lay, word, n = d.memory.layouts["A"]["l1"], ctx.word_bytes("l1"), d.platform.l1.n_banks
    assert n == 32
    first = (lay.address(a.indices[0]) - lay.base) // word % n
    assert first.tolist() == banks
    every = (lay.address(a.indices) - lay.base) // word % n  # every beat: the same story
    assert all(len(set(beat)) == (4 if spec is PADDED else 1) for beat in every.tolist())


def test_vecadd_under_padded_rows_is_unchanged():
    """1-D containers keep contiguous."""
    c = ctx()
    assert plan(c, PADDED).layouts == plan(c, MemorySpec()).layouts


def test_padded_rows_rounds_a_row_up_to_odd_words():
    """An odd row stays (63 int64); a narrow dtype's row is counted in whole words: 64 int32
    are 32 words, 63 int32 252 B, also 32; either becomes 33 words, 66 elements apart."""
    from snax_forge.design.memory import padded_rows_layout

    c = rowsum_2(N=63).context()
    assert padded_rows_layout(c, {"A": ["l1"]})["A"]["l1"].strides == (504, 8)
    c.graph.containers["A"].dtype = "int32"
    assert padded_rows_layout(c, {"A": ["l1"]})["A"]["l1"].strides == (264, 4)
    c = rowsum_2().context()
    c.graph.containers["A"].dtype = "int32"
    assert padded_rows_layout(c, {"A": ["l1"]})["A"]["l1"].strides == (264, 4)
