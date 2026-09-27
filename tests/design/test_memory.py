"""The memory plan (DP1b, D86): passes, pins, the context later policies read."""

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


def test_vecadd_default_is_contiguous_in_l2_and_l1():
    p = plan(ctx(), MemorySpec())
    assert p.passes == DEFAULT_PASSES and p.changes == {}
    assert bases(p) == {c: {"l2": b, "l1": b} for c, b in (("A", 0), ("B", 512), ("C", 1024))}
    assert all(
        lay.shape == (64,) and lay.strides == (8,)
        for mems in p.layouts.values()
        for lay in mems.values()
    )
    assert list(p.layouts["A"]) == ["l2", "l1"]


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


def test_pins_give_scenarios_vecadd():
    spec = MemorySpec().with_changes([("B.l1.base", 576), ("C.l1.base", 1152)])
    assert bases(plan(ctx(), spec)) == {
        "A": {"l2": 0, "l1": 0}, "B": {"l2": 512, "l1": 576}, "C": {"l2": 1024, "l1": 1152}
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
