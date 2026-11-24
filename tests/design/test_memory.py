"""The memory plan: passes, pins, the context later policies read."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snax_forge.design import DesignError
from snax_forge.design.memory import (
    MEMORY_PASSES,
    MemoryContext,
    MemoryPlan,
    MemorySpec,
    plan,
    register_memory_pass,
)

from .helpers import design


def ctx(**kw) -> MemoryContext:
    return design(**kw).context()


def bases(p: MemoryPlan) -> dict[str, dict[str, int]]:
    return {c: {m: lay.base for m, lay in mems.items()} for c, mems in p.layouts.items()}


# =============================================================================
# 1. The default plan
# =============================================================================


def test_without_an_l2_everything_is_in_l1_word_aligned():
    def edit(d):
        d["l2"] = None

    p = plan(ctx(edit_platform=edit), MemorySpec())
    assert bases(p) == {"A": {"l1": 0}, "B": {"l1": 512}, "C": {"l1": 1024}}


# =============================================================================
# 2. Pins
# =============================================================================


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


# =============================================================================
# 4. padded_rows: rows an odd number of words apart (rowsum 2's lane accumulator)
# =============================================================================

PADDED = MemorySpec().with_changes([("passes.layout", "padded_rows")])


def rowsum_v2(spec: MemorySpec | None = None, **params: int):
    """rowsum_v2 bound by its recipe (W 4, M 8, N 64 unless given) as a Design under ``spec``,
    its checks run (they make the plan)."""
    from snax_forge.design import Design, Platform, run_checks
    from snax_forge.sandbox import Recipe, apply_recipe
    from tests.helpers import imported, recipe_path

    from .helpers import SNAX_FORGE

    r = Recipe.load(recipe_path("rowsum_v2")).with_params(params)
    g = apply_recipe(r, imported("rowsum"))[-1].graph
    d = Design(Platform.load(SNAX_FORGE), g, memory_spec=spec or MemorySpec())
    assert run_checks(d) == []
    return d


@pytest.mark.parametrize(
    "spec, banks", [(MemorySpec(), [0, 0, 0, 0]), (PADDED, [0, 1, 2, 3])],
    ids=["R=64", "R=65"],
)  # fmt: skip
def test_each_lanes_first_read_one_bank_at_64_distinct_at_65(spec, banks):
    """The lanes read down a column, A[4 i_t + w, j]: rows R words apart put lane w in bank
    (w R + j) mod 32, all one bank at R = 64, w + j at R = 65 (L1's 32 banks)."""
    d = rowsum_v2(spec)
    ctx = d.context()
    (a,) = [x for x in ctx.accesses if x.connector == "a"]
    lay, word, n = d.memory.layouts["A"]["l1"], ctx.word_bytes("l1"), d.platform.l1.n_banks
    assert n == 32
    first = (lay.address(a.indices[0]) - lay.base) // word % n
    assert first.tolist() == banks
    every = (lay.address(a.indices) - lay.base) // word % n  # every beat: the same story
    assert all(len(set(beat)) == (4 if spec is PADDED else 1) for beat in every.tolist())
