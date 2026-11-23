"""A carried fold (D192): ``carry_seed`` and its inverse, ``bind`` giving the unit its own
output as its seed, the reference executor's identity on a first visit, and LOWER's CARRY and
its refusal when the queues could overtake it, on ``matvec_v9``."""

from __future__ import annotations

import pytest

from snax_forge.dfg import DfgError, Reduction, Seed
from snax_forge.lower.derive import LowerError, unit_firings
from snax_forge.sandbox import SandboxError
from snax_forge.sandbox.seed import carry_seed, uncarry_seed
from tests.helpers import steps


def _carried():
    gs = steps("matvec_v9")
    k = next(
        i
        for i, g in enumerate(gs)
        if any(
            m.reduce is not None and m.reduce.carry for n, _ in g.walk() for m in n.outputs.values()
        )
    )
    return gs, k


def test_a_carry_reduction_round_trips_and_excludes_a_seed():
    r = Reduction("add", 0, None, True)
    assert r.to_dict() == {"op": "add", "identity": 0, "carry": True}
    assert Reduction.from_dict(r.to_dict(), "x") == r
    with pytest.raises(DfgError, match="carry and seed are mutually exclusive"):
        Reduction("add", None, Seed("q", ("m",)), True)
    with pytest.raises(DfgError, match="carry: true when given"):
        Reduction.from_dict({"op": "add", "identity": 0, "carry": False}, "x")


def test_carry_seed_needs_a_loop_that_revisits_the_output():
    g = steps("matvec_v2")[4]  # sum's nest split, no strip loop outside it
    with pytest.raises(SandboxError, match="written once per element"):
        carry_seed(g, "sum_map_1")


def test_uncarry_seed_undoes_carry_seed():
    gs, k = _carried()
    before = gs[k - 1]
    assert uncarry_seed(carry_seed(before, "sum_map_1"), "sum_map_1").to_json() == before.to_json()
    with pytest.raises(SandboxError, match="carried already"):
        carry_seed(gs[k], "sum_map_1")


def test_bind_gives_the_unit_its_own_output_as_its_seed():
    gs, _ = _carried()
    (n,) = [n for n, _ in gs[-1].walk() if n.kind == "accelerated" and n.attrs["instance"] == "sum"]
    assert n.attrs["carry"] is True
    assert n.inputs["seed"].data == n.outputs["out"].data == "q"
    assert list(n.inputs["seed"].subset) == list(n.outputs["out"].subset)


def test_carry_is_the_jobs_before_the_first_revisit():
    from snax_forge.design.streamers import instance_of

    gs, _ = _carried()
    (n,) = [n for n, _ in gs[-1].walk() if n.kind == "accelerated" and n.attrs["instance"] == "sum"]
    values, loops = unit_firings(gs[-1], n, instance_of(n))
    assert [lp.var for lp in loops][:2] == ["n_1_t", "m_t"]  # the strip outermost
    assert values == {"n": 1024, "T": 4, "carry": 64}  # 4 strips of 64 rows, 4 beats a row


def test_lower_refuses_a_carry_the_queues_could_overtake(tmp_path):
    from snax_forge.design.point import DesignPoint
    from snax_forge.lower.derive import task_list
    from tests.gen.helpers import designed

    path = designed(tmp_path, "matvec_v9", "matvec_v9_m16", ("M=16",))
    with pytest.raises(LowerError, match="read back as its seed 16 jobs after it is written"):
        task_list(DesignPoint.load(tmp_path / path))
