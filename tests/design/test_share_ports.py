"""Shared movers (D169, PORT1): the crossbar's inputs by hand, the movers and their masks, the
fit rule, and the crossbar's warning and limit (D85)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from snax_forge.design import Design, DesignError
from snax_forge.design.check import NOTES, crossbar_inputs, run_checks
from snax_forge.design.platform import Platform
from snax_forge.design.streamers import (
    ShareError,
    instances,
    resolve,
    set_channels,
    shared_movers,
)
from tests.helpers import steps

from .helpers import SNAX_FORGE


def _inputs(name: str, **params: int) -> tuple[int, int]:
    g = steps(name, **params)[-1]
    return crossbar_inputs(Platform.load(SNAX_FORGE), g, instances(g))


@pytest.mark.parametrize(
    "name, params, dedicated, shared",
    [
        ("dot_1", {"W": 4}, 19, 14),
        ("mvt_1_dot2", {"W": 8}, 43, 26),
        ("mvt_2", {"W_x1": 8, "W_x2": 8}, 84, 50),
    ],
)  # fmt: skip
def test_the_crossbar_inputs_by_hand(name, params, dedicated, shared):
    """n_in = Σ movers' channels + the core (one since D170) + AXI: dot 1 at W = 4 is 4 + 4 + 4 (reader0 vmul.a /
    vsum.a, reader1 vmul.b, writer0 vmul.out / vsum.out) against 4 · 3 + 4 + 1; mvt_1_dot2 at
    W = 8 is 8 · 3 against 8 · 5 + 1; mvt_2 two such sets against twice 41."""
    assert sum(_inputs(name, **params)) == dedicated
    assert sum(_inputs(f"{name}_shared", **params)) == shared


def test_the_movers_and_their_masks():
    """mvt_1_dot2_shared at W = 8: per role the k-th widest port of each group on mover k; the
    writer masked, vsum_x1.out taking 1 of its 8 channels."""
    g = steps("mvt_1_dot2_shared", W=8)[-1]
    found = resolve(Platform.load(SNAX_FORGE), g, instances(g))
    (movers,) = shared_movers(g, found).values()
    assert [
        (m.name, m.num_ch, m.masked, [(u.group, u.port, u.num_ch) for u in m.users]) for m in movers
    ] == [
        ("reader0", 8, False, [("mvt", "vmul_x1.a", 8), ("vadd", "vadd.a", 8)]),
        ("reader1", 8, False, [("mvt", "vmul_x1.b", 8), ("vadd", "vadd.b", 8)]),
        ("writer0", 8, True, [("mvt", "vsum_x1.out", 1), ("vadd", "vadd.out", 8)]),
    ]
    # vmul_x1's two loops (rows, beats) on the readers, vadd's one: the mover takes the most
    assert [m.temporal_dims for m in movers] == [2, 2, 1]
    assert set_channels(g, found) == {"mvt_vadd": 24}


def test_a_port_that_does_not_fit_its_mover_is_refused():
    """A user's spatial bounds must be the mover's but the slowest: vsum.a as [2, 2] on reader0's
    [4] does not fit (share.shape)."""
    g = steps("dot_1_shared", W=4)[-1]
    found = resolve(Platform.load(SNAX_FORGE), g, instances(g))
    found["vsum_a"] = replace(found["vsum_a"], spatial_bounds=(2, 2))
    with pytest.raises(ShareError, match=r"vsum_a's spatial bounds \[2, 2\] \(in words\) do not "
                       r"fit in the first channels of vmul_a's \[4\]"):  # fmt: skip
        shared_movers(g, found)


def test_unshared_designs_pass_unchanged():
    """No marker, no shared accelerator: every port its own mover, as before."""
    g = steps("dot_1", W=4)[-1]
    found = resolve(Platform.load(SNAX_FORGE), g, instances(g))
    assert shared_movers(g, found) == {}
    assert set_channels(g, found) == {"vmul": 12, "vsum": 5}


@pytest.mark.slow
def test_above_64_inputs_a_note_above_128_a_refusal():
    """D85 amended: mvt_2 at W = 8 (84 inputs) and vecmul at W = 32 (98) pass with the note
    design.xbar_inputs; vecmul at W = 64 (194) is refused (connect.xbar_inputs); shared, mvt_2
    takes 50 and gets no note."""
    pf = Platform.load(SNAX_FORGE)
    for name, params, n in (("mvt_2", {"W_x1": 8, "W_x2": 8}, 84),
                            ("vecmul", {"W": 32, "N": 2048}, 98)):  # fmt: skip
        d = Design(pf, steps(name, **params)[-1])
        assert run_checks(d) == []
        (note,) = NOTES["design.xbar_inputs"](d)
        assert note.message.startswith(f"{n} TCDM crossbar inputs, above 64 (l1.warn_xbar_inputs)")
    d = Design(pf, steps("mvt_2_shared", W_x1=8, W_x2=8)[-1])
    assert list(NOTES["design.xbar_inputs"](d)) == []
    wide = Design(pf, steps("vecmul", W=64, N=2048)[-1])
    (p,) = [p for p in run_checks(wide) if p.code == "connect.xbar_inputs"]
    assert p.message.startswith("194 TCDM crossbar inputs, the platform takes at most 128")
    assert p.fix == "fewer lanes, or share_ports on a container between two accelerators (D169)"


def test_the_warning_is_a_platform_value():
    pf = Platform.load(SNAX_FORGE)
    assert (pf.l1.max_xbar_inputs, pf.l1.warn_xbar_inputs) == (128, 64)
    with pytest.raises(DesignError, match="warn_xbar_inputs must be 1 to max_xbar_inputs"):
        pf.with_changes([("l1.warn_xbar_inputs", 129)])
