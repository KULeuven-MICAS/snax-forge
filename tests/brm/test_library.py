"""The BRM library and its first file, elementwise_add (BRM3).

Accepted against the elementwise stub on scenarios/vecadd: the entry the
BRM resolves to is alu4's ``acc`` entry, and vecadd run with it gives the
same cycles, profile and data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from snax_forge.brm import LIBRARY, Brm, BrmError, library_names, load_brm, task_nest
from snax_forge.snax_model.scenario import ClusterConfig, Scenario, run

SCEN = Path(__file__).resolve().parents[2] / "scenarios"
IMPL = "chisel_tiled_spatial"


def acc_spec(cluster: ClusterConfig):
    (spec,) = [c for c in cluster.components if c.name == "acc"]
    return spec


# =============================================================================
# The library
# =============================================================================


@pytest.mark.parametrize("name", library_names())
def test_library_files_are_canonical(name):
    """Every file loads, validates, and is written exactly as ``to_json`` writes it."""
    brm = load_brm(name)
    assert (LIBRARY / f"{name}.json").read_text() == brm.to_json()
    assert Brm.from_dict(brm.to_dict()) == brm


def test_library_has_elementwise_add():
    assert "elementwise_add" in library_names()


def test_unknown_name():
    with pytest.raises(BrmError, match="no BRM 'gemm' in the library"):
        load_brm("gemm")


# =============================================================================
# elementwise_add against the elementwise stub
# =============================================================================


def test_default_entry_is_alu4s():
    """No params: W = 4 by default, the entry of clusters/alu4.json, in its key order."""
    inst = load_brm("elementwise_add").resolve(IMPL)
    spec = acc_spec(ClusterConfig.load(SCEN / "clusters" / "alu4.json"))
    kind, params = inst.accel_entry()
    assert kind == spec.accel
    assert list(params.items()) == list(spec.params.items())


def test_other_lane_counts():
    brm = load_brm("elementwise_add")
    for w in (1, 2, 8):
        inst = brm.resolve(IMPL, {"W": w})
        assert inst.accel_entry()[1]["lanes"] == w
        assert [p.lanes for p in inst.accel_config().ports] == [w, w, w]


def test_only_add():
    with pytest.raises(BrmError, match="not in the param's values"):
        load_brm("elementwise_add").resolve(IMPL, {"op": "mul"})


def test_ports_agree_on_element_positions():
    """The D70 assumption, checked for this BRM: a, b and out take the same element."""
    inst = load_brm("elementwise_add").resolve(IMPL)
    idx = [task_nest(inst, p, {"n": 16}).indices() for p in ("a", "b", "out")]
    assert np.array_equal(idx[0], idx[1]) and np.array_equal(idx[0], idx[2])


def test_vecadd_same_cycles_and_data_as_the_stub():
    """vecadd with its acc entry taken from the BRM runs exactly as with the stub."""
    stub = Scenario.load(SCEN / "vecadd" / "scenario.json")
    brm = Scenario.load(SCEN / "vecadd" / "scenario.json")
    spec = acc_spec(brm.cluster)
    spec.accel, spec.params = load_brm("elementwise_add").resolve(IMPL).accel_entry()
    a, b = run(stub), run(brm)
    assert a.total_cycles == b.total_cycles
    assert a.profile.to_dict() == b.profile.to_dict()
    assert np.array_equal(a.l1, b.l1) and np.array_equal(a.l2, b.l2)
    assert a.reads == b.reads
    x, y = (np.load(SCEN / "vecadd" / f) for f in ("a.npy", "b.npy"))
    assert np.array_equal(b.l2[2048 // 8 : 2048 // 8 + 64, 0], x + y)  # c = a + b at L2 2048
