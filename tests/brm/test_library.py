"""The BRM library: elementwise_add (BRM3), elementwise_mul and accumulate (BRM4).

elementwise_add is accepted against the elementwise stub on scenarios/vecadd:
the entry the BRM resolves to is alu4's ``acc`` entry, and vecadd run with it
gives the same cycles, profile and data. accumulate is accepted against the
reduce stub on scenarios/reduce the same way (D103), and its Chisel
implementation declares the Accumulator's drain.
"""

from __future__ import annotations

import json
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


def test_library_has_dots_blocks():
    assert {"elementwise_add", "elementwise_mul", "accumulate"} <= set(library_names())


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
    # c = a + b in L2 where vecadd's store_C puts it (1024 since NAME1, D83)
    c = store_base(SCEN / "vecadd" / "tasks.json", "store_C") // 8
    assert np.array_equal(b.l2[c : c + 64, 0], x + y)


def store_base(tasks: Path, task: str) -> int:
    """The L2 byte address a store task writes to, read off the task list."""
    steps = json.loads(tasks.read_text())["steps"]
    (step,) = [s for s in steps if s.get("task_name") == task]
    return step["values"]["dst"]["base"]


@pytest.mark.parametrize("w", [1, 4, 8])
def test_function_code_agrees_with_the_model(w):
    """The BRM's code and the kind SNAX-MODEL runs compute the same lanes (D82)."""
    import numpy as np

    from snax_forge import expr

    inst = load_brm("elementwise_add").resolve("chisel_tiled_spatial", {"W": w})
    cfg = inst.accel_config()
    rng = np.random.default_rng(w)
    ins = {p.name: rng.integers(-1000, 1000, p.lanes) for p in cfg.inputs}
    got = cfg.fn(0, ins, {}, {"n": 1})
    for target, e in expr.statements(inst.brm.function.code, "code"):
        assert np.array_equal(got[target], expr.evaluate(e, ins))


def test_elementwise_mul_is_add_with_mul():
    inst = load_brm("elementwise_mul").resolve(IMPL)
    assert inst.accel_entry() == (
        "elementwise",
        {"lanes": 4, "n_inputs": 2, "op": "mul", "latency": 0, "ii": 1},
    )
    assert inst.brm.pattern.attrs == {"op": "mul", "arity": 2}


@pytest.mark.parametrize("w", [1, 4, 8])
def test_mul_code_agrees_with_the_model(w):
    """Per lane, elementwise_mul's code and its kind compute the same (D82)."""
    from snax_forge import expr

    inst = load_brm("elementwise_mul").resolve(IMPL, {"W": w})
    cfg = inst.accel_config()
    rng = np.random.default_rng(w)
    ins = {p.name: rng.integers(-1000, 1000, p.lanes) for p in cfg.inputs}
    got = cfg.fn(0, ins, {}, {"n": 1})
    for target, e in expr.statements(inst.brm.function.code, "code"):
        assert np.array_equal(got[target], expr.evaluate(e, ins))


# =============================================================================
# accumulate against the reduce stub (BRM4, D103)
# =============================================================================


def test_accumulate_is_red4s_entry():
    """The adder tree at W = 4 is the reduce stub entry red4 had, in its key order."""
    inst = load_brm("accumulate").resolve("chisel_adder_tree")
    spec = acc_spec(ClusterConfig.load(SCEN / "clusters" / "red4.json"))
    assert inst.accel_entry() == (spec.accel, spec.params)
    assert list(spec.params.items()) == [
        ("lanes", 4), ("lanes_out", 1), ("op", "add"), ("latency", 1), ("ii", 1)
    ]  # fmt: skip


def test_reduce_same_cycles_and_data_as_the_stub():
    """scenarios/reduce with its acc entry written as the stub's gives 35 cycles and the sums."""
    brm = Scenario.load(SCEN / "reduce" / "scenario.json")
    stub = Scenario.load(SCEN / "reduce" / "scenario.json")
    spec = acc_spec(stub.cluster)
    spec.params = {"lanes": 4, "lanes_out": 1, "op": "add", "latency": 1, "ii": 1}
    a, b = run(stub), run(brm)
    assert a.total_cycles == b.total_cycles == 35
    assert a.profile.to_dict() == b.profile.to_dict()
    assert np.array_equal(a.l1, b.l1)
    x = np.load(SCEN / "reduce" / "x.npy")
    assert b.l1[128:132, 0].tolist() == x.reshape(4, 16).sum(axis=1).tolist()


def test_the_chisel_accumulator_declares_its_drain():
    """One lane, drain 1: in the entry only because it is not 0 (D103)."""
    brm = load_brm("accumulate")
    inst = brm.resolve("chisel_accumulator", {"W": 1})
    assert (inst.latency, inst.initiation_interval, inst.drain) == (1, 1, 1)
    assert inst.accel_entry()[1] == {
        "lanes": 1, "lanes_out": 1, "op": "add", "latency": 1, "ii": 1, "drain": 1
    }  # fmt: skip
    assert inst.accel_config().drain == 1
    assert "drain" not in brm.resolve("chisel_adder_tree").accel_entry()[1]
    with pytest.raises(BrmError, match="W = 4 not supported"):
        brm.resolve("chisel_accumulator", {"W": 4})


def test_the_drain_costs_a_cycle_per_back_to_back_sum():
    """scenarios/reduce (4 sums of 16) with drain 1 and 2: 38 and 41 cycles, the same sums."""
    base = run(Scenario.load(SCEN / "reduce" / "scenario.json"))
    for drain, cycles in ((1, 38), (2, 41)):
        sc = Scenario.load(SCEN / "reduce" / "scenario.json")
        spec = acc_spec(sc.cluster)
        spec.params = {**spec.params, "drain": drain}
        r = run(sc)
        assert r.total_cycles == cycles
        assert np.array_equal(r.l1, base.l1)


@pytest.mark.parametrize(("impl", "w"), [("chisel_adder_tree", 4), ("chisel_accumulator", 1)])
def test_accumulate_code_folded_agrees_with_the_model(impl, w):
    """``out = a`` folded with the pattern's op over every lane of T beats is what the kind gives."""
    from snax_forge.dfg import REDUCE_OPS

    inst = load_brm("accumulate").resolve(impl, {"W": w})
    cfg = inst.accel_config()
    T = 5
    beats = np.random.default_rng(w).integers(-1000, 1000, (T, w))
    state: dict = {}
    outs = [cfg.fn(k, {"a": beats[k]}, state, {"n": T, "T": T}) for k in range(T)]
    assert outs[:-1] == [{}] * (T - 1)
    op = REDUCE_OPS[inst.brm.pattern.attrs["op"]]
    assert inst.brm.function.code == "out = a"
    assert outs[-1]["out"].tolist() == [op.ufunc.reduce(beats.ravel())]


def test_accumulate_ports_and_beats():
    """a takes W elements per beat; out gives one element per T beats."""
    inst = load_brm("accumulate").resolve("chisel_adder_tree")
    assert task_nest(inst, "a", {"n": 16, "T": 4}).n_beats == 16
    out = task_nest(inst, "out", {"n": 16, "T": 4})
    assert (out.n_beats, out.n_lanes) == (4, 1)
