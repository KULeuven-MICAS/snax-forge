"""dot end to end (E2E2, D106): a multiplier chained into an accumulator through L1, M5's acceptance.

``recipes/dot.json`` (the adder tree, W lanes) and ``recipes/dot_serial.json``
(the Chisel Accumulator, one lane) run from the import fixture to a checked
model run. Accepted when every run equals the kernel's reference and REF1 at
its pinned cycle count, the task list holds the one wait between the two
accelerators, ``run.md`` names it as a chaining wait, and on a beat trace a
product's journey goes from ``mul`` through L1 into ``sum``. The cycle counts
are the first data of the paper's claim C3: behind a one-lane accumulator a
wider multiplier buys little.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.flow import run_flow
from snax_forge.report import reports_of
from snax_forge.report.run import ChainRow
from snax_forge.viz import api
from tests.design.helpers import FIXTURES, RECIPE, REPO, SMALL16

from .helpers import PLAIN as VECADD

DOT = REPO / "recipes" / "dot.json"
SERIAL = REPO / "recipes" / "dot_serial.json"
PLAIN = FIXTURES / "dot.snaxdfg"  # the import of the dot kernel (DFG3)

# (recipe, W) -> cycles on small16 at N = 64
CYCLES = {
    (DOT, 4): 99,
    (DOT, 8): 87,
    (DOT, 16): 79,
    (DOT, 1): 219,
    (SERIAL, 4): 147,
    (SERIAL, 8): 143,
    (SERIAL, 1): 219,
}


def _flow(base: Path, recipe: Path, w: int, level: str = "task"):
    out = base / f"{recipe.stem}_W{w}"
    return run_flow(
        recipe, SMALL16, recipe_sets={"W": w}, graph_path=PLAIN, out=out, trace_level=level
    )


@pytest.fixture(scope="module")
def flows(tmp_path_factory):
    base = tmp_path_factory.mktemp("dot")
    return {(r, w): _flow(base, r, w, "beat" if (r, w) == (DOT, 4) else "task") for r, w in CYCLES}


# =============================================================================
# The runs (the acceptance)
# =============================================================================


@pytest.mark.parametrize(("recipe", "w"), list(CYCLES), ids=[f"{r.stem}_W{w}" for r, w in CYCLES])
def test_every_run_is_right_at_its_cycles(flows, recipe, w):
    f = flows[(recipe, w)]
    assert f.passed
    assert f.check["containers"]["out"]["reference"] and f.check["containers"]["out"]["ref1"]
    assert f.result.total_cycles == CYCLES[(recipe, w)]


def test_dot_from_the_kernel(tmp_path):
    """The whole path from kernels/polybench/dot.py, as `pixi run flow recipes/dot.json` runs it."""
    f = run_flow(DOT, SMALL16, out=tmp_path / "dot")
    assert f.passed and f.result.total_cycles == 99


def test_a_wider_multiplier_buys_little_behind_one_lane(flows):
    """C3's first data: 4 -> 8 multiplier lanes save 4 cycles behind the one-lane accumulator,
    12 when the accumulator widens with it."""
    cycles = {k: f.result.total_cycles for k, f in flows.items()}
    serial = cycles[(SERIAL, 4)] - cycles[(SERIAL, 8)]
    both = cycles[(DOT, 4)] - cycles[(DOT, 8)]
    assert (serial, both) == (4, 12)


# =============================================================================
# The chain: one wait between the accelerators, named in run.md
# =============================================================================


def test_the_task_list_waits_once_between_the_accelerators(flows):
    f = flows[(DOT, 4)]
    names = [s.task_name for s in f.tasks.steps if s.op == "configure"]
    assert names == [
        "load_A", "load_B", "mult_mul_a", "mult_mul_b", "mult_mul_out", "mult_mul",
        "sum_sum_a", "sum_sum_out", "sum_sum", "store_out",
    ]  # fmt: skip
    assert f.tasks.configured()["sum_sum_a"].after == ["mult_mul_out"]


def test_run_md_names_the_chaining_wait(flows):
    f = flows[(DOT, 4)]
    run = reports_of(f.out).run
    assert run.controller.chains == [ChainRow("mul", "sum", "mul_out", 17, ["sum_a"])]
    waits = [(w.block, w.cycles, w.then) for w in run.controller.waits]
    assert waits == [
        ("dma", 1, "load_B"),
        ("dma", 1, "mult_mul_a"),
        ("mul_out", 17, "sum_sum_a"),
        ("sum_out", 9, "store_out"),
        ("dma", 5, None),
    ]
    text = (f.out / "report" / "run.md").read_text()
    assert "- mul → sum: 17 cycles waiting on mul_out, which sum_a reads" in text


def test_vecadd_has_no_chaining_wait(tmp_path):
    f = run_flow(RECIPE, SMALL16, graph_path=VECADD, out=tmp_path / "vecadd")
    run = reports_of(f.out).run
    assert run.controller.chains == []
    assert "Chaining waits" not in (f.out / "report" / "run.md").read_text()


# =============================================================================
# What the design report says: the drain, the padding
# =============================================================================


def test_the_serial_accumulator_declares_its_drain(flows):
    """drain 1 in the cluster file and the design report; dot's one sum per task does not pay it."""
    f = flows[(SERIAL, 4)]
    cluster = json.loads((f.out / "cluster.json").read_text())
    (sum_,) = [c for c in cluster["components"] if c["name"] == "sum"]
    assert sum_["params"] == {
        "lanes": 1,
        "lanes_out": 1,
        "op": "add",
        "latency": 1,
        "ii": 1,
        "drain": 1,
    }
    design = reports_of(f.out).design
    assert [(a.instance, a.implementation, a.drain) for a in design.accelerators] == [
        ("mul", "chisel_tiled_spatial", 0),
        ("sum", "chisel_accumulator", 1),
    ]


def test_out_is_moved_as_one_padded_beat(flows):
    design = reports_of(flows[(DOT, 4)].out).design
    assert (
        "out: 8 B, moved by the DMA as 1 whole 64-byte beat; bytes 1032–1088 are kept free in "
        "L1 and L2."
    ) in design.notes


# =============================================================================
# Data movement on the beat trace: through L1 from one accelerator to the next
# =============================================================================


def test_a_product_goes_from_mul_through_l1_into_sum(flows):
    rv = api.load_run(flows[(DOT, 4)].out / "run", "dot")
    j = api.journey_view(rv, "tmp0", 5)
    assert [(h["act"], h["owner"]) for h in j["hops"]] == [
        ("write", "mul_out"),
        ("read", "sum_a"),
        ("data back", "sum_a"),
    ]
    assert [(x["acc"], x["role"], x["inputs"]) for x in j["firings"]] == [
        ("mul", "produced", {"a": ["A[5]"], "b": ["B[5]"]}),
        ("sum", "consumed", {"a": ["tmp0[5]"]}),  # its lane of sum's firing 1
    ]
    write, read = j["hops"][0]["t"], j["hops"][1]["t"]
    assert read > write  # it waits in L1 for the chaining wait to end


def test_the_sum_comes_from_all_sixteen_firings(flows):
    rv = api.load_run(flows[(DOT, 4)].out / "run", "dot")
    j = api.journey_view(rv, "out", 0)
    firings = [x for x in j["firings"] if x["role"] == "produced"]
    assert [x["n"] for x in firings] == list(range(16))
    assert {x["acc"] for x in firings} == {"sum"}
    assert [h["act"] for h in j["hops"]][-1] == "write" and j["hops"][-1]["mem"] == "l2"
