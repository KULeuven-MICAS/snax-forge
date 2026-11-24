"""Design point -> task list."""

from __future__ import annotations

import json

import pytest

from snax_forge.design import Design, DesignPoint, MemorySpec, Platform, run_checks, streamers
from snax_forge.design.streamers import Loop
from snax_forge.dfg import Graph, Memlet
from snax_forge.lower import (
    Layout,
    LowerError,
    TaskList,
    memlet_values,
    task_list,
)
from snax_forge.lower import __main__ as cli
from tests.brm.helpers import affine_simd
from tests.design.helpers import (
    SNAX_FORGE,
    bound_rowsum,
    bound_vecadd,
    design,
)

from .helpers import VECADD_TASKS

PINS = [("B.l1.base", 576), ("C.l1.base", 1152)]


def point(graph="vecadd_accelerated", memory=(), sets=None, edit_graph=None) -> DesignPoint:
    d = design(graph, sets=sets, edit_graph=edit_graph)
    d.memory_spec = MemorySpec().with_changes(list(memory))
    assert run_checks(d) == []
    return DesignPoint.of(d, "vecadd")


# =============================================================================
# 1. vecadd
# =============================================================================


def test_vecadd_on_snax_forge_gives_its_checked_in_task_list():
    """Equal to tests/lower/fixtures/vecadd_v1_tasks.json."""
    d = Design(Platform.load(SNAX_FORGE), bound_vecadd(4))
    assert run_checks(d) == []
    got = task_list(DesignPoint.of(d, "vecadd_v1"))
    want = TaskList.load(VECADD_TASKS)
    assert got == want and got.to_dict() == want.to_dict()


# =============================================================================
# 2. More than one loop, more than one node
# =============================================================================


@pytest.mark.parametrize(
    "name, reader, writer, n",
    [
        ("rowsum_v1", ([16, 8], [32, 512]), ([8], [8]), 128),
        ("rowsum_v3", ([16, 4], [8, 128]), ([4], [8]), 64),
    ],
)
def test_rowsum_restarts_every_row(name, reader, writer, n):
    """rowsum on snax_forge: A's reader runs the columns innermost (T = N / W
    beats) and the rows outer, two temporal loops; y's writer has one firing loop, the rows,
    one lane, one word per row; n = M * N / W beats and len = T = 16 (M sums)."""
    d = Design(Platform.load(SNAX_FORGE), bound_rowsum(name))
    assert run_checks(d) == []
    p = DesignPoint.of(d, name)
    assert {s.name: s.temporal_dims for s in p.streamers.values()} == {"acc_a": 2, "acc_out": 1}
    c = task_list(p).configured()
    for port, cont, (bounds, strides) in (("a", "A", reader), ("out", "y", writer)):
        assert c[f"sum_acc_{port}"].values == {
            "base": p.memory.layouts[cont]["l1"].base, "temporal_bounds": bounds,
            "temporal_strides": strides, "spatial_strides": [8],
        }  # fmt: skip
    assert c["sum_acc"].values == {"n": n, "T": 16}


# =============================================================================
# in place: a unit that writes the container it reads
# =============================================================================


def in_place(gd: dict, out_subset: str | None = None, tasklet_subset: str | None = None) -> None:
    """vecadd writing B in place of C: ``B = A + B``. With subsets, the write goes there
    instead (in a B of 2N), so it is no longer the read's elements."""
    del gd["containers"]["C"]
    if out_subset is not None:
        gd["containers"]["B"]["shape"] = ["2 * N"]
    (node,) = gd["body"][0]["body"]
    node["outputs"]["out"]["data"] = "B"
    (tasklet,) = node["attrs"]["replaced"]["body"]
    tasklet["outputs"]["out"]["data"] = "B"
    if out_subset is not None:
        node["outputs"]["out"]["subset"] = [out_subset]
        tasklet["outputs"]["out"]["subset"] = [tasklet_subset]


# =============================================================================
# dot: a multiplier chained into an accumulator through L1
# =============================================================================


# =============================================================================
# dot_v1_1: dot 1's two groups in one accelerator, a task each
# =============================================================================


def dot1() -> DesignPoint:
    from tests.gen.helpers import bound_dot1_enc

    d = Design(Platform.load(SNAX_FORGE), bound_dot1_enc())
    assert run_checks(d) == []
    return DesignPoint.of(d, "dot_v1_1")


# every temporal stride a word: a reader at stride 0 on its first loop would start (Reader.scala)
IDLE = {"base": 0, "temporal_bounds": [0], "temporal_strides": [8], "spatial_strides": [0]}


def test_dot_1_configures_every_mover_per_task_the_other_groups_idle():
    """Each group's task configures all five movers in the shell's order and starts them with
    its unit; the other group's are idle, every bound 0."""
    tl = task_list(dot1())
    c = tl.configured()
    starts = [s.tasks for s in tl.steps if s.op == "start" and len(s.tasks) > 1]
    assert starts == [
        ["mult_vmul_a", "mult_vmul_b", "mult_vmul_out", "mult_vsum_a", "mult_vsum_out", "mult_vmul"],
        ["sum_vmul_a", "sum_vmul_b", "sum_vmul_out", "sum_vsum_a", "sum_vsum_out", "sum_vsum"],
    ]  # fmt: skip
    for t in ("mult_vsum_a", "mult_vsum_out", "sum_vmul_a", "sum_vmul_b", "sum_vmul_out"):
        assert (c[t].values, c[t].after) == (IDLE, []), t
    assert c["sum_vsum_a"].after == ["mult_vmul_out"]  # tmp0 from mul, through L1
    assert c["sum_vsum_a"].values["base"] == c["mult_vmul_out"].values["base"]
    assert (c["mult_vmul"].values, c["sum_vsum"].values) == ({"n": 16}, {"n": 16, "T": 16})
    # one accelerator, dot: its groups follow through their data, no serial sync
    assert [s.task for s in tl.steps if s.op == "sync"] == ["store_out"]


def _backwards(g):
    """C written back to front: beat t at elements 60 - 4t .. 63 - 4t."""
    g["body"][0]["body"][0]["outputs"]["out"]["subset"] = ["60 - 4 * n_1_t:64 - 4 * n_1_t"]


def test_a_memlet_the_nest_disagrees_with_is_refused(monkeypatch):
    """An affine BRM: the addresses are right, the order is not the BRM's nest's."""
    monkeypatch.setattr(streamers, "load_brm", lambda name: affine_simd())
    with pytest.raises(LowerError, match="different orders"):
        task_list(point(edit_graph=_backwards))


# =============================================================================
# 3. memlet_values
# =============================================================================


def test_memlet_values_refuse_what_is_not_affine():
    with pytest.raises(LowerError, match="not affine"):
        memlet_values(Memlet("X", ["i * i"]), [Loop("m", "i", 0, 4, 1)],
                      Layout(0, (64,), (8,)), {}, "x")  # fmt: skip
    with pytest.raises(LowerError, match="changes length"):
        memlet_values(Memlet("X", ["i:2 * i"]), [Loop("m", "i", 0, 4, 1)],
                      Layout(0, (64,), (8,)), {}, "x")  # fmt: skip


# =============================================================================
# 4. Command line
# =============================================================================


def test_cli_writes_the_task_list_beside_the_point(tmp_path, capsys):
    point(memory=PINS).save(tmp_path / "design_point.json")
    assert cli.main(["tasks", str(tmp_path / "design_point.json")]) == 0
    assert TaskList.load(tmp_path / "tasks.json") == task_list(point(memory=PINS))
    out = capsys.readouterr().out
    assert "task list of vecadd (12 steps):" in out
    assert "sync      store_C (poll)" in out


def test_cli_rejects_a_bad_point(tmp_path, capsys):
    (tmp_path / "dp.json").write_text("{}")
    assert cli.main(["tasks", str(tmp_path / "dp.json")]) == 1
    assert "[point.keys]" in capsys.readouterr().err
    assert cli.main(["other"]) == 2


@pytest.mark.parametrize(
    "first",
    ["snax_forge.design", "snax_forge.lower", "snax_forge.design.memory",
     "snax_forge.lower.layout", "snax_forge.design.point", "snax_forge.lower.derive"],
)  # fmt: skip
def test_design_and_lower_import_in_any_order(first):
    """design.memory reads lower.layout and lower reads design: no import cycle either way."""
    import subprocess
    import sys

    code = f"import {first}; import snax_forge.design, snax_forge.lower"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


# =============================================================================
# The fold's seed: matvec_acc_v1's q, read once per job, written in place
# =============================================================================


def _seeded(name: str, edit=None) -> DesignPoint:
    from snax_forge.dfg import Graph
    from snax_forge.dfg.import_sdfg import import_kernel
    from snax_forge.sandbox import Recipe, apply_recipe
    from tests.helpers import recipe_path

    r = Recipe.load(recipe_path(name))
    g = apply_recipe(r, import_kernel(r.kernel))[-1].graph
    if edit is not None:
        d = g.to_dict()
        edit(d)
        g = Graph.from_dict(d)
    d = Design(Platform.load(SNAX_FORGE), g)
    assert run_checks(d) == []
    return DesignPoint.of(d, name)


@pytest.mark.parametrize(("attrs", "message"), [
    ({"perm": [0], "by": "dma"}, "a copy of rank 1; a transpose lowers at rank 2 only for now"),
    ({"perm": [0], "by": "core"}, "a copy by core has no task; use by=dma"),
])  # fmt: skip
def test_a_copy_lowers_by_dma_at_rank_2_only(attrs, message):
    """vecadd reading A through a copy A_T of rank 1: refused by name, with the fix."""
    d = bound_vecadd(4).to_dict()
    d["containers"]["A_T"] = dict(d["containers"]["A"], transient=True)
    d["body"] = json.loads(json.dumps(d["body"]).replace('"data": "A"', '"data": "A_T"'))
    d["body"].insert(0, {"id": "transpose_A", "kind": "transpose",
                         "inputs": {"src": {"data": "A", "subset": ["0:N"]}},
                         "outputs": {"dst": {"data": "A_T", "subset": ["0:N"]}},
                         "attrs": attrs})  # fmt: skip
    dp = Design(Platform.load(SNAX_FORGE), Graph.from_dict(d))
    assert run_checks(dp) == []
    with pytest.raises(LowerError, match=rf"^transpose_A: {message}$"):
        task_list(DesignPoint.of(dp, "vecadd"))
