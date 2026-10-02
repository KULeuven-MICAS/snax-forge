"""EX2: every number and name examples/dot/README.md states, checked on what its commands make.

The README's two sandbox commands and two flow commands are run as written,
from a temporary folder that stands for the repo root: the recipes and the
platform are the checked-in ones, and ``out/`` is the temporary one, so the
bound graphs are ``out/sandbox/<name>/4_bind.snaxdfg`` as the README has
them. Each quoted table is marked ``<!-- excerpt: FOLDER/report/FILE -->``
and must be part of that report line for line. What the README says the DFG
viewer shows is checked on the viewer's API (the drawing itself is checked
by eye, D55); the claims made in prose are checked below in its order.
"""

from __future__ import annotations

import json
import shlex

import pytest

from snax_forge.flow import __main__ as flow_cli
from snax_forge.report import reports_of
from snax_forge.sandbox import __main__ as sandbox_cli
from snax_forge.viz import api
from snax_forge.viz.dfg import api as dfg
from tests.design.helpers import REPO

from .helpers import excerpts_hold

README = REPO / "examples" / "dot" / "README.md"
NAMES = ("dot", "dot_serial")
SANDBOX = {n: f"pixi run sandbox recipes/{n}.json" for n in NAMES}
FLOW = {
    n: f"pixi run flow out/sandbox/{n}/4_bind.snaxdfg --platform platforms/small16.json"
    for n in NAMES
}
STEPS = ["0_input", "1_split_map", "2_bind", "3_split_map", "4_bind"]


def _args(cmd: str) -> list[str]:
    """The arguments after ``pixi run <tool>``, the checked-in inputs by their full path."""
    return [
        str(REPO / a) if a.startswith(("recipes/", "platforms/")) else a
        for a in shlex.split(cmd)[3:]
    ]


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """The README's commands, run in order; (root, name -> RunView of its flow)."""
    root = tmp_path_factory.mktemp("ex2")
    runs = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        for n in NAMES:
            assert sandbox_cli.main([*_args(SANDBOX[n]), "--out", f"out/sandbox/{n}"]) == 0
            assert flow_cli.main([*_args(FLOW[n]), "--out", f"out/flow/{n}"]) == 0
            runs[n] = api.load_run(root / "out" / "flow" / n / "run", n)
    return root, runs


def text() -> str:
    return README.read_text()


def view(root, name: str, step: str) -> dict:
    e = dfg.GraphSet([root / "out" / "sandbox" / name]).get(step)
    return dfg.graph_view(e)


def node(v: dict, node_id: str, kind: str | None = None) -> dict:
    """The node ``node_id`` (of ``kind``, when a map and its tasklet share the id) of a view."""

    def walk(n):
        yield n
        for c in n.get("body") or []:
            yield from walk(c)

    found = [
        n
        for row in v["rows"]
        if row["kind"] == "node"
        for n in walk(row["node"])
        if n["id"] == node_id and kind in (None, n["kind"])
    ]
    (n,) = found
    return n


def edges(v: dict) -> list[str]:
    return [e["text"] for e in v["edges"]]


def task(rv, block: str) -> dict:
    (t,) = api.tasks(rv)[block]
    return t


# =============================================================================
# The commands, the folders and the excerpts
# =============================================================================


def test_the_readme_gives_the_commands_and_folders():
    t = text()
    for n in NAMES:
        assert f"\n{SANDBOX[n]}\n" in t and f"\n{FLOW[n]}\n" in t
        assert f"`out/sandbox/{n}/`" in t and f"`out/flow/{n}/`" in t
    assert "\npixi run view-dfg out/sandbox/dot/\n" in t
    pair = "pixi run view-dfg out/sandbox/dot/4_bind.snaxdfg out/sandbox/dot_serial/4_bind.snaxdfg"
    assert f"\n{pair}\n" in t
    assert "\npixi run view out/flow/dot/run out/flow/dot_serial/run\n" in t


def test_the_flows_are_from_the_bound_graphs_into_the_folders_the_readme_names(made):
    root, _ = made
    for n in NAMES:
        log = (root / "out" / "flow" / n / "flow.log").read_text()
        head = f"flow {n} (graph out/sandbox/{n}/4_bind.snaxdfg; kernel dot; N=64) on small16"
        assert log.startswith(head)
        assert "  sandbox   skipped: the bound graph is the input (D108)" in log
        assert not (root / "out" / "flow" / n / "sandbox").exists()


def test_every_excerpt_is_a_part_of_its_report(made):
    root, _ = made
    assert excerpts_hold(text(), root / "out" / "flow") == 12


# =============================================================================
# 1. The graphs
# =============================================================================


def test_each_recipe_writes_the_five_steps_the_table_lists(made):
    root, _ = made
    for n in NAMES:
        files = sorted(p.stem for p in (root / "out" / "sandbox" / n).glob("*.snaxdfg"))
        assert files == STEPS
    for s in STEPS:
        assert f"| `{s}.snaxdfg` |" in text()
    v = view(root, "dot", "0_input")
    assert v["symbols"] == {"N": 64}
    # 4_bind: every tasklet is an accelerator
    for n in NAMES:
        last = json.loads((root / "out" / "sandbox" / n / "4_bind.snaxdfg").read_text())
        kinds = [b["kind"] for top in last["body"] for b in top["body"]]
        assert kinds == ["accelerated", "accelerated"]


def test_the_recipes_differ_in_two_values_of_the_last_two_steps():
    a, b = (json.loads((REPO / "recipes" / f"{n}.json").read_text()) for n in NAMES)
    assert a["steps"][:2] == b["steps"][:2] and a["params"] == b["params"]
    assert (a["steps"][2]["params"]["factor"], b["steps"][2]["params"]["factor"]) == ("W", 1)
    impl = [r["steps"][3]["params"].pop("implementation") for r in (a, b)]
    assert impl == ["chisel_adder_tree", "chisel_accumulator"]
    a["steps"][2]["params"].pop("factor"), b["steps"][2]["params"].pop("factor")
    assert a["steps"] == b["steps"]


# =============================================================================
# 2. What the DFG viewer shows
# =============================================================================


def test_the_viewer_names_the_steps_after_their_files(made):
    root, _ = made
    gs = dfg.GraphSet([root / "out" / "sandbox" / "dot"])
    assert [s["name"] for s in gs.summaries()] == STEPS
    assert "http://127.0.0.1:8766/" in text()


def test_the_input_graph_is_two_maps_with_a_transient_between(made):
    root, _ = made
    v = view(root, "dot", "0_input")
    tops = [r["node"] for r in v["rows"] if r["kind"] == "node"]
    assert [(n["id"], n["title"], n["iterations"]) for n in tops] == [
        ("mult_map", "i in 0:N", 64),
        ("sum_map", "i in 0:N", 64),
    ]
    assert node(v, "mult")["lines"] == ["out = in1 * in2"]
    assert node(v, "sum")["lines"] == ["out = in1"]
    assert v["containers"]["tmp0"]["transient"] and not v["containers"]["A"]["transient"]
    # tmp0 sits in the row between the two maps
    assert [r["boxes"] for r in v["rows"] if r["kind"] == "containers"][1] == ["tmp0@1"]
    assert edges(v)[-1] == "out[0] (reduce add)"


def test_the_split_and_the_bind_of_the_multiplier(made):
    root, _ = made
    v = view(root, "dot", "1_split_map")
    outer, inner = node(v, "mult_map"), node(v, "mult_map_s")
    assert (outer["title"], outer["iterations"], outer["loop_kind"]) == (
        "i_t in 0:N // 4",
        16,
        "temporal",
    )
    assert (inner["title"], inner["loop_kind"]) == ("i_s in 0:4", "spatial")
    v = view(root, "dot", "2_bind")
    mul = node(v, "mult")
    assert (mul["kind"], mul["title"]) == ("accelerated", "mul = elementwise_mul")
    assert mul["replaced"] == "mult_map_s › mult" and "`mult_map_s › mult`" in text()
    assert mul["heading"] == ["impl = chisel_tiled_spatial", "W = 4, op = mul"]
    assert "A[4 * i_t:4 * i_t + 4]" in edges(v)
    ids = {n["id"] for r in v["rows"] if r["kind"] == "node" for n in r["node"]["body"]}
    assert ids == {"mult", "sum"}  # the spatial loop is gone


def test_the_reduce_label_leaves_the_edge_when_sum_is_bound(made):
    root, _ = made
    assert any("(reduce add)" in e for e in edges(view(root, "dot", "3_split_map")))
    v = view(root, "dot", "4_bind")
    assert node(v, "sum")["title"] == "sum = accumulate"
    assert edges(v)[-1] == "out[0:1]" and not any("reduce" in e for e in edges(v))


def test_the_two_bound_graphs_side_by_side(made):
    root, _ = made
    files = [root / "out" / "sandbox" / n / "4_bind.snaxdfg" for n in NAMES]
    gs = dfg.GraphSet(files)
    assert [s["name"] for s in gs.summaries()] == ["4_bind", "4_bind-2"]
    a, b = (dfg.graph_view(gs.get(n)) for n in ("4_bind", "4_bind-2"))
    assert node(a, "mult_map") == node(b, "mult_map")  # the multiplier half is the same
    rows = {
        "map": [(node(v, "sum_map")["title"], node(v, "sum_map")["iterations"]) for v in (a, b)],
        "sum": [node(v, "sum")["heading"] for v in (a, b)],
        "edge": [node(v, "sum")["inputs"][0]["text"] for v in (a, b)],
    }
    assert rows["map"] == [("i_t_1 in 0:N // 4", 16), ("i_t_1 in 0:N // 1", 64)]
    assert rows["sum"] == [
        ["impl = chisel_adder_tree", "W = 4, op = add"],
        ["impl = chisel_accumulator", "W = 1, op = add"],
    ]
    assert rows["edge"] == ["tmp0[4 * i_t_1:4 * i_t_1 + 4]", "tmp0[i_t_1:i_t_1 + 1]"]
    t = text()
    assert "| `sum_map` | `i_t_1 in 0:N // 4`, `16×` | `i_t_1 in 0:N // 1`, `64×` |" in t
    assert (
        "| `sum` | `impl = chisel_adder_tree`, `W = 4, op = add` "
        "| `impl = chisel_accumulator`, `W = 1, op = add` |" in t
    )
    assert "| input of `sum` | `tmp0[4 * i_t_1:4 * i_t_1 + 4]` | `tmp0[i_t_1:i_t_1 + 1]` |" in t


# =============================================================================
# 3-5. The two runs
# =============================================================================


def test_the_comparison_table(made):
    root, rv = made
    prof = {n: v.outputs.profile.to_dict() for n, v in rv.items()}
    lanes, firings, cycles, wait, total = [], [], [], [], []
    for n in NAMES:
        cl = {c.name: c for c in rv[n].cluster.components}
        lanes.append(cl["sum_a"].config["n_ports"])
        t = task(rv[n], "sum")
        firings.append(prof[n]["accelerators"]["sum"]["firings"])
        cycles.append(t["done"] - t["start"])
        (chain,) = reports_of(root / "out" / "flow" / n).run.controller.chains
        assert (chain.src, chain.dst, chain.block, chain.readers) == (
            "mul",
            "sum",
            "mul_out",
            ["sum_a"],
        )
        wait.append(chain.cycles)
        total.append(rv[n].outputs.run["total_cycles"])
    assert (lanes, firings, cycles, total) == ([4, 1], [16, 64], [19, 67], [99, 147])
    assert wait == [17, 17]
    t = text()
    assert "| `sum` lanes | 4 | 1 |" in t and "| `sum` firings | 16 | 64 |" in t
    assert "| `sum_sum` task (cycles) | 19 | 67 |" in t and "| total (cycles) | 99 | 147 |" in t
    assert "| chaining wait `mul → sum` (cycles) | 17 | 17 |" in t


def test_the_runs_are_the_same_until_sum_starts_at_72(made):
    _, rv = made
    starts, done = [], []
    for n in NAMES:
        ts = [(b, t["start"], t["done"]) for b, rows in api.tasks(rv[n]).items() for t in rows]
        starts.append(sorted((b, s) for b, s, _ in ts if s <= 72))
        done.append(sorted((b, s, d) for b, s, d in ts if d < 72))
        assert task(rv[n], "sum")["start"] == 72
    # what started by cycle 72 started at the same cycle, and what was done by then is the same
    assert starts[0] == starts[1] and len(starts[0]) == 9
    assert done[0] == done[1] and len(done[0]) == 6


def test_the_48_cycles_are_the_48_more_firings(made):
    _, rv = made
    total = [rv[n].outputs.run["total_cycles"] for n in NAMES]
    cycles = [task(rv[n], "sum")["done"] - task(rv[n], "sum")["start"] for n in NAMES]
    prof = [rv[n].outputs.profile.to_dict()["accelerators"]["sum"] for n in NAMES]
    assert total[1] - total[0] == cycles[1] - cycles[0] == 48
    assert prof[1]["firings"] - prof[0]["firings"] == 48
    assert "The\n48 cycles between 99 and 147 are all in that one task: 48 more firings" in text()


def test_the_drain_is_declared_and_costs_nothing_here(made):
    _, rv = made
    drain = {}
    for n in NAMES:
        (acc,) = [c for c in rv[n].cluster.components if c.name == "sum"]
        drain[n] = acc.params.get("drain", 0)
        # firings + 3 cycles in both: the drain adds none
        t = task(rv[n], "sum")
        firings = rv[n].outputs.profile.to_dict()["accelerators"]["sum"]["firings"]
        assert t["done"] - t["start"] == firings + 3
    assert drain == {"dot": 0, "dot_serial": 1}


def test_the_multiplier_is_held_back_as_vecadd_is_in_loop1(made):
    _, rv = made
    for n in NAMES:
        held = rv[n].outputs.profile.streamers["mul_b"].cycles.get("stall_xbar", 0)
        assert held == 7  # acc_b's 7 cycles in examples/loop1
        regions = {(r.name, r.mem): r.base for r in rv[n].outputs.regions}
        assert (regions["A", "l1"], regions["B", "l1"]) == (0, 512)
