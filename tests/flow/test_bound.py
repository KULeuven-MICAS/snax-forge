"""The flow from a bound graph (FLOW2, D108): the path without its sandbox stage.

1. the last step of a recipe flow runs as the recipe did, for vecadd and dot
2. a checked-in bound graph runs on its own; what the folder, the summary, the
   log and the design report hold without a recipe
3. what stops it: a graph that is not bound, one edited to compute something
   else, a kernel the graph's name does not give
4. names, and a folder an earlier recipe flow used
5. the command line: the suffix picks the form
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.design import DesignError
from snax_forge.flow import FlowError, bound_name, run_bound, run_flow, summary
from snax_forge.flow import __main__ as cli
from snax_forge.report import reports_of
from tests.design.helpers import BOUND, FIXTURES, RECIPE, REPO, SMALL16

from .helpers import PINS, PLAIN

DOT = REPO / "recipes" / "dot.json"
DOT_PLAIN = FIXTURES / "dot.snaxdfg"
DOT_BOUND = FIXTURES / "dot_accelerated.snaxdfg"


def files(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def edited(tmp_path: Path, name: str, edit) -> Path:
    """``BOUND`` (vecadd on ``elementwise_add``, W = 4) edited as a dict, as a person would."""
    d = json.loads(BOUND.read_text())
    edit(d)
    path = tmp_path / name
    path.write_text(json.dumps(d))
    return path


# =============================================================================
# 1. The acceptance: the last step of a recipe flow
# =============================================================================


@pytest.mark.parametrize(
    ("recipe", "plain", "kw", "cycles"),
    [(RECIPE, PLAIN, {"memory_sets": PINS}, 77), (DOT, DOT_PLAIN, {}, 99)],
    ids=["vecadd_pinned", "dot"],
)
def test_the_last_step_runs_as_the_recipe_did(tmp_path, recipe, plain, kw, cycles):
    """Same cluster file, task list, scenario, run and run report, byte for byte, in folders
    of one name; only the design point's ``graph_from`` differs, and there is no sandbox/."""
    a = run_flow(recipe, SMALL16, graph_path=plain, out=tmp_path / "a" / "x", name="x", **kw)
    b = run_bound(a.steps[-1], SMALL16, out=tmp_path / "b" / "x", name="x", **kw)
    assert b.passed and a.result.total_cycles == b.result.total_cycles == cycles
    assert b.check == a.check
    same = files(a.out) - {"flow.log", "report/design.md", "design/design_point.json"}
    same = {f for f in same if not f.startswith("sandbox/")}
    assert files(b.out) == same | {"flow.log", "report/design.md", "design/design_point.json"}
    for f in sorted(same):
        assert (b.out / f).read_bytes() == (a.out / f).read_bytes(), f
    pa, pb = (json.loads((f.out / "design" / "design_point.json").read_text()) for f in (a, b))
    assert pa.pop("graph_from") == pb.pop("graph_from") == str(a.steps[-1]) and pa == pb


# =============================================================================
# 2. A bound graph on its own
# =============================================================================


def test_a_checked_in_bound_graph_runs(tmp_path):
    f = run_bound(DOT_BOUND, SMALL16, out=tmp_path / "f")
    assert f.passed and f.result.total_cycles == 99  # recipes/dot.json at W = 4
    assert (f.name, f.point.name, f.recipe, f.steps) == (
        "dot_accelerated",
        "dot_accelerated",
        None,
        [],
    )
    assert f.check["kernel"] == "dot" and f.point.graph_from == str(DOT_BOUND)
    assert not (tmp_path / "f" / "sandbox").exists()
    assert list(f.times) == ["design", "lower", "scenario", "run", "check", "report"]


def test_the_summary_and_the_log_name_the_graph(tmp_path):
    f = run_bound(DOT_BOUND, SMALL16, out=tmp_path / "f")
    head, sandbox, *_ = summary(f).splitlines()
    assert head.startswith("flow dot_accelerated (graph ") and "dot_accelerated.snaxdfg; " in head
    assert "; kernel dot; N=64) on small16 -> " in head
    assert sandbox == "  sandbox   skipped: the bound graph is the input (D108)"
    log = (tmp_path / "f" / "flow.log").read_text()
    assert (
        log.startswith(summary(f)) and "  graph     graph.load, graph.symbols, graph.unbound" in log
    )
    assert "\n  design    " in log.partition("stage times")[2]
    assert "\n  sandbox   " not in log.partition("stage times")[2]


def test_the_design_report_has_the_graph_in_place_of_the_recipe(tmp_path):
    f = run_bound(DOT_BOUND, SMALL16, out=tmp_path / "f")
    h = reports_of(f.out).design.header
    assert (h.kernel, h.graph_from, h.params, h.symbols) == ("dot", str(DOT_BOUND), {}, {"N": 64})
    md = (f.out / "report" / "design.md").read_text()
    assert f"| Kernel | dot |\n| Graph | {DOT_BOUND} |\n| Symbols | N=64 |" in md
    assert "| Recipe |" not in md
    # a recipe flow's header is as it was
    r = run_flow(DOT, SMALL16, graph_path=DOT_PLAIN, out=tmp_path / "r")
    md = (r.out / "report" / "design.md").read_text()
    assert "| Kernel | dot |\n| Recipe | dot (W=4) |\n| Symbols | N=64 |" in md
    assert "| Graph |" not in md and reports_of(r.out).design.header.graph_from == ""


# =============================================================================
# 3. What stops it
# =============================================================================


def test_a_graph_that_is_not_bound_stops_at_the_design_step(tmp_path):
    """The plain import: its symbol and its tasklet named at once (D85), nothing lowered or run."""
    with pytest.raises(DesignError) as e:
        run_bound(DOT_PLAIN, SMALL16, out=tmp_path / "f")
    assert e.value.codes == ["graph.symbols", "graph.unbound", "graph.unbound"]
    out = tmp_path / "f"
    assert files(out) == {"flow.log"}
    log = (out / "flow.log").read_text()
    assert log.startswith("flow dot FAILED\n") and "[graph.unbound] tasklet mult" in log


def test_a_graph_edited_to_compute_something_else_fails_the_check(tmp_path):
    """vecadd's adder swapped for the multiplier: every design check passes, the output is not
    the kernel's."""

    def swap(d):
        attrs = d["body"][0]["body"][0]["attrs"]
        attrs.update(brm="elementwise_mul", code="out = a * b", params={"W": 4, "op": "mul"})

    f = run_bound(edited(tmp_path, "vecadd.snaxdfg", swap), SMALL16, out=tmp_path / "f")
    c = f.check["containers"]["C"]
    assert not f.passed and not c["reference"] and c["mismatches"] > 0
    assert "  check     C (64 elements, from l2) DIFFERS: reference False" in summary(f)


def test_the_kernel_is_the_graphs_name_unless_given(tmp_path):
    path = edited(tmp_path, "mine.snaxdfg", lambda d: d.update(name="mine"))
    with pytest.raises(FlowError) as e:
        run_bound(path, SMALL16, out=tmp_path / "f")
    msg = str(e.value)
    assert msg.startswith(
        f"kernel (the name of {path}; --kernel K names it): unknown kernel 'mine'"
    )
    assert "'vecadd'" in msg and (tmp_path / "f" / "flow.log").read_text().startswith(
        "flow mine FAILED"
    )
    f = run_bound(path, SMALL16, kernel="vecadd", out=tmp_path / "f")
    assert f.passed and f.result.total_cycles == 85 and f.check["kernel"] == "vecadd"
    with pytest.raises(FlowError, match=r"kernel \(--kernel\): unknown kernel 'nope'"):
        run_bound(path, SMALL16, kernel="nope", out=tmp_path / "f")


# =============================================================================
# 4. Names and reused folders
# =============================================================================


def test_bound_name():
    """The design step's name for the graph, then every --set as the recipe form appends them."""
    assert bound_name("out/sandbox/vecadd_w8/2_bind.snaxdfg") == "vecadd_w8"
    assert bound_name("tests/dfg/fixtures/dot_accelerated.snaxdfg") == "dot_accelerated"
    # a flow's own sandbox/ gives the flow folder, not "sandbox"
    assert bound_name("out/flow/dot_W8/sandbox/4_bind.snaxdfg") == "dot_W8"
    assert bound_name("out/sandbox/sandbox.snaxdfg") == "sandbox"
    sets = bound_name("my/dot.snaxdfg", [("l1.n_banks", 32)], [("B.l1.base", 576)])
    assert sets == "dot_l1.n_banks32_B.l1.base576"


def test_the_design_point_keeps_the_graphs_name_the_folder_every_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = run_bound(BOUND, SMALL16, memory_sets=PINS, out=tmp_path / "f")
    assert f.name == "vecadd_accelerated_B.l1.base576_C.l1.base1152"
    assert f.point.name == "vecadd_accelerated" and f.result.total_cycles == 77


def test_an_earlier_recipes_sandbox_does_not_stay_in_the_folder(tmp_path):
    """The folder of a recipe flow, used again for another bound graph: its recipe and steps go,
    so the report cannot name a recipe the run did not come from."""
    out = tmp_path / "f"
    run_flow(RECIPE, SMALL16, graph_path=PLAIN, out=out)
    assert (out / "sandbox" / "recipe.json").is_file()
    run_bound(BOUND, SMALL16, out=out)
    assert not (out / "sandbox").exists()
    assert reports_of(out).design.header.graph_from == str(BOUND)


def test_a_graph_in_the_folders_own_sandbox_is_kept(tmp_path):
    """A recipe flow's last step run again in its own folder: the same name, the sandbox and
    its recipe stay, and the report still names the recipe."""
    out = tmp_path / "vecadd"
    a = run_flow(RECIPE, SMALL16, graph_path=PLAIN, out=out)
    b = run_bound(a.steps[-1], SMALL16, out=out)
    assert (b.name, b.point.name) == ("vecadd", "vecadd") and b.result.total_cycles == 85
    assert (out / "sandbox" / "recipe.json").is_file() and a.steps[-1].is_file()
    assert reports_of(out).design.header.recipe == "vecadd"
    assert "| Recipe | vecadd (W=4) |" in (out / "report" / "design.md").read_text()


# =============================================================================
# 5. Command line
# =============================================================================


def main(*argv) -> int:
    return cli.main([str(a) for a in argv])


def test_cli_takes_a_bound_graph_by_its_suffix(tmp_path, capsys):
    args = [DOT_BOUND, "--platform", SMALL16, "--out", tmp_path / "f"]
    assert main(*args) == 0
    out = capsys.readouterr().out
    assert out.startswith("flow dot_accelerated (graph ") and "run       99 cycles" in out
    assert "check     out (1 elements, from l2) equals the dot reference and REF1" in out
    assert (tmp_path / "f" / "flow.log").read_text().startswith(out)
    # platform and memory --set as with a recipe
    args = [BOUND, "--platform", SMALL16, "--out", tmp_path / "g"]
    assert main(*args, "--set", "memory.B.l1.base=576", "--set", "memory.C.l1.base=1152") == 0
    out = capsys.readouterr().out
    assert "flow vecadd_accelerated_B.l1.base576_C.l1.base1152 (graph " in out
    assert "run       77 cycles" in out


def test_cli_fails_as_the_recipe_form_does(tmp_path, capsys):
    assert main(DOT_PLAIN, "--platform", SMALL16, "--out", tmp_path / "f") == 1
    assert "[graph.unbound] tasklet mult" in capsys.readouterr().err
    path = edited(tmp_path, "mine.snaxdfg", lambda d: d.update(name="mine"))
    assert main(path, "--platform", SMALL16, "--out", tmp_path / "g") == 1
    assert "unknown kernel 'mine'" in capsys.readouterr().err
    assert main(path, "--platform", SMALL16, "--out", tmp_path / "g", "--kernel", "vecadd") == 0


@pytest.mark.parametrize(
    ("first", "extra", "says"),
    [
        (BOUND, ["--set", "W=8"], "is a bound graph, it has no recipe params"),
        (BOUND, ["--graph", PLAIN], "is a bound graph already"),
        (RECIPE, ["--kernel", "vecadd"], "is a recipe, which names its kernel"),
    ],
    ids=["recipe_param", "graph", "kernel_with_recipe"],
)
def test_cli_refuses_what_belongs_to_the_other_form(tmp_path, capsys, first, extra, says):
    with pytest.raises(SystemExit) as e:
        main(first, "--platform", SMALL16, "--out", tmp_path / "f", *extra)
    assert e.value.code == 2 and says in capsys.readouterr().err
    assert not (tmp_path / "f").exists()
