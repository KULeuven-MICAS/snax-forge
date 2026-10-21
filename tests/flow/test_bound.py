"""The flow from a bound graph (D108): the path without its sandbox stage, SNAX-GVSoC
stubbed (helpers.stub).

1. the last step of a recipe flow runs as the recipe did
2. a checked-in bound graph runs on its own; what the folder, the summary and
   the log hold without a recipe
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
from tests.design.helpers import BOUND, RECIPE, SNAX_ALU

from .helpers import PLAIN, built, stub

FIFO = [("streamers.default.fifo_depth", 4)]


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


def test_the_last_step_runs_as_the_recipe_did(tmp_path, monkeypatch):
    """Same task list, generated folder and run, byte for byte, in folders of one name; only
    the design point's ``graph_from`` differs, and there is no sandbox/."""
    fake = stub(monkeypatch, tmp_path)
    a = run_flow(RECIPE, SNAX_ALU, graph_path=PLAIN, out=tmp_path / "a" / "x", name="x")
    b = run_bound(a.steps[-1], SNAX_ALU, out=tmp_path / "b" / "x", name="x")
    assert b.passed and a.total_cycles == b.total_cycles and b.check == a.check
    keep = {"flow.log", "design/design_point.json", "report/design.md"}  # graph, not recipe
    same = {f for f in files(a.out) - keep - {"build.log"} if not f.startswith("sandbox/")}
    assert files(b.out) == same | keep
    for f in sorted(same):
        assert (b.out / f).read_bytes() == (a.out / f).read_bytes(), f
    pa, pb = (json.loads((f.out / "design" / "design_point.json").read_text()) for f in (a, b))
    assert pa.pop("graph_from") == pb.pop("graph_from") == str(a.steps[-1]) and pa == pb
    assert [g for g, _ in fake.calls] == ["build", "sw", "run", "sw", "run"]  # built once


# =============================================================================
# 2. A bound graph on its own
# =============================================================================


def test_a_checked_in_bound_graph_runs(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    f = run_bound(BOUND, SNAX_ALU, out=tmp_path / "f")
    assert f.passed and (f.name, f.point.name, f.recipe, f.steps) == (
        "vecadd_accelerated", "vecadd_accelerated", None, []
    )  # fmt: skip
    assert f.check["kernel"] == "vecadd" and f.point.graph_from == str(BOUND)
    assert not (tmp_path / "f" / "sandbox").exists()
    assert list(f.times) == [
        "design",
        "lower",
        "gen",
        "build",
        "sw",
        "gvsoc",
        "check",
        "convert",
        "report",
    ]


def test_the_summary_and_the_log_name_the_graph(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    f = run_bound(BOUND, SNAX_ALU, out=tmp_path / "f")
    head, sandbox, *_ = summary(f).splitlines()
    assert head.startswith("flow vecadd_accelerated (graph ")
    assert "vecadd_accelerated.snaxdfg; kernel vecadd; N=64) on snax_alu -> " in head
    assert sandbox == "  sandbox   skipped: the bound graph is the input (D108)"
    log = (tmp_path / "f" / "flow.log").read_text()
    assert log.startswith(summary(f)) and "  graph     graph.load, graph.symbols" in log
    assert "\n  design    " in log.partition("stage times")[2]
    assert "\n  sandbox   " not in log.partition("stage times")[2]


# =============================================================================
# 3. What stops it
# =============================================================================


def test_a_graph_that_is_not_bound_stops_at_the_design_step(tmp_path, monkeypatch):
    """The plain import: its symbol and its tasklet named at once (D85), nothing lowered or run."""
    fake = stub(monkeypatch, tmp_path)
    with pytest.raises(DesignError) as e:
        run_bound(PLAIN, SNAX_ALU, out=tmp_path / "f")
    assert e.value.codes == ["graph.symbols", "graph.unbound"]
    out = tmp_path / "f"
    assert files(out) == {"flow.log"} and fake.calls == []
    log = (out / "flow.log").read_text()
    assert log.startswith("flow vecadd FAILED\n") and "[graph.unbound] tasklet" in log


def test_a_graph_edited_to_compute_something_else_stops_at_gen(tmp_path, monkeypatch):
    """vecadd's adder swapped for the multiplier: every design check passes, and the goldens
    would be the multiplier's; gen's reference check stops it before anything is built."""

    def swap(d):
        attrs = d["body"][0]["body"][0]["attrs"]
        attrs.update(brm="elementwise_mul", code="out = a * b", params={"W": 4, "op": "mul"})

    fake = stub(monkeypatch, tmp_path)
    with pytest.raises(FlowError) as e:
        run_bound(edited(tmp_path, "vecadd.snaxdfg", swap), SNAX_ALU, out=tmp_path / "f")
    assert str(e.value).startswith(
        "gen: the graph's reference executor differs from the vecadd reference: C in "
    )
    assert not (tmp_path / "f" / "gen").exists() and fake.calls == []


def test_the_kernel_is_the_graphs_name_unless_given(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    path = edited(tmp_path, "mine.snaxdfg", lambda d: d.update(name="mine"))
    with pytest.raises(FlowError) as e:
        run_bound(path, SNAX_ALU, out=tmp_path / "f")
    msg = str(e.value)
    assert msg.startswith(
        f"kernel (the name of {path}; --kernel K names it): unknown kernel 'mine'"
    )
    assert "'vecadd'" in msg
    assert (tmp_path / "f" / "flow.log").read_text().startswith("flow mine FAILED")
    f = run_bound(path, SNAX_ALU, kernel="vecadd", out=tmp_path / "f")
    assert f.passed and f.check["kernel"] == "vecadd"
    with pytest.raises(FlowError, match=r"kernel \(--kernel\): unknown kernel 'nope'"):
        run_bound(path, SNAX_ALU, kernel="nope", out=tmp_path / "f")


# =============================================================================
# 4. Names and reused folders
# =============================================================================


def test_bound_name():
    """The design step's name for the graph, then every --set as the recipe form appends them."""
    assert bound_name("out/sandbox/vecadd_w8/2_bind.snaxdfg") == "vecadd_w8"
    assert bound_name("tests/dfg/fixtures/vecadd_accelerated.snaxdfg") == "vecadd_accelerated"
    # a flow's own sandbox/ gives the flow folder, not "sandbox"
    assert bound_name("out/flow/vecadd_W8/sandbox/2_bind.snaxdfg") == "vecadd_W8"
    assert bound_name("out/sandbox/sandbox.snaxdfg") == "sandbox"
    sets = bound_name("my/vsum.snaxdfg", [("l1.n_banks", 32)], [("B.l1.base", 576)])
    assert sets == "vsum_l1.n_banks32_B.l1.base576"


def test_the_design_point_keeps_the_graphs_name_the_folder_every_set(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    f = run_bound(BOUND, SNAX_ALU, platform_sets=FIFO, out=tmp_path / "f")
    assert f.name == "vecadd_accelerated_streamers.default.fifo_depth4"
    assert f.point.name == "vecadd_accelerated" and f.passed


def test_an_earlier_recipes_sandbox_does_not_stay_in_the_folder(tmp_path, monkeypatch):
    """The folder of a recipe flow, used again for another bound graph: its recipe and steps go,
    so nothing names a recipe the run did not come from."""
    stub(monkeypatch, tmp_path)
    out = tmp_path / "f"
    run_flow(RECIPE, SNAX_ALU, graph_path=PLAIN, out=out)
    assert (out / "sandbox" / "recipe.json").is_file()
    run_bound(BOUND, SNAX_ALU, out=out)
    assert not (out / "sandbox").exists()


def test_a_graph_in_the_folders_own_sandbox_is_kept(tmp_path, monkeypatch):
    """A recipe flow's last step run again in its own folder: the same name, the sandbox and
    its recipe stay."""
    stub(monkeypatch, tmp_path)
    out = tmp_path / "vecadd"
    a = run_flow(RECIPE, SNAX_ALU, graph_path=PLAIN, out=out)
    b = run_bound(a.steps[-1], SNAX_ALU, out=out)
    assert (b.name, b.point.name) == ("vecadd", "vecadd") and b.passed
    assert (out / "sandbox" / "recipe.json").is_file() and a.steps[-1].is_file()


# =============================================================================
# 5. Command line
# =============================================================================


def main(*argv) -> int:
    return cli.main([str(a) for a in argv])


def test_cli_takes_a_bound_graph_by_its_suffix(tmp_path, monkeypatch, capsys):
    built(stub(monkeypatch, tmp_path))  # a build would print its lines first
    assert main(BOUND, "--platform", SNAX_ALU, "--out", tmp_path / "f") == 0
    out = capsys.readouterr().out
    assert out.startswith("flow vecadd_accelerated (graph ") and "  check     passed: " in out
    assert (tmp_path / "f" / "flow.log").read_text().startswith(out)
    # platform --set as with a recipe
    args = [BOUND, "--platform", SNAX_ALU, "--out", tmp_path / "g"]
    assert main(*args, "--set", "platform.streamers.default.fifo_depth=4") == 0
    assert (
        "flow vecadd_accelerated_streamers.default.fifo_depth4 (graph " in capsys.readouterr().out
    )


def test_cli_fails_as_the_recipe_form_does(tmp_path, monkeypatch, capsys):
    stub(monkeypatch, tmp_path)
    assert main(PLAIN, "--platform", SNAX_ALU, "--out", tmp_path / "f") == 1
    assert "[graph.unbound] tasklet" in capsys.readouterr().err
    path = edited(tmp_path, "mine.snaxdfg", lambda d: d.update(name="mine"))
    assert main(path, "--platform", SNAX_ALU, "--out", tmp_path / "g") == 1
    assert "unknown kernel 'mine'" in capsys.readouterr().err
    assert main(path, "--platform", SNAX_ALU, "--out", tmp_path / "g", "--kernel", "vecadd") == 0


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
        main(first, "--platform", SNAX_ALU, "--out", tmp_path / "f", *extra)
    assert e.value.code == 2 and says in capsys.readouterr().err
    assert not (tmp_path / "f").exists()
