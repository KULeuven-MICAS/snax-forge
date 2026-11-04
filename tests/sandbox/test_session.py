"""The editor's session (D55, C§12, sandbox/session.py), without HTTP: built step by step it ends
in the hand-written recipes' graphs; the start from a kernel's symbols, the cursor, the params (a
new one per split), Save, the sites with the bind list and its default instance, the marks, the
diff and the estimate. Every kernel is imported (DaCe, once per kernel), hence
slow; the roots are in ``tmp_path``."""

from __future__ import annotations

from dataclasses import replace

import pytest

from snax_forge.design import Design, DesignPoint, Platform, run_checks
from snax_forge.lower import task_list
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, session
from snax_forge.sandbox import __main__ as sandbox_cli
from snax_forge.sandbox.session import Roots, Session
from tests.helpers import imported

from .helpers import RECIPES, REPO, recipe_graphs, renamed, tagged_names

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def cached_import(monkeypatch):
    monkeypatch.setattr(session, "_import", imported)


@pytest.fixture
def roots(tmp_path):
    return Roots(tmp_path / "sessions", tmp_path / "recipes", tmp_path / "sandbox")


def recipe(name: str) -> Recipe:
    return Recipe.load(REPO / "recipes" / f"{name}.json")


def opened(name: str, roots: Roots) -> Session:
    return Session.from_recipe(REPO / "recipes" / f"{name}.json", roots=roots)


def ok(result: dict) -> dict:
    assert result["ok"], result
    return result


def split_and_tag(s: Session, map: str, factor: str | int, new: dict | None = None) -> str:
    """``split`` then the inner map spatial and the outer temporal, as the editor does them; the
    inner map's id."""
    got = ok(s.apply("split", {"map": map, "factor": factor}, new_params=new))
    (inner,) = got["diff"]["nodes"]["added"]
    ok(s.apply("tag_map", {"map": inner, "kind": "spatial"}))
    ok(s.apply("tag_map", {"map": map, "kind": "temporal"}))
    return inner


def bind_default(s: Session, node: str) -> dict:
    """``bind`` on ``node`` with the bind form's defaults: the first binding that is ok and the
    default instance."""
    f = {x["name"]: x for x in s.sites(s.cursor, node)["bind"]["form"]}
    return ok(s.apply("bind", {"node": node, **f["binding"]["value"],
                               "instance": f["instance"]["value"]}))  # fmt: skip


# =============================================================================
# Built in a session, a recipe's graphs
# =============================================================================


@pytest.mark.parametrize("name", ["vecadd", "dot_2", "dot_4"])
def test_built_by_hand_it_ends_in_the_recipes_graph(name, roots):
    """Each step of the recipe applied in the session, one by one; the last graph is the
    recipe's, file for file. ``split_map`` stays ``split_map`` here: the later steps name its
    ``<id>_s``, and a second split (dot 4's ``split_reduce``) names its variables after the
    first's, so a ``split`` and two tags would differ in more than the three names
    (test_sites.py checks ``split_map`` against them, a step at a time)."""
    r = recipe(name)
    s = Session.from_kernel(r.kernel, r.params, r.symbols, roots=roots)
    for step in r.steps:
        ok(s.apply(step.transform, step.params))
    want = apply_recipe(r, imported(r.kernel))[-1].graph
    assert s.graphs[-1].to_json() == want.to_json()
    assert s.name == r.kernel and (roots.sessions / r.kernel / "recipe.json").is_file()


def test_a_taken_name_gets_a_suffix(roots):
    names = [Session.from_kernel("vecadd", {"N": 64}, roots=roots).name for _ in range(3)]
    assert names == ["vecadd", "vecadd_2", "vecadd_3"]


def test_a_number_given_for_a_factor_becomes_a_param(roots, tmp_path):
    """C§12: every factor the editor writes names a param. vecmul's split given 8 stores W with
    W = 8 in the params, so after the bind its Bindings entry has W editable and W = 4 replays
    to 512 cycles; a second split gets W_2; a link's depth given as a number becomes D; the
    saved recipe sweeps with --set W=…; a recipe opened from a file keeps its numbers."""
    from snax_forge.viz.dfg.api import bindings_section

    s = Session.from_kernel("vecmul", {"N": 2048}, roots=roots)
    got = ok(s.apply("split", {"map": "mult_map", "factor": 8}))
    assert got["params"] == {"map": "mult_map", "factor": "W"} and got["new_params"] == {"W": 8}
    assert s.recipe.steps[0].params["factor"] == "W" and s.recipe.params == {"N": 2048, "W": 8}
    ok(s.apply("tag_map", {"map": "mult_map", "kind": "temporal"}))
    ok(s.apply("tag_map", {"map": "mult_map_inner", "kind": "spatial"}))
    got = bind_default(s, "mult")  # the editor's W, this unit's alone: W_acc_1 (D156)
    assert got["renamed"] == {"W": "W_acc_1"} and s.recipe.params == {"N": 2048, "W_acc_1": 8}
    assert s.recipe.steps[0].params["factor"] == "W_acc_1"
    (unit,) = bindings_section(s.graphs[4], s.recipe, editable=True)["entries"]
    assert unit["params"] == [{"name": "W_acc_1", "value": 8, "editable": True, "shared_with": []}]
    ok(s.set_params({"W_acc_1": 4}))
    assert s.estimate(4).accelerator_cycles == 512
    ok(s.save("vecmul_w"))
    assert sandbox_cli.main([str(roots.recipes / "vecmul_w.json"), "--set", "W_acc_1=2", "--out",
                             str(tmp_path / "w2")]) == 0  # fmt: skip
    dot = Session.from_kernel("dot", {"N": 512}, roots=roots)
    ok(dot.apply("split", {"map": "mult_map", "factor": 8}))
    ok(dot.apply("split", {"map": "sum_map", "factor": 4}))
    assert [x.params["factor"] for x in dot.recipe.steps] == ["W", "W_2"]
    assert dot.recipe.params == {"N": 512, "W": 8, "W_2": 4}
    two = opened("dot_2", roots)
    assert two.recipe.steps[-1].params["depth"] == 1  # the file's number stays
    ok(two.move(two.n - 1))
    got = ok(two.apply("link", {"container": "tmp0", "depth": 1, "group": "dot"}, drop=True))
    assert got["params"]["depth"] == "D" and two.recipe.params["D"] == 1


def test_a_kernel_session_needs_every_symbol_and_no_w(roots):
    """The kernel's symbols only, each a param of its name; one without a value is refused
    with one line naming it."""
    s = Session.from_kernel("vecmul", {"N": 2048}, roots=roots)
    assert (s.recipe.params, s.recipe.symbols) == ({"N": 2048}, {"N": "N"})
    with pytest.raises(SandboxError, match=r"^vecmul: give N, e\.g\. --set N=2048$"):
        Session.from_kernel("vecmul", {}, roots=roots)
    with pytest.raises(
        SandboxError, match=r"^matvec: give M, N, e\.g\. --set M=2048 --set N=2048$"
    ):
        Session.from_kernel("matvec", {"W": 4}, roots=roots)
    assert Session.from_kernel("vecmul", {}, {"N": 64}, roots=roots).recipe.params == {}


def test_vecmul_from_its_kernel_with_a_new_param_is_the_recipes_graph(roots):
    """vecmul at N = 2048 alone, ``split`` by a new param W = 8, the tags, ``bind`` at the form's
    defaults: ``recipes/vecmul.json``'s last graph at N = 2048, W = 8 up to the split's names;
    the default instance ``acc_1`` gives the task ``mult_acc_1`` and the streamers ``acc_1_a``,
    ``acc_1_b`` and ``acc_1_out``, and W becomes ``W_acc_1``."""
    s = Session.from_kernel("vecmul", {"N": 2048}, roots=roots)
    refused = s.apply("split", {"map": "mult_map", "factor": "W"})
    assert refused["param"] == "W"
    assert (refused["code"], refused["message"]) == (
        "session.param",
        "split: factor 'W' is not a param; give its value",
    )
    assert (s.n, s.recipe.params) == (0, {"N": 2048})
    assert split_and_tag(s, "mult_map", "W", {"W": 8}) == "mult_map_inner"
    assert s.recipe.params == {"N": 2048, "W": 8}
    got = bind_default(s, "mult")
    assert got["params"]["instance"] == "acc_1" and got["renamed"] == {"W": "W_acc_1"}
    r = recipe("vecmul").with_params({"N": 2048, "W": 8})
    want = apply_recipe(r, imported("vecmul"))[-1].graph
    names = tagged_names(s.graphs[0], "mult_map", 8)
    assert names == {"i_1": "i_t", "i_2": "i_s", "mult_map_inner": "mult_map_s"}
    assert renamed(s.graphs[-1], names | {"acc_1": "acc"}) == want.to_json()
    d = Design(s.platform, s.graphs[-1])
    assert run_checks(d) == []
    point = DesignPoint.of(d, "vecmul")
    assert sorted(point.streamers) == ["acc_1_a", "acc_1_b", "acc_1_out"]
    accel = [x.task_name for x in task_list(point).steps if getattr(x, "type", None) == "accel"]
    assert accel == ["mult_acc_1"]


def test_dot_1_in_the_editor_has_a_param_per_split(roots, tmp_path):
    """dot 1 with ``W_mul`` 8 and ``W_sum`` 4, each added by its split (names given, so never
    renamed): the binds default to ``acc_1`` and ``acc_2``, Save writes params ``{N: 2048, W_mul:
    8, W_sum: 4}``, and the saved recipe passes ``pixi run sandbox``. Undoing the step that
    added a param keeps it."""
    s = Session.from_kernel("dot", {"N": 2048}, roots=roots)
    split_and_tag(s, "mult_map", "W_mul", {"W_mul": 8})
    got = bind_default(s, "mult")
    assert got["params"]["instance"] == "acc_1" and got["renamed"] == {}
    split_and_tag(s, "sum_map", "W_sum", {"W_sum": 4})
    assert bind_default(s, "sum")["params"]["instance"] == "acc_2"
    f = {x["name"]: x for x in s.sites(s.cursor, "mult")["encapsulate"]["form"]}
    ok(s.apply("encapsulate", {"accelerator": f["accelerator"]["value"],
                               "units": ["acc_1", "acc_2"]}))  # fmt: skip
    got = ok(s.save("dot_mine"))
    saved = Recipe.load(roots.recipes / "dot_mine.json")
    assert saved.params == {"N": 2048, "W_mul": 8, "W_sum": 4}
    assert [x.params.get("factor") for x in saved.steps if x.transform == "split"] == [
        "W_mul", "W_sum"]  # fmt: skip
    assert got["problems"] == []
    assert sandbox_cli.main([str(roots.recipes / "dot_mine.json"), "--out",
                             str(tmp_path / "steps")]) == 0  # fmt: skip
    ok(s.move(1))
    assert s.recipe.params == {"N": 2048, "W_mul": 8, "W_sum": 4}  # undo keeps the params


# =============================================================================
# The cursor
# =============================================================================


def test_the_cursor_moves_and_a_step_before_the_last_drops_the_rest(roots):
    s = Session.from_kernel("vecadd", {"N": 64, "W": 4}, roots=roots)
    split_and_tag(s, "add_map", "W")
    assert (s.n, s.cursor) == (3, 3)
    ok(s.move(1))
    ok(s.move(3))
    ok(s.move(1))
    steps = list(s.recipe.steps)
    refused = s.apply("tag_map", {"map": "add_map", "kind": "tile"})
    assert (refused["ok"], refused["code"]) == (False, "session.drop")
    assert (s.recipe.steps, s.cursor, len(s.graphs)) == (steps, 1, 4)
    got = ok(s.apply("tag_map", {"map": "add_map", "kind": "tile"}, drop=True))
    assert (got["index"], s.n, s.cursor) == (2, 2, 2)
    assert [x.transform for x in s.recipe.steps] == ["split", "tag_map"]
    assert s.export(2) == roots.sessions / "vecadd" / "export" / "2_tag_map.snaxdfg"
    assert s.move(5)["code"] == "session.move"
    ok(s.move(1))
    back = Session.resume(roots.sessions / "vecadd")
    assert (back.recipe.steps, back.cursor, back.platform_path) == (
        s.recipe.steps,
        1,
        session.PLATFORM,
    )
    assert [g.to_json() for g in back.graphs] == [g.to_json() for g in s.graphs]


def test_a_refused_step_changes_nothing(roots):
    s = Session.from_kernel("vecadd", {"N": 64}, roots=roots)
    got = s.apply("bind", {"node": "add", "brm": "elementwise_add",
                           "implementation": "chisel_tiled_spatial", "instance": "acc"})  # fmt: skip
    assert (got["ok"], got["code"]) == (False, "sandbox") and "split the map first" in got[
        "message"
    ]
    assert (s.n, s.cursor) == (0, 0)


# =============================================================================
# The params
# =============================================================================


def test_new_params_replay_every_step(roots):
    s = opened("vecadd", roots)
    ok(s.set_params({"W": 8}))
    assert s.graphs[-1].node("add").attrs["params"]["W"] == 8
    assert Recipe.load(roots.sessions / "vecadd" / "recipe.json").params["W"] == 8


def test_a_step_that_refuses_the_new_params_keeps_the_old(roots):
    s = opened("dot_4", roots)
    before = [g.to_json() for g in s.graphs]
    got = s.set_params({"N": 510, "W": 4})
    assert (got["ok"], got["index"]) == (False, 2)
    assert "step 2 (split_reduce)" in got["message"] and "not a multiple of 4" in got["message"]
    assert s.recipe.params == {"W": 4, "N": 512}
    assert [g.to_json() for g in s.graphs] == before


# =============================================================================
# Save
# =============================================================================


def test_save_refuses_unbound_work(roots):
    s = Session.from_kernel("vecadd", {"N": 64}, roots=roots)
    got = s.save("vecadd_mine")
    assert got["code"] == "save.graph" and any("[graph.unbound]" in p for p in got["problems"])
    assert not (roots.recipes / "vecadd_mine.json").exists()


def test_save_refuses_two_accelerators_unless_forced(roots):
    s = opened("dot_1", roots)
    assert s.recipe.steps[-1].transform == "encapsulate"
    ok(s.move(s.n - 1))
    got = s.save("dot_two")
    assert got["code"] == "save.design" and any("[gen.instances]" in p for p in got["problems"])
    assert ok(s.save("dot_two", force=True))["recipe"] == str(roots.recipes / "dot_two.json")


def test_save_lists_too_many_crossbar_inputs_and_saves_anyway(roots):
    """vecmul at N = 2048, W = 32 from the editor: Save names connect.xbar_inputs (99 inputs
    over 64) as a design problem and saves when forced (C§12)."""
    s = Session.from_kernel("vecmul", {"N": 2048}, roots=roots)
    split_and_tag(s, "mult_map", "W", {"W": 32})
    bind_default(s, "mult")
    got = s.save("vecmul_w32")
    assert got["code"] == "save.design" and not (roots.recipes / "vecmul_w32.json").exists()
    (p,) = got["problems"]
    assert p.startswith("[connect.xbar_inputs] acc_1: 99 TCDM crossbar inputs, the platform "
                        "takes at most 64")  # fmt: skip
    assert ok(s.save("vecmul_w32", force=True))["problems"] == got["problems"]


def test_a_saved_recipe_gives_the_sessions_graphs(roots):
    """dot 2 saved at its last step (one accelerator, its two units linked) replays to the
    session's graphs; a name taken or not a name is refused."""
    s = opened("dot_2", roots)
    got = ok(s.save("dot_mine"))
    assert got["commands"][0] == {"command": f"pixi run sandbox {roots.recipes / 'dot_mine.json'}",
                                  "what": "writes the steps and checks each against the reference"}  # fmt: skip
    assert got["commands"][1]["command"].endswith("--platform platforms/snax_forge.json")
    assert [c["what"] for c in got["commands"][1:]] == [
        "runs the recipe on SNAX-GVSoC and writes the reports",
        "the flow plus the bundle for snax_cluster's RTL"]  # fmt: skip
    assert s.save("dot_mine")["code"] == "save.name"
    assert s.save("2dot")["code"] == "save.name"
    saved = Recipe.load(roots.recipes / "dot_mine.json")
    assert (saved.name, saved.steps) == ("dot_mine", s.recipe.steps)
    graphs = [x.graph.to_json() for x in apply_recipe(saved, imported("dot"))]
    assert graphs == [g.to_json() for g in s.graphs]
    files = sorted(p.name for p in (roots.sandbox / "dot_mine").iterdir())
    assert files == ["0_input.snaxdfg", "1_split_map.snaxdfg", "2_bind.snaxdfg",
                     "3_split_map.snaxdfg", "4_bind.snaxdfg", "5_link.snaxdfg", "recipe.json"]  # fmt: skip


# =============================================================================
# The sites, the bind list and the marks
# =============================================================================


def test_the_sites_at_vecadds_import(roots):
    s = Session.from_kernel("vecadd", {"N": 64}, roots=roots)
    at_map, at_add = s.sites(0, "add_map"), s.sites(0, "add")
    assert (at_map["split"]["ok"], at_map["split"]["params"]) == (True, {"map": "add_map"})
    assert not at_add["bind"]["ok"] and "split the map first" in at_add["bind"]["message"]
    assert s.sites(0, "add") is at_add  # cached
    marks = s.marks(0)
    assert "add_map" in marks["nodes"] and "add" not in marks["nodes"]


def _ok(entries):
    return [(e["brm"], e["implementation"]) for e in entries if e["ok"]]


def test_the_bind_list_of_a_split_multiply(roots):
    s = Session.from_kernel("dot", {"N": 512, "W": 4}, roots=roots)
    ok(s.apply("split_map", {"map": "mult_map", "factor": "W"}))
    entries = s.sites(1, "mult")["bind"]["list"]
    assert _ok(entries) == [("elementwise_mul", "chisel_tiled_spatial")]
    (mul,) = [e for e in entries if e["ok"]]
    assert mul["params"]["instance"] == "acc_1" and mul["timing"] == {
        "latency": 0,
        "ii": 1,
        "span": 1,
    }


def test_the_bind_list_of_dot_4s_final_sum(roots):
    s = opened("dot_4", roots)
    assert s.recipe.steps[2].transform == "fuse_map"
    entries = s.sites(3, "sum_final")["bind"]["list"]
    assert _ok(entries) == [("accumulate", "chisel_adder_tree"), ("accumulate", "chisel_serial")]
    (acc,) = [e for e in entries if e["implementation"] == "chisel_accumulator"]
    assert not acc["ok"] and "W" in acc["message"]


# =============================================================================
# The diff and the estimate
# =============================================================================


def test_the_diff_of_a_split_and_of_a_link(roots):
    s = Session.from_kernel("vecadd", {"N": 64, "W": 4}, roots=roots)
    d = ok(s.apply("split", {"map": "add_map", "factor": "W"}))["diff"]
    # the map is split, its new inner map added, the tasklet's memlets rewritten over both
    assert (d["nodes"]["changed"], d["nodes"]["added"], d["nodes"]["removed"]) == (
        ["add_map", "add"], ["add_map_inner"], [])  # fmt: skip
    assert s.diff(0) == {"nodes": {"added": [], "removed": [], "changed": []},
                         "containers": {"added": [], "removed": [], "changed": []}}  # fmt: skip
    two = opened("dot_2", roots)
    assert two.recipe.steps[-1].transform == "link"
    assert two.diff(two.n)["containers"]["changed"] == ["tmp0"]
    assert two.estimate(two.n).containers["tmp0"].placement == "link"


@pytest.mark.parametrize("name", RECIPES)
def test_the_estimate_of_every_step_of_every_recipe(name, roots):
    r = recipe(name)
    if name == "vecadd_undo":  # starts from vecadd's bound graph, as README.md runs it
        s = Session(name, r, recipe_graphs("vecadd")[1][-1], session.PLATFORM, roots)
    else:
        s = opened(name, roots)
    for k in range(s.n + 1):
        s.estimate(k)
    assert s.estimate(0) is s.estimate(0)
    assert s.platform == Platform.load(REPO / session.PLATFORM)


# =============================================================================
# The forms
# =============================================================================


def _form(answer):
    return {f["name"]: f for f in answer["form"]}


def test_the_forms_of_vecadds_import(roots):
    """split's: the map, not editable, and a sweepable factor defaulting to a param of its own,
    W_2 here since the session has W (its value 4 beside it); tag_map's kind a choice of TAGS; a
    refused transform has no form."""
    s = Session.from_kernel("vecadd", {"N": 64, "W": 4}, roots=roots)
    sites = s.sites(0, "add_map")
    split = _form(sites["split"])
    assert list(split) == ["map", "factor"]
    assert (split["map"]["value"], split["map"]["readonly"], split["map"]["type"]) == (
        "add_map",
        True,
        "str",
    )
    assert (split["factor"]["type"], split["factor"]["sweepable"], split["factor"]["value"]) == (
        "int",
        True,
        "W_2",
    )
    assert split["factor"]["new_value"] == 4
    kind = _form(sites["tag_map"])["kind"]
    assert (kind["type"], kind["value"], [c["value"] for c in kind["choices"]]) == (
        "choice", "temporal", ["temporal", "tile", "spatial"])  # fmt: skip
    assert "form" not in sites["bind"]


def test_the_bind_form_lists_the_library_with_timing(roots):
    s = Session.from_kernel("dot", {"N": 512, "W": 4}, roots=roots)
    ok(s.apply("split_map", {"map": "mult_map", "factor": "W"}))
    f = _form(s.sites(1, "mult")["bind"])
    assert list(f) == ["node", "binding", "instance"] and f["node"]["readonly"]
    assert f["binding"]["value"] == {
        "brm": "elementwise_mul",
        "implementation": "chisel_tiled_spatial",
    }
    (good,) = [c for c in f["binding"]["choices"] if c["ok"]]
    assert (good["label"], good["timing"]) == ("elementwise_mul / chisel_tiled_spatial",
                                               {"latency": 0, "ii": 1, "span": 1})  # fmt: skip
    assert all(c["message"] for c in f["binding"]["choices"] if not c["ok"])
    assert f["instance"]["value"] == "acc_1"


def test_the_encapsulate_and_link_forms(roots):
    """dot 1 before encapsulate: the units unencapsulated in graph order, every one ticked
    (D156), not only the selected one, and a free accelerator name; dot 2 before its link: the depth a param of its own, D,
    at 1 by default, the group from the site."""
    one = opened("dot_1", roots)
    ok(one.move(one.n - 1))
    f = _form(one.sites(one.cursor, "mult")["encapsulate"])
    assert (f["units"]["type"], f["units"]["value"], [c["value"] for c in f["units"]["choices"]]) == (
        "checklist", ["mul", "sum"], ["mul", "sum"])  # fmt: skip
    assert f["accelerator"]["value"] == "dot" and not f["units"]["readonly"]
    assert _form(one.sites(one.cursor, "sum")["encapsulate"])["units"]["value"] == ["mul", "sum"]
    two = opened("dot_2", roots)
    ok(two.move(two.n - 1))
    f = _form(two.sites(two.cursor, "tmp0")["link"])
    assert (f["container"]["readonly"], f["depth"]["value"], f["depth"]["sweepable"], f["group"]["value"]) == (
        True, "D", True, "dot")  # fmt: skip
    assert f["depth"]["new_value"] == 1


# =============================================================================
# D156: names, the checks on every step, a param change up to the cursor
# =============================================================================


def dot1_by_hand(roots: Roots, w_mul: int = 8, w_sum: int = 8) -> tuple[Session, list[dict]]:
    """dot 1 at N = 2048 from the kernel with the editor's defaults (the dry run's session):
    split mult_map by a number, both tags, bind; split sum_map, both tags, bind accumulate /
    chisel_adder_tree; encapsulate. The session and each step's apply answer."""
    s = Session.from_kernel("dot", {"N": 2048}, roots=roots)
    out = []
    for m, node, w in (("mult_map", "mult", w_mul), ("sum_map", "sum", w_sum)):
        out.append(ok(s.apply("split", {"map": m, "factor": w})))
        (inner,) = out[-1]["diff"]["nodes"]["added"]
        out.append(ok(s.apply("tag_map", {"map": inner, "kind": "spatial"})))
        out.append(ok(s.apply("tag_map", {"map": m, "kind": "temporal"})))
        f = {x["name"]: x for x in s.sites(s.cursor, node)["bind"]["form"]}
        binding = {"brm": "accumulate", "implementation": "chisel_adder_tree"} if node == "sum" \
            else f["binding"]["value"]  # fmt: skip
        out.append(ok(s.apply("bind", {"node": node, **binding,
                                       "instance": f["instance"]["value"]})))  # fmt: skip
    f = {x["name"]: x for x in s.sites(s.cursor, "mult")["encapsulate"]["form"]}
    out.append(ok(s.apply("encapsulate", {"accelerator": f["accelerator"]["value"],
                                          "units": ["acc_1", "acc_2"]})))  # fmt: skip
    return s, out


def test_dot_1_by_hand_names_its_units_params_and_loops(roots):
    """D156 and D73: the binds' instances ``acc_1`` and ``acc_2`` by their order, each split's
    factor named ``W`` and renamed ``W_<instance>`` at its bind, the second split's variables
    the next free pair ``i_3`` and ``i_4`` (not ``i_1_1`` and ``i_2``)."""
    s, out = dot1_by_hand(roots)
    assert [x["params"].get("instance") for x in out if x["transform"] == "bind"] == [
        "acc_1",
        "acc_2",
    ]
    assert [x["renamed"] for x in out if x["transform"] == "bind"] == [
        {"W": "W_acc_1"}, {"W": "W_acc_2"}]  # fmt: skip
    assert s.recipe.params == {"N": 2048, "W_acc_1": 8, "W_acc_2": 8}
    assert [x.params["factor"] for x in s.recipe.steps if x.transform == "split"] == [
        "W_acc_1", "W_acc_2"]  # fmt: skip
    vars_ = {n.id: n.attrs["var"] for n, _ in s.graphs[7].walk() if n.kind == "map"}
    assert (vars_["sum_map"], vars_["sum_map_inner"]) == ("i_3", "i_4")
    assert s.recipe.steps[-1].params["accelerator"] == "dot"


def test_a_bind_keeps_a_shared_or_a_recipes_own_param(roots):
    """A factor the editor named that another unit's split names too stays (dot 2–6's W), and so
    does a recipe's own param (vecmul's W, opened from its file)."""
    s = Session.from_kernel("dot", {"N": 512}, roots=roots)
    split_and_tag(s, "mult_map", 4)
    split_and_tag(s, "sum_map", "W")  # the same W
    assert bind_default(s, "mult")["renamed"] == {} and "W" in s.recipe.params
    v = opened("vecmul", roots)
    ok(v.move(v.n - 1))
    step = v.recipe.steps[-1]
    got = ok(v.apply("bind", dict(step.params), drop=True))
    assert got["renamed"] == {} and "W" in v.recipe.params


def test_every_step_lists_saves_design_checks(roots):
    """D156: dot 1 at W = 8 and 8 before encapsulate has no problem and one note, gen.instances
    before Save; after it neither; at W = 10 and 10 none either (D144), at 8 and 10 connect.pad,
    naming open item 106. Save still refuses at step 8 unless forced."""
    s, out = dot1_by_hand(roots)
    eight = s.checks(8)
    assert eight["problems"] == [] and [n["code"] for n in eight["notes"]] == ["gen.instances"]
    assert eight["notes"][0]["message"].startswith(
        "before Save: encapsulate acc_1, acc_2 into one accelerator (2 accelerators"
    )
    assert out[7]["checks"] == eight  # the bind's answer carries them
    assert s.checks(9) == {"problems": [], "notes": []} == out[8]["checks"]
    ok(s.move(8))
    assert s.save("dot_early")["code"] == "save.design"
    ten = dot1_by_hand(Roots(roots.sessions / "b", roots.recipes, roots.sandbox), 10, 10)[0]
    assert ten.checks(9) == {"problems": [], "notes": []}
    bad = dot1_by_hand(Roots(roots.sessions / "c", roots.recipes, roots.sandbox), 8, 10)[0]
    (p,) = bad.checks(9)["problems"]
    assert (p["code"], p["where"]) == ("connect.pad", "container tmp0")
    assert "open item 106" in p["message"] and p["fix"]


def test_while_building_unbound_is_a_note_and_notes_show_beside_problems(roots):
    """D156: before the bind graph.unbound is a note ("before Save: bind it …"), not a problem,
    and Save still refuses it; vecadd_i32's int32 adder, not in nangate45's table, keeps its
    char.missing note beside a problem (W = 32: too many crossbar inputs)."""
    s = Session.from_kernel("vecadd", {"N": 64}, roots=roots)
    split_and_tag(s, "add_map", 4)
    c = s.checks(3)
    assert c["problems"] == [] and [n["code"] for n in c["notes"]] == ["graph.unbound"]
    assert c["notes"][0]["message"].startswith("before Save: bind it (")
    assert c["notes"][0]["before_save"] is True
    assert s.save("vecadd_unbound")["code"] == "save.graph"
    i32 = Session.from_kernel("vecadd", {"N": 2048}, dtypes={"A": "int32", "B": "int32", "C": "int32"},
                              roots=Roots(roots.sessions / "i32", roots.recipes, roots.sandbox))  # fmt: skip
    split_and_tag(i32, "add_map", 64)
    bind_default(i32, "add")
    c = i32.checks(4)
    assert [p["code"] for p in c["problems"]] == ["connect.xbar_inputs"]
    assert [(n["code"], n["before_save"]) for n in c["notes"]] == [("char.missing", False)]
    assert not c["problems"][0]["before_save"]


def test_a_param_change_replays_up_to_the_cursor(roots):
    """D156: dot 1 at step 9, a link as step 10, back to 9, W_acc_2 = 4: accepted, step 10 a
    redo step; moving onto it replays it, and the link refuses (its writer's beats are not its
    reader's), the cursor staying at 9."""
    s, _ = dot1_by_hand(roots)
    ok(s.apply("link", {"container": "tmp0", "depth": 1, "group": "dot"}))
    assert s.n == 10
    ok(s.move(9))
    got = ok(s.set_params({"W_acc_2": 4}))
    assert (got["replayed"], got["redo"], s.replayed) == (9, [10], 9)
    with pytest.raises(SandboxError, match="redo step"):
        s.diff(10)
    refused = s.move(10)
    assert (refused["ok"], refused["code"], refused["index"]) == (False, "link.order", 10)
    assert (s.cursor, s.replayed, s.n) == (9, 9, 10)
    ok(s.set_params({"W_acc_2": 8}))
    assert ok(s.move(10))["replayed"] == [10] and s.cursor == 10


def test_save_keeps_only_the_params_its_steps_use(roots):
    """D156: dot 1 saved at step 9 after a link was tried at step 10 and stepped back from keeps
    N, W_acc_1 and W_acc_2, not the link's D; a recipe opened from a file keeps its own params
    (dot_1's N and W) whatever its saved steps use."""
    s, _ = dot1_by_hand(roots)
    ok(s.apply("link", {"container": "tmp0", "depth": 1, "group": "dot"}))
    assert "D" in s.recipe.params
    ok(s.move(9))
    ok(s.save("dot_1_kept"))
    saved = Recipe.load(roots.recipes / "dot_1_kept.json")
    assert saved.params == {"N": 2048, "W_acc_1": 8, "W_acc_2": 8}
    extra = roots.recipes / "vecmul_extra.json"
    roots.recipes.mkdir(parents=True, exist_ok=True)
    replace(recipe("vecmul"), name="vecmul_extra", params={**recipe("vecmul").params, "K": 3}).save(
        extra
    )
    one = Session.from_recipe(extra, roots=roots)
    ok(one.save("vecmul_kept"))
    assert Recipe.load(roots.recipes / "vecmul_kept.json").params["K"] == 3  # its own, kept
