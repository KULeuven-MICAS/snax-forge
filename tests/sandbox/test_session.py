"""The editor's session (sandbox/session.py), without HTTP: built step by step it ends
in the hand-written recipes' graphs; the start from a kernel's symbols, the cursor, the params (a
new one per split), Save, the sites with the bind list and its default instance, the marks, the
diff and the estimate. Every kernel is imported (DaCe, once per kernel), hence
slow; the roots are in ``tmp_path``."""

from __future__ import annotations

from dataclasses import replace

import pytest

from snax_forge.char import DEFAULT_TECH
from snax_forge.design import Platform
from snax_forge.estimate import estimate
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, session
from snax_forge.sandbox.session import Roots, Session
from tests.helpers import REPO, imported, recipe_path

from .helpers import RECIPES, recipe_graphs

pytestmark = pytest.mark.slow


def recipe(name: str) -> Recipe:
    return Recipe.load(recipe_path(name))


def opened(name: str, roots: Roots) -> Session:
    return Session.from_recipe(recipe_path(name), roots=roots)


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


def test_a_kernel_session_at_a_type_preset(roots):
    """``edit-dfg --kernel mvt_types --set types=i8 --set dtypes.x2=int64``: the preset and the
    dtype reach the import and the recipe, the rest are params; the session's recipe file
    carries them."""
    from snax_forge.viz.dfg.edit import kernel_start

    body = kernel_start({"M": 64, "N": 64, "types": "i8", "dtypes.x2": "int64"})
    assert body == {"types": "i8", "dtypes": {"x2": "int64"}, "params": {"M": 64, "N": 64}}
    s = Session.from_kernel("mvt_types", body["params"], dtypes=body["dtypes"], roots=roots,
                            types=body["types"])  # fmt: skip
    g = s.graphs[0]
    assert (g.containers["A"].dtype, g.containers["x1"].dtype, g.containers["x2"].dtype) == (
        "int8", "int32", "int64")  # fmt: skip
    assert s.recipe.types == "i8" and s.recipe.dtypes == {"x2": "int64"}
    saved = Recipe.load(roots.sessions / s.name / "recipe.json")
    assert (saved.types, saved.dtypes, saved.params) == ("i8", {"x2": "int64"}, {"M": 64, "N": 64})


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
    got = s.apply("bind", {"node": "add", "brm": "simd",
                           "implementation": "chisel_tiled_spatial", "instance": "acc"})  # fmt: skip
    assert (got["ok"], got["code"]) == (False, "sandbox") and "split the map first" in got[
        "message"
    ]
    assert (s.n, s.cursor) == (0, 0)


# =============================================================================
# The params
# =============================================================================


def test_new_params_replay_every_step(roots):
    s = opened("vecadd_v1", roots)
    ok(s.set_params({"W": 8}))
    assert s.graphs[-1].node("add").attrs["params"]["W"] == 8
    assert Recipe.load(roots.sessions / "vecadd_v1" / "recipe.json").params["W"] == 8


def test_a_step_that_refuses_the_new_params_keeps_the_old(roots):
    s = opened("dot_v4", roots)
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
    assert not (roots.recipes / "custom" / "vecadd_mine.json").exists()


def test_save_lists_too_many_crossbar_inputs_and_saves_anyway(roots):
    """vecmul at N = 2048, W = 64 from the editor: Save names connect.xbar_inputs (194 inputs
    over 128) as a design problem and saves when forced."""
    s = Session.from_kernel("vecmul", {"N": 2048}, roots=roots)
    split_and_tag(s, "mult_map", "W", {"W": 64})
    bind_default(s, "mult")
    got = s.save("vecmul_w64")
    assert (
        got["code"] == "save.design" and not (roots.recipes / "custom" / "vecmul_w64.json").exists()
    )
    (p,) = got["problems"]
    assert p.startswith("[connect.xbar_inputs] acc_1: 194 TCDM crossbar inputs, the platform "
                        "takes at most 128")  # fmt: skip
    assert ok(s.save("vecmul_w64", force=True))["problems"] == got["problems"]


def test_a_saved_recipe_gives_the_sessions_graphs(roots):
    """dot 2 saved at its last step (one accelerator, its two units linked) replays to the
    session's graphs; a name taken or not a name is refused."""
    s = opened("dot_v2", roots)
    got = ok(s.save("dot_mine"))
    assert got["commands"][0] == {"command": f"pixi run sandbox {roots.recipes / 'custom' / 'dot_mine.json'}",
                                  "what": "writes the steps and checks each against the reference"}  # fmt: skip
    assert got["commands"][1]["command"].endswith("--platform platforms/snax_forge.json")
    assert [c["what"] for c in got["commands"][1:]] == [
        "runs the recipe on SNAX-GVSoC and writes the reports",
        "the flow plus the bundle for snax_cluster's RTL"]  # fmt: skip
    assert s.save("dot_mine")["code"] == "save.name"
    assert s.save("2dot")["code"] == "save.name"
    saved = Recipe.load(roots.recipes / "custom" / "dot_mine.json")
    assert (saved.name, saved.steps) == ("dot_mine", s.recipe.steps)
    graphs = [x.graph.to_json() for x in apply_recipe(saved, imported("dot"))]
    assert graphs == [g.to_json() for g in s.graphs]
    files = sorted(p.name for p in (roots.sandbox / "dot_mine").iterdir())
    assert files == ["0_input.snaxdfg", "1_split_map.snaxdfg", "2_bind.snaxdfg",
                     "3_split_map.snaxdfg", "4_bind.snaxdfg", "5_link.snaxdfg", "recipe.json"]  # fmt: skip


# =============================================================================
# The sites, the bind list and the marks
# =============================================================================


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
    two = opened("dot_v2", roots)
    assert two.recipe.steps[-1].transform == "link"
    assert two.diff(two.n)["containers"]["changed"] == ["tmp0"]
    assert two.estimate(two.n).containers["tmp0"].placement == "link"


SESSION_CHECKED = ("vecadd_v1", "vecadd_v1_4")  # the recipes the sweep also opens as a session


@pytest.mark.parametrize("name", RECIPES)
def test_the_estimate_of_every_step_of_every_recipe(name, roots):
    """Every step's estimate on the editor's platform and technology, from recipe_graphs'
    graphs: a session replays the same steps (apply_recipe's) and would build them again. On
    SESSION_CHECKED a session too: its graphs are those, its estimate of a step is theirs,
    cached; vecadd_v1_4 starts from vecadd's bound graph, as README.md runs it."""
    r, graphs = recipe_graphs(name)
    platform = Platform.load(REPO / session.PLATFORM)
    assert len(graphs) == len(r.steps) + 1
    estimates = [estimate(g, platform, DEFAULT_TECH) for g in graphs]
    if name not in SESSION_CHECKED:
        return
    if name == "vecadd_v1_4":
        s = Session(name, recipe(name), recipe_graphs("vecadd_v1")[1][-1], session.PLATFORM, roots)
    else:
        s = opened(name, roots)
    assert [g.to_json() for g in s.graphs] == [g.to_json() for g in graphs]
    assert s.platform == platform
    assert [s.estimate(k) for k in range(s.n + 1)] == estimates
    assert s.estimate(0) is s.estimate(0)


# =============================================================================
# The forms
# =============================================================================


# =============================================================================
# Names, the checks on every step, a param change up to the cursor
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


def test_a_bind_keeps_a_shared_or_a_recipes_own_param(roots):
    """A factor the editor named that another unit's split names too stays (dot 2–6's W), and so
    does a recipe's own param (vecmul's W, opened from its file)."""
    s = Session.from_kernel("dot", {"N": 512}, roots=roots)
    split_and_tag(s, "mult_map", 4)
    split_and_tag(s, "sum_map", "W")  # the same W
    assert bind_default(s, "mult")["renamed"] == {} and "W" in s.recipe.params
    v = opened("vecmul_v1", roots)
    ok(v.move(v.n - 1))
    step = v.recipe.steps[-1]
    got = ok(v.apply("bind", dict(step.params), drop=True))
    assert got["renamed"] == {} and "W" in v.recipe.params


def test_every_step_lists_saves_design_checks(roots):
    """Dot 1 at W = 8 and 8 has no problem and no note before encapsulate (two
    accelerators) or after it; at W = 10 and 10 none either, at 8 and 10
    connect.pad. Save takes step 8."""
    s, out = dot1_by_hand(roots)
    eight = s.checks(8)
    assert eight == {"problems": [], "notes": []}
    assert out[7]["checks"] == eight  # the bind's answer carries them
    assert s.checks(9) == {"problems": [], "notes": []} == out[8]["checks"]
    ok(s.move(8))
    assert ok(s.save("dot_early"))["recipe"] == str(roots.recipes / "custom" / "dot_early.json")
    ten = dot1_by_hand(Roots(roots.sessions / "b", roots.recipes, roots.sandbox), 10, 10)[0]
    assert ten.checks(9) == {"problems": [], "notes": []}
    bad = dot1_by_hand(Roots(roots.sessions / "c", roots.recipes, roots.sandbox), 8, 10)[0]
    (p,) = bad.checks(9)["problems"]
    assert (p["code"], p["where"]) == ("connect.pad", "container tmp0")
    assert "never written" in p["message"] and p["fix"]


def test_a_param_change_replays_up_to_the_cursor(roots):
    """Dot 1 at step 9, a link as step 10, back to 9, W_acc_2 = 4: accepted, step 10 a
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


def test_save_keeps_a_param_named_only_inside_apply_recipes_params():
    """``apply_recipe``'s ``{"W": "W_x1"}``: W_x1 is kept, the step's own names are not."""
    from snax_forge.sandbox.recipe import Step

    steps = [Step("lift", {"kernel": "matvec_acc", "at": "tmp1_mult_map"}),
             Step("apply_recipe", {"block": "matvec_acc", "recipe": "matvec/matvec_acc_v1",
                                   "params": {"W": "W_x1"}})]  # fmt: skip
    keep = session.used_params({"M": 64, "N": 64, "W_x1": 8, "D": 1}, steps,
                               {"M": "M", "N": "N"}, [])  # fmt: skip
    assert keep == {"M", "N", "W_x1"}


def test_a_param_is_dropped_only_when_nothing_names_it(roots):
    """vecmul with an extra param K nothing names: dropped, and Save does not need it; W, which
    its split names, is refused naming the step; an unknown name is refused."""
    extra = roots.recipes / "elementwise" / "vecmul_extra.json"
    extra.parent.mkdir(parents=True, exist_ok=True)
    replace(recipe("vecmul_v1"), name="vecmul_extra",
            params={**recipe("vecmul_v1").params, "K": 3}).save(extra)  # fmt: skip
    s = Session.from_recipe(extra, roots=roots)
    assert ok(s.drop_param("K"))["params"] == recipe("vecmul_v1").params
    assert "K" not in s.own_params and "K" not in Recipe.load(s.folder / "recipe.json").params
    refused = s.drop_param("W")
    assert refused["code"] == "params.drop" and "step 1 (split_map)" in refused["message"]
    assert s.drop_param("nope")["code"] == "params.drop"


def test_a_param_renamed_in_every_step_and_inside_apply_recipe(roots):
    """dot 1 by hand: W_acc_1 renamed W_m in its split and the params, the graphs the same;
    mvt_v5_1's W_x1 renamed W inside both apply_recipe steps' params, replayed; a
    taken name or a non-identifier is refused."""
    s, _ = dot1_by_hand(roots)
    before = [g.to_dict() for g in s.graphs]
    ok(s.rename_param("W_acc_1", "W_m"))
    assert "W_m" in s.recipe.params and "W_acc_1" not in s.recipe.params
    assert s.recipe.steps[0].params["factor"] == "W_m"
    assert [g.to_dict() for g in s.graphs] == before
    assert s.rename_param("W_m", "W_acc_2")["code"] == "params.rename"
    assert s.rename_param("W_m", "2W")["code"] == "params.rename"
    m = opened("mvt_v5_1", roots)
    ok(m.rename_param("W_x1", "W"))
    assert [st.params["params"]["W"] for st in m.recipe.steps if st.transform == "apply_recipe"] \
        == ["W", "W"]  # fmt: skip
    assert m.recipe.params["W"] == 8 and "W_x1" not in m.recipe.params


def test_the_editor_steps_through_fuse_seed(roots):
    """edit-dfg on matvec_acc: after the sum's tag and split, a map of the sum's nest offers
    fuse_seed with the elementwise nest's drop-down, as do the two maps selected; applied, the
    step checks, and unfuse_seed takes it back."""
    s = Session.from_kernel("matvec_acc", {"M": 8, "N": 512}, roots=roots)
    ok(s.apply("tag_map", {"map": "sum_map", "kind": "temporal"}))
    ok(s.apply("split_map", {"map": "sum_map_1", "factor": 4}))
    site = s.sites(s.cursor, "sum_map_1_s")["fuse_seed"]
    assert site["ok"] and site["params"] == {"first": "sum_map_1_s",
                                             "second": "augassign_19_4_map"}  # fmt: skip
    (pair,) = site["form"]
    assert (pair["name"], pair["value"]) == ("pair", site["params"])
    back = s.sites(s.cursor, "augassign_19_4_map")["fuse_seed"]  # the other side offers it too
    assert back["ok"] and back["params"] == {"first": "sum_map", "second": "augassign_19_4_map"}
    both = s.sites(s.cursor, ["sum_map", "augassign_19_4_map"])["fuse_seed"]
    assert both["ok"] and both["params"] == {"first": "sum_map", "second": "augassign_19_4_map"}
    ok(s.apply("fuse_seed", {"first": "sum_map_1", "second": pair["value"]["second"]}))
    assert s.sites(s.cursor, "sum")["fuse_seed"]["ok"] is False
    assert s.sites(s.cursor, "sum_map")["unfuse_seed"]["ok"]
    ok(s.apply("unfuse_seed", {"map": "sum_map_1"}))
    assert s.graphs[s.cursor] == s.graphs[s.cursor - 2]


# =============================================================================
# The highlight and the editor agree: every partner shown is offered
# =============================================================================


def _offered(answer: dict, transform: str, map: str, partner: str) -> bool:
    """The editor's answer on ``map`` offers ``transform`` with ``partner``: in fuse_map's and
    fuse_seed's pair drop-down (either side), interchange_map's ``with``."""
    if not answer["ok"]:
        return False
    (field,) = [f for f in answer["form"] if f["name"] in ("pair", "with")]
    if transform == "interchange_map":
        return any(c["value"] == partner and c["ok"] for c in field["choices"])
    return any(set(c["value"].values()) == {map, partner} for c in field["choices"])


SCENES = [
    ("matvec_acc", {"M": 8, "N": 512}, []),
    ("matvec_acc", {"M": 8, "N": 512}, [("tag_map", {"map": "sum_map", "kind": "temporal"})]),
    ("matvec_acc", {"M": 8, "N": 512}, [("tag_map", {"map": "sum_map", "kind": "temporal"}),
                                        ("split_map", {"map": "sum_map_1", "factor": 4}),
                                        ("split_map", {"map": "mult_map_1", "factor": 4})]),
    ("axpy", {"N": 64}, []),
]  # fmt: skip


@pytest.mark.parametrize(("kernel", "symbols", "steps"), SCENES,
                         ids=["matvec_acc", "matvec_acc-tag", "matvec_acc-splits", "axpy"])  # fmt: skip
def test_every_highlighted_partner_is_offered_by_the_editor(roots, kernel, symbols, steps):
    """For every map and every partner the highlight shows (viz/dfg/api.py's ``partners``,
    near partners left out), the editor's answer on that selected map offers that transform
    with that partner, whichever side of the pair the map is."""
    from pathlib import Path

    from snax_forge.viz.dfg.api import Entry, graph_view

    s = Session.from_kernel(kernel, symbols, roots=roots)
    for name, params in steps:
        ok(s.apply(name, params))
    v = graph_view(Entry("g", Path("g"), s.graphs[s.cursor], None))
    shown = [(m, p["transform"], p["map"]) for m, ps in v["partners"].items() for p in ps
             if "near" not in p]  # fmt: skip
    assert shown
    missing = [(m, t, p) for m, t, p in shown
               if not _offered(s.sites(s.cursor, m)[t], t, m, p)]  # fmt: skip
    assert missing == []
