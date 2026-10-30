"""The editor's session (D55, C§12, sandbox/session.py), without HTTP: built step by step it ends
in the hand-written recipes' graphs; the cursor, the params, Save, the sites with the bind list,
the marks, the diff and the estimate. Every kernel is imported (DaCe, once per kernel), hence
slow; the roots are in ``tmp_path``."""

from __future__ import annotations

import functools

import pytest

from snax_forge.design import Platform
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe, session
from snax_forge.sandbox.session import Roots, Session

from .helpers import RECIPES, REPO, recipe_graphs

pytestmark = pytest.mark.slow


@functools.cache
def _imported(kernel: str, dtypes: tuple[tuple[str, str], ...]):
    return import_kernel(kernel, dict(dtypes) or None)


def imported(kernel: str, dtypes: dict[str, str] | None = None):
    """A kernel's import, once per kernel and dtypes (the session never changes it)."""
    return _imported(kernel, tuple(sorted((dtypes or {}).items())))


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


def split_and_tag(s: Session, map: str, factor: str) -> None:
    """``split`` then the inner map spatial and the outer temporal, as the editor does them."""
    (inner,) = ok(s.apply("split", {"map": map, "factor": factor}))["diff"]["nodes"]["added"]
    ok(s.apply("tag_map", {"map": inner, "kind": "spatial"}))
    ok(s.apply("tag_map", {"map": map, "kind": "temporal"}))


# =============================================================================
# Built in a session, a recipe's graphs
# =============================================================================


@pytest.mark.parametrize("name", ["vecadd", "dot_2", "dot_4"])
def test_built_by_hand_it_ends_in_the_recipes_graph(name, roots):
    """Each split_map of the recipe done as split and two tags; the last graph is the
    recipe's, file for file."""
    r = recipe(name)
    s = Session.from_kernel(r.kernel, r.params, r.symbols, roots=roots)
    for step in r.steps:
        if step.transform == "split_map":
            split_and_tag(s, step.params["map"], step.params["factor"])
        else:
            ok(s.apply(step.transform, step.params))
    want = apply_recipe(r, imported(r.kernel))[-1].graph
    assert s.graphs[-1].to_json() == want.to_json()
    assert s.name == r.kernel and (roots.sessions / r.kernel / "recipe.json").is_file()


def test_a_taken_name_gets_a_suffix(roots):
    names = [Session.from_kernel("vecadd", {"N": 64}, roots=roots).name for _ in range(3)]
    assert names == ["vecadd", "vecadd_2", "vecadd_3"]


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


def test_a_saved_recipe_gives_the_sessions_graphs(roots):
    """dot 2 saved at its last step (one accelerator, its two units linked) replays to the
    session's graphs; a name taken or not a name is refused."""
    s = opened("dot_2", roots)
    got = ok(s.save("dot_mine"))
    assert got["commands"][0] == f"pixi run sandbox {roots.recipes / 'dot_mine.json'}"
    assert got["commands"][1].endswith("--platform platforms/snax_forge.json")
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
    assert mul["params"]["instance"] == "mult" and mul["timing"] == {
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
        ["add_map", "add"], ["add_map_s"], [])  # fmt: skip
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
    """split's: the map, not editable, and a sweepable factor defaulting to the param W;
    tag_map's kind a choice of TAGS; a refused transform has no form."""
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
        "W",
    )
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
    assert f["instance"]["value"] == "mult"


def test_the_encapsulate_and_link_forms(roots):
    """dot 1 before encapsulate: the units unencapsulated in graph order, the selected one
    ticked, and a free accelerator name; dot 2 before its link: depth 1 by default, the group
    from the site."""
    one = opened("dot_1", roots)
    ok(one.move(one.n - 1))
    f = _form(one.sites(one.cursor, "mult")["encapsulate"])
    assert (f["units"]["type"], f["units"]["value"], [c["value"] for c in f["units"]["choices"]]) == (
        "checklist", ["mul"], ["mul", "sum"])  # fmt: skip
    assert f["accelerator"]["value"] == "dot" and not f["units"]["readonly"]
    two = opened("dot_2", roots)
    ok(two.move(two.n - 1))
    f = _form(two.sites(two.cursor, "tmp0")["link"])
    assert (f["container"]["readonly"], f["depth"]["value"], f["depth"]["sweepable"], f["group"]["value"]) == (
        True, 1, True, "dot")  # fmt: skip
