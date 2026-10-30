"""The graph page's server in both modes (D55, viz/dfg/server.py), on servers at port 0: view
mode over a recipe's step files, edit mode over a session (sandbox/session.py) with its roots in
``tmp_path``, the JSON-only POSTs on both viewers, and every static file the page imports."""

from __future__ import annotations

import functools
import re
from pathlib import Path

import pytest

from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe, session, write_steps
from snax_forge.sandbox.session import Roots
from snax_forge.viz.dfg.server import make_dfg_server, make_edit_server
from tests.helpers import REPO

from .helpers import get, get_json, post, run_dir, served, serving

PLATFORM = REPO / "platforms" / "snax_forge.json"
STATIC = REPO / "snax_forge" / "viz" / "static"


def _texts(node):
    yield node["text"]
    for child in node.get("body", []):
        yield from _texts(child)


# =============================================================================
# View mode
# =============================================================================


@pytest.mark.slow
def test_view_mode_over_dot_2s_steps(tmp_path):
    """dot 2's six step files: step 5 (the link) in compact texts, with Details and the
    estimate's 130 cycles."""
    r = Recipe.load(REPO / "recipes" / "dot_2.json")
    write_steps(r, apply_recipe(r, import_kernel(r.kernel)), tmp_path / "sandbox" / "dot_2")
    with served(make_dfg_server([tmp_path / "sandbox" / "dot_2"], 0, platform=PLATFORM)) as srv:
        assert get_json(srv, "/api/mode") == (200, {"mode": "view"})
        status, steps = get_json(srv, "/api/steps")
        assert status == 200 and [s["label"] for s in steps] == [
            "0 input", "1 split_map", "2 bind", "3 split_map", "4 bind", "5 link"]  # fmt: skip
        status, v = get_json(srv, "/api/step/5")
        texts = [t for row in v["rows"] if row["kind"] == "node" for t in _texts(row["node"])]
        assert {"tmp0 · FIFO 1", "mul · a * b · W 4", "i_t · 128 · temporal"} <= set(texts)
        assert v["containers"]["A"]["badge"] == "L1 · 4 KiB"
        # tmp0 is the link's: no box, the edges from mult into the link and out of it into sum
        boxes = [b for row in v["rows"] if row["kind"] == "containers" for b in row["boxes"]]
        assert boxes == ["A@0", "B@0", "out@1"]
        assert [(e["from"], e["to"]) for e in v["edges"] if e["data"] == "tmp0"] == [
            ("mult.out", "tmp0.in"), ("tmp0.out", "sum.a")]  # fmt: skip
        four = get_json(srv, "/api/step/4")[1]  # before the link, tmp0 in L1 keeps its box
        assert [b for row in four["rows"] if row["kind"] == "containers" for b in row["boxes"]] == [
            "A@0", "B@0", "tmp0@1", "out@1"]  # fmt: skip
        (edge,) = [e for e in v["edges"] if e["data"] == "tmp0" and e["to"].startswith("sum.")]
        d = v["details"][f"edge:{edge['id']}"]
        assert (d["subset"], d["packing"], d["loops"], d["beats"], d["streamer"]) == (
            ["4 * i_t_1:4 * i_t_1 + 4"], "int64 · num_slc 1 · num_ch 4", [["i_t_1", 128]], 128,
            False)  # fmt: skip
        assert [g["cycles"] for g in v["estimate"]["groups"]] == [130]
        assert "tmp0" in v["marks"]["containers"] and "diff" not in v
        s = v["summary"]
        assert (s["accelerator_cycles"], s["not_mapped"], s["groups"][0]["formula"], s["groups"][0]["counter"]) == (
            130, 0, "130 = 128 + 2", 133)  # fmt: skip
        assert (s["l1"]["footprint_text"], s["l1"]["size_text"], s["reads"], s["writes"]) == (
            "8200 B", "128 KiB", 1024, 1)  # fmt: skip
        # Compare: the link step against the one before, the session's diff and the fields
        status, c = get_json(srv, "/api/compare/4/5")
        assert status == 200 and (c["a"], c["b"]) == (4, 5)
        assert c["diff"]["nodes"]["added"] == ["tmp0"] and c["diff"]["containers"]["changed"] == [
            "tmp0"
        ]
        assert "placement" in c["changed"]["container:tmp0"] and "node:tmp0" in c["changed"]
        assert get_json(srv, "/api/compare/4/9")[0] == 404
        assert get_json(srv, "/api/graphs")[0] == 404  # the old page's routes are gone
        assert get_json(srv, "/api/step/6")[0] == 404


# =============================================================================
# Edit mode
# =============================================================================


@functools.cache
def _imported(kernel, dtypes):
    return import_kernel(kernel, dict(dtypes) or None)


@pytest.mark.slow
def test_edit_mode_over_a_vecadd_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        session, "_import", lambda k, d: _imported(k, tuple(sorted((d or {}).items())))
    )
    roots = Roots(tmp_path / "sessions", tmp_path / "recipes", tmp_path / "sandbox")
    with served(make_edit_server(0, roots=roots)) as srv:
        assert get_json(srv, "/api/mode")[1]["mode"] == "edit"
        assert get_json(srv, "/api/session") == (400, {"code": "session.none", "message":
                                                       "no session is open: POST /api/session first"})  # fmt: skip
        assert get_json(srv, "/api/mode")[1]["platform"] == session.PLATFORM
        status, s = post(srv, "/api/session", {"kernel": "vecadd", "params": {"N": 64, "W": 4}})
        assert status == 200 and (s["name"], s["cursor"], s["steps"]) == ("vecadd", 0, [])
        assert (s["labels"], s["undo"], s["redo"], s["recipe"]["kernel"]) == (
            ["0 input"],
            None,
            None,
            "vecadd",
        )
        assert s["command"].startswith("pixi run sandbox ") and s["command"].endswith(
            "vecadd/recipe.json"
        )
        split = get_json(srv, "/api/step/0/site/add_map")[1]["split"]
        assert split["ok"] and [(f["name"], f["value"]) for f in split["form"]] == [
            ("map", "add_map"),
            ("factor", "W"),
        ]
        bind = get_json(srv, "/api/step/0/site/add")[1]["bind"]
        assert not bind["ok"] and "split the map first" in bind["message"]
        logs = []
        for transform, params in (
            ("split", {"map": "add_map", "factor": "W"}),
            ("tag_map", {"map": "add_map_s", "kind": "spatial"}),
            ("tag_map", {"map": "add_map", "kind": "temporal"}),
            ("bind", {"node": "add", "brm": "elementwise_add",
                      "implementation": "chisel_tiled_spatial", "instance": "acc"}),
        ):  # fmt: skip
            status, got = post(srv, "/api/apply", {"transform": transform, "params": params})
            assert status == 200 and got["transform"] == transform, got
            assert got["log"].startswith(f"step {got['index']}  {transform} ")
            logs.append(got["log"])
        # the cycles before and after; the not-mapped count only where it changes
        assert [line.split(" · C ")[1] for line in logs] == [
            "0 → 0", "0 → 0 · not mapped 64 → 16", "0 → 0", "0 → 16 · not mapped 16 → 0"]  # fmt: skip
        assert "changed add" in logs[-1]
        status, v = get_json(srv, "/api/step/4")
        # the tasklet add became the unit add, its spatial map gone
        assert (v["diff"]["nodes"]["removed"], "add" in v["diff"]["nodes"]["changed"]) == (
            ["add_map_s"],
            True,
        )
        assert v["estimate"]["groups"][0]["cycles"] == 16
        assert post(srv, "/api/cursor", {"k": 2})[1] == {
            "ok": True,
            "cursor": 2,
            "log": "cursor → step 2",
        }
        status, got = post(srv, "/api/apply", {"transform": "tag_map",
                                               "params": {"map": "add_map", "kind": "tile"}})  # fmt: skip
        assert (
            status == 400
            and got["code"] == "session.drop"
            and got["question"] == "Drop steps 3..4?"
        )
        assert get_json(srv, "/api/session")[1]["labels"][1:] == [
            "1 split",
            "2 tag_map",
            "3 tag_map",
            "4 bind",
        ]
        assert get_json(srv, "/api/session")[1]["undo"] == 1
        assert get_json(srv, "/api/compare/2/4")[1]["diff"]["nodes"]["removed"] == ["add_map_s"]
        assert post(srv, "/api/cursor", {"k": 4})[0] == 200
        assert post(srv, "/api/cursor", {"k": 9})[0] == 400
        assert post(srv, "/api/params", {"params": {"W": 8}})[0] == 200
        assert get_json(srv, "/api/session")[1]["params"] == {"N": 64, "W": 8}
        assert get_json(srv, "/api/step/4")[1]["estimate"]["groups"][0]["cycles"] == 8
        status, saved = post(srv, "/api/save", {"name": "vecadd_mine"})
        assert status == 200 and (roots.recipes / "vecadd_mine.json").is_file(), saved
        assert post(srv, "/api/save", {"name": "vecadd_mine"})[1]["code"] == "save.name"
        status, out = post(srv, "/api/export", {"k": 1})
        assert status == 200 and Path(out["path"]).name == "1_split.snaxdfg"
        assert [x["name"] for x in get_json(srv, "/api/recipes")[1]] == ["vecadd_mine"]
        assert get_json(srv, "/api/sessions")[1][0]["cursor"] == 4
        assert get_json(srv, "/api/nope")[0] == 404 and post(srv, "/api/nope")[0] == 404


# =============================================================================
# Both viewers: JSON POSTs, and the page's files
# =============================================================================


def test_a_post_that_is_not_json_is_refused(tmp_path):
    """415 without ``Content-Type: application/json``, on the run viewer, the DFG viewer in
    view mode and in edit mode."""
    fixture = REPO / "tests" / "dfg" / "fixtures" / "vecadd.snaxdfg"
    roots = Roots(tmp_path / "s", tmp_path / "r", tmp_path / "b")
    with serving([run_dir(tmp_path / "vecadd", "vecadd", "off")]) as runs, \
            served(make_dfg_server([fixture], 0)) as view, served(make_edit_server(0, roots=roots)) as edit:  # fmt: skip
        for srv, path in ((runs, "/api/reload"), (view, "/api/reload"), (edit, "/api/session")):
            assert post(srv, path, ctype=None)[0] == 415
            assert post(srv, path, ctype="text/plain")[0] == 415
        assert post(runs, "/api/reload")[1] == {"runs": ["vecadd"]}
        assert post(view, "/api/reload")[1] == {"graphs": ["vecadd"]}


def test_the_page_and_every_file_it_imports_are_served(tmp_path):
    """dfg.html, and each file it and its modules import (``src``, ``href``, ``from "./x"``),
    from the server in both modes."""
    html = (STATIC / "dfg.html").read_text()
    wanted, todo = set(), re.findall(r'(?:src|href)="([\w.-]+)"', html)
    while todo:
        name = todo.pop()
        if name in wanted:
            continue
        wanted.add(name)
        if name.endswith(".js"):
            todo += re.findall(r'from "\./([\w.-]+)"', (STATIC / name).read_text())
    assert {"dfg.js", "style.css"} <= wanted
    fixture = REPO / "tests" / "dfg" / "fixtures" / "vecadd.snaxdfg"
    roots = Roots(tmp_path / "s", tmp_path / "r", tmp_path / "b")
    for srv in (make_dfg_server([fixture], 0), make_edit_server(0, roots=roots)):
        with served(srv):
            status, _, body = get(srv, "/")
            assert status == 200 and b"dfg.js" in body
            for name in sorted(wanted):
                assert get(srv, f"/{name}")[0] == 200, name
