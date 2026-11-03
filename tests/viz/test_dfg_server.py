"""The graph page's server in both modes (D55, viz/dfg/server.py), on servers at port 0: view
mode over a recipe's step files, edit mode over a session (sandbox/session.py) with its roots in
``tmp_path``, the JSON-only POSTs on both viewers, and every static file the page imports."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import quote

import pytest

from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe, session, write_steps
from snax_forge.sandbox.session import Roots
from snax_forge.viz.dfg.server import make_dfg_server, make_edit_server
from tests.helpers import REPO, imported

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
        assert get_json(srv, "/api/mode") == (200, {"mode": "view", "techs": ["nangate45", "nangate15"],
                                                    "tech": "nangate45"})  # fmt: skip
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
        # the cost (C§20): a unit's, a buffer link's and a group's Details, the panel's figures,
        # an area to 3 significant figures, an energy to 3 decimals; ?tech= picks the
        # technology, an unknown one 400
        assert v["details"]["node:sum"]["cost"] == {
            "found": "measured on nangate45", "area": "1 640 µm² (comb 1 300, flops 340)",
            "per fire": "2.495 pJ", "holding": "0.457 pJ per cycle",
            "per task": "0.379 nJ (128 fires, 64 bits)"}  # fmt: skip
        assert list(v["details"]["node:sum"])[-2:] == ["timing", "cost"]
        assert v["details"]["node:tmp0"]["cost"]["found"] == "copy on nangate45"
        assert v["details"]["group:dot"]["cost"] == {
            "area": "32 200 µm²", "energy": "6.945 nJ", "ctrl": "196 bits, 1 040 µm², 0.193 nJ"}  # fmt: skip
        assert "cost" not in v["details"]["group:dot"]["estimate"]
        assert (s["groups"][0]["area_text"], s["groups"][0]["energy_text"]) == (
            "32 200 µm²",
            "6.945 nJ",
        )
        assert (s["cost"]["heading"], s["cost"]["missing"]) == (
            "Cost estimate (first-order) on nangate45", [])  # fmt: skip
        assert "about 20 %" in s["cost"]["caveat"] and v["tech"] == "nangate45"
        status, v15 = get_json(srv, "/api/step/5?tech=nangate15")
        assert status == 200 and v15["tech"] == "nangate15"
        assert v15["summary"]["cost"]["heading"].endswith("nangate15")
        assert v15["details"]["node:sum"]["cost"]["found"] == "measured on nangate15"
        assert v15["summary"]["groups"][0]["cycles"] == 130  # the cycles do not change
        assert get_json(srv, "/api/compare/4/5?tech=nangate15")[0] == 200
        status, bad = get_json(srv, "/api/step/5?tech=tsmc7")
        assert status == 400 and bad["code"] == "char.tech"
        assert get_json(srv, "/api/compare/4/9")[0] == 404
        assert get_json(srv, "/api/graphs")[0] == 404  # the old page's routes are gone
        assert get_json(srv, "/api/step/6")[0] == 404


def test_view_mode_over_two_designs(tmp_path):
    """Two recipe folders with the same stems: the steps route names each by folder, with its
    label and folder; Compare between them heads each column with folder and label; /api/find
    reads the hash's name, a unique bare stem or a number."""
    fixtures = REPO / "tests" / "dfg" / "fixtures"
    for d in ("dot4_w8", "dot5_w8"):
        (tmp_path / d).mkdir()
        shutil.copy(fixtures / "vecadd.snaxdfg", tmp_path / d / "0_input.snaxdfg")
        shutil.copy(fixtures / "vecadd_accelerated.snaxdfg", tmp_path / d / "1_bind.snaxdfg")
    shutil.copy(fixtures / "vecadd_accelerated.snaxdfg", tmp_path / "dot5_w8" / "2_link.snaxdfg")
    with served(make_dfg_server([tmp_path / "dot4_w8", tmp_path / "dot5_w8"], 0)) as srv:
        status, steps = get_json(srv, "/api/steps")
        assert status == 200 and [(s["name"], s["folder"], s["label"]) for s in steps] == [
            ("dot4_w8/0_input", "dot4_w8", "0 input"), ("dot4_w8/1_bind", "dot4_w8", "1 bind"),
            ("dot5_w8/0_input", "dot5_w8", "0 input"), ("dot5_w8/1_bind", "dot5_w8", "1 bind"),
            ("dot5_w8/2_link", "dot5_w8", "2 link")]  # fmt: skip
        status, v = get_json(srv, "/api/step/3")
        assert status == 200 and (v["name"], v["folder"], v["label"]) == (
            "dot5_w8/1_bind", "dot5_w8", "1 bind")  # fmt: skip
        status, c = get_json(srv, "/api/compare/1/3")
        assert status == 200 and c["columns"]["labels"] == ["dot4_w8 · 1 bind", "dot5_w8 · 1 bind"]
        assert c["diff"] is not None and c["columns"]["details"]
        find = {ref: get_json(srv, f"/api/find?step={quote(ref, safe='')}")[1]["k"]
                for ref in ("dot5_w8/1_bind", "2_link", "1_bind", "4", "9")}  # fmt: skip
        assert find == {"dot5_w8/1_bind": 3, "2_link": 4, "1_bind": None, "4": 4, "9": None}
        status, bad = get_json(srv, "/api/find")
        assert status == 400 and bad["code"] == "request.query"


# =============================================================================
# Edit mode
# =============================================================================


@pytest.mark.slow
def test_edit_mode_over_a_vecadd_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_import", imported)
    roots = Roots(tmp_path / "sessions", tmp_path / "recipes", tmp_path / "sandbox")
    with served(make_edit_server(0, roots=roots)) as srv:
        assert get_json(srv, "/api/mode")[1]["mode"] == "edit"
        assert get_json(srv, "/api/session") == (400, {"code": "session.none", "message":
                                                       "no session is open: POST /api/session first"})  # fmt: skip
        assert get_json(srv, "/api/mode")[1]["platform"] == session.PLATFORM
        assert get_json(srv, "/api/mode")[1]["tech"] == "nangate45"
        status, bad = post(srv, "/api/session", {"kernel": "vecadd", "params": {}})
        assert status == 400 and bad["message"] == "vecadd: give N, e.g. --set N=2048"
        status, s = post(srv, "/api/session", {"kernel": "vecadd", "params": {"N": 64}})
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
            ("factor", 4),
        ]
        bind = get_json(srv, "/api/step/0/site/add")[1]["bind"]
        assert not bind["ok"] and "split the map first" in bind["message"]
        status, got = post(srv, "/api/apply", {"transform": "split",
                                               "params": {"map": "add_map", "factor": "W"}})  # fmt: skip
        assert (
            status == 400
            and got["message"] == "split: factor 'W' is not a param; give its value"
            and got["param"] == "W"
        )
        logs = []
        for transform, params in (
            ("split", {"map": "add_map", "factor": "W"}),
            ("tag_map", {"map": "add_map_inner", "kind": "spatial"}),
            ("tag_map", {"map": "add_map", "kind": "temporal"}),
            ("bind", {"node": "add", "brm": "elementwise_add",
                      "implementation": "chisel_tiled_spatial", "instance": "acc"}),
        ):  # fmt: skip
            new = {"W": 4} if transform == "split" else None
            status, got = post(srv, "/api/apply", {"transform": transform, "params": params,
                                                   "new_params": new})  # fmt: skip
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
            ["add_map_inner"],
            True,
        )
        assert v["estimate"]["groups"][0]["cycles"] == 16
        # 3 ports x 16 beats x 4 channels over 32 banks: 6 cycles, under 16, no note (D151)
        assert v["estimate"]["groups"][0]["bank_bound"] == 6
        assert (v["summary"]["groups"][0]["bank_bound"], v["summary"]["groups"][0]["bank_note"]) == (
            6, None)  # fmt: skip
        assert (
            v["summary"]["cost"]["heading"].endswith("nangate45")
            and "cost" in v["details"]["node:add"]
        )
        v15 = get_json(srv, "/api/step/4?tech=nangate15")[1]  # the session caches per technology
        assert v15["summary"]["cost"]["heading"].endswith("nangate15")
        assert v15["summary"]["cost"]["area_text"] != v["summary"]["cost"]["area_text"]
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
        assert get_json(srv, "/api/session")[1]["drop"] == "Drop steps 3..4?"  # the page asks first
        assert get_json(srv, "/api/session")[1]["labels"][1:] == [
            "1 split",
            "2 tag_map",
            "3 tag_map",
            "4 bind",
        ]
        assert get_json(srv, "/api/session")[1]["undo"] == 1
        assert get_json(srv, "/api/compare/2/4")[1]["diff"]["nodes"]["removed"] == ["add_map_inner"]
        assert post(srv, "/api/cursor", {"k": 4})[0] == 200
        assert post(srv, "/api/cursor", {"k": 9})[0] == 400
        assert post(srv, "/api/params", {"params": {"W": 8}})[0] == 200
        assert get_json(srv, "/api/session")[1]["params"] == {"N": 64, "W": 8}
        assert get_json(srv, "/api/step/4")[1]["estimate"]["groups"][0]["cycles"] == 8
        status, saved = post(srv, "/api/save", {"name": "vecadd_mine"})
        assert status == 200 and (roots.recipes / "vecadd_mine.json").is_file(), saved
        assert saved["log"] == f"saved {saved['shown']}" and saved["commands"][2]["command"] == (
            f"pixi run forge {saved['shown']}")  # fmt: skip
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
