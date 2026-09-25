"""The DFG viewer (VIS5, D76, D81): loading files, the view of a graph, the routes.

The rows, boxes and edges are what the panel draws, so they are checked here
exactly; where the browser puts them, and the SVG curves, are checked by eye
(D55) on the three vecadd fixtures and a recipe's output directory.
"""

from __future__ import annotations

import json
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from snax_forge.dfg import Graph
from snax_forge.viz.dfg.__main__ import main
from snax_forge.viz.dfg.api import GraphSet, entry_names, expand, graph_view, load_entry
from snax_forge.viz.dfg.server import make_dfg_server

from .helpers import REPO

FIXTURES = REPO / "tests" / "dfg" / "fixtures"


def view(path: Path) -> dict:
    return graph_view(load_entry(path.stem, path))


def tasklet(tid, ins, outs, code="out = in1 + in2"):
    return {
        "id": tid,
        "kind": "tasklet",
        "inputs": {c: {"data": d, "subset": ["i"]} for c, d in ins.items()},
        "outputs": {c: {"data": d, "subset": ["i"]} for c, d in outs.items()},
        "attrs": {"code": code},
    }


def graph_file(tmp_path: Path, name: str, containers: dict, body: list) -> Path:
    cont = {k: {"shape": ["N"], "dtype": "int64", "transient": t} for k, t in containers.items()}
    g = Graph.from_dict({"name": name, "symbols": {"N": None}, "containers": cont, "body": body})
    path = tmp_path / f"{name}.snaxdfg"
    g.save(path)
    return path


def a_map(mid, *nodes):
    return {"id": mid, "kind": "map", "attrs": {"var": "i", "range": "0:N"}, "body": list(nodes)}


# =============================================================================
# Loading
# =============================================================================


def test_files_and_directories(tmp_path):
    for name in ("10_last", "2_bind", "0_input"):
        shutil.copy(FIXTURES / "vecadd.snaxdfg", tmp_path / f"{name}.snaxdfg")
    (tmp_path / "recipe.json").write_text("{}")
    files = expand([tmp_path, FIXTURES / "vecadd.snaxdfg"])
    assert [f.name for f in files] == [
        "0_input.snaxdfg", "2_bind.snaxdfg", "10_last.snaxdfg", "vecadd.snaxdfg"
    ]  # fmt: skip
    assert entry_names([Path("a/x.snaxdfg"), Path("b/x.snaxdfg"), Path("x.snaxdfg")]) == [
        "x", "x-2", "x-3"
    ]  # fmt: skip
    with pytest.raises(FileNotFoundError, match="no such file"):
        expand([tmp_path / "nope"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no .snaxdfg file"):
        expand([tmp_path / "empty"])


def test_a_broken_file_keeps_its_error(tmp_path):
    bad = tmp_path / "bad.snaxdfg"
    bad.write_text((FIXTURES / "vecadd.snaxdfg").read_text().replace('"data": "B"', '"data": "X"'))
    (tmp_path / "notjson.snaxdfg").write_text("{")
    gs = GraphSet([tmp_path])
    assert [(e["name"], e["error"]) for e in gs.summaries()] == [
        ("bad", "node 'add'.inputs.in2: unknown container 'X'"),
        ("notjson", "invalid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"),
    ]  # fmt: skip
    assert "rows" not in graph_view(gs.get("bad"))
    bad.write_text((FIXTURES / "vecadd.snaxdfg").read_text())
    gs.reload()
    assert gs.get("bad").error is None and graph_view(gs.get("bad"))["rows"]


# =============================================================================
# The view of a graph
# =============================================================================


def test_vecadd_accelerated():
    v = view(FIXTURES / "vecadd_accelerated.snaxdfg")
    assert v["graph"] == "vecadd" and v["symbols"] == {"N": 64}
    assert [r["kind"] for r in v["rows"]] == ["containers", "node", "containers"]
    assert v["rows"][0]["boxes"] == ["A@0", "B@0"] and v["rows"][2]["boxes"] == ["C@1"]
    assert v["boxes"]["C@1"] == {
        "key": "C@1", "container": "C", "version": 1, "written_by": ["add"], "again": False
    }  # fmt: skip
    assert v["containers"]["A"]["sizes"] == [64]
    top = v["rows"][1]["node"]
    assert (top["id"], top["loop_kind"], top["iterations"], top["title"]) == (
        "add_map", "temporal", 16, "i_t in 0:N // 4"
    )  # fmt: skip
    acc = top["body"][0]
    assert acc["title"] == "acc = elementwise_add / chisel_tiled_spatial"
    assert acc["lines"] == ["W = 4, op = add"]
    assert [p["text"] for p in acc["inputs"]] == [
        "A[4 * i_t:4 * i_t + 4]",
        "B[4 * i_t:4 * i_t + 4]",
    ]
    assert v["edges"] == [
        {"from": "A@0", "to": "add.a", "data": "A", "text": "A[4 * i_t:4 * i_t + 4]", "dir": "read"},
        {"from": "B@0", "to": "add.b", "data": "B", "text": "B[4 * i_t:4 * i_t + 4]", "dir": "read"},
        {"from": "add.out", "to": "C@1", "data": "C", "text": "C[4 * i_t:4 * i_t + 4]", "dir": "write"},
    ]  # fmt: skip


def test_iterations_need_bound_symbols():
    plain = view(FIXTURES / "vecadd.snaxdfg")
    assert plain["rows"][1]["node"]["iterations"] is None
    assert plain["containers"]["A"]["sizes"] is None
    split = view(FIXTURES / "vecadd_split.snaxdfg")
    inner = split["rows"][1]["node"]["body"][0]
    assert (inner["id"], inner["loop_kind"], inner["iterations"]) == ("add_map_s", "spatial", 4)


def test_a_transient_between_two_maps_and_a_container_shown_again(tmp_path):
    """C = (A + B) * B: tmp0 between the maps, B shown again above the second map."""
    path = graph_file(
        tmp_path, "two", {"A": False, "B": False, "C": False, "tmp0": True},
        [a_map("add_map", tasklet("add", {"in1": "A", "in2": "B"}, {"out": "tmp0"})),
         a_map("mult_map", tasklet("mult", {"in1": "tmp0", "in2": "B"}, {"out": "C"}, "out = in1 * in2"))],
    )  # fmt: skip
    v = view(path)
    assert [r.get("boxes") for r in v["rows"]] == [
        ["A@0", "B@0"], None, ["tmp0@1", "B@0~1"], None, ["C@1"]
    ]  # fmt: skip
    assert v["boxes"]["B@0~1"]["again"] and v["boxes"]["tmp0@1"]["written_by"] == ["add"]
    assert [(e["from"], e["to"]) for e in v["edges"]] == [
        ("A@0", "add.in1"), ("B@0", "add.in2"), ("add.out", "tmp0@1"),
        ("tmp0@1", "mult.in1"), ("B@0~1", "mult.in2"), ("mult.out", "C@1"),
    ]  # fmt: skip


def test_in_place_gets_a_new_version(tmp_path):
    """A read by the first map and written by the second: A@0 at the top, A@1 at the bottom."""
    path = graph_file(
        tmp_path, "inplace", {"A": False, "B": False, "tmp0": True},
        [a_map("add_map", tasklet("add", {"in1": "A", "in2": "B"}, {"out": "tmp0"})),
         a_map("copy_map", tasklet("copy", {"in1": "tmp0"}, {"out": "A"}, "out = in1"))],
    )  # fmt: skip
    v = view(path)
    assert [r.get("boxes") for r in v["rows"]] == [["A@0", "B@0"], None, ["tmp0@1"], None, ["A@1"]]
    assert ("copy.out", "A@1") in [(e["from"], e["to"]) for e in v["edges"]]


def test_inside_one_node_an_edge_goes_straight_from_the_writer(tmp_path):
    path = graph_file(
        tmp_path, "chain", {"A": False, "B": False, "tmp0": True},
        [a_map("m", tasklet("t1", {"in1": "A"}, {"out": "tmp0"}, "out = in1"),
               tasklet("t2", {"in1": "tmp0"}, {"out": "B"}, "out = in1"))],
    )  # fmt: skip
    v = view(path)
    assert [(e["from"], e["to"], e["data"]) for e in v["edges"]] == [
        ("A@0", "t1.in1", "A"), ("t1.out", "t2.in1", "tmp0"),
        ("t1.out", "tmp0@1", "tmp0"), ("t2.out", "B@1", "B"),
    ]  # fmt: skip
    assert v["rows"][-1]["boxes"] == ["tmp0@1", "B@1"]


def test_unused_containers_are_in_the_first_row(tmp_path):
    path = graph_file(
        tmp_path, "unused", {"A": False, "B": False, "Z": False},
        [a_map("m", tasklet("t", {"in1": "A"}, {"out": "B"}, "out = in1"))],
    )  # fmt: skip
    assert view(path)["rows"][0]["boxes"] == ["A@0", "Z@0"]


# =============================================================================
# Server and command line
# =============================================================================


@contextmanager
def serving(paths):
    srv = make_dfg_server(list(paths), port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)


def get(srv, path):
    with urlopen(srv.url.rstrip("/") + path, timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_routes(tmp_path):
    path = tmp_path / "vecadd.snaxdfg"
    shutil.copy(FIXTURES / "vecadd.snaxdfg", path)
    with serving([path]) as srv:
        status, ctype, body = get(srv, "/")
        assert status == 200 and ctype == "text/html" and b"dfg.js" in body
        assert get(srv, "/dfg.js")[1] == "text/javascript"
        assert json.loads(get(srv, "/api/graphs")[2]) == [
            {"name": "vecadd", "path": str(path), "graph": "vecadd", "error": None}
        ]
        assert json.loads(get(srv, "/api/graph/vecadd")[2])["rows"][0]["boxes"] == ["A@0", "B@0"]
        with pytest.raises(HTTPError) as e:
            get(srv, "/api/graph/nope")
        assert e.value.code == 404
        path.write_text(path.read_text().replace('"N": null', '"N": 8'))
        with urlopen(Request(srv.url + "api/reload", method="POST"), timeout=10) as r:
            assert json.loads(r.read()) == {"graphs": ["vecadd"]}
        assert json.loads(get(srv, "/api/graph/vecadd")[2])["symbols"] == {"N": 8}


def test_the_run_viewer_still_serves_its_own_page(tmp_path):
    from .helpers import run_dir
    from .helpers import serving as run_serving

    with run_serving([run_dir(tmp_path / "vecadd", "vecadd", "off")]) as srv:
        assert b"app.js" in get(srv, "/")[2]


def test_cli_rejects_a_missing_path(tmp_path, capsys):
    assert main([str(tmp_path / "nope.snaxdfg"), "--port", "0"]) == 2
    assert "no such file" in capsys.readouterr().err
