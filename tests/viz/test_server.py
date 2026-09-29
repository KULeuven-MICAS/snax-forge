"""Tests for the visualiser's HTTP server (VIS1, D55): it starts, serves the
viewer and the run list, answers an events query, and a reload picks up a
changed directory. Only the plumbing; api.py's answers are test_api.py's."""

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .helpers import run_dir, serving


def get(srv, path):
    with urlopen(srv.url.rstrip("/") + path, timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def get_json(srv, path):
    return json.loads(get(srv, path)[2])


def test_server_serves_viewer_and_runs(tmp_path):
    d = run_dir(tmp_path / "vecadd", "vecadd", "task")
    with serving([d]) as srv:
        assert srv.server_address[0] == "127.0.0.1"
        status, ctype, body = get(srv, "/")
        assert status == 200 and ctype == "text/html" and b"app.js" in body
        assert get(srv, "/app.js")[1] == "text/javascript"  # modules need a JS type
        runs = get_json(srv, "/api/runs")
        assert runs == [
            {"name": "vecadd", "path": str(d), "scenario": "vecadd",
             "trace_level": "task", "total_cycles": 77}
        ]  # fmt: skip
        evs = get_json(srv, "/api/run/vecadd/events?from=0&to=5&src=ctl")["events"]
        assert evs and all(e["src"] == "ctl" and e["t"] < 5 for e in evs)
        evs = get_json(srv, "/api/run/vecadd/events?from=0&to=200&k=start,done")["events"]
        assert evs and {e["k"] for e in evs} == {"start", "done"}


def test_reload_picks_up_a_changed_directory(tmp_path):
    d = run_dir(tmp_path / "run", "vecadd", "off")
    with serving([d]) as srv:
        assert get_json(srv, "/api/runs")[0]["total_cycles"] == 77
        run_dir(d, "vecadd_conflict", "task")  # overwrite the directory
        assert get_json(srv, "/api/runs")[0]["total_cycles"] == 77  # read once, not watched
        req = Request(srv.url + "api/reload", method="POST")
        with urlopen(req, timeout=10) as r:
            assert json.loads(r.read()) == {"runs": ["run"]}
        assert get_json(srv, "/api/runs")[0]["total_cycles"] == 85
        assert get_json(srv, "/api/run/run")["trace"]["level"] == "task"


def status_of(srv, path):
    try:
        return get(srv, path)[0]
    except HTTPError as e:
        return e.code


def test_memory_routes(tmp_path):
    """The memory tab's two routes (D96): the folded view and rows on request."""
    d = run_dir(tmp_path / "vecadd", "vecadd", "off")
    with serving([d]) as srv:
        view = get_json(srv, "/api/run/vecadd/memory")
        assert view["has_regions"] and [m["mem"] for m in view["memories"]] == ["l1", "l2"]
        rows = get_json(srv, "/api/run/vecadd/memory/l1/rows?from=4&to=6")
        assert [r["row"] for r in rows["rows"]] == [4, 5]
        # B's first element in bank 8 of row 4; regions are numbered per memory (A, B, C in L1)
        assert rows["rows"][0]["cells"][8] == [[1, 0]]
        assert len(get_json(srv, "/api/run/vecadd/memory/l2/rows")["rows"]) == 256  # default
        assert status_of(srv, "/api/run/vecadd/memory/l1/rows?from=0&to=300") == 400
        assert status_of(srv, "/api/run/vecadd/memory/l1/rows?from=x") == 400
        assert status_of(srv, "/api/run/vecadd/memory/l3/rows") == 404
        assert status_of(srv, "/api/run/nope/memory") == 404
