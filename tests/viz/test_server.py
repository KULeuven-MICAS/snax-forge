"""Tests for the visualiser's HTTP server (VIS1, D55): it starts, serves the
viewer and the run list, answers an events query, and a reload picks up a
changed directory. Only the plumbing; api.py's answers are test_api.py's."""

import json
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
