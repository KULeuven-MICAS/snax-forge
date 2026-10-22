"""Tests for the visualiser's HTTP server: it starts, serves the viewer and
the run list, answers an events query, and a reload picks up a changed
directory. Only the plumbing; api.py's answers are test_api.py's."""

import json
import re
from pathlib import Path
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
        assert get(srv, "/memory.js")[1] == "text/javascript"  # the memory tab
        runs = get_json(srv, "/api/runs")
        assert runs == [
            {"name": "vecadd", "path": str(d), "design": "vecadd",
             "trace_level": "task", "total_cycles": 226}
        ]  # fmt: skip
        evs = get_json(srv, "/api/run/vecadd/events?from=0&to=60&src=core0")["events"]
        assert evs and all(e["src"] == "core0" and e["t"] < 60 for e in evs)
        evs = get_json(srv, "/api/run/vecadd/events?from=0&to=226&k=start,done")["events"]
        assert evs and {e["k"] for e in evs} == {"start", "done"}
        detail = get_json(srv, "/api/run/vecadd")
        assert detail["profile"]["total_cycles"] == 226 and detail["cluster"]["design"] == "vecadd"


def test_every_module_the_viewer_imports_is_served(tmp_path):
    """Each ``from "./x.js"`` in the static modules names a file the server gives out as JS."""
    static = Path(__file__).parents[2] / "snax_forge" / "viz" / "static"
    wanted = {
        m for f in static.glob("*.js") for m in re.findall(r'from "\./([\w-]+\.js)"', f.read_text())
    }
    assert "player.js" in wanted  # the play button
    d = run_dir(tmp_path / "vecadd", "vecadd", "task")
    with serving([d]) as srv:
        for m in sorted(wanted):
            assert get(srv, f"/{m}")[:2] == (200, "text/javascript"), m


def test_reload_picks_up_a_changed_directory(tmp_path):
    d = run_dir(tmp_path / "run", "vecadd", "off")
    with serving([d]) as srv:
        assert get_json(srv, "/api/runs")[0]["trace_level"] == "off"
        run_dir(d, "vecadd", "task")  # the folder made again
        assert get_json(srv, "/api/runs")[0]["trace_level"] == "off"  # read once, not watched
        req = Request(srv.url + "api/reload", method="POST")
        with urlopen(req, timeout=10) as r:
            assert json.loads(r.read()) == {"runs": ["run"]}
        assert get_json(srv, "/api/run/run")["trace"]["level"] == "task"


def status_of(srv, path):
    try:
        return get(srv, path)[0]
    except HTTPError as e:
        return e.code


def test_memory_routes_answer_and_movement_says_why_not(tmp_path):
    d = run_dir(tmp_path / "vecadd", "vecadd", "task")
    with serving([d]) as srv:
        view = get_json(srv, "/api/run/vecadd/memory")
        assert [m["mem"] for m in view["memories"]] == ["l1", "l2"]
        assert "needs --trace stall" in view["marks_missing"]["conflicts"]
        rows = get_json(srv, "/api/run/vecadd/memory/l1/rows?from=0&to=4")["rows"]
        assert [r["row"] for r in rows] == [0, 1, 2, 3]
        assert status_of(srv, "/api/run/vecadd/memory?marks=conflicts") == 400  # needs stall
        assert status_of(srv, "/api/run/vecadd/memory?marks=later") == 400
        assert status_of(srv, "/api/run/vecadd/memory/l1/rows?from=0&to=999") == 400
        assert status_of(srv, "/api/run/vecadd/memory/l3/rows?from=0&to=4") == 404
        assert get_json(srv, "/api/run/vecadd/memory/cycle?t=5") == {
            "t": 5,
            "level": "task",
            "events": [],
            "returns": [],
        }  # no beat lines at task
        assert status_of(srv, "/api/run/vecadd/memory/cycle") == 400
        assert status_of(srv, "/api/run/nope/memory/cycle?t=5") == 404
        assert get_json(srv, "/api/run/vecadd/movement")["available"] is False
        assert get_json(srv, "/api/run/vecadd/journey?region=A&index=0")["available"] is False
        assert status_of(srv, "/api/run/vecadd/journey?region=Q&index=0") == 400
        assert get_json(srv, "/api/run/vecadd/fifo")["available"] is False
        assert status_of(srv, "/api/run/nope/memory") == 404


def test_schedule_routes_answer_and_say_why_not(tmp_path):
    """The schedule at task (its detail needs beat), and at off neither."""
    task = run_dir(tmp_path / "vecadd", "vecadd", "task")
    off = run_dir(tmp_path / "vsum", "vsum", "off")
    with serving([task, off]) as srv:
        s = get_json(srv, "/api/run/vecadd/schedule")
        assert s["available"] and s["segments"][0]["from"] == 0 and s["segments"][-1]["to"] == 226
        folds = [x["from"] for x in s["segments"] if x["folded"]]
        opened = get_json(srv, f"/api/run/vecadd/schedule?open={folds[0]}")["segments"]
        assert [x["from"] for x in opened if x["folded"]] == folds[1:]
        held = get_json(srv, f"/api/run/vecadd/schedule?cycle={folds[0]}")["segments"]
        assert [x["from"] for x in held if x["folded"]] == folds[1:]
        d = get_json(srv, "/api/run/vecadd/schedule/detail?from=0&to=50")
        assert d["available"] is False and "needs --trace beat" in d["reason"]
        assert status_of(srv, "/api/run/vecadd/schedule/detail?from=0") == 400
        assert status_of(srv, "/api/run/vecadd/schedule?open=x") == 400
        s = get_json(srv, "/api/run/vsum/schedule")
        assert s["available"] is False and "traced at off" in s["reason"]


def test_cluster_and_step_routes_answer(tmp_path):
    d = run_dir(tmp_path / "vecadd", "vecadd", "task")
    with serving([d]) as srv:
        first = get_json(srv, "/api/run/vecadd/step?t=0&dir=1&by=boundary")["t"]
        assert first > 0
        second = get_json(srv, f"/api/run/vecadd/step?t={first}&dir=1&by=boundary")["t"]
        assert second > first
        assert get_json(srv, f"/api/run/vecadd/step?t={second}&dir=-1&by=boundary")["t"] == first
        assert get_json(srv, f"/api/run/vecadd/step?t={first}&dir=-1&by=boundary")["t"] == first
        at = get_json(srv, f"/api/run/vecadd/cluster?t={first}")
        assert at["t"] == first and at["events"] and all(e["t"] == first for e in at["events"])
        assert at["needs"]["grant"] == "beat"
        assert status_of(srv, "/api/run/vecadd/cluster") == 400
        assert status_of(srv, "/api/run/vecadd/step?t=0&dir=2&by=event") == 400
        assert status_of(srv, "/api/run/vecadd/step?t=0&dir=1&by=later") == 400
