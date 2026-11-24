"""run/snapshots.py: SNAX-GVSoC's snapshots read in the order taken, and the windows between
them named by the quiet build's marks."""

import pytest

from snax_forge.run import RunError, parse_line
from snax_forge.run.snapshots import Window, snapshots, windows

from .helpers import snap_lines

MARKS = [{"mark": 1, "kind": "dma_in", "label": "dma load_A"},
         {"mark": 2, "kind": "run", "label": "run acc"}]  # fmt: skip
LANES = ("acc_a", "acc_b", "acc_out")


def counters(grants: int, stalls: int, firings: int, busy: int) -> dict:
    """Running totals: every lane of every mover ``grants`` and ``stalls``, core0 7 grants, the
    unit ``firings``, the group ``busy`` cycles and one of flush."""
    inputs = {f"{m}.lane{k}": (grants, stalls) for m in LANES for k in range(4)}
    return {"inputs": inputs | {"core0": (7, 0)},
            "group": {"busy": busy, "stall_in": 0, "stall_out": 0, "flush": int(busy > 0)},
            "units": {"acc": (firings, firings)},
            "movers": {m: {"busy": busy, "stall_xbar": stalls, "stall_fifo": 0, "idle": 0}
                       for m in LANES}}  # fmt: skip


RUN = [(1, 100, counters(0, 0, 0, 0)), (2, 400, counters(0, 0, 0, 0)),
       (-1, 600, counters(16, 2, 16, 18))]  # fmt: skip


def read(cluster, snaps=RUN):
    return [parse_line(x) for x in snap_lines(cluster, snaps)]


def test_a_window_round_trips(cluster):
    w = windows(snapshots(read(cluster), cluster), MARKS)[2]
    assert Window.from_dict(w.to_dict()) == w


def test_a_run_without_snap_lines_has_none(cluster):
    assert snapshots([x for x in read(cluster) if x.kind != "snap"], cluster) == []
    assert windows([], MARKS) == []


def test_components_that_took_different_marks_are_refused(cluster):
    lines = [x for x in read(cluster)
             if not (x.path.endswith("streamer_acc") and x.fields.get("mark") == 2)]  # fmt: skip
    with pytest.raises(RunError, match="snapshots.marks"):
        snapshots(lines, cluster)


def test_a_run_whose_last_snapshot_is_not_the_end_is_refused(cluster):
    lines = [x for x in read(cluster) if x.kind != "snap" or x.fields["mark"] != -1]
    with pytest.raises(RunError, match="snapshots.end"):
        snapshots(lines, cluster)


def test_an_end_that_is_not_the_whole_run_is_refused(cluster):
    lines = [parse_line(x.replace("grants=16 stalls=2 stalls_dma", "grants=15 stalls=2 stalls_dma"))
             for x in snap_lines(cluster, RUN)]  # fmt: skip
    with pytest.raises(RunError, match="whole run"):
        snapshots(lines, cluster)
