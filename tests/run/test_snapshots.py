"""run/snapshots.py: SNAX-GVSoC's snapshots read in the order taken, and the windows between
them named by the quiet build's marks."""

import pytest

from snax_forge.run import RunError, parse_line
from snax_forge.run.snapshots import Window, snapshots, windows

from .helpers import line, snap_lines

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


def test_a_snap_line_reads_in_each_form():
    got = parse_line(line("snax_acc0/unit", "summary", "5 snap mark=2 unit=mul firings=3 pushes=1"))
    assert (got.kind, got.fields) == ("snap", {"mark": 2, "unit": "mul", "firings": 3, "pushes": 1})
    got = parse_line(line("tcdm/snax_xbar", "summary", "5 snap mark=-1 input=9 grants=4 stalls=0"))
    assert got.fields == {"mark": -1, "input": 9, "grants": 4, "stalls": 0}
    got = parse_line(line("tcdm/snax_xbar", "summary", "5 snap mark=3 copy_reads=64 copy_writes=8"))
    assert got.fields == {"mark": 3, "copy_reads": 64, "copy_writes": 8}
    got = parse_line(line("tcdm/snax_xbar", "summary", "5 copies copy_reads=64 copy_writes=8"))
    assert (got.kind, got.fields) == ("copies", {"copy_reads": 64, "copy_writes": 8})


def test_snapshots_come_in_the_order_taken_named_as_the_converter_names(cluster):
    snaps = snapshots(read(cluster), cluster)
    assert [(s.mark, s.cycle) for s in snaps] == [(1, 100), (2, 400), (-1, 600)]
    end = snaps[-1].counters
    assert end.inputs["acc_a.lane0"] == (16, 2) and end.inputs["core0"] == (7, 0)
    assert end.units == {"acc": (16, 16)}
    assert end.groups == {"acc": {"busy": 18, "stall_in": 0, "stall_out": 0, "flush": 1}}
    assert end.movers["acc_out"]["stall_xbar"] == 2 and len(end.banks) == 2


def test_windows_are_differences_named_by_their_marks_and_add_up_to_the_end(cluster):
    snaps = snapshots(read(cluster), cluster)
    ws = windows(snaps, MARKS)
    assert [(w.index, w.mark, w.kind, w.label, w.first, w.last) for w in ws] == [
        (0, 0, "boot", "boot", 0, 100), (1, 1, "dma_in", "dma load_A", 100, 400),
        (2, 2, "run", "run acc", 400, 600)]  # fmt: skip
    assert ws[0].counters.inputs["core0"] == (7, 0) and ws[2].counters.inputs["core0"] == (0, 0)
    assert ws[2].counters.units == {"acc": (16, 16)} and ws[2].counters.busy() == 19
    for port, (grants, stalls) in snaps[-1].counters.inputs.items():
        assert sum(w.counters.inputs[port][0] for w in ws) == grants
        assert sum(w.counters.inputs[port][1] for w in ws) == stalls


def test_a_copy_window_counts_the_copys_bank_words_and_a_run_without_them_none(cluster):
    copy = [{"mark": 1, "kind": "dma_copy", "label": "dma transpose_A_T"}, MARKS[1]]
    snaps = [(m, c, x | {"copies": cp}) for (m, c, x), cp in
             zip(RUN, [(0, 0), (4096, 4096), (4096, 4096)], strict=True)]  # fmt: skip
    ws = windows(snapshots(read(cluster, snaps), cluster), copy)
    assert [(w.kind, w.counters.copies) for w in ws] == [
        ("boot", (0, 0)), ("dma_copy", (4096, 4096)), ("run", (0, 0))]  # fmt: skip
    assert Window.from_dict(ws[1].to_dict()) == ws[1]
    old = ws[1].to_dict()
    del old["counters"]["copies"]
    assert Window.from_dict(old).counters.copies == (0, 0)
    plain = windows(snapshots(read(cluster), cluster), copy)
    assert all(w.counters.copies == (0, 0) for w in plain)


def test_a_mark_the_marks_file_lacks_is_named_by_its_value(cluster):
    ws = windows(snapshots(read(cluster), cluster), MARKS[:1])
    assert (ws[2].kind, ws[2].label) == ("mark", "mark 2")


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


def test_snap_lines_leave_the_converted_run_as_it_was(cluster, tasks):
    """Every run now writes the end's snapshot: the profile and the trace do not read it."""
    from snax_forge.run.convert import convert

    from .helpers import synthetic

    plain = [parse_line(x) for x in synthetic(cluster, tasks)]
    end = [x for x in read(cluster) if x.kind == "snap"]
    for level in ("off", "task"):
        a = convert(plain, cluster, tasks, 226, level, {"passed": True})
        b = convert(plain + end, cluster, tasks, 226, level, {"passed": True})
        assert a[0] == b[0] and (a[1] is None or a[1].events == b[1].events)
