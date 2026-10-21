"""run/trace.py: the trace's events checked against their kinds, and its round trip (C§7)."""

import json

import pytest

from snax_forge.run import LEVELS, RunError, Trace


def events():
    return [
        {"t": 0, "k": "func", "src": "core0", "name": "main", "pc": 0x800001E0},
        {"t": 113, "k": "barrier", "src": "core0", "last": 131},
        {"t": 132, "k": "start", "src": "acc_a", "task": "add_acc_a"},
        {"t": 133, "k": "start", "src": "acc", "task": "add_acc"},
        {"t": 134, "k": "poll", "src": "core0", "pc": 0x80000378, "csr": 0x3D4, "block": "acc",
         "value": 1, "count": 10, "last": 152},
        {"t": 132, "k": "csr", "src": "core0", "pc": 0x80000370, "csr": 0x3CF,
         "block": "acc_streamer", "op": "w", "value": 1},
        {"t": 133, "k": "csr", "src": "core0", "pc": 0x80000374, "csr": 0x3D3, "block": "acc",
         "op": "w", "value": 1, "held": 2},
        {"t": 153, "k": "done", "src": "acc", "task": "add_acc", "firings": 16, "first": 137,
         "last": 152, "cycles": {"busy": 16, "stall_in": 3, "stall_out": 0, "flush": 0}},
        {"t": 150, "k": "done", "src": "acc_a", "task": "add_acc_a",
         "cycles": {"busy": 16, "stall_xbar": 0, "stall_fifo": 2, "idle": 1}},
        {"t": 88, "k": "issue", "src": "dma", "task": "load_A", "id": 2, "dir": "l2_to_l1",
         "from": 0x80002900, "to": 0x10000000, "size": 512, "from_stride": 0, "to_stride": 0,
         "reps": 1},
        {"t": 110, "k": "done", "src": "dma", "task": "load_A", "id": 2},
        {"t": 0, "k": "state", "src": "core1", "mode": "run"},
    ]  # fmt: skip


def test_round_trip():
    t = Trace("task", 226, 2812, ["acc", "acc_a", "dma", "core0", "core1"], events())
    assert Trace.from_dict(json.loads(json.dumps(t.to_dict()))) == t
    assert t.meta() == {"level": "task", "total_cycles": 226, "boot": 2812,
                        "sources": ["acc", "acc_a", "dma", "core0", "core1"], "filter": []}  # fmt: skip
    assert [e["task"] for e in t.of_kind("done")] == ["add_acc", "add_acc_a", "load_A"]
    assert [e["task"] for e in t.of_kind("done", "dma")] == ["load_A"]


def beat_events():
    """One event of each beat, stall and instruction kind, as the converter writes them."""
    return [
        {"t": 121, "k": "grant", "src": "acc_a.lane0", "task": "add_acc_a", "op": "r",
         "mem": "l1", "addr": 0x10000000, "bank": 0, "row": 0, "value": 0x2BD, "back": 122,
         "b": 0, "region": "A", "elems": [0], "vals": [701]},
        {"t": 0, "k": "grant", "src": "core0", "task": None, "op": "w", "mem": "l1",
         "addr": 0x1001FF8C, "bank": 17, "row": 511, "value": 0, "back": 0, "size": 4,
         "region": None, "elems": [], "vals": []},
        {"t": 92, "k": "dma", "src": "dma", "task": "load_A", "id": 2, "i": 0, "op": "w",
         "mem": "l1", "addr": 0x10000000, "size": 64, "bank": 0, "row": 0, "value": [0x2BD] * 8,
         "l2": 0x80002900, "region": "A", "elems": list(range(8)), "vals": [701] * 8},
        {"t": 78, "k": "burst", "src": "dma", "task": "load_A", "id": 2, "op": "r", "mem": "l2",
         "addr": 0x80002900, "size": 512, "back": 90, "region": "A", "elems": list(range(64))},
        {"t": 124, "k": "fire", "src": "acc", "task": "add_acc", "n": 0,
         "b": {"acc_a": 0, "acc_b": 0}, "value": {"acc_a": [0x2BD], "acc_b": [0x2EE]},
         "vals": {"acc_a": [701], "acc_b": [750]}},
        {"t": 124, "k": "push", "src": "acc", "task": "add_acc", "n": 0, "first": 0, "last": 0,
         "value": [0x5AB], "vals": [1451]},
        {"t": 121, "k": "stall", "src": "acc_b.lane0", "task": "add_acc_b", "addr": 0x10000200,
         "bank": 0, "row": 2, "by": "acc_a.lane0", "by_addr": 0x10000000, "b": 0,
         "region": "B", "elems": [0]},
        {"t": 123, "k": "fifo", "src": "acc_a", "task": "add_acc_a", "lane": 0, "count": 1},
        {"t": 0, "k": "insn", "src": "core0", "pc": 0x800001E0, "func": "main",
         "text": "addi sp, sp, -16 sp=1001ffb0 sp:1001ffc0"},
    ]  # fmt: skip


def test_the_levels_and_the_beat_kinds():
    assert LEVELS == ("off", "task", "beat", "stall", "finegrained")
    t = Trace("finegrained", 226, 2812, ["acc", "acc_a", "dma", "core0"], events() + beat_events(),
              ["acc", "acc_a", "dma", "core0"])  # fmt: skip
    back = Trace.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back == t and back.filter == ["acc", "acc_a", "dma", "core0"]
    assert t.meta()["filter"] == ["acc", "acc_a", "dma", "core0"]
    assert [e["src"] for e in t.of_kind("grant")] == ["acc_a.lane0", "core0"]


@pytest.mark.parametrize(
    "event, code",
    [
        ({"t": 0, "k": "cmd", "src": "core0"}, "trace.kind"),
        ({"k": "state", "src": "core0", "mode": "run"}, "trace.kind"),
        ({"t": 0, "k": "state", "src": "core0"}, "trace.field"),
        ({"t": 0, "k": "state", "src": "core0", "mode": "run", "why": "wfi"}, "trace.field"),
        ({"t": 0, "k": "done", "src": "acc", "task": "x", "firings": 1}, "trace.field"),
        ({"t": 0, "k": "fifo", "src": "acc_a", "task": "x", "lane": 0}, "trace.field"),
        ({"t": 0, "k": "insn", "src": "core0", "pc": 0, "text": "nop"}, "trace.field"),
    ],
)
def test_an_event_not_of_its_kind_is_a_named_error(event, code):
    with pytest.raises(RunError) as e:
        Trace("task", 1, 0, [], [event])
    assert e.value.code == code


def test_off_has_no_trace():
    with pytest.raises(RunError) as e:
        Trace("off", 1, 0)
    assert e.value.code == "trace.level"
