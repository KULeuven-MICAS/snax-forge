"""run/trace.py: the trace's events checked against their kinds, and its round trip (C§7)."""

import json

import pytest

from snax_forge.run import RunError, Trace


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
         "last": 152},
        {"t": 150, "k": "done", "src": "acc_a", "task": "add_acc_a"},
        {"t": 88, "k": "issue", "src": "dma", "task": "load_A", "id": 2, "dir": "l2_to_l1",
         "from": 0x80002900, "to": 0x10000000, "size": 512, "from_stride": 0, "to_stride": 0,
         "reps": 1},
        {"t": 110, "k": "done", "src": "dma", "task": "load_A", "id": 2},
        {"t": 0, "k": "state", "src": "core1", "mode": "run"},
    ]  # fmt: skip


def test_round_trip():
    t = Trace("task", 226, 2799, ["acc", "acc_a", "dma", "core0", "core1"], events())
    assert Trace.from_dict(json.loads(json.dumps(t.to_dict()))) == t
    assert t.meta() == {"level": "task", "total_cycles": 226, "boot": 2799,
                        "sources": ["acc", "acc_a", "dma", "core0", "core1"]}  # fmt: skip
    assert [e["task"] for e in t.of_kind("done")] == ["add_acc", "add_acc_a", "load_A"]
    assert [e["task"] for e in t.of_kind("done", "dma")] == ["load_A"]


@pytest.mark.parametrize(
    "event, code",
    [
        ({"t": 0, "k": "cmd", "src": "core0"}, "trace.kind"),
        ({"k": "state", "src": "core0", "mode": "run"}, "trace.kind"),
        ({"t": 0, "k": "state", "src": "core0"}, "trace.field"),
        ({"t": 0, "k": "state", "src": "core0", "mode": "run", "why": "wfi"}, "trace.field"),
        ({"t": 0, "k": "done", "src": "acc", "task": "x", "firings": 1}, "trace.field"),
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
