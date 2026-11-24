"""run/ on a recorded run of the one pass in strips (D189–D192): bicg_v4_1 at 40 × 32, W = 4,
R = 4, trace level beat (fixtures/bicg_v4_1/README.md), converted without GVSoC.

The group ``bicg_types`` is one SNAX-GVSoC component, read at its boundary: its firings are its
head's (``mul``, whose outer inputs are the forked A and p, each sliced two to a word, D180), and
each push names its sink, ``sum`` (q, packed two int32 results to a word, D174, its seed carried
from the first strip to the second) or ``mac`` (s, a bank of R rows, r held a row). Each output is
checked on its own: its pushes against its rate, its writer's grants against its pushes, each push
against the slice or the word its writer wrote. A sliced input's firing gives the k-th slice of
its word, k counted over the firings that name the word.
"""

import json

import numpy as np
import pytest

from snax_forge.run import RunError, parse_line
from snax_forge.run.convert import convert
from snax_forge.run.sources import all_sources
from snax_forge.run.trace import Trace

from .helpers import RECORDED, load_recorded

GROUP = "bicg_types"
TEXT = (RECORDED / "bicg_v4_1" / "events_beat.log").read_text().splitlines()
M, N, W, STRIP = 40, 32, 4, 16  # STRIP = R·W columns a strip


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("bicg_v4_1", "events_beat.log")


def convert_text(recorded, lines):
    cluster, tasks, check, _, regions, _ = recorded
    return convert([parse_line(x) for x in lines], cluster, tasks, check["total_cycles"], "beat",
                   check["functional_check"], regions, all_sources(cluster))  # fmt: skip


@pytest.fixture(scope="module")
def converted(recorded):
    return convert_text(recorded, TEXT)


@pytest.fixture(scope="module")
def trace(converted):
    return converted[1]


def grants(trace, mover):
    return sorted((e for e in trace.of_kind("grant") if e["src"].split(".")[0] == mover),
                  key=lambda e: e["t"])  # fmt: skip


def test_the_group_is_read_at_its_boundary(recorded):
    a = recorded.cluster.accelerator
    assert [m.name for m in a.movers_of(GROUP)] == [
        "mul_a", "mul_b", "sum_seed", "mac_b", "sum_out", "mac_out"]  # fmt: skip
    assert [m.name for m in a.fire_readers(GROUP)] == ["mul_a", "mul_b"]  # the head's: A and p
    assert a.sinks(GROUP) == ["sum", "mac"]
    assert [a.writer_of(GROUP, k).name for k in ("sum", "mac")] == ["sum_out", "mac_out"]
    assert [a.d_out(a.writer_of(GROUP, k), GROUP) for k in ("sum", "mac")] == [2, 1]
    readers = [m for m in a.movers_of(GROUP) if not m.write]
    assert [(m.name, a.d_in(m, GROUP)) for m in readers] == [
        ("mul_a", 2), ("mul_b", 2), ("sum_seed", 2), ("mac_b", 8)]  # fmt: skip


def test_each_output_converts_on_its_own(trace):
    """320 firings (2 strips × 40 rows × 4 beats), each with A's and p's beat; q's 80 pushes at
    rate 4, two to sum_out's word (40 grants), and s's 8 pushes at rate 40 (16 grants)."""
    fires, pushes = trace.of_kind("fire", GROUP), trace.of_kind("push", GROUP)
    assert len(fires) == 320 and {tuple(e["b"]) for e in fires} == {("mul_a", "mul_b")}
    q = [e for e in pushes if e["unit"] == "sum"]
    assert [(e["n"], e["first"], e["last"]) for e in q] == [
        (n, 4 * n, 4 * n + 3) for n in range(80)
    ]
    writes = {}
    for e in trace.of_kind("grant"):
        if e["op"] == "w":
            writes[e["src"].split(".")[0]] = writes.get(e["src"].split(".")[0], 0) + 1
    assert writes == {"sum_out": 40, "mac_out": 16}


def test_two_firings_take_the_two_slices_of_a_word(trace):
    """Each firing names its word for two firings; the last strip's q is A p (its seed the first
    strip's), A and p rebuilt from the fired slices."""
    fires = trace.of_kind("fire", GROUP)
    assert [e["b"]["mul_a"] for e in fires[:6]] == [0, 0, 1, 1, 2, 2]
    A, p = np.zeros((M, N), int), np.zeros(N, int)
    for i, e in enumerate(fires):
        s, rest = divmod(i, 4 * M)
        m, j = divmod(rest, 4)
        c = s * STRIP + j * W
        A[m, c : c + W], p[c : c + W] = e["vals"]["mul_a"][:W], e["vals"]["mul_b"][:W]
    q = [e["vals"][0] for e in trace.of_kind("push", GROUP) if e["unit"] == "sum"]
    assert len(q) == 2 * M and q[M:] == [int(x) for x in A @ p]


def test_the_bank_drains_its_rows_at_each_strips_end(trace):
    """R = 4 results a strip, their firings divided evenly over the job, pushed one after the
    other once the strip's last row is folded."""
    mac = [e for e in trace.of_kind("push", GROUP) if e["unit"] == "mac"]
    assert [(e["n"], e["first"], e["last"]) for e in mac] == [
        (n, 40 * n, 40 * n + 39) for n in range(8)]  # fmt: skip
    t = [e["t"] for e in mac]
    assert t[:4] == [t[0] + k for k in range(4)] and t[4:] == [t[4] + k for k in range(4)]
    assert t[4] - t[3] > M
    assert len(grants(trace, "mac_out")) == 16  # 8 results of 4 int32 lanes, two words each


def test_the_held_input_is_read_once_a_strip(trace):
    """r's 40 int8, 8 to a word: 5 words a strip, read again on the second."""
    r = grants(trace, "mac_b")
    assert [e["b"] for e in r] == list(range(10))
    assert [e["value"] for e in r[:5]] == [e["value"] for e in r[5:]]


def test_the_carried_seed_reads_back_the_first_strips_q(trace):
    """q's seed, 40 int32 two to a word: 20 words a strip at q's own addresses; the second
    strip's are the first strip's writes, each read after it is written."""
    seed, out = grants(trace, "sum_seed"), grants(trace, "sum_out")
    assert len(seed) == len(out) == 40
    assert [e["addr"] for e in seed[:20]] == [e["addr"] for e in seed[20:]]
    assert [e["addr"] for e in seed[:20]] == [e["addr"] for e in out[:20]]
    for read, wrote in zip(seed[20:], out[:20], strict=True):
        assert read["value"] == wrote["value"] and read["t"] > wrote["t"]


def test_the_counter_is_its_classes(converted):
    """BUSY_COUNTER 327: 321 busy, 5 stall_in before the first firing, 1 flush."""
    profile, trace = converted
    (done,) = trace.of_kind("done", GROUP)
    assert done["cycles"] == {"busy": 321, "stall_in": 5, "stall_out": 0, "flush": 1}
    assert profile.functional_check["tasks"]["q_mult_bicg_types"]["counter"] == 327
    runs = [(e["t"], e["mode"]) for e in trace.of_kind("class", GROUP)]
    assert runs == [(238, "stall_in"), (243, "busy"), (564, "flush")]


def test_the_trace_names_the_sink_and_round_trips(trace):
    again = Trace.from_dict(json.loads(json.dumps(trace.to_dict())))
    assert again == trace
    assert {e.get("unit") for e in again.of_kind("push")} == {"sum", "mac"}


def _edit(match: str, new: str | None) -> list[str]:
    """The lines with the first containing ``match`` replaced by ``new`` (None drops it)."""
    i = next(i for i, x in enumerate(TEXT) if match in x)
    return TEXT[:i] + ([] if new is None else [new]) + TEXT[i + 1 :]


def _sub(match: str, old: str, new: str) -> list[str]:
    i = next(i for i, x in enumerate(TEXT) if match in x)
    return _edit(match, TEXT[i].replace(old, new))


BROKEN = [  # each one line changed, the check's code and what it says
    (_edit("push n=5 first=20", None), "invariant", "79 push lines of sum, 320 firings at rate 4"),
    (_edit("unit=mac", None), "invariant", "7 push lines of mac, 320 firings at rate 40"),
    (
        _edit("unit=mac", None),
        "invariant",
        r"mac_out \(s_mult_mac_out\): 16 grants, 7 pushes of mac",
    ),
    (
        _sub("start count=320", "rate=4,40", "rate=4,20"),
        "invariant",
        "8 push lines of mac, 320 firings at rate 20",
    ),
    (
        _sub("start count=320", "rate=4,40", "rate=4"),
        "invariant",
        "1 rates at its start, 2 outputs",
    ),
    (
        _edit("grant in=4 op=w addr=0x540", None),
        "invariant",
        r"sum_out \(sum_sum_out\): 39 grants, 80 pushes of sum in 40 words of 2 of 1 lanes",
    ),
    (
        _sub("push n=3 first=12", "value=0x1 ", "value=0x2 "),
        "invariant",
        "push 3 at 260: slice 1 of sum_out's word 1 is 0x1, pushed as 0x2",
    ),
    (
        _sub("unit=mac", "0x5900000032", "0x5900000033"),
        "invariant",
        "push 0 at 400: written by mac_out as",
    ),
    (
        _sub("fire n=1 ", "value=0xf8f9f8fc", "value=0xf8f9f8fd"),
        "invariant",
        "fire 1 at 244: slice 1 of mul_a's word 0 is 0xf8f9f8fc, fired as 0xf8f9f8fd",
    ),
    (
        _sub("fire n=2 ", "b=1,1", "b=0,1"),
        "invariant",
        "fire 2 at 245: mul_a's word 0 holds 2 slices, this is firing 3 to name it",
    ),
    (
        _sub("push n=0 first=0 last=3", " unit=sum", ""),
        "beats",
        r"push at 248: unit None, where bicg_types's outputs are \['sum', 'mac'\]",
    ),
    (_sub("push n=0 first=0 last=3", " unit=sum", " unit=mul"), "beats", "unit 'mul', where"),
]


@pytest.mark.parametrize(("lines", "code", "message"), BROKEN)
def test_each_output_is_checked_on_its_own(recorded, lines, code, message):
    with pytest.raises(RunError, match=message) as e:
        convert_text(recorded, lines)
    assert e.value.code == f"convert.{code}"
