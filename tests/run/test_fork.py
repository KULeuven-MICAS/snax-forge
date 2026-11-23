"""run/ on a recorded run of a group of two chains on a forked input (D189): bicg_v3 at trace
level stall (fixtures/bicg_v3/README.md), converted without GVSoC.

The group ``bicg_types`` is one SNAX-GVSoC component, read at its boundary: its firings are its
head's (``mul``, whose outer inputs are the forked A and p), and each push names its sink, ``sum``
(q, packed two int32 results to a word, D174) or ``mac`` (s). Each output is checked on its own:
its pushes against its rate, its writer's grants against its pushes, each push against the slice
or the word its writer wrote.
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
TEXT = (RECORDED / "bicg_v3" / "events_stall.log").read_text().splitlines()


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("bicg_v3", "events_stall.log")


def convert_text(recorded, lines, level="stall"):
    cluster, tasks, check, _, regions, _ = recorded
    if level == "beat":
        lines = [x for x in lines if "/stalls" not in x.split("] ", 1)[0]]
    return convert([parse_line(x) for x in lines], cluster, tasks, check["total_cycles"], level,
                   check["functional_check"], regions, all_sources(cluster))  # fmt: skip


@pytest.fixture(scope="module")
def stalled(recorded):
    return convert_text(recorded, TEXT)


def test_the_group_is_read_at_its_boundary(recorded):
    a = recorded.cluster.accelerator
    assert [m.name for m in a.movers_of(GROUP)] == ["mul_a", "mul_b", "mac_b", "sum_out", "mac_out"]
    assert [m.name for m in a.fire_readers(GROUP)] == ["mul_a", "mul_b"]  # the head's: A and p
    assert a.sinks(GROUP) == ["sum", "mac"]
    assert [a.writer_of(GROUP, k).name for k in ("sum", "mac")] == ["sum_out", "mac_out"]
    assert [a.d_out(a.writer_of(GROUP, k), GROUP) for k in ("sum", "mac")] == [2, 1]


@pytest.mark.parametrize("level", ["beat", "stall"])
def test_each_output_converts_on_its_own(recorded, level):
    """64 firings (M = 64 rows of A, N = W = 32), each with A's and p's beat; q's 64 pushes at
    rate 1, two to sum_out's word (32 grants), and s's one push at rate 64 (16 grants)."""
    _, trace = convert_text(recorded, TEXT, level)
    fires, pushes = trace.of_kind("fire", GROUP), trace.of_kind("push", GROUP)
    assert len(fires) == 64 and {tuple(e["b"]) for e in fires} == {("mul_a", "mul_b")}
    q = [e for e in pushes if e["unit"] == "sum"]
    (s,) = [e for e in pushes if e["unit"] == "mac"]
    assert [(e["n"], e["first"], e["last"]) for e in q] == [(n, n, n) for n in range(64)]
    assert (s["n"], s["first"], s["last"], len(s["value"]), len(s["vals"])) == (0, 0, 63, 16, 32)
    writes = {}
    for e in trace.of_kind("grant"):
        if e["op"] == "w":
            writes[e["src"].split(".")[0]] = writes.get(e["src"].split(".")[0], 0) + 1
    assert writes == {"sum_out": 32, "mac_out": 16}


def test_the_pushes_are_q_and_s_of_the_operands(stalled):
    """q = A p from each firing's own operands, s = Aᵀ r from mac_b's grants (r is no head's
    input, so in no fire line)."""
    _, trace = stalled
    fires = trace.of_kind("fire", GROUP)
    A = np.array([e["vals"]["mul_a"] for e in fires])
    p = np.array(fires[0]["vals"]["mul_b"])
    r = [v for e in sorted((e for e in trace.of_kind("grant") if e["src"] == "mac_b.lane0"),
                           key=lambda e: e["b"]) for v in e["vals"]]  # fmt: skip
    pushes = trace.of_kind("push", GROUP)
    assert [e["vals"] for e in pushes if e["unit"] == "sum"] == [[int(x)] for x in A @ p]
    assert next(e["vals"] for e in pushes if e["unit"] == "mac") == [int(x) for x in A.T @ r]


def test_the_counter_is_its_classes(stalled):
    """BUSY_COUNTER 72: 66 busy, 5 stall_in (one before the first firing, four in a run from
    t = 254), 1 flush."""
    profile, trace = stalled
    (done,) = trace.of_kind("done", GROUP)
    assert done["cycles"] == {"busy": 66, "stall_in": 5, "stall_out": 0, "flush": 1}
    assert profile.functional_check["tasks"]["q_mult_bicg_types"]["counter"] == 72
    runs = [(e["t"], e["mode"]) for e in trace.of_kind("class", GROUP)]
    assert runs == [(196, "stall_in"), (197, "busy"), (254, "stall_in"), (258, "busy"),
                    (267, "flush")]  # fmt: skip


def test_the_trace_names_the_sink_and_round_trips(stalled):
    _, trace = stalled
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


BROKEN = [  # each one line changed, and what the check says
    (_edit("push n=5 first=5", None), "63 push lines of sum, 64 firings at rate 1"),
    (_edit("unit=mac", None), "0 push lines of mac, 64 firings at rate 64"),
    (_edit("unit=mac", None), r"mac_out \(s_mult_mac_out\): 16 grants, 0 pushes of mac"),
    (
        _sub("start count=64", "rate=1,64", "rate=1,32"),
        "1 push lines of mac, 64 firings at rate 32",
    ),
    (_sub("start count=64", "rate=1,64", "rate=1"), "1 rates at its start, 2 outputs"),
    (
        _edit("grant in=9 op=w", None),
        r"sum_out \(sum_sum_out\): 31 grants, 64 pushes of sum in 32 words of 2 of 1 lanes",
    ),
    (
        _sub("push n=3 first=3", "value=0xffffffd3", "value=0xffffffd4"),
        "push 3 at 202: slice 1 of sum_out's word 1 is 0xffffffd3, pushed as 0xffffffd4",
    ),
    (
        _sub("unit=mac", "0xffffff43ffffffe6", "0xffffff43ffffffe7"),
        "push 0 at 266: written by mac_out as",
    ),
]


@pytest.mark.parametrize(("lines", "message"), BROKEN)
def test_each_output_is_checked_on_its_own(recorded, lines, message):
    with pytest.raises(RunError, match=message) as e:
        convert_text(recorded, lines)
    assert e.value.code == "convert.invariant"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            " unit=sum",
            "",
            r"push at 199: unit None, where bicg_types's outputs are \['sum', 'mac'\]",
        ),
        (" unit=sum", " unit=mul", "unit 'mul', where"),
    ],
)
def test_a_push_names_one_of_its_groups_outputs(recorded, old, new, message):
    with pytest.raises(RunError, match=message) as e:
        convert_text(recorded, _sub("push n=0 first=0 last=0", old, new))
    assert e.value.code == "convert.beats"
