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

import pytest

from snax_forge.run import RunError, parse_line
from snax_forge.run.convert import convert
from snax_forge.run.sources import all_sources
from snax_forge.run.trace import Trace
from tests.run.helpers import RECORDED, load_recorded

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
