"""run/ on a recorded run of the one pass in strips (D190–D192): bicg_v4_1 at 40 × 16, W = 4,
R = 2, trace level beat (fixtures/bicg_v4_1/README.md), converted without GVSoC.

The group ``bicg_types`` is read at its boundary: its firings are its head's (``mul``, whose
outer inputs are the forked A and p, each sliced two to a word, D180), and each push names its
sink, ``sum`` (q, its seed carried from the first strip to the second) or ``mac`` (s, a bank of R
rows, r held a row). A sliced input's firing gives the k-th slice of its word, k counted over the
firings that name the word.
"""

import numpy as np
import pytest

from snax_forge.run import RunError, parse_line
from snax_forge.run.convert import convert
from snax_forge.run.sources import all_sources

from .helpers import RECORDED, load_recorded

GROUP = "bicg_types"
TEXT = (RECORDED / "bicg_v4_1" / "events_beat.log").read_text().splitlines()
M, N, W, STRIP = 40, 16, 4, 8  # STRIP = R·W columns a strip


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("bicg_v4_1", "events_beat.log")


def convert_text(recorded, lines):
    cluster, tasks, check, _, regions, _ = recorded
    return convert([parse_line(x) for x in lines], cluster, tasks, check["total_cycles"], "beat",
                   check["functional_check"], regions, all_sources(cluster))  # fmt: skip


@pytest.fixture(scope="module")
def trace(recorded):
    return convert_text(recorded, TEXT)[1]


def grants(trace, mover):
    return sorted((e for e in trace.of_kind("grant") if e["src"].split(".")[0] == mover),
                  key=lambda e: e["t"])  # fmt: skip


def test_the_inputs_are_sliced(recorded):
    a = recorded.cluster.accelerator
    readers = [m for m in a.movers_of(GROUP) if not m.write]
    assert [(m.name, a.d_in(m, GROUP)) for m in readers] == [
        ("mul_a", 2), ("mul_b", 2), ("sum_seed", 2), ("mac_b", 8)]  # fmt: skip
    assert [m.name for m in a.fire_readers(GROUP)] == ["mul_a", "mul_b"]


def test_two_firings_take_the_two_slices_of_a_word(trace):
    """160 firings (2 strips × 40 rows × 2 beats), each naming its word for two firings; the
    last strip's q is A p, A and p rebuilt from the fired slices."""
    fires = trace.of_kind("fire", GROUP)
    assert len(fires) == 160
    assert [e["b"]["mul_a"] for e in fires[:4]] == [0, 0, 1, 1]
    A, p = np.zeros((M, N), int), np.zeros(N, int)
    for i, e in enumerate(fires):
        s, rest = divmod(i, 2 * M)
        m, j = divmod(rest, 2)
        c = s * STRIP + j * W
        A[m, c : c + W], p[c : c + W] = e["vals"]["mul_a"][:W], e["vals"]["mul_b"][:W]
    q = [e["vals"][0] for e in trace.of_kind("push", GROUP) if e["unit"] == "sum"]
    assert len(q) == 2 * M and q[M:] == [int(x) for x in A @ p]


def test_the_bank_drains_its_rows_at_each_strips_end(trace):
    """R = 2 results a strip, their firings divided evenly over the job, pushed one after the
    other once the strip's last row is folded; s = Aᵀ r on each strip."""
    mac = [e for e in trace.of_kind("push", GROUP) if e["unit"] == "mac"]
    assert [(e["n"], e["first"], e["last"]) for e in mac] == [
        (0, 0, 39), (1, 40, 79), (2, 80, 119), (3, 120, 159)]  # fmt: skip
    t = [e["t"] for e in mac]
    assert t[1] == t[0] + 1 and t[3] == t[2] + 1 and t[2] - t[1] > M
    assert len(grants(trace, "mac_out")) == 8  # 4 results of 4 int32 lanes, two words each


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


def _sub(match: str, old: str, new: str) -> list[str]:
    i = next(i for i, x in enumerate(TEXT) if match in x)
    return TEXT[:i] + [TEXT[i].replace(old, new)] + TEXT[i + 1 :]


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (
            _sub("fire n=1 ", "value=0xf8f9f8fc", "value=0xf8f9f8fd"),
            "fire 1 at 225: slice 1 of mul_a's word 0 is 0xf8f9f8fc, fired as 0xf8f9f8fd",
        ),
        (
            _sub("fire n=2 ", "b=1,1", "b=0,1"),
            "fire 2 at 226: mul_a's word 0 holds 2 slices, this is firing 3 to name it",
        ),
    ],
)
def test_a_wrong_slice_is_refused(recorded, lines, message):
    with pytest.raises(RunError, match=message) as e:
        convert_text(recorded, lines)
    assert e.value.code == "convert.invariant"
