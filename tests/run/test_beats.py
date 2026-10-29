"""run/beats.py: the beat, stall and instruction events and every invariant they hold, on a
synthetic run (helpers.synthetic at stall), then each invariant broken by one hand-made
change to its lines."""

import re

import pytest

from snax_forge.run import RunError, parse_line
from snax_forge.run.beats import agu_nest, beats
from snax_forge.run.convert import convert

from .helpers import synthetic, synthetic_regions

SOURCES = ["acc", "acc_a", "acc_b", "acc_out", "dma"]


@pytest.fixture
def run(cluster, tasks):
    return cluster, tasks


def text(x: str) -> str:
    return x.split("] ", 1)[1]


def convert_lines(run, lines, level="stall", sources=SOURCES):
    cluster, tasks = run
    sources = sources if level != "task" else None
    return convert([parse_line(x) for x in lines], cluster, tasks, 226, level, None,
                   synthetic_regions(tasks), sources)  # fmt: skip


def test_a_synthetic_run_holds_every_invariant(run):
    for level in ("beat", "stall", "finegrained"):
        sources = SOURCES + (["core0", "core1"] if level == "finegrained" else [])
        _, trace = convert_lines(run, synthetic(*run, level), level, sources)
        kinds = {e["k"] for e in trace.events}
        assert {"grant", "dma", "burst", "fire", "push"} <= kinds
        assert ("stall" in kinds) == ("fifo" in kinds) == (level != "beat")
        assert ("insn" in kinds) == (level == "finegrained")
        assert trace.filter == sources


def test_the_events_carry_their_lookups(run):
    _, trace = convert_lines(run, synthetic(*run, "finegrained"), "finegrained",
                             SOURCES + ["core0"])  # fmt: skip
    g = next(e for e in trace.events if e["k"] == "grant" and e["src"] == "acc_a.lane1")
    assert (g["task"], g["mem"], g["addr"], g["b"], g["back"]) == (
        "add_acc_a", "l1", 0x10000008, 0, g["t"] + 1)  # fmt: skip
    f = next(e for e in trace.events if e["k"] == "fire")
    assert f["b"] == {"acc_a": 0, "acc_b": 0} and f["value"]["acc_a"] == [1, 2, 3, 4]
    p = next(e for e in trace.events if e["k"] == "push")
    assert (p["n"], p["first"], p["last"], p["value"]) == (0, 0, 0, [101, 103, 105, 107])
    s = next(e for e in trace.events if e["k"] == "stall")
    assert (s["src"], s["by"], s["task"]) == ("acc_b.lane0", "acc_a.lane0", "add_acc_b")
    d = next(e for e in trace.events if e["k"] == "dma")
    assert (d["task"], d["mem"], len(d["value"])) == ("load_A", "l1", 8)
    b = next(e for e in trace.events if e["k"] == "burst")
    assert (b["task"], b["mem"], b["region"], b["elems"]) == ("load_A", "l2", "A", list(range(64)))
    assert (g["region"], g["elems"], d["region"], d["elems"]) == ("A", [1], "A", list(range(8)))
    i = next(e for e in trace.events if e["k"] == "insn")
    assert (i["src"], i["func"], i["pc"], i["text"]) == ("core0", "main", 0x800001E4,
                                                          "sw ra, 12(sp)")  # fmt: skip
    # core1 is not a source: none of its instructions
    assert {e["src"] for e in trace.events if e["k"] == "insn"} == {"core0"}


def test_a_source_not_named_has_no_lines(run):
    _, trace = convert_lines(run, synthetic(*run, "stall"), "stall", ["acc_a"])
    assert {e["src"] for e in trace.events if e["k"] in ("grant", "stall", "fifo")} == {
        "acc_a", *(f"acc_a.lane{i}" for i in range(4))}  # fmt: skip
    assert not [e for e in trace.events if e["k"] in ("fire", "push", "dma", "burst")]


def test_the_nest_and_the_beats():
    # 4 lanes 8 B apart, 3 beats 32 B apart; masked to the TCDM
    assert agu_nest(0x10000000, [4], [8], [3], [32], 4, 0x1FFFF, False) == [
        [0, 32, 64], [8, 40, 72], [16, 48, 80], [24, 56, 88]]  # fmt: skip
    # a reader's repeat walks loop 0 once
    assert agu_nest(0, [1], [8], [4, 2], [0, 8], 1, 0xFFFF, True) == [[0, 8]]
    assert agu_nest(0, [1], [8], [4, 2], [0, 8], 1, 0xFFFF, False) == [[0] * 4 + [8] * 4]
    assert agu_nest(0, [1], [8], [0], [8], 1, 0xFFFF, False) == [[]]
    assert (beats(0x80002900, 512, 64), beats(0x10, 64, 64), beats(0, 0, 64)) == (8, 2, 0)


def drop(pattern: str, count: int = 1):
    """A change: the first ``count`` lines matching ``pattern`` removed."""

    def change(lines):
        out, left = [], count
        for x in lines:
            if left and re.search(pattern, text(x)):
                left -= 1
                continue
            out.append(x)
        return out

    return change


def sub(pattern: str, old: str, new: str):
    """A change: in the first line matching ``pattern``, ``old`` replaced by ``new``."""

    def change(lines):
        out, done = [], False
        for x in lines:
            if not done and re.search(pattern, text(x)):
                assert old in x, (old, x)
                x, done = x.replace(old, new, 1), True
            out.append(x)
        assert done, pattern
        return out

    return change


BROKEN = [
    (drop(r"grant in=0 "), "acc_a.lane0: 15 grant lines in the window, 16 in its summary"),
    (sub(r"bank index=5 ", "reads=", "reads=1"), "bank 5: "),
    (drop(r"dma id=2 i=3 "), "DMA copy 2 (load_A): 7 `w` beats on L1, its writes make 8"),
    (sub(r"dma transfers", "r_beats=24", "r_beats=25"), "the DMA's copies make 24 read"),
    (drop(r"fire n=3 "), "add_acc: 15 fire lines, its done says 16 firings"),
    (drop(r"push n=3 "), "add_acc: 15 push lines, 16 firings at rate 1"),
    (drop(r"grant in=200 .*b=5$"), "acc_out (add_acc_out): 63 grants, 16 pushes of 4 lanes"),
    (sub(r"fire n=2 ", "value=0x9,", "value=0x7,"), "fire 2 at "),
    (sub(r"push n=2 ", "value=0x75,", "value=0x76,"), "push 2 at "),
    (sub(r"grant in=1 .*b=0$", "row=0", "row=9"), "bank 1 row 9, its address 0x10000008"),
    (sub(r"csr=0x3c0 op=w", "value=268435456", "value=268435968"),
     "acc_a.lane0 (add_acc_a): grant addresses"),
    (drop(r"stall in=101 "), "acc_b.lane1: 0 stall lines (0 by the DMA), its summary 1 (0)"),
    (sub(r"bank index=0 ", "conflicts=1", "conflicts=2"),
     "bank 0: 1 cycles with two or more requesters, 2 conflicts"),
    (sub(r"fifo mover=reader0 lane=0 count=2", "count=2", "count=3"), "acc_a lane 0: fifo counts"),
    (sub(r"mover mover=reader0 ", "qmax=2,", "qmax=3,"),
     "acc_a lane 0: fifo counts up to 2, its summary's qmax 3"),
]  # fmt: skip


@pytest.mark.parametrize("change, said", BROKEN)
def test_each_invariant_broken_is_named(run, change, said):
    lines = synthetic(*run, "stall")
    convert_lines(run, lines)  # holds
    with pytest.raises(RunError) as e:
        convert_lines(run, change(lines))
    assert e.value.code == "convert.invariant"
    assert said in str(e.value), str(e.value)


def test_a_copy_outside_its_containers_l2_array_is_named(run):
    """From task on: each container's L2 bytes are its copies', and no copy leaves them."""
    cluster, tasks = run
    for level in ("task", "stall"):
        lines = synthetic(cluster, tasks, level)
        convert_lines(run, lines, level)  # holds
        moved = sub(r"issue id=2 ", "src=0x80000000", "src=0x80000040")(lines)
        with pytest.raises(RunError) as e:
            convert_lines(run, moved, level)
        assert e.value.code == "convert.invariant"
        assert (
            "container A: its copies touch L2 [('0x80000040', '0x80000240')], its array "
            "(64 int64 elements, its padded shape in 64-byte beats) is [0x80000000, 0x80000200)"
        ) in str(e.value)
    # a run without L2 regions: every copy touching L2 is named
    lines = [parse_line(x) for x in synthetic(cluster, tasks, "task")]
    with pytest.raises(RunError, match=r"DMA copy 2 \(load_A\) touches L2 and is no container"):
        convert(lines, cluster, tasks, 226, "task", None, [])


def test_a_fifo_above_its_depth_or_not_empty_at_the_end_is_named(run):
    lines = synthetic(*run, "stall")
    last = [i for i, x in enumerate(lines) if "fifo mover=writer0 lane=2 count=0" in x][-1]
    broken = lines[:last] + lines[last + 1 :]
    with pytest.raises(RunError, match="acc_out lane 2: fifo counts above its depth 8 or not 0"):
        convert_lines(run, broken)


# =============================================================================
# Packed beats (D142): a beat's words are its mover's num_channel
# =============================================================================


def _packed(cluster, channels: dict[str, int], lines: list[str]):
    """The fire and push events of ``lines`` on ``cluster`` with its movers at ``channels``
    (a mover's lanes are its channels: words per beat)."""
    from dataclasses import replace

    from snax_forge.run.beats import Context, beat_events
    from snax_forge.run.convert import source_of

    from .helpers import line

    acc = cluster.accelerator
    movers = [replace(m, lanes=channels[m.name]) for m in acc.movers if m.name in channels]
    c = replace(cluster, accelerator=replace(acc, movers=movers))
    parsed = [parse_line(line("snax_acc0/unit", "beats", x, 2613)) for x in lines]
    ctx = Context(c, [], 0, 10_000, "beat", [acc.name, *channels])
    return [e for _, _, e in beat_events([(source_of(x, c), x) for x in parsed], ctx)
            if e["k"] in ("fire", "push")]  # fmt: skip


def test_vecadd_i32s_first_firing_two_words_per_port(cluster):
    """As traced: two readers of 2 words (4 int32 lanes, two per word) and a writer of 2."""
    fire, push = _packed(cluster, {"acc_a": 2, "acc_b": 2, "acc_out": 2}, [
        (
            "2613 fire n=0 b=0,0 value=0x111000002bd,0xfffffe3300000016,0xfffffd26000002ee,"
            "0x1ba0000009d"
        ),
        "2613 push n=0 first=0 last=0 value=0xfffffe37000005ab,0xffffffed000000b3",
    ])  # fmt: skip
    assert fire["value"] == {"acc_a": [0x111000002BD, 0xFFFFFE3300000016],
                             "acc_b": [0xFFFFFD26000002EE, 0x1BA0000009D]}  # fmt: skip
    assert push["value"] == [0xFFFFFE37000005AB, 0xFFFFFFED000000B3]


def test_a_fold_two_words_in_one_out(cluster):
    """vsum_tree_i32's shape: a reader of 2 words, a writer of 1 (the fold's one word)."""
    fire, push = _packed(cluster, {"acc_a": 2, "acc_out": 1}, [
        "2613 fire n=0 b=0 value=0x111000002bd,0xfffffe3300000016",
        "2613 push n=0 first=0 last=0 value=0xffffffffffffffbd",
    ])  # fmt: skip
    assert fire["value"] == {"acc_a": [0x111000002BD, 0xFFFFFE3300000016]}
    assert push["value"] == [0xFFFFFFFFFFFFFFBD]
    with pytest.raises(RunError, match="push at 2613: 2 words, the writer moves 1"):
        _packed(cluster, {"acc_a": 2, "acc_out": 1}, [
            "2613 push n=0 first=0 last=0 value=0x1,0x2",
        ])  # fmt: skip
