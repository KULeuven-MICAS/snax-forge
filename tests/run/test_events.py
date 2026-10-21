"""run/events.py: events.log lines read in snax-gvsoc's grammar (docs/EVENTS.md)."""

import pytest

from snax_forge.run import RunError, parse_line, read_events

from .helpers import PATH, line


def test_a_unit_start():
    got = parse_line(line("snax_acc0/unit", "events", "2932 start count=16 rate=1", 2932))
    assert got.path == f"{PATH}/snax_acc0/unit"
    assert (got.trace, got.cycle, got.kind) == ("events", 2932, "start")
    assert got.fields == {"count": 16, "rate": 1}


@pytest.mark.parametrize(
    "path, trace, text, fields",
    [
        ("snax_acc0/streamer", "events", "2931 start mover=reader0", {"mover": "reader0"}),
        ("snax_acc0/unit", "events", "2951 done firings=16 first=2936 last=2951",
         {"firings": 16, "first": 2936, "last": 2951}),
        ("snax_acc0/streamer", "summary",
         "2953 mover mover=writer0 run=0 busy=16 stall_xbar=0 stall_fifo=4 idle=2 qmax=2,2,2,2",
         {"mover": "writer0", "run": 0, "busy": 16, "stall_xbar": 0, "stall_fifo": 4, "idle": 2,
          "qmax": [2, 2, 2, 2]}),
        ("snax_acc0/streamer", "summary",
         "9 mover mover=reader0 run=0 busy=1 stall_xbar=0 stall_fifo=0 idle=0 qmax=3",
         {"mover": "reader0", "run": 0, "busy": 1, "stall_xbar": 0, "stall_fifo": 0, "idle": 0,
          "qmax": [3]}),
        ("tcdm/snax_xbar", "summary",
         "20001 input index=12 name=core0 grants=322 stalls=0 stalls_dma=0 whole_run=1",
         {"index": 12, "name": "core0", "grants": 322, "stalls": 0, "stalls_dma": 0,
          "whole_run": 1}),
        ("pe0", "events", "2874 csr pc=0x80000324 csr=0x3c0 op=w value=268435456",
         {"pc": 0x80000324, "csr": 0x3C0, "op": "w", "value": 268435456}),
        ("pe0", "events", "2874 csr pc=0x80000324 csr=0x3d3 op=w value=1 held=3",
         {"pc": 0x80000324, "csr": 0x3D3, "op": "w", "value": 1, "held": 3}),
        ("pe0", "events", "2933 poll pc=0x80000378 csr=0x3d4 value=1 first=2933 last=2951 count=10",
         {"pc": 0x80000378, "csr": 0x3D4, "value": 1, "first": 2933, "last": 2951, "count": 10}),
        ("pe0", "events", "2912 barrier first=2912 last=2930", {"first": 2912, "last": 2930}),
        ("pe0", "events", "1712 state mode=run", {"mode": "run"}),
        ("pe0", "events", "2799 func name=main pc=0x800001e0", {"name": "main", "pc": 0x800001E0}),
        ("pe0", "events", "1713 func name=$d pc=0x1000", {"name": "$d", "pc": 0x1000}),
        ("idma/fe", "events",
         "2887 issue id=2 src=0x80002900 dst=0x10000000 size=512 src_stride=0 dst_stride=0 reps=1",
         {"id": 2, "src": 0x80002900, "dst": 0x10000000, "size": 512, "src_stride": 0,
          "dst_stride": 0, "reps": 1}),
        ("idma/fe", "events", "2909 done id=2", {"id": 2}),
        ("idma/fe", "summary",
         "20001 dma transfers=3 busy=54 r_beats=24 w_beats=24 bytes_to_l1=1024 bytes_from_l1=512",
         {"transfers": 3, "busy": 54, "r_beats": 24, "w_beats": 24, "bytes_to_l1": 1024,
          "bytes_from_l1": 512}),
        ("pe1", "summary", "20001 csrs writes=0 reads=0", {"writes": 0, "reads": 0}),
        ("tcdm/snax_xbar", "beats",
         "2933 grant in=0 op=r addr=0x0 bank=0 row=0 value=0x2bd back=2934 b=0",
         {"in": 0, "op": "r", "addr": 0, "bank": 0, "row": 0, "value": 0x2BD, "back": 2934,
          "b": 0}),
        ("tcdm/snax_xbar", "beats",
         "1876 grant in=12 op=w addr=0x1ff8c bank=17 row=511 value=0x0 back=1876 size=4",
         {"in": 12, "op": "w", "addr": 0x1FF8C, "bank": 17, "row": 511, "value": 0, "back": 1876,
          "size": 4}),
        ("tcdm/snax_xbar", "beats",
         ("2977 dma id=4 i=0 op=r addr=0x400 size=16 bank=0 row=4 value=0x5ab,0xfffffffffffffe37 "
          "l2=0x80003300 back=2977"),
         {"id": 4, "i": 0, "op": "r", "addr": 0x400, "size": 16, "bank": 0, "row": 4,
          "value": [0x5AB, 0xFFFFFFFFFFFFFE37], "l2": 0x80003300, "back": 2977}),
        ("idma/axi_read", "beats", "2890 burst id=2 op=r addr=0x80002900 size=512 back=2902",
         {"id": 2, "op": "r", "addr": 0x80002900, "size": 512, "back": 2902}),
        ("snax_acc0/unit", "beats", "2936 fire n=0 b=0,0 value=0x2bd,0x111,0x2ee,0x9d",
         {"n": 0, "b": [0, 0], "value": [0x2BD, 0x111, 0x2EE, 0x9D]}),
        ("snax_acc0/unit", "beats", "2299 push n=0 first=0 last=15 value=0x84",
         {"n": 0, "first": 0, "last": 15, "value": [0x84]}),
        ("tcdm/snax_xbar", "stalls", "2933 stall in=4 addr=0x200 bank=0 row=2 by=0 by_addr=0x0 b=0",
         {"in": 4, "addr": 0x200, "bank": 0, "row": 2, "by": 0, "by_addr": 0, "b": 0}),
        ("tcdm/snax_xbar", "stalls", "2950 stall in=13 addr=0x8 bank=1 row=0 by=-1 by_addr=0x0",
         {"in": 13, "addr": 8, "bank": 1, "row": 0, "by": -1, "by_addr": 0}),
        ("snax_acc0/streamer", "stalls", "2935 fifo mover=reader0 lane=0 count=1",
         {"mover": "reader0", "lane": 0, "count": 1}),
    ],
)  # fmt: skip
def test_every_kind_reads(path, trace, text, fields):
    got = parse_line(line(path, trace, text))
    assert (got.path, got.trace, got.fields) == (f"{PATH}/{path}", trace, fields)


def test_an_instruction_line_is_gvsoc_s_own():
    text = ("main:0                           M 800001e0 addi                sp, sp, -16"
            "                  sp=1001ffb0  sp:1001ffc0 ")  # fmt: skip
    got = parse_line(line("pe0", "insn", text, 2799))
    assert (got.path, got.trace, got.cycle, got.kind) == (f"{PATH}/pe0", "insn", 2799, "insn")
    assert got.fields == {"pc": 0x800001E0, "text": "addi sp, sp, -16 sp=1001ffb0 sp:1001ffc0"}


@pytest.mark.parametrize(
    "text, code",
    [
        (line("pe0", "events", "5 jump to=main"), "events.kind"),
        (line("pe0", "events", "5 state mode=run why=wfi"), "events.field"),
        (line("pe0", "events", "5 state"), "events.field"),
        (line("pe0", "events", "5 barrier first=x last=7"), "events.value"),
        (line("pe0", "events", "5 state mode"), "events.field"),
        (line("pe0", "trace", "5 state mode=run"), "events.trace"),
        (line("pe0", "events", "five state mode=run"), "events.line"),
        ("forge cycles: 226", "events.line"),
        (line("tcdm/snax_xbar", "beats", "5 grant in=0 op=r addr=0x0 bank=0 row=0 value=0x1"),
         "events.field"),
        (line("snax_acc0/unit", "beats", "5 fire n=0 b=0,x value=0x1"), "events.value"),
        (line("snax_acc0/streamer", "stalls", "5 fifo mover=reader0 lane=0"), "events.field"),
        (line("pe0", "insn", "main:0 M"), "events.line"),
        (line("pe0", "insn", "main:0 M zz addi sp"), "events.line"),
    ],
)  # fmt: skip
def test_what_is_not_the_grammar_is_a_named_error(text, code):
    with pytest.raises(RunError) as e:
        parse_line(text)
    assert e.value.code == code


def test_a_file_reads_in_order_and_names_its_line(tmp_path):
    f = tmp_path / "events.log"
    f.write_text(
        line("pe0", "events", "1 state mode=run", 1)
        + "\n\n"
        + line("pe0", "events", "2 func name=main pc=0x10", 2)
        + "\n"
    )
    assert [x.kind for x in read_events(f)] == ["state", "func"]
    f.write_text(
        line("pe0", "events", "1 state mode=run") + "\n" + line("pe0", "events", "2 x") + "\n"
    )
    with pytest.raises(RunError, match="events.log:2"):
        read_events(f)
