"""Shared snippets for the run tests: a cluster's three hjson files and design point, as the
generator writes them for vecadd on snax_forge (cut to what run/ reads), and GVSoC header lines."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

from snax_forge.run.trace import LEVELS

PATH = "/chip/soc/cluster_0"
RECORDED = Path(__file__).parent / "fixtures"  # the recorded runs, a folder each
TASKS = RECORDED / "vecadd" / "tasks.json"  # the recorded vecadd's


class Recorded(NamedTuple):
    """A recorded run of tests/run/fixtures/ as the converter takes it, with its regions (L1
    from its design point, L2 from its ELF's symbols) and those symbols."""

    cluster: Any
    tasks: Any
    check: dict[str, Any]
    lines: list[Any]
    regions: Any
    elf_symbols: dict[str, Any]


def load_recorded(name: str | Path, events: str = "events.log") -> Recorded:
    """The recorded run ``name`` (a folder of tests/run/fixtures/, or its path), its
    ``events`` file read."""
    from snax_forge.design import DesignPoint
    from snax_forge.lower.tasks import TaskList
    from snax_forge.run import load_cluster, read_events, regions_of

    d = RECORDED / name if isinstance(name, str) else Path(name)
    syms = json.loads((d / "elf_symbols.json").read_text())
    return Recorded(
        load_cluster(d / "snax_forge.hjson", d / "design_point.json"),
        TaskList.from_dict(json.loads((d / "tasks.json").read_text())),
        json.loads((d / "check.json").read_text()),
        read_events(d / events),
        regions_of(DesignPoint.load(d / "design_point.json"), syms),
        syms,
    )


CLUSTER: dict[str, Any] = {
    "cluster": {
        "cluster_base_addr": 0x10000000,
        "data_width": 64,
        "dma_data_width": 512,
        "tcdm": {"size": 128, "banks": 32},
        "hives": [{"cores": [{"$ref": "#/acc_core"}, {"$ref": "#/dma_core"}]}],
    },
    "acc_core": {
        "xdma": False,
        "snax_acc_cfg": [{
            "snax_acc_name": "snax_forge",
            "snax_streamer_cfg": {"$ref": "snax_forge_streamer.hjson"},
            "snax_datapath_cfg": {"$ref": "accelerator.json"},
        }],
    },
    "dma_core": {"xdma": True},
}  # fmt: skip

STREAMER: dict[str, Any] = {
    "data_reader_params": {"spatial_bounds": [[4], [4]], "temporal_dim": [1, 1],
                           "num_channel": [4, 4], "fifo_depth": [8, 8]},
    "data_writer_params": {"spatial_bounds": [[4]], "temporal_dim": [1], "num_channel": [4],
                           "fifo_depth": [8]},
}  # fmt: skip

# vecadd's accelerator.json (C§17), as the recorded run's
UNIT: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "vecadd" / "accelerator.json").read_text()
)

POINT: dict[str, Any] = {
    "name": "vecadd",
    "platform": {"accelerator_slot": "snax_forge"},
    "streamers": {
        "acc_a": {"instance": "acc", "port": "a", "write": False},
        "acc_b": {"instance": "acc", "port": "b", "write": False},
        "acc_out": {"instance": "acc", "port": "out", "write": True},
    },
}


def files() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fresh copies of the cluster file, the streamer file, accelerator.json, the point."""
    return tuple(copy.deepcopy(x) for x in (CLUSTER, STREAMER, UNIT, POINT))  # type: ignore[return-value]


def line(path: str, trace: str, text: str, cycle: int = 0) -> str:
    """One events.log line as GVSoC writes it, ANSI colours and padding included."""
    full = f"{PATH}/{path}/{trace}"
    return f"{cycle * 100000}: {cycle}: [\x1b[34m{full:<80}\x1b[0m] {text}"


MAIN = 1000  # main's entry in the synthetic runs
READ = 5  # the cycles from a function's entry to its mcycle read
BOOT = MAIN + READ  # the mcycle read in main: the window's start
L2 = 0x8000_0000  # snax_forge's L2, where the synthetic link puts data.h's arrays


def synthetic_link(tasks: Any) -> dict[str, int]:
    """The address a synthetic program's link gives each array its DMA tasks name: one after
    another from ``L2``, in the order they are first named, each its moved bytes."""
    from snax_forge.lower.tasks import Configure

    out, at = {}, L2
    for s in tasks.steps:
        if isinstance(s, Configure) and s.type == "dma":
            side = s.values["src" if s.values["direction"] == "l2_to_l1" else "dst"]
            if side["symbol"] not in out:
                out[side["symbol"]] = at
                at += side["bounds"][0] * side["strides"][0]
    return out


def synthetic(
    cluster: Any, tasks: Any, level: str = "task", cycles: int = 226,
    link: dict[str, int] | None = None,
) -> list[str]:  # fmt: skip
    """An events.log a passing program on ``cluster`` could write, for ``tasks`` (a TaskList):
    boot, then from main's entry (``MAIN``) and its mcycle read (``BOOT``) each DMA copy, each
    accelerator task with its movers and the core's poll, forge_check's entry and its mcycle
    read ``cycles`` later; the summary lines to match. ``level`` off writes the summary lines
    only; from ``beat`` on, the streamer's CSR writes before each start and the lines of each
    beat (``beat_lines``), from ``finegrained`` three instructions per core. The summary's
    grants, stalls, banks and queue maxima are the beats' at every level. A DMA task's L2
    side is at its array's address in ``link`` (default ``synthetic_link``)."""
    from snax_forge.lower.tasks import Configure

    acc = cluster.accelerator
    unit, streamer = f"snax_acc{acc.core}/unit", f"snax_acc{acc.core}/streamer"
    pa, pd = f"pe{acc.core}", f"pe{cluster.dma_core}"
    ev: list[tuple[int, str, str, str]] = []  # cycle, path, trace, text
    on = LEVELS.index(level) if level in LEVELS else 0

    def add(cycle: int, path: str, trace: str, text: str) -> None:
        ev.append((cycle, path, trace, f"{cycle} {text}"))

    for i in range(cluster.nb_cores):
        add(0, f"pe{i}", "events", "state mode=sleep")
        add(100, f"pe{i}", "events", "state mode=run")
        add(MAIN - 50, f"pe{i}", "events", "func name=snrt_main pc=0x80002000")
        add(MAIN - 40, f"pe{i}", "events", f"barrier first={MAIN - 40} last={MAIN - 20}")
        add(MAIN, f"pe{i}", "events", "func name=main pc=0x800001e0")
        add(BOOT, f"pe{i}", "events", f"mcycle pc=0x800001f0 value={BOOT}")
        add(MAIN + 10, f"pe{i}", "events", f"func name=forge_core_{i} pc=0x80000300")
        if on >= LEVELS.index("finegrained"):
            for k, text in enumerate(INSNS):  # GVSoC's own line, no cycle first
                ev.append((BOOT + 1 + k, f"pe{i}", "insn", text))
    traffic = Traffic(cluster)
    steps = [s for s in tasks.steps if isinstance(s, Configure)]
    link = synthetic_link(tasks) if link is None else link
    t, ident = MAIN + 20, 2
    copies = 0
    for s in steps:
        if s.type != "dma":
            continue
        src, dst = (side["base"] if "base" in side else link[side["symbol"]]
                    for side in (s.values["src"], s.values["dst"]))  # fmt: skip
        size = s.values["src"]["bounds"][0] * s.values["src"]["strides"][0]
        add(t, "idma/fe", "events", f"issue id={ident} src={src:#x} dst={dst:#x} "
            f"size={size} src_stride=0 dst_stride=0 reps=1")  # fmt: skip
        traffic.copy(t, ident, src, dst, size)
        add(t + 10, "idma/fe", "events", f"done id={ident}")
        t, ident, copies = t + 12, ident + 1, copies + 1
    firings, runs = 0, 0
    config = {s.component: s.values for s in steps if s.type == "streamer"}
    for s in steps:
        if s.type != "accel":
            continue
        n = s.values["n"]
        if on >= LEVELS.index("beat"):
            for csr, value in streamer_csrs(cluster, config):
                add(t - 1, pa, "events", f"csr pc=0x80000360 csr={csr:#x} op=w value={value}")
        add(t, pa, "events", f"csr pc=0x80000370 csr={acc.csrs.streamer_start:#x} op=w value=1")
        for m in acc.movers:
            add(t, streamer, "events", f"start mover={m.gvsoc}")
        add(t + 1, pa, "events", f"csr pc=0x80000374 csr={acc.csrs.start:#x} op=w value=1")
        add(t + 1, unit, "events", f"start count={n} rate=1")
        e = t + 1 + n + 2
        add(t + 2, pa, "events", f"poll pc=0x80000378 csr={acc.csrs.busy:#x} value=1 first={t + 2} "
            f"last={e} count={(e - t) // 2}")  # fmt: skip
        add(e, unit, "events", f"done firings={n} first={t + 4} last={e}")
        add(e, unit, "summary", f"run index={runs} firings={n} busy={n} stall_in=2 "
            "stall_out=0 flush=0")  # fmt: skip
        traffic.run(t, n, config)
        for m in acc.movers:
            add(e + 1, streamer, "events", f"done mover={m.gvsoc}")
            qmax = ",".join(str(traffic.qmax(m, lane, runs)) for lane in range(m.lanes))
            add(e + 2, streamer, "summary", f"mover mover={m.gvsoc} run={runs} busy={n} stall_xbar=1 "
                f"stall_fifo=0 idle=2 qmax={qmax}")  # fmt: skip
        add(e + 1, pa, "events", f"csr pc=0x80000378 csr={acc.csrs.busy:#x} op=r value=0")
        firings, runs = firings + n, runs + 1
        t = e + 4
    add(t, pa, "events", f"barrier first={t} last={t + 3}")
    add(t + 3, pd, "events", f"barrier first={t + 3} last={t + 3}")
    add(MAIN + cycles - 20, pa, "events", "func name=main pc=0x80000218")
    add(MAIN + cycles, pa, "events", "func name=forge_check pc=0x80000254")
    add(BOOT + cycles, pa, "events", f"mcycle pc=0x80000264 value={BOOT + cycles}")
    add(MAIN + cycles + 30, pa, "events", "func name=printf_ pc=0x80000c18")
    end = MAIN + 20000
    add(end - 5, pa, "events", "state mode=sleep")
    for i, p in enumerate(acc.movers):
        for lane, port in enumerate(p.ports):
            index = input_index(cluster, port)
            add(end, "tcdm/snax_xbar", "summary", f"input index={index} name=snax_acc"
                f"{acc.core}.streamer.{p.gvsoc}.lane{lane} grants={traffic.grants[index]} "
                f"stalls={traffic.stalls[index]} stalls_dma=0 whole_run=0")  # fmt: skip
    for i in range(cluster.nb_cores):
        add(end, "tcdm/snax_xbar", "summary", f"input index={1000 + i} name=core{i} grants=300 "
            "stalls=0 stalls_dma=0 whole_run=1")  # fmt: skip
    for b in range(cluster.banks):
        add(end, "tcdm/snax_xbar", "summary", f"bank index={b} reads={traffic.reads[b]} "
            f"writes={traffic.writes[b]} conflicts={traffic.conflicts[b]}")  # fmt: skip
    for i in range(cluster.nb_cores):
        add(end, f"pe{i}", "summary", f"csrs writes={18 if i == acc.core else 0} "
            f"reads={13 if i == acc.core else 0}")  # fmt: skip
    add(end, "idma/fe", "summary", f"dma transfers={copies} busy={copies * 18} "
        f"r_beats={traffic.r_beats} w_beats={traffic.w_beats} bytes_to_l1=1024 "
        "bytes_from_l1=512")  # fmt: skip
    keep = {"summary", "events"} if level != "off" else {"summary"}
    keep |= {"beats", "stalls", "insn"} if on >= LEVELS.index("beat") else set()
    ev += [x for x in traffic.lines if on >= LEVELS.index(TRAFFIC_LEVEL[x[2]])]
    ev.sort(key=lambda x: x[0])  # as GVSoC writes them, nearly
    return [line(path, trace, text, cycle) for cycle, path, trace, text in ev if trace in keep]


INSNS = (
    "main:0                           M 800001e4 sw                  ra, 12(sp)",
    "main:0                           M 800001e8 sw                  s0, 8(sp)",
    "main:0                           M 800001ec addi                s0, sp, 16",
)
TRAFFIC_LEVEL = {"beats": "beat", "stalls": "stall"}


def input_index(cluster: Any, port: str) -> int:
    """The synthetic runs' crossbar index of a mover's lane: 100 per mover, then the lane."""
    for i, m in enumerate(cluster.accelerator.movers):
        if port in m.ports:
            return i * 100 + m.ports.index(port)
    raise KeyError(port)


def streamer_csrs(cluster: Any, config: dict[str, Any]) -> list[tuple[int, int]]:
    """Each mover's CSR writes for its task's values: pointer low and high, spatial strides,
    temporal bounds and strides, in the streamer's layout."""
    from snax_forge.run.beats import mover_csrs

    out = []
    for m in cluster.accelerator.movers:
        v = config[m.name]
        csr = mover_csrs(cluster)[m.name]
        regs = [v["base"], 0, *v["spatial_strides"], *v["temporal_bounds"], *v["temporal_strides"]]
        out += [(csr + i, r) for i, r in enumerate(regs)]
    return out


class Traffic:
    """The beat and stall lines of a synthetic run, and the summary counts they make. Per
    accelerator run of n firings started at t (streamer start t, unit start t + 1): the first
    reader's beat b granted at t + 1 + b on every lane, the others' one cycle later (their
    beat 0 stalls behind it at t + 1 on the same banks), firing b at t + 4 + b pushing at once,
    the writer's grant at t + 5 + b; words A[i] = i + 1 on reader 0, 100 + i on the others,
    their sum written; the class runs of the summary lines ``synthetic`` writes (the unit
    stall_in for 2 cycles from t + 2, then busy; each mover busy for n cycles from t + 1, then
    stall_xbar for 1, then idle for 2). A copy of ``size`` bytes: one beat of 64 bytes a cycle from t + 1, its
    bursts (one read, or one write per beat)."""

    def __init__(self, cluster: Any) -> None:
        self.c = cluster
        self.lines: list[tuple[int, str, str, str]] = []
        self.grants: dict[int, int] = defaultdict(int)
        self.stalls: dict[int, int] = defaultdict(int)
        self.reads = [0] * cluster.banks
        self.writes = [0] * cluster.banks
        self.conflicts = [0] * cluster.banks
        self.r_beats = self.w_beats = 0
        self._qmax: dict[tuple[str, int, int], int] = {}

    def add(self, cycle: int, path: str, trace: str, text: str) -> None:
        self.lines.append((cycle, path, trace, f"{cycle} {text}"))

    def copy(self, t: int, ident: int, src: int, dst: int, size: int) -> None:
        c = self.c
        into = c.is_l1(dst)
        l1, l2 = (dst, src) if into else (src, dst)
        nb = size // c.dma_beat_bytes
        self.r_beats += nb
        self.w_beats += nb
        if into:
            self.add(t + 1, "idma/axi_read", "beats", f"burst id={ident} op=r addr={l2:#x} "
                     f"size={size} back={t + 1}")  # fmt: skip
        for j in range(nb):
            a = l1 + j * c.dma_beat_bytes
            bank, row = c.bank_row(a)
            words = c.dma_beat_bytes // c.bank_bytes
            for w in range(words):
                (self.writes if into else self.reads)[(bank + w) % c.banks] += 1
            value = ",".join(hex(j * words + w + 1) for w in range(words))
            back = "" if into else f" back={t + 1 + j}"
            self.add(t + 1 + j, "tcdm/snax_xbar", "beats", f"dma id={ident} i={j} "
                     f"op={'w' if into else 'r'} addr={a - c.l1_base:#x} size={c.dma_beat_bytes} "
                     f"bank={bank} row={row} value={value} l2={l2 + j * c.dma_beat_bytes:#x}"
                     f"{back}")  # fmt: skip
            if not into:
                self.add(t + 1 + j, "idma/axi_write", "beats", f"burst id={ident} op=w "
                         f"addr={l2 + j * c.dma_beat_bytes:#x} size={c.dma_beat_bytes} "
                         f"back={t + 1 + j}")  # fmt: skip

    def _grant(self, cycle: int, m: Any, lane: int, addr: int, op: str, value: int, b: int) -> None:
        c = self.c
        index = input_index(c, m.ports[lane])
        bank, row = c.bank_row(addr)
        self.grants[index] += 1
        (self.reads if op == "r" else self.writes)[bank] += 1
        self.add(cycle, "tcdm/snax_xbar", "beats", f"grant in={index} op={op} "
                 f"addr={addr - c.l1_base:#x} bank={bank} row={row} value={value:#x} "
                 f"back={cycle + 1} b={b}")  # fmt: skip

    def run(self, t: int, n: int, config: dict[str, Any]) -> None:
        c = self.c
        acc = c.accelerator
        readers = [m for m in acc.movers if not m.write]
        writer = next(m for m in acc.movers if m.write)
        streamer = f"snax_acc{acc.core}/streamer"
        unit = f"snax_acc{acc.core}/unit"

        def addr(m: Any, b: int, lane: int) -> int:
            v = config[m.name]
            return v["base"] + b * v["temporal_strides"][0] + lane * v["spatial_strides"][0]

        def word(r: int, b: int, lane: int) -> int:
            return (1 if r == 0 else 100) + b * readers[0].lanes + lane

        run = sum(1 for k in self._qmax if k[0] == writer.name and k[1] == 0)
        self.add(t + 2, unit, "beats", "class mode=stall_in")
        self.add(t + 4, unit, "beats", "class mode=busy")
        for m in acc.movers:
            for cyc, mode in ((t + 1, "busy"), (t + 1 + n, "stall_xbar"), (t + 2 + n, "idle")):
                self.add(cyc, streamer, "beats", f"class mover={m.gvsoc} mode={mode}")
        for r, m in enumerate(readers):
            for b in range(n):
                g = t + 1 + b + (r > 0)
                for lane in range(m.lanes):
                    self._grant(g, m, lane, addr(m, b, lane), "r", word(r, b, lane), b)
            if r > 0:  # beat 0 stalls behind reader 0's on the same banks
                for lane in range(m.lanes):
                    a, by = addr(m, 0, lane), addr(readers[0], 0, lane)
                    bank, row = c.bank_row(a)
                    index = input_index(c, m.ports[lane])
                    self.stalls[index] += 1
                    self.conflicts[bank] += 1
                    self.add(t + 1, "tcdm/snax_xbar", "stalls", f"stall in={index} "
                             f"addr={a - c.l1_base:#x} bank={bank} row={row} "
                             f"by={input_index(c, readers[0].ports[lane])} "
                             f"by_addr={by - c.l1_base:#x} b=0")  # fmt: skip
            # visible at back + 1, popped by firing b at t + 4 + b
            self._fifo(streamer, m, run, [(t + 3 + b + (r > 0), t + 4 + b) for b in range(n)])
        for b in range(n):
            f = t + 4 + b
            beats = ",".join(str(b) for _ in readers)
            values = ",".join(hex(word(r, b, lane)) for r, m in enumerate(readers)
                              for lane in range(m.lanes))  # fmt: skip
            self.add(f, unit, "beats", f"fire n={b} b={beats} value={values}")
            out = [sum(word(r, b, lane) for r in range(len(readers)))
                   for lane in range(writer.lanes)]  # fmt: skip
            self.add(f, unit, "beats", f"push n={b} first={b} last={b} "
                     f"value={','.join(hex(x) for x in out)}")  # fmt: skip
            for lane in range(writer.lanes):
                self._grant(f + 1, writer, lane, addr(writer, b, lane), "w", out[lane], b)
        self._fifo(streamer, writer, run, [(t + 5 + b, t + 5 + b) for b in range(n)])

    def _fifo(self, streamer: str, m: Any, run: int, words: list[tuple[int, int]]) -> None:
        """A lane's count lines from its words (visible, popped), the same on every lane."""
        prev, top = 0, 0
        for cyc in range(min(v for v, _ in words), max(p for _, p in words) + 2):
            count = sum(1 for v, p in words if v <= cyc <= p)
            if count != prev:
                for lane in range(m.lanes):
                    self.add(cyc, streamer, "stalls", f"fifo mover={m.gvsoc} lane={lane} "
                             f"count={count}")  # fmt: skip
                prev, top = count, max(top, count)
        for lane in range(m.lanes):
            self._qmax[(m.name, lane, run)] = top

    def qmax(self, m: Any, lane: int, run: int) -> int:
        return self._qmax.get((m.name, lane, run), 0)


def tiny_elf(symbols: dict[str, tuple[int, int]], wide: bool = False) -> bytes:
    """A minimal little-endian ELF (32-bit, or 64-bit when ``wide``) with only a symbol table:
    each ``name -> (value, size)`` an object symbol, as a linker writes them."""
    import struct

    strtab = b"\0" + b"".join(n.encode() + b"\0" for n in symbols)
    entry = "<IBBHQQ" if wide else "<IIIBBH"
    syms, off = [struct.pack(entry, *([0] * 6))], 1
    for name, (value, size) in symbols.items():
        info = 0x11  # GLOBAL OBJECT
        syms.append(struct.pack(entry, off, info, 0, 1, value, size) if wide
                    else struct.pack(entry, off, value, size, info, 0, 1))  # fmt: skip
        off += len(name) + 1
    symtab = b"".join(syms)
    ehsize, shentsize = (64, 64) if wide else (52, 40)
    sym_off = ehsize
    str_off = sym_off + len(symtab)
    sh_off = str_off + len(strtab)
    sect = "<IIQQQQIIQQ" if wide else "<IIIIIIIIII"
    sections = [
        struct.pack(sect, *([0] * 10)),
        struct.pack(sect, 0, 2, 0, 0, sym_off, len(symtab), 2, 1, 4, struct.calcsize(entry)),
        struct.pack(sect, 0, 3, 0, 0, str_off, len(strtab), 0, 0, 1, 0),
    ]
    ident = b"\x7fELF" + bytes([2 if wide else 1, 1, 1]) + bytes(9)
    hdr = "<16sHHIQQQIHHHHHH" if wide else "<16sHHIIIIIHHHHHH"
    header = struct.pack(hdr, ident, 2, 0xF3, 1, 0, 0, sh_off, 0, ehsize, 0, 0, shentsize, 3, 0)
    return header + symtab + strtab + b"".join(sections)


def point_symbols(point: Any) -> dict[str, tuple[int, int]]:
    """The symbols a program of ``point`` would link: each container in L2 one after another
    from the L2's first byte in the plan's order, its L2 bytes (``lower.layout.l2_bytes``, as
    gen/data.py declares it), and a golden beside an output."""
    from snax_forge.lower.layout import l2_bytes

    out, at = {}, point.platform.l2.base_addr
    for name, lays in point.memory.layouts.items():
        if "l2" in lays:
            n = l2_bytes(point, name)
            out[name] = (at, n)
            out[f"{name}_golden"] = (0x90000000 + at % 0x10000, n)
            at += n
    return out


def synthetic_regions(tasks: Any) -> list[Any]:
    """The regions of a synthetic run (``synthetic``): each container a DMA task moves, in L1 at
    the task's base and in L2 at ``synthetic_link``'s, its elements int64 words."""
    from snax_forge.lower.tasks import Configure
    from snax_forge.run import Region

    link = synthetic_link(tasks)
    out = []
    for s in tasks.steps:
        if not isinstance(s, Configure) or s.type != "dma":
            continue
        l1, l2 = ("dst", "src") if s.values["direction"] == "l2_to_l1" else ("src", "dst")
        side = s.values[l2]
        name, size = side["symbol"], side["bounds"][0] * side["strides"][0]
        n = size // 8
        out += [Region(name, "l1", s.values[l1]["base"], (n,), (8,), "int64"),
                Region(name, "l2", link[name], (n,), (8,), "int64", size)]  # fmt: skip
    return out
