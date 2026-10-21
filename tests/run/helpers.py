"""Shared snippets for the run tests: a cluster's three hjson files and design point, as the
generator writes them for vecadd on snax_alu (cut to what run/ reads), and GVSoC header lines."""

from __future__ import annotations

import copy
from typing import Any

PATH = "/chip/soc/cluster_0"

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
            "snax_acc_name": "snax_alu",
            "snax_streamer_cfg": {"$ref": "snax_alu_streamer.hjson"},
            "snax_datapath_cfg": {"$ref": "snax_alu_acc.hjson"},
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

UNIT: dict[str, Any] = {
    "unit": "SnaxUnit", "inputs": ["a", "b"], "outputs": ["out"], "lanes": 4, "data_width": 64,
    "op": 0, "signed": True, "control": "counted", "wait": "lockstep", "latency": 0, "ii": 1,
    "fields": ["loopCount"],
}  # fmt: skip

POINT: dict[str, Any] = {
    "name": "vecadd",
    "platform": {"accelerator_slot": "snax_alu"},
    "streamers": {
        "acc_a": {"instance": "acc", "port": "a", "write": False},
        "acc_b": {"instance": "acc", "port": "b", "write": False},
        "acc_out": {"instance": "acc", "port": "out", "write": True},
    },
}


def files() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fresh copies of the cluster file, the streamer file, the datapath file, the point."""
    return tuple(copy.deepcopy(x) for x in (CLUSTER, STREAMER, UNIT, POINT))  # type: ignore[return-value]


def line(path: str, trace: str, text: str, cycle: int = 0) -> str:
    """One events.log line as GVSoC writes it, ANSI colours and padding included."""
    full = f"{PATH}/{path}/{trace}"
    return f"{cycle * 100000}: {cycle}: [\x1b[34m{full:<80}\x1b[0m] {text}"


MAIN = 1000  # main's entry in the synthetic runs
READ = 5  # the cycles from a function's entry to its mcycle read
BOOT = MAIN + READ  # the mcycle read in main: the window's start


def synthetic(cluster: Any, tasks: Any, level: str = "task", cycles: int = 226) -> list[str]:
    """An events.log a passing program on ``cluster`` could write, for ``tasks`` (a TaskList):
    boot, then from main's entry (``MAIN``) and its mcycle read (``BOOT``) each DMA copy, each
    accelerator task with its movers and the core's poll, forge_check's entry and its mcycle
    read ``cycles`` later; the summary lines to match. ``level`` off writes the summary lines only."""
    from snax_forge.lower.tasks import Configure

    acc = cluster.accelerator
    unit, streamer = f"snax_acc{acc.core}/unit", f"snax_acc{acc.core}/streamer"
    pa, pd = f"pe{acc.core}", f"pe{cluster.dma_core}"
    ev: list[tuple[int, str, str, str]] = []  # cycle, path, trace, text

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
    steps = [s for s in tasks.steps if isinstance(s, Configure)]
    t, ident = MAIN + 20, 2
    copies = 0
    for s in steps:
        if s.type != "dma":
            continue
        src, dst = s.values["src"], s.values["dst"]
        size = src["bounds"][0] * src["strides"][0]
        add(t, "idma/fe", "events", f"issue id={ident} src={src['base']:#x} dst={dst['base']:#x} "
            f"size={size} src_stride=0 dst_stride=0 reps=1")  # fmt: skip
        add(t + 10, "idma/fe", "events", f"done id={ident}")
        t, ident, copies = t + 12, ident + 1, copies + 1
    firings, runs = 0, 0
    for s in steps:
        if s.type != "accel":
            continue
        n = s.values["n"]
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
        for m in acc.movers:
            add(e + 1, streamer, "events", f"done mover={m.gvsoc}")
            add(e + 2, streamer, "summary", f"mover mover={m.gvsoc} run={runs} busy={n} stall_xbar=1 "
                f"stall_fifo=0 idle=2 qmax={','.join(['2'] * m.lanes)}")  # fmt: skip
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
            add(end, "tcdm/snax_xbar", "summary", f"input index={i * 100 + lane} name=snax_acc"
                f"{acc.core}.streamer.{p.gvsoc}.lane{lane} grants={firings} stalls=1 "
                "stalls_dma=0 whole_run=0")  # fmt: skip
    for i in range(cluster.nb_cores):
        add(end, "tcdm/snax_xbar", "summary", f"input index={1000 + i} name=core{i} grants=300 "
            "stalls=0 stalls_dma=0 whole_run=1")  # fmt: skip
    for b in range(cluster.banks):
        add(end, "tcdm/snax_xbar", "summary", f"bank index={b} reads=6 writes=6 conflicts=0")
    for i in range(cluster.nb_cores):
        add(end, f"pe{i}", "summary", f"csrs writes={18 if i == acc.core else 0} "
            f"reads={13 if i == acc.core else 0}")  # fmt: skip
    add(end, "idma/fe", "summary", f"dma transfers={copies} busy={copies * 18} "
        f"r_beats={copies * 8} w_beats={copies * 8} bytes_to_l1=1024 bytes_from_l1=512")  # fmt: skip
    keep = {"summary", "events"} if level != "off" else {"summary"}
    ev.sort(key=lambda x: x[0])  # as GVSoC writes them, nearly
    return [line(path, trace, text, cycle) for cycle, path, trace, text in ev if trace in keep]
