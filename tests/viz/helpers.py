"""Shared helpers for the visualiser tests.

Runs are flow folders made the way the flow makes them, SNAX-GVSoC stubbed
(tests/flow/helpers.py: ``stubbed_flow``, its synthetic events), so the
viewer reads exactly what the flow writes.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

from snax_forge.viz.server import make_server
from tests.flow.helpers import stubbed_flow
from tests.helpers import REPO
from tests.run.helpers import load_recorded


def run_dir(out: Path, name: str = "vecadd", level: str = "task", **kw) -> Path:
    """The flow of recipe ``name`` at trace ``level`` into the folder ``out`` (``kw`` to
    ``run_flow``: ``memory_sets``, ``trace_sources``, ...); returns it."""
    return stubbed_flow(out.parent, name, level, out=out, **kw)


@contextmanager
def serving(dirs):
    """A viewer server on a free port, serving in a thread; yields it."""
    srv = make_server(list(dirs), port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)


# =============================================================================
# Hand-written events (C§7) on the run tests' cluster, for the index and the views
# =============================================================================

L1 = 0x10000000  # the cluster's L1 base (tests/run/helpers.CLUSTER)
UNIT_CLASSES = ("busy", "stall_in", "stall_out", "flush")
MOVER_CLASSES = ("busy", "stall_xbar", "stall_fifo", "idle")
L2 = 0x80000000


def cluster():
    """The run tests' vecadd cluster: 32 banks of 8 bytes, a 64-byte DMA beat, acc with movers
    acc_a, acc_b (readers) and acc_out (writer) of 4 lanes."""
    from snax_forge.run import cluster_from
    from tests.run.helpers import files

    return cluster_from(*files())


def regions(n: int = 64):
    """A, B and C of ``n`` int64 elements, contiguous in L1 from its base and in L2 from
    0x80000000."""
    from snax_forge.run import Region

    out = []
    for i, name in enumerate("ABC"):
        out += [Region(name, "l1", L1 + i * 8 * n, (n,), (8,), "int64"),
                Region(name, "l2", L2 + i * 8 * n, (n,), (8,), "int64")]  # fmt: skip
    return out


def tasks(*runs: str, name: str = "vecadd"):
    """A vecadd task list: copies ``load_A`` and ``load_B``, then per run prefix ``p`` (``add``
    by default) the tasks ``p_acc_a``, ``p_acc_b``, ``p_acc_out`` and ``p_acc`` launched by one
    start, then the copy ``store_C``. Values are left empty: the index reads only the steps."""
    from snax_forge.lower.tasks import Configure, Start, TaskList

    def copy(task, direction):
        return [Configure(task, "dma", "dma", {"direction": direction}), Start([task])]

    steps = copy("load_A", "l2_to_l1") + copy("load_B", "l2_to_l1")
    for p in runs or ("add",):
        mine = [f"{p}_acc_a", f"{p}_acc_b", f"{p}_acc_out", f"{p}_acc"]
        steps += [Configure(t, "streamer", t.removeprefix(f"{p}_"), {}) for t in mine[:3]]
        steps += [Configure(mine[3], "accel", "acc", {}), Start(mine)]
    return TaskList(name, steps + copy("store_C", "l1_to_l2"))


class Events:
    """Builds C§7 events with the converter's lookups done from the cluster and the regions:
    ``bank`` and ``row``, ``region``, ``elems`` and ``vals`` (each word's value its element's)."""

    def __init__(self, cl=None, regs=None):
        self.c = cl or cluster()
        self.regions = regs if regs is not None else regions()
        self.events: list[dict] = []

    def _look(self, mem, addr, size):
        from snax_forge.run import region_at

        r = region_at(self.regions, mem, addr)
        if r is None:
            return None, [], []
        elems = r.elems(addr, size)
        return r.container, elems, [self.value(r.container, f) for f in elems]

    @staticmethod
    def value(container: str, flat: int) -> int:
        """Element ``flat``'s value: A[i] = i + 1, B[i] = 100 + i, C[i] = their sum."""
        return {"A": flat + 1, "B": 100 + flat, "C": 101 + 2 * flat}.get(container, flat)

    def add(self, e):
        self.events.append(e)
        return e

    def grant(self, t, src, addr, op="r", b=None, task=None, back=None):
        region, elems, vals = self._look("l1", addr, 8)
        bank, row = self.c.bank_row(addr)
        e = {"t": t, "k": "grant", "src": src, "task": task, "op": op, "mem": "l1", "addr": addr,
             "bank": bank, "row": row, "value": vals[0] if vals else 0,
             "back": t + 1 if back is None else back}  # fmt: skip
        if b is not None:
            e["b"] = b
        return self.add({**e, "region": region, "elems": elems, "vals": vals})

    def stall(self, t, src, addr, by, by_addr, b=None, task=None):
        region, elems, _ = self._look("l1", addr, 8)
        bank, row = self.c.bank_row(addr)
        e = {"t": t, "k": "stall", "src": src, "task": task, "addr": addr, "bank": bank,
             "row": row, "by": by, "by_addr": by_addr}  # fmt: skip
        if b is not None:
            e["b"] = b
        return self.add({**e, "region": region, "elems": elems})

    def dma(self, t, ident, i, op, addr, l2, task=None, size=64):
        region, elems, vals = self._look("l1", addr, size)
        bank, row = self.c.bank_row(addr)
        e = {"t": t, "k": "dma", "src": "dma", "task": task, "id": ident, "i": i, "op": op,
             "mem": "l1", "addr": addr, "size": size, "bank": bank, "row": row,
             "value": list(vals), "l2": l2}  # fmt: skip
        if op == "r":
            e["back"] = t
        return self.add({**e, "region": region, "elems": elems, "vals": vals})

    def burst(self, t, ident, op, addr, size, task=None):
        region, elems, _ = self._look("l2", addr, size)
        return self.add({"t": t, "k": "burst", "src": "dma", "task": task, "id": ident, "op": op,
                         "mem": "l2", "addr": addr, "size": size, "back": t, "region": region,
                         "elems": elems})  # fmt: skip

    def fire(self, t, n, b, vals, task="add_acc"):
        return self.add({"t": t, "k": "fire", "src": "acc", "task": task, "n": n, "b": dict(b),
                         "value": {m: list(v) for m, v in vals.items()},
                         "vals": {m: list(v) for m, v in vals.items()}})  # fmt: skip

    def push(self, t, n, first, last, vals, task="add_acc"):
        return self.add({"t": t, "k": "push", "src": "acc", "task": task, "n": n, "first": first,
                         "last": last, "value": list(vals), "vals": list(vals)})  # fmt: skip

    def task(self, t, src, task, done=None, cycles=None):
        """A task of ``src`` from ``t`` to ``done``; its run's ``cycles`` on the done, by
        default all ``busy`` from ``t + 1`` to ``done - 1``."""
        self.add({"t": t, "k": "start", "src": src, "task": task})
        if done is not None:
            names = UNIT_CLASSES if src == self.c.accelerator.name else MOVER_CLASSES
            default = {c: done - t - 1 if c == "busy" else 0 for c in names}
            self.add({"t": done, "k": "done", "src": src, "task": task,
                      "cycles": dict(cycles or default)})  # fmt: skip

    def csr(self, t, csr, value, block="acc_streamer"):
        return self.add({"t": t, "k": "csr", "src": "core0", "pc": 0x80000360, "csr": csr,
                         "block": block, "op": "w", "value": value})  # fmt: skip

    def copy(self, t, ident, task, done, size=512):
        """A DMA copy of ``size`` bytes from L2 to L1 issued at ``t``, done at ``done``."""
        self.add({"t": t, "k": "issue", "src": "dma", "task": task, "id": ident,
                  "dir": "l2_to_l1", "from": L2, "to": L1, "size": size, "from_stride": 0,
                  "to_stride": 0, "reps": 1})  # fmt: skip
        self.add({"t": done, "k": "done", "src": "dma", "task": task, "id": ident})

    def core(self, t, k, src="core0", **kw):
        """A core's task-level event (``state``, ``poll``, ``barrier``, ``func``)."""
        return self.add({"t": t, "k": k, "src": src, **kw})

    def index(self, level="stall", filter=None, task_list=None, total=1000):
        """The RunIndex of these events (sorted by t, stable) at ``level`` in a window of
        ``total`` cycles, paired by ``task_list`` (``tasks()``, one run, by default)."""
        from snax_forge.run import RunOutputs, Trace
        from snax_forge.viz.index import RunIndex

        if filter is None:
            filter = ["acc", *(m.name for m in self.c.accelerator.movers), "dma"]
        evs = sorted(self.events, key=lambda e: e["t"])
        trace = Trace(level, total, 0, self.c.sources, evs, list(filter))
        return RunIndex.from_run(RunOutputs({}, None, trace), self.c, self.regions,
                                 task_list or tasks())  # fmt: skip


def two_runs():
    """Two runs of acc and its movers (tasks add_* then add2_*): b and n restart at 0 in the
    second, which reads A[1] where the first read A[0]."""
    ev = Events()
    for run, (p, t) in enumerate((("add", 0), ("add2", 100))):
        for m in ("acc_a", "acc_b", "acc_out"):
            ev.task(t, m, f"{p}_{m}", t + 20)
        ev.task(t, "acc", f"{p}_acc", t + 20)
        ev.grant(t + 1, "acc_a.lane0", L1 + 8 * run, b=0, task=f"{p}_acc_a")
        ev.grant(t + 1, "acc_b.lane0", L1 + 512 + 8 * run, b=0, task=f"{p}_acc_b")
        ev.fire(t + 3, 0, {"acc_a": 0, "acc_b": 0}, {"acc_a": [run + 1], "acc_b": [100 + run]},
                task=f"{p}_acc")  # fmt: skip
        ev.push(t + 3, 0, 0, 0, [101 + 2 * run], task=f"{p}_acc")
        ev.grant(t + 4, "acc_out.lane0", L1 + 1024 + 8 * run, op="w", b=0, task=f"{p}_acc_out")
    return ev


# =============================================================================
# The recorded run (tests/run/fixtures/vecadd), converted into a flow folder
# =============================================================================

RECORDED = REPO / "tests" / "run" / "fixtures" / "vecadd"


def recorded_run(out: Path, level: str = "stall", fixture: Path | None = None) -> Path:
    """The recorded vecadd run (or the run of ``fixture``, a folder of tests/run/fixtures/, at
    task) converted at ``level`` into the flow folder ``out``, laid out as the flow lays it out
    (``gen/``, ``design/``, ``run/``, ``tasks.json``): ``task`` from its task-level events,
    ``beat`` and ``stall`` from its stall-level ones (at ``beat`` without the ``stalls``
    lines). Returns ``out``."""
    recorded = RECORDED if fixture is None else fixture
    import shutil

    from snax_forge.common import to_json
    from snax_forge.run import write_run
    from snax_forge.run.convert import convert

    (out / "gen").mkdir(parents=True)
    (out / "design").mkdir()
    for name in ("snax_forge.hjson", "snax_forge_streamer.hjson", "accelerator.json"):
        shutil.copy(recorded / name, out / "gen" / name)
    shutil.copy(recorded / "design_point.json", out / "design" / "design_point.json")
    shutil.copy(recorded / "tasks.json", out / "tasks.json")
    cluster, tasks, check, lines, regions, syms = load_recorded(
        recorded, "events.log" if level == "task" else "events_stall.log"
    )
    sources = (
        []
        if level == "task"
        else [*cluster.accelerator.unit_names, *(m.name for m in cluster.accelerator.movers), "dma"]
    )
    if level == "beat":
        lines = [x for x in lines if x.trace != "stalls"]
    profile, trace = convert(lines, cluster, tasks, check["total_cycles"], level,
                             check["functional_check"], regions, sources)  # fmt: skip
    (out / "run").mkdir()
    (out / "run" / "run.json").write_text(to_json({
        "design_point": "design/design_point.json",
        "hjson": ["gen/snax_forge.hjson", "gen/snax_forge_streamer.hjson", "gen/accelerator.json"],
        "elf_symbols": syms, "trace_level": level, "trace_sources": sources,
    }))  # fmt: skip
    write_run(out / "run", profile, trace)
    return out


def get(srv, path: str):
    """A GET on a running viewer: status, content type and body."""
    with urlopen(srv.url.rstrip("/") + path, timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()
