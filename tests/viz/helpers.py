"""Shared helpers for the visualiser tests.

Runs are flow folders made the way the flow makes them, SNAX-GVSoC stubbed
(tests/flow/helpers.py: ``stubbed_flow``, its synthetic events), so the
viewer reads exactly what the flow writes.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

from snax_forge.viz.server import make_server
from tests.flow.helpers import stubbed_flow

REPO = Path(__file__).resolve().parents[2]


def run_dir(out: Path, name: str = "vecadd", level: str = "task") -> Path:
    """The flow of recipe ``name`` at trace ``level`` into the folder ``out``; returns it."""
    return stubbed_flow(out.parent, name, level, out=out)


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

    def task(self, t, src, task, done=None):
        self.add({"t": t, "k": "start", "src": src, "task": task})
        if done is not None:
            self.add({"t": done, "k": "done", "src": src, "task": task})

    def csr(self, t, csr, value, block="acc_streamer"):
        return self.add({"t": t, "k": "csr", "src": "core0", "pc": 0x80000360, "csr": csr,
                         "block": block, "op": "w", "value": value})  # fmt: skip

    def index(self, level="stall", filter=None):
        """The RunIndex of these events (sorted by t, stable) at ``level``."""
        from snax_forge.run import RunOutputs, Trace
        from snax_forge.viz.index import RunIndex

        if filter is None:
            filter = ["acc", *(m.name for m in self.c.accelerator.movers), "dma"]
        evs = sorted(self.events, key=lambda e: e["t"])
        trace = Trace(level, 1000, 0, self.c.sources, evs, list(filter))
        return RunIndex.from_run(RunOutputs({}, None, trace), self.c, self.regions)
