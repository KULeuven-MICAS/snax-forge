"""The energy run (D177; CONTRACTS, "Profile and trace"): a run's snapshots priced.

    energy_of(lines, cluster, marks, graph, est) -> EnergyRun

``lines`` are a run's ``summary`` events on SNAX-GVSoC: the quiet build's (``forge_quiet.elf``)
with ``marks`` its ``obs_marks.json``, or, ``marks`` None, the default build's (``forge.elf``),
which writes no mark: one window, the whole run (a pipelined schedule's, D179). ``est`` is the
design's costed estimate. Each window between two snapshots (run/snapshots.py) is priced at the
instances' prices (estimate/dynamic.py): its units' firings and pushes, its groups' busy cycles,
its movers' requests and grants, the crossbar's counts of the DMA's copies inside L1. ``static``
is the estimate's own five parts with the counts they were priced on, for comparison.
``EnergyRun`` round-trips through ``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from snax_forge.common import JsonFile
from snax_forge.estimate.dynamic import (
    Counts,
    Energy,
    MoverSpec,
    Prices,
    graph_counts,
    port_movers,
    price,
    prices_of,
)
from snax_forge.run import Cluster, RunError
from snax_forge.run.snapshots import BOOT, WHOLE, Window, snapshots, windows

if TYPE_CHECKING:
    from snax_forge.dfg import Graph
    from snax_forge.estimate import Estimate
    from snax_forge.run.events import Line

ENERGY_JSON = "energy.json"


@dataclass
class EnergyRun(JsonFile):
    """A run's windows, each with its energy, and the static estimate's (module doc)."""

    tech: str
    source: dict[str, str]
    prices: Prices
    windows: list[Window]
    energies: list[Energy]
    static: Energy

    @property
    def total(self) -> dict[str, float]:
        """nJ per part over every window."""
        return {k: sum(e.parts[k] for e in self.energies) for k in self.static.parts}

    def of_kind(self, kind: str) -> list[tuple[Window, Energy]]:
        return [(w, e) for w, e in zip(self.windows, self.energies, strict=True) if w.kind == kind]

    def runs(self) -> list[tuple[Window, Energy]]:
        """The accelerators' windows: the quiet build's ``run`` windows, or the default build's
        one window of the whole run."""
        return self.of_kind("run") or self.of_kind(WHOLE[0])

    def to_dict(self) -> dict[str, Any]:
        return {"tech": self.tech, "source": dict(self.source), "prices": self.prices.to_dict(),
                "windows": [w.to_dict() | {"energy": e.to_dict()}
                            for w, e in zip(self.windows, self.energies, strict=True)],
                "total": self.total, "static": self.static.to_dict()}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnergyRun:
        return cls(d["tech"], dict(d["source"]), Prices.from_dict(d["prices"]),
                   [Window.from_dict(w) for w in d["windows"]],
                   [Energy.from_dict(w["energy"]) for w in d["windows"]],
                   Energy.from_dict(d["static"]))  # fmt: skip


def mover_specs(cluster: Cluster) -> list[MoverSpec]:
    """The cluster's movers as the tables price them: depth, loops and channels its own."""
    return [MoverSpec(m.name, m.write, m.fifo_depth, m.temporal_dim, m.lanes)
            for m in cluster.accelerator.movers]  # fmt: skip


def counts_of(w: Window, cluster: Cluster) -> Counts:
    """A window's counts as the pricing reads them: per unit its firings and pushes, the groups'
    busy cycles added up and per group (a group's line is written by one of its units), per mover
    its lanes' grants and stalls (requests) and grants, and the crossbar's counts of the DMA's
    copies inside L1 (D184)."""
    movers = {}
    for m in cluster.accelerator.movers:
        lanes = [w.counters.inputs[p] for p in m.ports if p in w.counters.inputs]
        movers[m.name] = (sum(g + s for g, s in lanes), sum(g for g, _ in lanes))
    group = {u.name: u.group for u in cluster.accelerator.units}
    groups: dict[str, int] = {}
    for k, c in w.counters.groups.items():
        g = group.get(k, k)
        groups[g] = groups.get(g, 0) + sum(c.values())
    return Counts(dict(w.counters.units), w.counters.busy(), movers, w.counters.copies, groups)


def energy_of(lines: Sequence[Line], cluster: Cluster, marks: Sequence[dict[str, Any]] | None,
              graph: Graph, est: Estimate) -> EnergyRun:  # fmt: skip
    """The windows of a run, priced (module doc): the quiet build's by its ``marks``, the default
    build's (``marks`` None) in one. A ``RunError`` (``energy.snapshots``) when the run wrote no
    snapshot, or (``energy.marks``) a mark the default build should not have written."""
    snaps = snapshots(lines, cluster)
    if not snaps:
        raise RunError("energy.snapshots", "the run wrote no `snap` lines: SNAX-GVSoC is older "
                       "than its snapshots (the submodule)")  # fmt: skip
    if marks is None and len(snaps) > 1:
        raise RunError("energy.marks", f"the default build wrote marks {[s.mark for s in snaps]}: "
                       "only the quiet build marks its sections; price that with its "
                       "obs_marks.json")  # fmt: skip
    ws = windows(snaps, marks or [], WHOLE if marks is None else BOOT)
    p = prices_of(graph, est, mover_specs(cluster))
    static = price(prices_of(graph, est, port_movers(est)), graph_counts(graph, est))
    return EnergyRun(est.tech or "", dict(est.char_source or {}), p, ws,
                     [price(p, counts_of(w, cluster)) for w in ws], static)  # fmt: skip
