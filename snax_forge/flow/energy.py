"""The energy run (D177; CONTRACTS, "Profile and trace"): the quiet build's snapshots priced.

    energy_of(lines, cluster, marks, graph, est) -> EnergyRun

``lines`` are the quiet build's events (``forge_quiet.elf`` on SNAX-GVSoC, its ``summary``
lines), ``marks`` its ``obs_marks.json``, ``est`` the design's costed estimate. Each window
between two snapshots (run/snapshots.py) is priced at the instances' prices
(estimate/dynamic.py): its units' firings and pushes, its groups' busy cycles, its movers'
requests and grants. ``static`` is the estimate's own five parts with the counts they were
priced on, for comparison. ``EnergyRun`` round-trips through ``to_dict`` / ``from_dict``.
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
from snax_forge.run.snapshots import Window, snapshots, windows

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
    busy cycles added up, per mover its lanes' grants and stalls (requests) and grants."""
    movers = {}
    for m in cluster.accelerator.movers:
        lanes = [w.counters.inputs[p] for p in m.ports if p in w.counters.inputs]
        movers[m.name] = (sum(g + s for g, s in lanes), sum(g for g, _ in lanes))
    return Counts(dict(w.counters.units), w.counters.busy(), movers)


def energy_of(lines: Sequence[Line], cluster: Cluster, marks: Sequence[dict[str, Any]],
              graph: Graph, est: Estimate) -> EnergyRun:  # fmt: skip
    """The windows of a quiet run, priced (module doc). A ``RunError`` (``energy.snapshots``)
    when the run wrote no snapshot."""
    snaps = snapshots(lines, cluster)
    if not snaps:
        raise RunError("energy.snapshots", "the run wrote no `snap` lines: SNAX-GVSoC is older "
                       "than its snapshots (the submodule)")  # fmt: skip
    ws = windows(snaps, marks)
    p = prices_of(graph, est, mover_specs(cluster))
    static = price(prices_of(graph, est, port_movers(est)), graph_counts(graph, est))
    return EnergyRun(est.tech or "", dict(est.char_source or {}), p, ws,
                     [price(p, counts_of(w, cluster)) for w in ws], static)  # fmt: skip
