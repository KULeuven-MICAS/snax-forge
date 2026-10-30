"""The ideal cycles of a bound graph's groups, and the work no unit runs (D151, C§19).

    group_rule() -> module                    snax-gvsoc's snax/snax_group_rule.py (GroupRule)
    group_cycles(units, links, owner, jobs, ts) -> int | None
    groups(graph) -> list[GroupEstimate]
    not_mapped(graph) -> int

**The rule.** A group's cycles are snax-gvsoc's group rule stepped with every outer input
valid and the outer output ready: the COUNTER of C§17 on ideal streams, which a run on
SNAX-GVSoC gives plus the streamers' fill. ``group_cycles`` is that stepping, the one place it
is done: gen/rtl.py's ``group_counter`` calls it for a fixture's counter. The rule is loaded by
path from the submodule (``gvsoc/snax-gvsoc``, D134), once; without it, ``estimate.submodule``
names the fix.

**A group** (D146, D148): its units, the accelerated nodes of one ``group`` (``link``'s), else
of one instance, in the graph's order, and its links, from the link nodes. Per unit its L, II
and span are its BRM implementation's timing at its params; its firings n and named rates are
SNAX-LOWER's (lower/derive.py's ``unit_firings``), and JOBS and each unit's T are what the
program writes (lower/c.py's ``job_writes``): T a fold's named rate, else 1. A unit alone runs
one task per node; a linked unit has one node. Then:

    cycles       the rule's
    bottleneck   max over the units of n · max(II, span)
    ramp         the cycles to start and finish the pipeline: Σ over the units of
                 L + span − 1, plus 1 per buffer link (depth >= 1), 0 per wire
    stall        cycles − bottleneck − ramp
    fill         the streamers' fill a run adds: 3, or 2 for a group whose units have one
                 outer input between them (C§17)

**Not mapped**: the iterations of the nests no unit runs, counted once per innermost map that
holds tasklets (a fused nest's two count once), over every map around it but the spatial ones,
itself included; a tasklet outside every map counts 1. Never added to cycles.
"""

from __future__ import annotations

import functools
import importlib.util
from dataclasses import dataclass, field
from math import prod
from types import ModuleType
from typing import Any

from snax_forge.common import CodedError, repo_root
from snax_forge.design.streamers import groups as graph_groups
from snax_forge.design.streamers import instances, link_ends, links
from snax_forge.dfg import Graph, Node
from snax_forge.dfg.subset import map_range
from snax_forge.lower.c import fold_rate, job_writes
from snax_forge.lower.derive import unit_firings

SUBMODULE = repo_root() / "gvsoc" / "snax-gvsoc"
# The streamers' fill a run adds to a group's cycles (C§17): 3, or 2 with one outer input
FILL, FILL_ONE_INPUT = 3, 2


class EstimateError(CodedError):
    """A graph the estimate cannot judge; ``code`` names the rule (``estimate.<rule>``)."""


@functools.cache
def group_rule() -> ModuleType:
    """snax-gvsoc's ``snax/snax_group_rule.py``: the group control rule over edges, the same as
    the C++ SnaxGroupRule and the Chisel GroupRule (the shared group cases check all three)."""
    path = SUBMODULE / "snax" / "snax_group_rule.py"
    if not path.is_file():
        raise EstimateError("estimate.submodule", f"{path} is not there: the submodule "
                            "gvsoc/snax-gvsoc is not checked out (git submodule update --init)")  # fmt: skip
    spec = importlib.util.spec_from_file_location("snax_group_rule", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group_cycles(units: list[dict[str, Any]], links: list[dict[str, Any]], owner: str,
                 jobs: int, ts: list[int]) -> int | None:  # fmt: skip
    """The busy cycles of one run of a group, E - s: the group rule (``units`` with ``name``,
    ``L``, ``ii``, ``span`` and ``inputs``, its outer inputs; ``links`` ``{from, to, to_input,
    depth}``; ``owner`` the unit with the outer output) stepped with every input valid and the
    output ready, JOBS ``jobs`` and each unit's T ``ts``; 0 for the empty run (JOBS or a T 0),
    None for a run that does not end."""
    if jobs == 0 or 0 in ts:
        return 0
    rule = group_rule().GroupRule(units, links, owner)
    n = rule.firings_per_run(jobs, ts)
    bound = sum(n) * max(u["ii"] * u["span"] for u in units) + sum(u["L"] for u in units) + 16
    _, end = rule.run(jobs, ts, lambda c: [True], lambda c: True, bound)
    return end


@dataclass
class GroupEstimate:
    """One group: its units and links, its cycles (module doc) and, from data.py, the channels
    of its outer ports and the words they move per cycle."""

    name: str
    units: list[str]
    cycles: int
    bottleneck: int
    ramp: int
    stall: int
    fill: int
    links: list[str] = field(default_factory=list)
    channels: int = 0
    words_per_cycle: float = 0.0

    @property
    def counter(self) -> int:
        """The COUNTER a run on SNAX-GVSoC gives: the cycles plus the streamers' fill."""
        return self.cycles + self.fill

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "units": list(self.units), "links": list(self.links),
                "cycles": self.cycles, "bottleneck": self.bottleneck, "ramp": self.ramp,
                "stall": self.stall, "fill": self.fill, "counter": self.counter,
                "channels": self.channels, "words_per_cycle": self.words_per_cycle}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GroupEstimate:
        return cls(d["name"], list(d["units"]), d["cycles"], d["bottleneck"], d["ramp"],
                   d["stall"], d["fill"], list(d["links"]), d["channels"],
                   d["words_per_cycle"])  # fmt: skip


def _ins(inst: Any) -> list[str]:
    return [p.name for p in inst.brm.interface.ports if p.direction == "in"]


def groups(graph: Graph) -> list[GroupEstimate]:
    """Every group of a bound graph, in the graph's order (module doc)."""
    insts = instances(graph)
    nodes: dict[str, list[Node]] = {}
    for n, _ in graph.walk():
        if n.kind == "accelerated":
            nodes.setdefault(n.attrs["instance"], []).append(n)
    out = []
    for name, members in graph_groups(graph).items():
        ends = [(k, *link_ends(graph, k)) for k in links(graph) if k.attrs["from"] in members]
        fed = {to for _, _, to in ends}
        taken = {frm for _, frm, _ in ends}
        rule_units = [{"name": u, "L": insts[u].latency, "ii": insts[u].initiation_interval,
                       "span": insts[u].span,
                       "inputs": sum(f"{u}.{p}" not in fed for p in _ins(insts[u]))}
                      for u in members]  # fmt: skip
        rule_links = [{"from": frm.split(".")[0], "to": to.split(".")[0],
                       "to_input": _ins(insts[to.split(".")[0]]).index(to.split(".")[1]),
                       "depth": k.attrs["depth"]} for k, frm, to in ends]  # fmt: skip
        (owner,) = {u for u in members for p in insts[u].brm.interface.ports
                    if p.direction == "out" and f"{u}.{p.name}" not in taken}  # fmt: skip
        tasks = (
            [{u: nodes[u][0] for u in members}]
            if len(members) > 1
            else [{members[0]: n} for n in nodes[members[0]]]
        )
        cycles = bottleneck = 0
        for task in tasks:
            values = {u: unit_firings(graph, n, insts[u])[0] for u, n in task.items()}
            writes = job_writes({u: insts[u] for u in task}, values if len(task) > 1 else
                                values[members[0]], set())  # fmt: skip
            jobs = next(v for _, k, v in writes if k == "JOBS")
            ts = [values[u][r] if (r := fold_rate(insts[u])) else 1 for u in members]
            end = group_cycles(rule_units, rule_links, owner, jobs, ts)
            if end is None:
                raise EstimateError("estimate.counter", f"group {name}: the group rule does not "
                                    "end")  # fmt: skip
            cycles += end
            bottleneck = max(bottleneck, *(values[u]["n"] * max(insts[u].initiation_interval,
                                           insts[u].span) for u in members))  # fmt: skip
        ramp = sum(insts[u].latency + insts[u].span - 1 for u in members)
        ramp += sum(k.attrs["depth"] >= 1 for k, _, _ in ends)
        inputs = sum(u["inputs"] for u in rule_units)
        out.append(GroupEstimate(name, list(members), cycles, bottleneck, ramp,
                               cycles - bottleneck - ramp, FILL_ONE_INPUT if inputs == 1 else FILL,
                               [k.attrs["container"] for k, _, _ in ends]))  # fmt: skip
    return out


def not_mapped(graph: Graph) -> int:
    """The iterations no unit runs (module doc)."""
    symbols = graph.bound_symbols
    total = 0
    for n, outer in graph.walk():
        holds = n.kind == "map" and any(k.kind == "tasklet" for k in n.body or [])
        if holds or (n.kind == "tasklet" and not outer):
            loops = [m for m in (*outer, n) if m.kind == "map"] if holds else []
            total += prod(len(map_range(m.attrs["range"], symbols, m.id)) for m in loops
                          if m.attrs.get("loop.kind") != "spatial")  # fmt: skip
    return total
