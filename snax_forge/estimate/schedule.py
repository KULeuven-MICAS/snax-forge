"""The accelerator runs and the longest path through them (CONTRACTS, "Estimate").

``runs(graph, groups, placement)`` gives a ``RunEstimate`` per run of a group (one per node, or
per set of nodes a linked group joins), in the order LOWER starts them, each with its cycles, the
L1 words its streams move and the earlier runs it waits for: those of its own group or its
accelerator (one streamer set), and those that write what it reads or writes, or read what it
writes, in L1. ``longest_path`` adds the cycles along the slowest chain of waits: the
accelerator's cycles when independent accelerators overlap (D179). ``overlap`` gives, with the
runs started as early as their waits allow, the most words a cycle that runs of different
accelerators move at once; the estimate notes it beyond the banks. Ideal like the rest (D155):
no core configuration, so it cannot tell the two pipelined schedules apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snax_forge.design.streamers import accelerator_of

if TYPE_CHECKING:
    from snax_forge.dfg import Graph

    from .cycles import GroupEstimate


@dataclass
class RunEstimate:
    """One run: named after its first node, as its task is."""

    name: str
    group: str
    accelerator: str
    cycles: int
    words: int  # through L1
    after: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "group": self.group, "accelerator": self.accelerator,
                "cycles": self.cycles, "words": self.words, "after": list(self.after)}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunEstimate:
        return cls(d["name"], d["group"], d["accelerator"], d["cycles"], d["words"],
                   list(d["after"]))  # fmt: skip


def runs(graph: Graph, groups: list[GroupEstimate], in_l1: set[str],
         words: dict[str, int]) -> list[RunEstimate]:  # fmt: skip
    """The runs of ``groups`` (``run_cycles``), ``in_l1`` the containers placed in L1 and
    ``words`` each node's L1 words; see the module doc."""
    nodes = {n.id: n for n, _ in graph.walk() if n.kind == "accelerated"}
    pos = {k: i for i, k in enumerate(nodes)}
    found = []
    for g in groups:
        for ids, cycles in g.run_cycles:
            ids = sorted(ids, key=pos.__getitem__)
            first = nodes[ids[0]]
            reads = {m.data for k in ids for m in nodes[k].inputs.values()} & in_l1
            writes = {m.data for k in ids for m in nodes[k].outputs.values()} & in_l1
            found.append((pos[ids[0]], RunEstimate(first.id, g.name, accelerator_of(first), cycles,
                                                   sum(words.get(k, 0) for k in ids)),
                          reads, writes))  # fmt: skip
    found.sort(key=lambda x: x[0])
    for j, (_, r, reads, writes) in enumerate(found):
        r.after = [p.name for _, p, pr, pw in found[:j]
                   if p.group == r.group or p.accelerator == r.accelerator
                   or pw & (reads | writes) or pr & writes]  # fmt: skip
    return [r for _, r, _, _ in found]


def finish(runs: list[RunEstimate]) -> dict[str, tuple[int, int]]:
    """Each run's start and end, started as soon as every run it waits for ends."""
    out: dict[str, tuple[int, int]] = {}
    for r in runs:
        start = max((out[a][1] for a in r.after), default=0)
        out[r.name] = (start, start + r.cycles)
    return out


def longest_path(runs: list[RunEstimate]) -> int:
    """The cycles along the slowest chain of waits."""
    return max((end for _, end in finish(runs).values()), default=0)


def overlap(runs: list[RunEstimate]) -> tuple[float, list[str]]:
    """The most L1 words a cycle moved at once by runs of two or more accelerators, and those
    accelerators; (0.0, []) when no two accelerators overlap."""
    times = finish(runs)
    best: tuple[float, list[str]] = (0.0, [])
    for r in runs:  # the peak is at some run's start
        t = times[r.name][0]
        live = [x for x in runs if x.cycles and times[x.name][0] <= t < times[x.name][1]]
        accs = list(dict.fromkeys(x.accelerator for x in live))
        rate = sum(x.words / x.cycles for x in live)
        if len(accs) > 1 and rate > best[0]:
            best = (rate, accs)
    return best
