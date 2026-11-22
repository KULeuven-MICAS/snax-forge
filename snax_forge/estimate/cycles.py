"""The ideal cycles of a bound graph's groups, and the work no unit runs (CONTRACTS, "Estimate").

``groups(graph)`` gives a ``GroupEstimate`` per group (the units of one ``group``, else of one
instance, and its links) with the fields CONTRACTS lists; ``cost`` is set later by
``estimate``. ``group_cycles`` steps snax-gvsoc's group rule with every outer input valid and the
outer output ready, a sliced input through its slicer (one cycle a run, D180), the one place
this is done (gen/rtl.py's ``group_counter`` calls it);
``group_rule`` loads ``snax/snax_group_rule.py`` from the submodule once, and without it
``estimate.submodule`` names the fix. A unit's L, II and span are its implementation's timing
at its params, its firings and named rates SNAX-LOWER's (``unit_firings``), JOBS and each T what
the program writes (``job_writes``). ``group_blocks`` and ``datapath_blocks`` give the blocks to
cost, with and without the cycles; ``not_mapped`` counts the iterations no unit runs.
"""

from __future__ import annotations

import functools
import importlib.util
from dataclasses import dataclass, field
from math import prod
from types import ModuleType
from typing import Any

from snax_forge.brm import SEED
from snax_forge.common import CodedError, repo_root
from snax_forge.design.streamers import (
    forked_ports,
    forks,
    group_chains,
    group_links,
    group_runs,
    instance_nodes,
    instances,
    packed,
    port_dtype,
    sliced,
    stream_lanes,
)
from snax_forge.design.streamers import groups as graph_groups
from snax_forge.dfg import Graph, Node
from snax_forge.dfg.subset import map_range
from snax_forge.lower.c import fold_rate, job_writes
from snax_forge.lower.derive import unit_firings

from .bits import ctrl_bits, packer_bits, slicer_bits
from .cost import Block, GroupCost
from .data import WORD_BITS

SUBMODULE = repo_root() / "gvsoc" / "snax-gvsoc"
# The streamers' fill a run adds to a group's cycles: 3, or 2 with one outer input
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


def group_cycles(units: list[dict[str, Any]], links: list[dict[str, Any]],
                 owner: str | list[str], jobs: int | list[int], ts: list[int],
                 forks: list[list[tuple[str, int]]] | None = None) -> int | None:  # fmt: skip
    """The busy cycles of one run of a group, E - s: the group rule (``units`` with ``name``,
    ``L``, ``ii``, ``span`` and ``inputs``, its outer inputs, and optionally ``d_in``, their pops
    per word in port order, absent all 1; ``links`` ``{from, to, to_input, depth}``; ``owner``
    the unit with the outer output, or with several chains each sink in unit order, ``jobs``
    then one per chain, and ``forks`` each ``(unit, outer input)`` one reader feeds, D189)
    stepped with every input valid and every output ready, JOBS ``jobs`` and each unit's T
    ``ts``; 0 for the empty run (a JOBS or a T 0), None for a run that does not end."""
    several = not isinstance(jobs, int)
    if 0 in (jobs if several else [jobs]) or 0 in ts:
        return 0
    rule = group_rule().GroupRule(units, links, owner, forks or ())
    n = rule.firings_per_run(jobs, ts)
    d_in = [1] * rule.n_valid
    for u, places in zip(units, rule.place, strict=True):
        for d, i in zip(u.get("d_in", [1] * u["inputs"]), places, strict=True):
            d_in[i] = max(d_in[i], d)  # a forked input: its reader's slicer, at its first
    ready = (lambda c: [True] * len(rule.outputs)) if several else (lambda c: True)
    bound = sum(n) * max(u["ii"] * u["span"] for u in units) + sum(u["L"] for u in units) + 16
    _, end = rule.run(jobs, ts, lambda c: [True] * len(d_in), ready, bound, 1, d_in)
    return end


def rule_forks(graph: Graph, members: list[str], insts: dict[str, Any],
               fed: set[str]) -> list[list[tuple[str, int]]]:  # fmt: skip
    """The group's forks (D189) as the group rule takes them: per fork each reader's unit and
    the place of its port among the unit's outer inputs (its ``in`` ports no link feeds)."""
    out = []
    for f in forks(graph):
        mine = []
        for r in f.attrs["readers"]:
            u, p = r.split(".")
            if u in members:
                outer = [q for q in _ins(insts[u]) if f"{u}.{q}" not in fed]
                mine.append((u, outer.index(p)))
        if len(mine) > 1:
            out.append(mine)
    return out


@dataclass
class GroupEstimate:
    """One group: its units and links, its cycles and register bits (CONTRACTS, "Estimate") and,
    from data.py, the channels of its outer ports and the words they move per cycle; on a platform
    its ``bank_bound``, the fewest cycles L1's banks allow those words."""

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
    bits: dict[str, int] = field(default_factory=dict)
    ctrl_bits: int | None = None
    cost: GroupCost | None = None
    bank_bound: int | None = None  # ceil(words / l1.n_banks); None without a platform
    n_banks: int | None = None  # the platform's, which bank_bound divides by
    # per run, its nodes and its cycles (schedule.py's runs); not written, derived again
    run_cycles: list[tuple[list[str], int]] = field(default_factory=list, compare=False)

    @property
    def counter(self) -> int:
        """The expected BUSY_COUNTER: the cycles plus the streamers' fill, a guide and not a check:
        a run's counts from START until busy drops, with bank conflicts and the
        streamers' head start, so it may be above or below this."""
        return self.cycles + self.fill

    @property
    def bank_bound_note(self) -> str | None:
        """One line when the group's streams need more cycles of L1 time than its ideal cycles
        (``bank_bound`` > ``cycles``), else None (design.md, run.md, the page): a bound on the
        streams, not on the BUSY_COUNTER."""
        if self.bank_bound is None or self.bank_bound <= self.cycles:
            return None
        words = round(self.words_per_cycle * self.cycles)
        return (f"{self.name}: the streams move {words} words over {self.n_banks} banks, at "
                f"least {self.bank_bound} cycles of L1 time against the ideal {self.cycles}")  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "units": list(self.units), "links": list(self.links),
                "cycles": self.cycles, "bottleneck": self.bottleneck, "ramp": self.ramp,
                "stall": self.stall, "fill": self.fill, "counter": self.counter,
                "channels": self.channels, "words_per_cycle": self.words_per_cycle,
                "bits": dict(self.bits), "ctrl_bits": self.ctrl_bits,
                "cost": None if self.cost is None else self.cost.to_dict(),
                "bank_bound": self.bank_bound, "n_banks": self.n_banks}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GroupEstimate:
        return cls(d["name"], list(d["units"]), d["cycles"], d["bottleneck"], d["ramp"],
                   d["stall"], d["fill"], list(d["links"]), d["channels"],
                   d["words_per_cycle"], dict(d.get("bits", {})), d.get("ctrl_bits"),
                   GroupCost.from_dict(d["cost"]) if d.get("cost") else None,
                   d.get("bank_bound"), d.get("n_banks"))  # fmt: skip


def _ins(inst: Any) -> list[str]:
    return [p.name for p in inst.ports if p.direction == "in"]


def _datapath_blocks(graph: Graph, insts: dict[str, Any], members: list[str],
                     ends: list[tuple[Node, str, str]], firings: dict[str, int],
                     gives: dict[str, int],
                     ) -> tuple[dict[str, Block], dict[str, Block], dict[str, int]]:  # fmt: skip
    """Per unit and per buffer link (by container) the datapath to cost, its params as the generator
    fills them and its fires (cost.py: a unit's ``firings`` times its span, a buffer's its
    producer's ``gives``, a load per output), and per unit and link its register bits (CONTRACTS,
    "Estimate"): a wire 0 and no block; a unit the generator does not fill a block with no params
    and no bits. A packed outer output's ``FifoPacker`` joins the links by ``<unit>.<port>``, a
    fire per result; a sliced outer input's ``FifoSlicer`` (``_slicers``) too, a fire per pop: its
    unit's firings, a seed's one per job (its unit's results). The generator is imported here:
    gen/rtl.py imports this package."""
    from snax_forge.gen.accelerator import Unit
    from snax_forge.gen.blocks import DATAPATHS, GenError, fill_node
    from snax_forge.gen.rtl import buffer_params, catalogue, fill, flops

    cat, nodes, filled, units, links = catalogue(), instance_nodes(graph), {}, {}, {}
    for u in members:
        inst = insts[u]
        dp = (inst.impl.binding or {}).get("datapath")
        units[u] = Block(dp, {}, firings[u] * inst.span)
        if dp not in DATAPATHS or dp not in cat:
            continue
        try:
            f = fill_node(dp, inst, graph, nodes[u])
        except GenError:
            continue
        filled[u] = Unit(u, inst.brm.name, dp, f.ports, dict(f.params), inst.latency,
                         inst.initiation_interval, [], f.op_type, inst.span)  # fmt: skip
        units[u] = Block(dp, fill(None, filled[u], cat[dp]), firings[u] * inst.span)
    bits = {u: flops(b.datapath, b.params, cat) for u, b in units.items() if u in filled}
    for k, frm, _ in ends:
        unit, port = frm.split(".", 1)
        if unit not in filled:
            continue
        name, depth = k.attrs["container"], k.attrs["depth"]
        if depth == 0:
            bits[name] = 0
            continue
        end = filled[unit].ports[port]
        links[name] = Block("LinkBuffer", buffer_params(depth, end.lanes, end.dtype,
                                                        cat["LinkBuffer"]), gives[unit])  # fmt: skip
        bits[name] = flops("LinkBuffer", links[name].params, cat)
    taken = {frm for _, frm, _ in ends}
    for u, unit in filled.items():  # a packed outer output's packer, by <unit>.<port> (D174)
        for p, port in unit.ports.items():
            if port.dir != "out" or f"{u}.{p}" in taken or port.dtype is None:
                continue
            d_out = packed(port.dtype, port.lanes, True, WORD_BITS)
            if d_out > 1:
                params = {"d_out": d_out, "lanes": port.lanes, "dtype": port.dtype}
                links[f"{u}.{p}"] = Block("FifoPacker", params, gives[u])
                bits[f"{u}.{p}"] = packer_bits(d_out, port.lanes, port.dtype)
    for k, b in _slicers(graph, insts, members, ends).items():  # a sliced outer input's (D180)
        u, p = k.split(".", 1)
        links[k] = Block("FifoSlicer", b.params, gives[u] if p == SEED else firings[u])
        bits[k] = slicer_bits(b.params["d_in"])
    return units, links, bits


def _slicers(graph: Graph, insts: dict[str, Any], members: list[str],
             ends: list[tuple[Node, str, str]]) -> dict[str, Block]:  # fmt: skip
    """Each sliced outer input's ``FifoSlicer`` (D180) by ``<unit>.<port>``: its ``d_in``, lanes
    and dtype, no fires (``_datapath_blocks`` gives its pops)."""
    fed, nodes, out = {to for _, _, to in ends} | set(forked_ports(graph)), {}, {}
    for n, _ in graph.walk():
        if n.kind == "accelerated":
            nodes.setdefault(n.attrs["instance"], []).append(n)
    for u in members:
        for p in _ins(insts[u]):
            if f"{u}.{p}" in fed:
                continue
            dtype = port_dtype(graph, nodes[u], p)
            lanes = stream_lanes(graph, nodes[u][0], insts[u], p)  # one for a broadcast (D186)
            d_in = sliced(dtype, lanes, False, WORD_BITS)
            if d_in > 1:
                out[f"{u}.{p}"] = Block("FifoSlicer", {"d_in": d_in, "lanes": lanes,
                                                       "dtype": dtype}, 0)  # fmt: skip
    return out


def _fixed_ts(graph: Graph, insts: dict[str, Any], members: list[str]) -> dict[str, int | None]:
    """Per unit its T when fixed at design time, as accelerator.py decides it: a fold's constant
    rate, 1 without a fold, None when the start carries it (a JOB_BEATS)."""
    from snax_forge.gen.accelerator import fixed_fold

    nodes, out = instance_nodes(graph), {}
    for u in members:
        if fold_rate(insts[u]) is None:
            out[u] = 1
        else:
            out[u] = fixed_fold(graph, nodes[u], insts[u])
    return out


def datapath_blocks(graph: Graph) -> dict[str, Block]:
    """Every unit (by name) and buffer link (by container) of a bound graph with the datapath
    its cost looks up and its params, without stepping the group rule (fires 0): what the
    design note ``char.missing`` reads."""
    insts, out = instances(graph), {}
    for members in graph_groups(graph).values():
        ends = group_links(graph, members)
        zero = dict.fromkeys(members, 0)
        units, bufs, _ = _datapath_blocks(graph, insts, members, ends, zero, zero)
        out |= units | bufs
    return out


def groups(graph: Graph, job_width: int = 32) -> list[GroupEstimate]:
    """Every group of a bound graph, in the graph's order (CONTRACTS, "Estimate"); ``job_width`` the
    platform's ``acc_csr.job_width``."""
    return [g for g, _, _ in group_blocks(graph, job_width)]


def group_blocks(graph: Graph, job_width: int = 32,
                 ) -> list[tuple[GroupEstimate, dict[str, Block], dict[str, Block]]]:  # fmt: skip
    """``groups``, each with its units and buffer links to cost (by name and by container,
    cost.py's ``Block``): per unit its firings over the group's tasks times its span, per
    buffer link its producer's outputs, its firings over its T (a fold gives once per T beats:
    dot 6's Mac once per job)."""
    insts = instances(graph)
    nodes: dict[str, list[Node]] = {}
    for n, _ in graph.walk():
        if n.kind == "accelerated":
            nodes.setdefault(n.attrs["instance"], []).append(n)
    out = []
    runs = group_runs(graph)
    for name, members in graph_groups(graph).items():
        ends = group_links(graph, members)  # one per pair of ports
        fed = {to for _, _, to in ends}
        taken = {frm for _, frm, _ in ends}
        slicers = _slicers(graph, insts, members, ends)
        rule_units = [{"name": u, "L": insts[u].latency, "ii": insts[u].initiation_interval,
                       "span": insts[u].span,
                       "inputs": sum(f"{u}.{p}" not in fed for p in _ins(insts[u])),
                       "d_in": [slicers[k].params["d_in"] if k in slicers else 1
                                for p in _ins(insts[u]) if (k := f"{u}.{p}") not in fed]}
                      for u in members]  # fmt: skip
        rule_links = [{"from": frm.split(".")[0], "to": to.split(".")[0],
                       "to_input": _ins(insts[to.split(".")[0]]).index(to.split(".")[1]),
                       "depth": k.attrs["depth"]} for k, frm, to in ends]  # fmt: skip
        owners = [u for u in members for p in insts[u].ports
                  if p.direction == "out" and f"{u}.{p.name}" not in taken]  # fmt: skip
        owner: str | list[str] = owners[0] if len(owners) == 1 else owners
        chains = group_chains(graph, members) if len(owners) > 1 else None
        rule_fork = rule_forks(graph, members, insts, fed)
        tasks = (  # a linked group once per run (several when nests share it)
            runs[name] if len(members) > 1 else [{members[0]: n} for n in nodes[members[0]]]
        )
        cycles = bottleneck = 0
        run_cycles: list[tuple[list[str], int]] = []
        firings, gives = dict.fromkeys(members, 0), dict.fromkeys(members, 0)
        for task in tasks:
            values = {u: unit_firings(graph, n, insts[u])[0] for u, n in task.items()}
            writes = job_writes({u: insts[u] for u in task}, values if len(task) > 1 else
                                values[members[0]], set(), chains)  # fmt: skip
            all_jobs = [v for _, k, v in writes if k.startswith("JOBS")]
            jobs: int | list[int] = all_jobs[0] if chains is None else all_jobs
            ts = [values[u][r] if (r := fold_rate(insts[u])) else 1 for u in members]
            end = group_cycles(rule_units, rule_links, owner, jobs, ts, rule_fork)
            if end is None:
                raise EstimateError("estimate.counter", f"group {name}: the group rule does not "
                                    "end")  # fmt: skip
            cycles += end
            run_cycles.append(([n.id for n in task.values()], end))
            for u in task:
                firings[u] += values[u]["n"]
                t = ts[members.index(u)]
                gives[u] += values[u]["n"] // t if t else 0  # T = 0: the empty run
            # per run (a group several nests share runs once per nest): its slowest unit
            bottleneck += max(values[u]["n"] * max(insts[u].initiation_interval, insts[u].span)
                              for u in task)  # fmt: skip
        ramp = sum(insts[u].latency + insts[u].span - 1 for u in members)
        ramp += sum(k.attrs["depth"] >= 1 for k, _, _ in ends)
        ramp *= len(tasks)  # each run fills the group's pipeline again
        inputs = sum(u["inputs"] for u in rule_units)
        units, bufs, bits = _datapath_blocks(graph, insts, members, ends, firings, gives)
        g = GroupEstimate(name, list(members), cycles, bottleneck, ramp,
                          cycles - bottleneck - ramp, FILL_ONE_INPUT if inputs == 1 else FILL,
                          [k.attrs["container"] for k, _, _ in ends], bits=bits,
                          ctrl_bits=ctrl_bits(rule_units, rule_links, owner,
                                              _fixed_ts(graph, insts, members), job_width))  # fmt: skip
        g.run_cycles = run_cycles
        out.append((g, units, bufs))
    return out


def not_mapped(graph: Graph, spatial: bool = False) -> int:
    """The iterations no unit runs (module doc), over every map around them but the spatial
    ones; with ``spatial``, over every map, spatial ones included: the unmapped work as
    iterations, since a tag alone runs nothing (lanes only exist once a unit runs them)."""
    symbols = graph.bound_symbols
    total = 0
    for n, outer in graph.walk():
        holds = n.kind == "map" and any(k.kind == "tasklet" for k in n.body or [])
        if holds or (n.kind == "tasklet" and not outer):
            loops = [m for m in (*outer, n) if m.kind == "map"] if holds else []
            total += prod(len(map_range(m.attrs["range"], symbols, m.id)) for m in loops
                          if spatial or m.attrs.get("loop.kind") != "spatial")  # fmt: skip
    return total
