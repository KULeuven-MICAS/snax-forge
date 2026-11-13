"""Design point -> task list.

``task_list`` derives the task list (tasks.py) from a design point and decides nothing; what it
derives, loads, streamer and idle-mover tasks, accelerator tasks, serial syncs, starts, stores
and the final syncs, is CONTRACTS, "Task list" (its "Derived" part). Each accelerated node is
one tile: every map around it is a temporal firing loop (``unit_firings``). Every configure and
sync uses the platform's ``wait_mode``; a load moves the whole layout as contiguous beats.

``memlet_values`` gives a streamer task's values from a memlet and the container's L1 layout:
base ``layout.base + sum_d (c_d + sum_k a_dk * b_k) * strides[d]`` for firing loops ``v_k``
(begin ``b_k``, step ``s_k``); one temporal loop per firing loop, innermost first, stride
``sum_d a_dk * s_k * strides[d]``; one spatial loop per range dimension, the last range first.
The spatial bounds must be the streamer's, and for an ``affine`` BRM the addresses visited beat
by beat (``addresses``) must be the nest's through the same layout (streams.py). Accelerators
run in series (``serial_syncs``); a ``LowerError`` names what does not fit.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np

from snax_forge import expr
from snax_forge.brm import Instance, memlet_ordered
from snax_forge.design.streamers import (
    Loop,
    accelerated,
    accelerator_of,
    accelerators,
    firing_loops,
    group_runs,
    instances,
    linked_ports,
    port_loops,
    streamer_name,
)
from snax_forge.design.streamers import groups as unit_groups
from snax_forge.dfg import DfgError, Graph, Memlet, Node, named_rates
from snax_forge.dfg.pad import padded_shape
from snax_forge.dfg.subset import parse_dim

from .layout import Layout, dma_side, over
from .streams import StreamError, in_words, streamer_values
from .tasks import Configure, Start, Sync, TaskList

if TYPE_CHECKING:
    from snax_forge.design.point import DesignPoint


class LowerError(ValueError):
    """A design point whose task list cannot be derived; the message names the node."""


def memlet_values(
    m: Memlet, loops: list[Loop], layout: Layout, symbols: dict[str, int], what: str,
    num_slc: int = 1,
) -> tuple[dict[str, Any], list[int]]:  # fmt: skip
    """A streamer's values for memlet ``m`` over ``loops``, in words as streams.py's (the
    fastest spatial stride a channel's, ``num_slc`` elements on); and its spatial bounds, in
    elements, fastest first."""
    names = [lp.var for lp in loops]
    base = layout.base
    tstrides = [0] * len(loops)
    spatial: list[tuple[int, int]] = []
    for d, dim in enumerate(m.subset):
        parts = parse_dim(dim, what)
        try:
            const, co = expr.linear(parts[0], names)
        except expr.ExprError as e:
            raise LowerError(f"{what}: {dim!r} is not affine in {names}: {e}") from None
        stride = layout.strides[d]
        start = expr.evaluate(const, symbols) + sum(co[lp.var] * lp.begin for lp in loops)
        base += start * stride
        for i, lp in enumerate(loops):
            tstrides[i] += co[lp.var] * lp.step * stride
        if len(parts) == 3:
            _, end_co = expr.linear(parts[1], names)
            if end_co != co:
                raise LowerError(f"{what}: range {dim!r} changes length with {names}")
            length = expr.evaluate(expr.linear(f"({parts[1]}) - ({parts[0]})", names)[0], symbols)
            spatial.append((len(range(0, length, parts[2])), parts[2] * stride))
    temporal = list(zip([lp.count for lp in loops], tstrides, strict=True))[::-1]
    if not temporal:
        temporal = [(1, 0)]  # one beat
    fastest = spatial[::-1]
    values = {
        "base": base,
        "temporal_bounds": [b for b, _ in temporal],
        "temporal_strides": [s for _, s in temporal],
        "spatial_strides": in_words([s for _, s in fastest], num_slc),
    }
    return values, [b for b, _ in fastest]


def addresses(values: dict[str, Any], spatial_bounds: list[int]) -> np.ndarray:
    """The byte addresses a streamer's values visit, beat by beat: (beats, lanes). For a packed
    port give the bounds in channels (the fastest S_0 / num_slc): each address is then a word."""
    t = list(zip(values["temporal_bounds"], values["temporal_strides"], strict=True))[::-1]
    s = list(zip(spatial_bounds, values["spatial_strides"], strict=True))[::-1]
    grid = np.full([b for b, _ in t] + [b for b, _ in s], values["base"], dtype=np.int64)
    for axis, (b, stride) in enumerate(t + s):
        shape = [1] * grid.ndim
        shape[axis] = b
        grid = grid + (np.arange(b) * stride).reshape(shape)
    lanes = int(np.prod([b for b, _ in s])) if s else 1
    return grid.reshape(-1, lanes)


def task_list(point: DesignPoint) -> TaskList:
    """The task list of a design point; see the module doc. Raises LowerError."""
    return serial_syncs(_Derivation(point).run(), point)


def accelerator_components(point: DesignPoint) -> dict[str, str]:
    """Each component an accelerator task runs on (a unit, or a linked group) -> its
    accelerator, the streamer set it is in."""
    group = {u: k for k, units in unit_groups(point.graph).items() for u in units}
    out: dict[str, str] = {}
    for acc, units in accelerators(point.graph).items():
        for u in units:
            out[u] = out[group[u]] = acc
    return out


# NB (D158): accelerators run in series: the next one is configured while the current one runs
# and started after a sync on it.
def serial_syncs(tasks: TaskList, point: DesignPoint) -> TaskList:
    """``tasks`` with the serial syncs (CONTRACTS, "Task list"): before each start of an accelerator
    task whose accelerator is not the previous accelerator task's, a ``sync`` on that previous
    task, in its ``wait_mode``, unless an ``after`` of the start's tasks holds it, or a sync on it
    comes after its start already. The configures stay where they are, before the sync."""
    acc_of = accelerator_components(point)
    conf = tasks.configured()
    steps: list[Any] = []
    prev: tuple[str, str] | None = None  # (accelerator, its accelerator task)
    synced: set[str] = set()
    for step in tasks.steps:
        if isinstance(step, Sync):
            synced.add(step.task)
        if isinstance(step, Start):
            accel = [t for t in step.tasks if conf[t].type == "accel"]
            if accel:
                acc = acc_of[conf[accel[0]].component]
                after = {a for t in step.tasks for a in conf[t].after}
                if prev is not None and prev[0] != acc and prev[1] not in after | synced:
                    steps.append(Sync(prev[1], conf[prev[1]].wait_mode))
                    synced.add(prev[1])
                prev = (acc, accel[0])
        steps.append(step)
    return TaskList(tasks.name, steps)


def unit_firings(g: Graph, node: Node, inst: Instance) -> tuple[dict[str, Any], list[Loop]]:
    """An accelerated node's accelerator values, its firings n and its named rates, and
    its firing loops; the estimate reads the same. Raises LowerError."""
    what = f"node {node.id}"
    try:
        loops = firing_loops(g, node)
    except ValueError as e:
        raise LowerError(f"{what}: {e}") from None
    memlets = {**node.inputs, **node.outputs}
    try:  # named rates from the loops a memlet does not use, as REF and bind do
        rates = named_rates(
            {p.name: (memlets[p.name], p.rate) for p in inst.ports},
            [(lp.var, lp.count) for lp in loops],
            what,
        )
    except DfgError as e:
        raise LowerError(str(e)) from None
    return {"n": prod(lp.count for lp in loops)} | rates, loops


class _Derivation:
    """``task_list``'s state while it walks the accelerated nodes: the steps so far and, per
    container, the task that last put it in L1, the readers since, and the task that leaves it
    in its last memory. Each group of units is one pass of loads, streamers, the accelerator
    task and stores (``group``)."""

    def __init__(self, point: DesignPoint) -> None:
        g = point.graph
        self.point, self.g, self.pf, self.plan = point, g, point.platform, point.memory
        self.symbols = g.bound_symbols
        self.insts = instances(g)
        self.mode = self.pf.wait_mode
        self.beat = self.pf.l1.wide_bits // 8
        self.nodes = accelerated(g)
        self.groups = accelerators(g)
        self.group_of = {u: k for k, units in unit_groups(g).items() for u in units}
        self.linked = {k: units for k, units in unit_groups(g).items() if len(units) > 1}
        self.runs = {
            m.id: run for runs in group_runs(g).values() for run in runs for m in run.values()
        }  # a linked group's runs
        self.linked_ports = linked_ports(g)
        self.last_write = {m.data: n.id for n in self.nodes for m in n.outputs.values()}
        self.steps: list[Any] = []
        self.in_l1: dict[str, str] = {}  # container -> task that last put it in L1
        self.readers: dict[str, list[str]] = {}  # container -> reader tasks since that
        self.final: dict[str, str] = {}  # container -> task that leaves it in its last memory

    def run(self) -> TaskList:
        done: set[str] = set()
        for n in self.nodes:
            group = self.group_of[n.attrs["instance"]]
            if n.id in done:  # a linked group's units are one task, at its first node
                continue
            members = self.linked.get(group)
            self.run_nodes = None
            if members is not None:  # one run of the group: the nodes its links join
                self.run_nodes = self.runs[n.id]
                done.update(m.id for m in self.run_nodes.values())
            self.group(n, group, members)
        for c, cont in self.g.containers.items():
            if c in self.final and not cont.transient:
                self.steps.append(Sync(self.final[c], self.mode))
        return TaskList(self.point.name, self.steps)

    def group(self, n: Any, group: str, members: list[str] | None) -> None:
        """One group's steps, at its first node ``n``: loads, streamers, the accelerator task,
        then the writes in L1 and the stores."""
        unit_nodes = [n] if members is None else [self.unit_node(u) for u in members]
        starts, unit_loops = self.firings(unit_nodes)
        self.loads(unit_nodes)
        own = self.streamers(unit_nodes, starts, unit_loops)
        values = {u: v | self.slot_values(un) for (u, v), un in zip(starts.items(), unit_nodes,
                                                                      strict=True)}  # fmt: skip
        self.accelerator(n, group, members, values, own)
        self.stores(unit_nodes, own)

    # -- helpers ----------------------------------------------------------------------------
    run_nodes: dict[str, Any] | None = None

    def slot_values(self, un: Any) -> dict[str, dict[str, str]]:
        """A unit's slots read from a container (CSRs, ``Instance.read``), each
        ``{"from": <container>}``, in slot order."""
        inst = self.insts[un.attrs["instance"]]
        return {k: {"from": un.inputs[k].data} for k in inst.read}

    def unit_node(self, u: str) -> Any:
        """A linked unit's node in the group's run being lowered (streamers.py's
        ``group_runs``)."""
        if self.run_nodes is None or u not in self.run_nodes:
            raise LowerError(f"unit {u}: no node in this run of its group")
        return self.run_nodes[u]

    def layout(self, c: str, mem: str) -> Layout | None:
        return self.plan.layouts.get(c, {}).get(mem)

    def moved(self, c: str) -> bool:
        return self.layout(c, "l1") is not None and self.layout(c, "l2") is not None

    def outer(self, un: Any) -> list[str]:
        """The node's ports with a streamer: every port not on a link."""
        u = un.attrs["instance"]
        return [p.name for p in self.insts[u].ports
                if f"{u}.{p.name}" not in self.linked_ports]  # fmt: skip

    def dma(self, task: str, direction: str, c: str, after: list[str]) -> None:
        elem, padded = np.dtype(self.g.containers[c].dtype).itemsize, padded_shape(self.g, c)
        l1 = dma_side(over(self.layout(c, "l1"), padded), elem, self.beat)  # a tail's pads too
        l2 = {"symbol": c} | {k: v for k, v in dma_side(over(self.layout(c, "l2"), padded), elem,
                                                        self.beat).items() if k != "base"}  # fmt: skip
        src, dst = (l2, l1) if direction == "l2_to_l1" else (l1, l2)
        values = {"direction": direction, "src": src, "dst": dst}
        self.steps.append(Configure(task, "dma", "dma", values, after, self.mode))
        self.steps.append(Start([task]))

    def idle(self, node: str, s: Any) -> Configure:
        """Node ``node``'s task on another group's mover ``s``: every bound 0, so it never
        starts and moves nothing; every temporal stride a word (CONTRACTS, "Task list")."""
        word = self.pf.l1.width_bits // 8
        values = {"base": 0, "temporal_bounds": [0] * s.temporal_dims,
                  "temporal_strides": [word] * s.temporal_dims,
                  "spatial_strides": [0] * len(s.spatial_bounds)}  # fmt: skip
        return Configure(f"{node}_{s.name}", "streamer", s.name, values, [], self.mode)

    # -- the phases of a group ----------------------------------------------------------------

    def firings(
        self, unit_nodes: list[Any]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[Loop]]]:
        """Each unit's accelerator values (its firings n and its named rates) and its firing
        loops."""
        starts: dict[str, dict[str, Any]] = {}
        unit_loops: dict[str, list[Loop]] = {}
        for un in unit_nodes:
            u = un.attrs["instance"]
            starts[u], unit_loops[u] = unit_firings(self.g, un, self.insts[u])
        return starts, unit_loops

    def loads(self, unit_nodes: list[Any]) -> None:
        """A DMA into L1 for each input not there yet."""
        for un in unit_nodes:
            for p in self.outer(un):
                c = un.inputs[p].data if p in un.inputs else None
                if c is not None and c not in self.in_l1 and self.moved(c):
                    task = f"load_{c}"
                    self.dma(task, "l2_to_l1", c, [])
                    self.in_l1[c] = task

    def streamers(self, unit_nodes: list[Any], starts: dict[str, dict[str, Any]],
                  unit_loops: dict[str, list[Loop]]) -> dict[str, Configure]:  # fmt: skip
        """Each outer port's streamer task, by streamer, after what it needs in L1."""
        own: dict[str, Configure] = {}
        # this start's reader tasks -> (their unit's node, the addresses they visit)
        launched: dict[str, tuple[str, np.ndarray]] = {}
        for un in unit_nodes:
            u, what = un.attrs["instance"], f"node {un.id}"
            inst, loops = self.insts[u], unit_loops[u]
            memlets = {**un.inputs, **un.outputs}
            for p in self.outer(un):
                m, s = memlets[p], self.point.streamers[streamer_name(u, p)]
                lay = self.layout(m.data, "l1")
                if lay is None:
                    raise LowerError(f"{what}.{p}: container {m.data} has no L1 layout")
                values, sb = memlet_values(
                    m, port_loops(inst, p, m, loops), lay, self.symbols, f"{what}.{p}", s.num_slc
                )
                if sb != list(s.spatial_bounds):
                    raise LowerError(
                        f"{what}.{p}: the memlet gives spatial bounds {sb}, streamer {s.name} has "
                        f"{list(s.spatial_bounds)}"
                    )
                if not memlet_ordered(inst.brm):  # affine: the nest must give the same order
                    try:
                        want = streamer_values(inst, p, starts[u], lay, s, self.pf.l1)
                    except StreamError as e:
                        raise LowerError(f"{what}.{p}: {e}") from None
                    if not np.array_equal(addresses(values, sb), addresses(want, sb)):
                        raise LowerError(
                            f"{what}.{p}: the memlet gives {values}, the BRM's nest gives {want}: "
                            "they take the elements in different orders"
                        )
                task = f"{un.id}_{s.name}"
                c = m.data
                if p in un.inputs:
                    after = [self.in_l1[c]] if c in self.in_l1 else []
                    self.readers.setdefault(c, []).append(task)
                    launched[task] = (un.id, addresses(values, sb))
                else:
                    readers = self.readers.pop(c, [])
                    for r in (r for r in readers if r in launched):
                        self.in_place(f"{what}.{p}", c, un.id, addresses(values, sb), r,
                                      *launched[r])  # fmt: skip
                    after = [
                        *(r for r in readers if r not in launched),
                        *([self.in_l1[c]] if c in self.in_l1 else []),
                    ]
                own[s.name] = Configure(task, "streamer", s.name, values, after, self.mode)
        return own

    @staticmethod
    def in_place(what: str, c: str, node: str, writes: np.ndarray, reader: str,
                 reader_node: str, reads: np.ndarray) -> None:  # fmt: skip
        """A writer of ``c`` whose reader ``reader`` starts with it: allowed only as an
        in-place update, its own unit reading the same addresses in the same order
        (CONTRACTS, "Task list", its "Derived" part)."""
        if reader_node != node:
            raise LowerError(
                f"{what}: writes {c}, which {reader} (node {reader_node}) reads in the same "
                "start; an in-place update lowers only within one unit"
            )
        if not np.array_equal(writes, reads):
            raise LowerError(
                f"{what}: writes {c} in place of {reader}'s read, but not the same elements in "
                "the same order; an in-place update lowers only as the same subset over the "
                "same firing loops"
            )

    def accelerator(self, n: Any, group: str, members: list[str] | None,
                    starts: dict[str, dict[str, Any]], own: dict[str, Configure]) -> None:  # fmt: skip
        """Every mover of the group's accelerator, its streamer set (the set's other groups'
        idle, in an encapsulated one only), then the group's task, all started together."""
        name = n.attrs["instance"]
        movers = []
        for unit in self.groups[accelerator_of(n)]:
            for q in (q.name for q in self.insts[unit].ports):
                s = self.point.streamers.get(streamer_name(unit, q))
                if s is None:  # a port on a link
                    continue
                c = own[s.name] if s.name in own else self.idle(n.id, s)
                self.steps.append(c)
                movers.append(c.task_name)
        component = name if members is None else group
        acc_task = f"{n.id}_{component}"
        start = starts[name] if members is None else starts
        self.steps.append(Configure(acc_task, "accel", component, start, [], self.mode))
        self.steps.append(Start([*movers, acc_task]))

    def stores(self, unit_nodes: list[Any], own: dict[str, Configure]) -> None:
        """After the group, its writes land in L1; a DMA out of L1 for each container's last
        write that has an L2 home."""
        for un in unit_nodes:
            for p in self.outer(un):
                if p in un.outputs:
                    c = un.outputs[p].data
                    task = own[streamer_name(un.attrs["instance"], p)].task_name
                    self.in_l1[c] = task
                    self.final[c] = task
        for un in unit_nodes:
            for p in self.outer(un):
                c = un.outputs[p].data if p in un.outputs else None
                if (
                    c is not None
                    and self.last_write[c] == un.id
                    and self.layout(c, "l2") is not None
                ):
                    store = f"store_{c}"
                    self.dma(store, "l1_to_l2", c,
                             [own[streamer_name(un.attrs["instance"], p)].task_name])  # fmt: skip
                    self.final[c] = store
