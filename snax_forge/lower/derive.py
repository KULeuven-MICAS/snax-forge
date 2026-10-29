"""Design point -> task list (D73, D64).

The task list (tasks.py, CONTRACTS.md section 9) is derived from a design
point and decides nothing. The accelerated nodes run in the graph's
execution order, one group of tasks each (a single tile: every map around a
node is one of its temporal firing loops, ``firing_loops``); a linked group's
units (D148, one node each) are one group of tasks at its first node, its
ports on a link with no streamer (D112) and no load or store. For each group:

    loads        for each input in port order whose container has an L2 and an
                 L1 layout and is not in L1 yet: configure ``load_<C>`` (DMA,
                 L2 -> L1, the whole layout as contiguous beats; its L2 side
                 the container's data.h array, ``symbol`` C), start it
    streamers    configure ``<node>_<instance>_<port>`` for every port, in the
                 BRM's port order, with values from the memlet through the
                 container's L1 layout (below)
    idle movers  in an accelerator of several groups (D148), configure
                 ``<node>_<streamer>`` for every port of the other groups'
                 units too, every temporal bound 0, so they never start: the
                 streamer starts all its movers with one CSR; the task list
                 holds them in the shell's order, unit by unit. Their
                 temporal strides are a bank word, not 0: a reader whose
                 first loop has stride 0 runs that loop once whatever its
                 bound and hands each beat out bound times (snax_cluster's
                 Reader.scala), so it would start and read
    accelerator  configure ``<node>_<instance>`` with ``n``, the firing count, and its
                 named rates; a linked group's ``<node>_<group>`` with each unit's,
                 by unit
    start        the streamer tasks, then the accelerator task, together
    stores       for each output whose container has an L2 layout and is written
                 for the last time here: configure ``store_<C>`` (L1 -> L2)
                 after the writer's task, start it

and at the end one ``sync`` per non-transient container the graph writes,
on the task that leaves it in its last memory (its store, or its writer
without an L2). ``after`` holds the data dependences only: a reader waits
for the task that last put its container in L1 (a load or a writer), a
writer for the tasks that read or wrote its container since; the waits a
busy component needs are placed later (waits.py). Every configure uses the platform's
``wait_mode``; so does every sync.

**Streamer values from a memlet** (D73). With the firing loops ``v_k``
(begin ``b_k``, ``count_k`` iterations, step ``s_k``) and the container's L1
layout (``base``, byte ``strides``), each subset dimension ``d`` is an index
or a range whose begin is affine in the ``v_k`` (``expr.linear``):

    base              layout.base + sum_d (c_d + sum_k a_dk * b_k) * strides[d]
    temporal loop k   bound count_k, stride sum_d a_dk * s_k * strides[d]
                      (innermost first, as the streamer takes them)
    spatial loop      one per range dimension: bound its length, stride its
                      step * strides[d] (fastest, the last range, first)

The spatial bounds must be the streamer's. For a BRM in the ``affine``
notation the addresses the values visit, beat by beat, must be the ones the
BRM's nest gives through the same layout (streams.py): the memlet and the
accelerator agree on the order of the elements. In the ``stream`` notation
the memlet is the order (D70), so there is nothing to compare. The memlet's loop structure is the one written (one streamer
loop per firing loop), so two temporal maps give two streamer loops even
where the nest has one.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np

from snax_forge import expr
from snax_forge.brm import memlet_ordered
from snax_forge.design.streamers import (
    Loop,
    accelerated,
    accelerator_of,
    accelerators,
    firing_loops,
    instances,
    linked_ports,
    port_loops,
    streamer_name,
)
from snax_forge.design.streamers import groups as unit_groups
from snax_forge.dfg import DfgError, Memlet, named_rates
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
    g, pf, plan = point.graph, point.platform, point.memory
    symbols = g.bound_symbols
    insts = instances(g)
    mode = pf.wait_mode
    beat = pf.l1.wide_bits // 8
    nodes = accelerated(g)
    groups = accelerators(g)
    group_of = {u: k for k, units in unit_groups(g).items() for u in units}
    linked = {k: units for k, units in unit_groups(g).items() if len(units) > 1}
    linked_ports_ = linked_ports(g)

    def unit_node(u: str) -> Any:
        """A linked unit's one node."""
        mine = [m for m in nodes if m.attrs["instance"] == u]
        if len(mine) != 1:
            raise LowerError(f"unit {u}: {len(mine)} nodes; a linked unit runs as one node")
        return mine[0]

    def layout(c: str, mem: str) -> Layout | None:
        return plan.layouts.get(c, {}).get(mem)

    def moved(c: str) -> bool:
        return layout(c, "l1") is not None and layout(c, "l2") is not None

    last_write = {m.data: n.id for n in nodes for m in n.outputs.values()}
    steps: list[Any] = []
    in_l1: dict[str, str] = {}  # container -> task that last put it in L1
    readers: dict[str, list[str]] = {}  # container -> reader tasks since that
    final: dict[str, str] = {}  # container -> task that leaves it in its last memory

    def dma(task: str, direction: str, c: str, after: list[str]) -> None:
        elem, padded = np.dtype(g.containers[c].dtype).itemsize, padded_shape(g, c)
        l1 = dma_side(over(layout(c, "l1"), padded), elem, beat)  # a tail's pads too (D144)
        l2 = {"symbol": c} | {k: v for k, v in dma_side(over(layout(c, "l2"), padded), elem,
                                                        beat).items() if k != "base"}  # fmt: skip
        src, dst = (l2, l1) if direction == "l2_to_l1" else (l1, l2)
        values = {"direction": direction, "src": src, "dst": dst}
        steps.append(Configure(task, "dma", "dma", values, after, mode))
        steps.append(Start([task]))

    def idle(node: str, s: Any) -> Configure:
        """Node ``node``'s task on another group's mover ``s``: every bound 0, so it never
        starts and moves nothing (D148); every temporal stride a word (module doc)."""
        word = pf.l1.width_bits // 8
        values = {"base": 0, "temporal_bounds": [0] * s.temporal_dims,
                  "temporal_strides": [word] * s.temporal_dims,
                  "spatial_strides": [0] * len(s.spatial_bounds)}  # fmt: skip
        return Configure(f"{node}_{s.name}", "streamer", s.name, values, [], mode)

    done: set[str] = set()
    for n in nodes:
        name = n.attrs["instance"]
        group = group_of[name]
        if group in done:  # a linked group's units are one task, at its first node
            continue
        members = linked.get(group)
        if members is None:
            unit_nodes = [n]
        else:
            done.add(group)
            unit_nodes = [unit_node(u) for u in members]
        starts: dict[str, dict[str, Any]] = {}  # each unit's accelerator values
        unit_loops: dict[str, list[Loop]] = {}
        for un in unit_nodes:
            u, what = un.attrs["instance"], f"node {un.id}"
            try:
                loops = firing_loops(g, un)
            except ValueError as e:
                raise LowerError(f"{what}: {e}") from None
            memlets = {**un.inputs, **un.outputs}
            try:  # named rates from the loops a memlet does not use (D102), as REF and bind do
                rates = named_rates(
                    {p.name: (memlets[p.name], p.rate) for p in insts[u].brm.interface.ports},
                    [(lp.var, lp.count) for lp in loops],
                    what,
                )
            except DfgError as e:
                raise LowerError(str(e)) from None
            starts[u] = {"n": prod(lp.count for lp in loops)} | rates
            unit_loops[u] = loops

        def outer(un: Any) -> list[str]:
            """The node's ports with a streamer: every port not on a link (D112)."""
            u = un.attrs["instance"]
            return [
                p.name for p in insts[u].brm.interface.ports if f"{u}.{p.name}" not in linked_ports_
            ]

        for un in unit_nodes:  # loads
            for p in outer(un):
                c = un.inputs[p].data if p in un.inputs else None
                if c is not None and c not in in_l1 and moved(c):
                    task = f"load_{c}"
                    dma(task, "l2_to_l1", c, [])
                    in_l1[c] = task

        own: dict[str, Configure] = {}
        for un in unit_nodes:  # streamers
            u, what = un.attrs["instance"], f"node {un.id}"
            inst, loops = insts[u], unit_loops[u]
            memlets = {**un.inputs, **un.outputs}
            for p in outer(un):
                m, s = memlets[p], point.streamers[streamer_name(u, p)]
                lay = layout(m.data, "l1")
                if lay is None:
                    raise LowerError(f"{what}.{p}: container {m.data} has no L1 layout")
                values, sb = memlet_values(
                    m, port_loops(inst, p, m, loops), lay, symbols, f"{what}.{p}", s.num_slc
                )
                if sb != list(s.spatial_bounds):
                    raise LowerError(
                        f"{what}.{p}: the memlet gives spatial bounds {sb}, streamer {s.name} has "
                        f"{list(s.spatial_bounds)}"
                    )
                if not memlet_ordered(inst.brm):  # affine: the nest must give the same order
                    try:
                        want = streamer_values(inst, p, starts[u], lay, s, pf.l1)
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
                    after = [in_l1[c]] if c in in_l1 else []
                    readers.setdefault(c, []).append(task)
                else:
                    after = [*readers.pop(c, []), *([in_l1[c]] if c in in_l1 else [])]
                own[s.name] = Configure(task, "streamer", s.name, values, after, mode)
        movers = []  # the accelerator's every mover, its other groups' idle (D148)
        for unit in groups[accelerator_of(n)]:
            for q in (q.name for q in insts[unit].brm.interface.ports):
                s = point.streamers.get(streamer_name(unit, q))
                if s is None:  # a port on a link
                    continue
                c = own[s.name] if s.name in own else idle(n.id, s)
                steps.append(c)
                movers.append(c.task_name)
        component = name if members is None else group
        acc_task = f"{n.id}_{component}"
        start = starts[name] if members is None else starts
        steps.append(Configure(acc_task, "accel", component, start, [], mode))
        steps.append(Start([*movers, acc_task]))

        for un in unit_nodes:  # after the group: writes land in L1
            for p in outer(un):
                if p in un.outputs:
                    c = un.outputs[p].data
                    task = own[streamer_name(un.attrs["instance"], p)].task_name
                    in_l1[c] = task
                    final[c] = task
        for un in unit_nodes:  # stores
            for p in outer(un):
                c = un.outputs[p].data if p in un.outputs else None
                if c is not None and last_write[c] == un.id and layout(c, "l2") is not None:
                    store = f"store_{c}"
                    dma(
                        store,
                        "l1_to_l2",
                        c,
                        [own[streamer_name(un.attrs["instance"], p)].task_name],
                    )
                    final[c] = store

    for c, cont in g.containers.items():
        if c in final and not cont.transient:
            steps.append(Sync(final[c], mode))
    return TaskList(point.name, steps)
