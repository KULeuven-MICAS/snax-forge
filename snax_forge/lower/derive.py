"""Design point -> task list (LOW1a, D45, D73, D89).

The task list (tasks.py, CONTRACTS.md section 9) is derived from a design
point and decides nothing. The accelerated nodes run in the graph's
execution order, one group of tasks each (a single tile: every map around a
node is one of its temporal firing loops, ``firing_loops``). For each group:

    loads        for each input in port order whose container has an L2 and an
                 L1 layout and is not in L1 yet: configure ``load_<C>`` (DMA,
                 L2 -> L1, the whole layout as contiguous beats), start it
    streamers    configure ``<node>_<instance>_<port>`` for every port, in the
                 BRM's port order, with values from the memlet through the
                 container's L1 layout (below)
    accelerator  configure ``<node>_<instance>`` with ``n``, the firing count
    start        the streamer tasks, then the accelerator task, together
    stores       for each output whose container has an L2 layout and is written
                 for the last time here: configure ``store_<C>`` (L1 -> L2)
                 after the writer's task, start it

and at the end one ``sync`` per non-transient container the graph writes,
on the task that leaves it in its last memory (its store, or its writer
without an L2). ``after`` holds the data dependences only: a reader waits
for the task that last put its container in L1 (a load or a writer), a
writer for the tasks that read or wrote its container since; ``lower_program``
adds the waits a busy component needs. Every configure uses the platform's
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

The spatial bounds must be the streamer's, and the addresses the values
visit, beat by beat, must be the ones the BRM's nest gives through the same
layout (streams.py): the memlet and the accelerator agree on the order of
the elements. The memlet's loop structure is the one written (one streamer
loop per firing loop), so two temporal maps give two streamer loops even
where the nest has one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from snax_forge import expr
from snax_forge.design.streamers import Loop, firing_loops, instances
from snax_forge.dfg import DfgError, Memlet, dim_names, named_rates
from snax_forge.dfg.subset import parse_dim

from .cluster import cluster_file
from .layout import Layout, dma_side
from .streams import StreamError, streamer_values
from .tasks import Configure, Start, Sync, TaskList

if TYPE_CHECKING:
    from snax_forge.design.point import DesignPoint


class LowerError(ValueError):
    """A design point whose task list cannot be derived; the message names the node."""


def memlet_values(
    m: Memlet, loops: list[Loop], layout: Layout, symbols: dict[str, int], what: str
) -> tuple[dict[str, Any], list[int]]:
    """A streamer's values for memlet ``m`` over ``loops``; and its spatial bounds, fastest first."""
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
        for k, lp in enumerate(loops):
            tstrides[k] += co[lp.var] * lp.step * stride
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
        "spatial_strides": [s for _, s in fastest],
    }
    return values, [b for b, _ in fastest]


def addresses(values: dict[str, Any], spatial_bounds: list[int]) -> np.ndarray:
    """The byte addresses a streamer's values visit, beat by beat: (beats, lanes)."""
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
    symbols = {k: v for k, v in g.symbols.items() if v is not None}
    insts = instances(g)
    cluster = cluster_file(point)
    mode = pf.wait_mode
    word = pf.l1.width_bits // 8
    beat = pf.l1.wide_bits // 8
    nodes = [n for n, _ in g.walk() if n.kind == "accelerated"]

    def layout(c: str, mem: str) -> Layout | None:
        return plan.layouts.get(c, {}).get(mem)

    def moved(c: str) -> bool:
        return layout(c, "l1") is not None and layout(c, "l2") is not None

    last_write = {m.data: n.id for n in nodes for m in n.outputs.values()}
    steps: list[Any] = []
    in_l1: dict[str, str] = {}  # container -> task that last put it in L1
    readers: dict[str, list[str]] = {}  # container -> reader tasks since that
    final: dict[str, str] = {}  # container -> task that leaves it in its last memory

    def dma(task: str, direction: str, src: Layout, dst: Layout, after: list[str]) -> None:
        values = {
            "direction": direction,
            "src": dma_side(src, word, beat),
            "dst": dma_side(dst, word, beat),
        }
        steps.append(Configure(task, "dma", "dma", values, after, mode))
        steps.append(Start([task]))

    for n in nodes:
        name = n.attrs["instance"]
        inst = insts[name]
        what = f"node {n.id}"
        try:
            loops = firing_loops(g, n)
        except ValueError as e:
            raise LowerError(f"{what}: {e}") from None
        n_fire = 1
        for lp in loops:
            n_fire *= lp.count
        ports = [p.name for p in inst.brm.interface.ports]
        memlets = {**n.inputs, **n.outputs}
        try:  # named rates from the loops a memlet does not use (D104), as REF and bind do
            rates = named_rates(
                {p: (memlets[p], inst.brm.port(p).rate) for p in ports},
                [(lp.var, lp.count) for lp in loops],
                what,
            )
        except DfgError as e:
            raise LowerError(str(e)) from None
        start = {"n": n_fire} | rates  # the accelerator task's values

        for p in ports:  # loads
            c = memlets[p].data
            if p in n.inputs and c not in in_l1 and moved(c):
                task = f"load_{c}"
                dma(task, "l2_to_l1", layout(c, "l2"), layout(c, "l1"), [])
                in_l1[c] = task

        group = []
        for p in ports:  # streamers
            m, s = memlets[p], point.streamers[f"{name}_{p}"]
            lay = layout(m.data, "l1")
            if lay is None:
                raise LowerError(f"{what}.{p}: container {m.data} has no L1 layout")
            # a port with a named rate moves one beat per T firings: its streamer
            # runs over the firing loops its memlet uses, not the ones it folds
            used = [lp for lp in loops if any(lp.var in dim_names(d) for d in m.subset)]
            port_loops = loops if inst.brm.port(p).rate == 1 else used
            values, sb = memlet_values(m, port_loops, lay, symbols, f"{what}.{p}")
            if sb != list(s.spatial_bounds):
                raise LowerError(
                    f"{what}.{p}: the memlet gives spatial bounds {sb}, streamer {s.name} has "
                    f"{list(s.spatial_bounds)}"
                )
            try:
                want = streamer_values(inst, p, start, lay, cluster, s.name)
            except StreamError as e:
                raise LowerError(f"{what}.{p}: {e}") from None
            if not np.array_equal(addresses(values, sb), addresses(want, sb)):
                raise LowerError(
                    f"{what}.{p}: the memlet gives {values}, the BRM's nest gives {want}: "
                    "they take the elements in different orders"
                )
            task = f"{n.id}_{s.name}"
            c = m.data
            if p in n.inputs:
                after = [in_l1[c]] if c in in_l1 else []
                readers.setdefault(c, []).append(task)
            else:
                after = [*readers.pop(c, []), *([in_l1[c]] if c in in_l1 else [])]
            steps.append(Configure(task, "streamer", s.name, values, after, mode))
            group.append(task)
        acc_task = f"{n.id}_{name}"
        steps.append(Configure(acc_task, "accel", name, start, [], mode))
        steps.append(Start([*group, acc_task]))

        for p, task in zip(ports, group, strict=True):  # after the group: writes land in L1
            c = memlets[p].data
            if p in n.outputs:
                in_l1[c] = task
                final[c] = task
        for p, task in zip(ports, group, strict=True):  # stores
            c = memlets[p].data
            if p in n.outputs and last_write[c] == n.id and layout(c, "l2") is not None:
                store = f"store_{c}"
                dma(store, "l1_to_l2", layout(c, "l1"), layout(c, "l2"), [task])
                final[c] = store

    for c, cont in g.containers.items():
        if c in final and not cont.transient:
            steps.append(Sync(final[c], mode))
    return TaskList(point.name, steps)
