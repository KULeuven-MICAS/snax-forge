"""Reference executor (REF1, D20, D79): run a ``.snaxdfg`` in NumPy.

It gives the golden output of a graph at any step of the flow: as imported,
split, or with accelerated nodes, so SNAX-SANDBOX can check every transform
against it (D72), and SNAX-MODEL's output is compared with it (E2E1).

    execute(graph, inputs) -> {container: array}

**Inputs and symbols.** ``inputs`` holds one array per non-transient
container, of its dtype. A symbol the graph binds keeps its value; one that
is null is read off the input shapes (``A`` of shape ``["N"]`` gives ``N =
len(A)``), and the two must agree. Containers start as copies of the
inputs; transients start as zeros. Only integer containers are run (D28).

**Order.** The body runs in order, each node to completion before the next.
How a node runs is registered per kind (``register_executor``, principle
6):

``map``
    no loop in Python: its variable becomes an index array on an axis of
    its own (the nest ``i_t`` over 16, ``i_s`` over 4 is a 16 x 4 grid), and
    its body runs once over the whole grid. Map iterations are independent
    (SDFG's meaning of a map), so this is exact. A range must not depend on
    an enclosing variable.
``tasklet``
    gather every input at its subset's index array, evaluate ``code`` on
    the gathered arrays (``expr.evaluate``), write each output back at its
    index array, cast to the container's dtype (integer overflow wraps as in
    C). All reads come before all writes. Two iterations writing one
    element is an error (a write conflict), unless the output memlet has a
    ``reduce`` (D102): then the elements it writes are set to its identity and
    every iteration's value is folded in with the op's ufunc (``ufunc.at``,
    which wraps as C does), so dot's sum map leaves ``sum(tmp0)`` in
    ``out[0]`` whatever ``out`` held before.
``accelerated``
    one firing at a time, through the BRM's function, the same
    ``fn(k, ins, state, params)`` SNAX-MODEL runs (the instance's
    ``accel_config``). The ``temporal`` maps directly around the node are
    its firing loop, outermost first, as the BRM nest orders them (D70);
    the node's ``code`` must be the BRM's ``function.code`` (D82), and its
    ``replaced`` subtree is history, never run;
    ``k`` counts firings, ``state`` lives for one task, ``n`` is the number
    of firings. Maps further out (a tile, an untagged map) start a new task
    per iteration. Each port's beat is gathered from its memlet (the lanes,
    row-major), and each output beat is written back the same way. A port
    with a named rate (``T``, dot's accumulator) moves one beat every T
    firings, T from ``named_rates`` (D104): an input is read at firings 0, T,
    2T, ..., an output written at T-1, 2T-1, ... where its memlet points in
    the firing that completes it.

**Checks on the way.** Every index is inside its container (NumPy would
wrap a negative one silently); an accelerated node's connectors are its
BRM's ports, each beat has the port's lanes and the port's dtype is the
container's (open item 30); a spatial map around an accelerated node is
an error, since its lanes are in the memlet.

**Named rates** (``named_rates``, D104). A rate that names a start
parameter gets its value from the graph: the product of the counts of the
innermost firing loops the port's memlet does not depend on, the loops its
beat folds over. dot's sum port ``out[0:1]`` depends on no loop, so T is
every firing (N / W). A memlet that skips a loop outside one it uses would
move a beat per firing yet land on one element, and is an error. The
sandbox's ``bind`` and SNAX-LOWER use the same function, so the order check,
the reference and the task list agree on T.
Everything raises ``ExecutionError`` naming the node.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.expr import ExprError, Value

from .graph import Graph, Memlet, Node
from .kinds import DfgError
from .subset import dim_names, parse_dim


class ExecutionError(DfgError):
    """A graph the reference executor cannot run, or a run that went wrong."""


@dataclass
class Loop:
    """One enclosing map: its node, variable and values."""

    node: Node
    var: str
    values: np.ndarray


@dataclass
class Context:
    """What a node's executor sees: the data, the symbols and the enclosing maps."""

    graph: Graph
    symbols: dict[str, int]
    data: dict[str, np.ndarray]
    loops: list[Loop] = field(default_factory=list)

    def grid(self) -> dict[str, Any]:
        """Symbols plus every enclosing variable as a broadcasting index array."""
        env: dict[str, Any] = dict(self.symbols)
        depth = len(self.loops)
        for d, loop in enumerate(self.loops):
            shape = [1] * depth
            shape[d] = len(loop.values)
            env[loop.var] = loop.values.reshape(shape)
        return env

    def shape(self) -> tuple[int, ...]:
        return tuple(len(loop.values) for loop in self.loops)


Executor = Callable[[Node, Context], None]
EXECUTORS: dict[str, Executor] = {}


def register_executor(kind: str, fn: Executor) -> None:
    """How the reference executor runs nodes of ``kind`` (principle 6)."""
    if kind in EXECUTORS:
        raise ValueError(f"executor for {kind!r} already registered")
    EXECUTORS[kind] = fn


# =============================================================================
# Entry point
# =============================================================================


def execute(graph: Graph, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run ``graph`` on ``inputs``; every container's final contents, by name."""
    symbols = bind_symbols(graph, inputs)
    data: dict[str, np.ndarray] = {}
    for name, c in graph.containers.items():
        dtype = np.dtype(c.dtype)
        if dtype.kind not in "iu":
            raise ExecutionError(f"container {name!r}: {c.dtype} is not an integer type (D28)")
        shape = tuple(_int(s, symbols, f"container {name!r}.shape") for s in c.shape)
        if c.transient:
            data[name] = np.zeros(shape, dtype)
            continue
        arr = np.asarray(inputs[name])
        if arr.shape != shape or arr.dtype != dtype:
            raise ExecutionError(
                f"input {name!r}: {arr.dtype}{list(arr.shape)} given, "
                f"the graph has {c.dtype}{list(shape)}"
            )
        data[name] = arr.copy()
    ctx = Context(graph, symbols, data)
    run_body(graph.body, ctx)
    return data


def bind_symbols(graph: Graph, inputs: Mapping[str, np.ndarray]) -> dict[str, int]:
    """The graph's symbol values: bound in the graph, or read off the input shapes."""
    missing = [k for k, c in graph.containers.items() if not c.transient and k not in inputs]
    if missing:
        raise ExecutionError(f"{graph.name}: no input for container {missing[0]!r}")
    unknown = sorted(set(inputs) - set(graph.containers))
    if unknown:
        raise ExecutionError(f"{graph.name}: input {unknown[0]!r} is not a container")
    seen: dict[str, int] = {}
    for name, c in graph.containers.items():
        if c.transient:
            continue
        arr_shape = np.shape(inputs[name])
        if len(arr_shape) != len(c.shape):
            raise ExecutionError(
                f"input {name!r}: {len(arr_shape)} dimensions, the graph has {len(c.shape)}"
            )
        for s, n in zip(c.shape, arr_shape, strict=True):
            if isinstance(s, str) and expr.is_name(s) and seen.setdefault(s, n) != n:
                raise ExecutionError(f"symbol {s!r}: inputs give both {seen[s]} and {n}")
    out: dict[str, int] = {}
    for s, v in graph.symbols.items():
        if v is not None:
            if s in seen and seen[s] != v:
                raise ExecutionError(
                    f"symbol {s!r}: the graph binds {v}, the inputs give {seen[s]}"
                )
            out[s] = v
        elif s in seen:
            out[s] = seen[s]
        else:
            raise ExecutionError(f"symbol {s!r}: not bound by the graph nor by an input shape")
    return out


def run_body(body: list[Node], ctx: Context) -> None:
    for node in body:
        if node.kind not in EXECUTORS:
            raise ExecutionError(f"node {node.id!r}: no executor for kind {node.kind!r}")
        EXECUTORS[node.kind](node, ctx)


# =============================================================================
# Helpers
# =============================================================================


def _int(v: Value, env: Mapping[str, Any], what: str) -> int:
    try:
        x = expr.evaluate(v, env)
    except ExprError as e:
        raise ExecutionError(f"{what}: {e}") from None
    if not isinstance(x, (int, np.integer)):
        raise ExecutionError(f"{what}: {v!r} depends on an enclosing variable")
    return int(x)


def _check_bounds(idx: Any, size: int, m: Memlet, dim: Value, what: str) -> None:
    if np.size(idx) and (np.min(idx) < 0 or np.max(idx) >= size):
        raise ExecutionError(f"{what}: {m.data}[{dim}] leaves 0:{size}")


def _grid_indices(m: Memlet, env: Mapping[str, Any], ctx: Context, what: str) -> tuple[Any, ...]:
    """A tasklet's subset over the whole grid: one index array per dimension."""
    out = []
    for dim, size in zip(m.subset, ctx.data[m.data].shape, strict=True):
        try:
            idx = expr.evaluate(dim, env)
        except ExprError as e:
            raise ExecutionError(f"{what}: {e}") from None
        _check_bounds(idx, size, m, dim, what)
        out.append(idx)
    return tuple(out)


def _beat_indices(m: Memlet, env: Mapping[str, int], ctx: Context, what: str) -> tuple[Any, ...]:
    """An accelerated node's subset for one firing: an open mesh over its ranges."""
    axes = []
    for dim, size in zip(m.subset, ctx.data[m.data].shape, strict=True):
        parts = parse_dim(dim, what)
        if len(parts) == 1:
            ax = np.array([_int(parts[0], env, what)])
        else:
            ax = np.arange(_int(parts[0], env, what), _int(parts[1], env, what), parts[2])
        _check_bounds(ax, size, m, dim, what)
        axes.append(ax)
    return np.ix_(*axes)


# =============================================================================
# map
# =============================================================================


def _run_map(node: Node, ctx: Context) -> None:
    what = f"node {node.id!r}"
    begin, end, step = parse_dim(node.attrs["range"], f"{what}.range")
    env = dict(ctx.symbols)
    lo, hi = _int(begin, env, f"{what}.range"), _int(end, env, f"{what}.range")
    ctx.loops.append(Loop(node, node.attrs["var"], np.arange(lo, hi, step)))
    try:
        run_body(node.body or [], ctx)
    finally:
        ctx.loops.pop()


# =============================================================================
# tasklet
# =============================================================================


def _run_tasklet(node: Node, ctx: Context) -> None:
    what = f"node {node.id!r}"
    env = ctx.grid()
    grid = ctx.shape()
    values = {}
    for c, m in node.inputs.items():
        values[c] = ctx.data[m.data][_grid_indices(m, env, ctx, f"{what}.inputs.{c}")]
    results = {}
    for line in node.attrs["code"].split("\n"):
        target, rhs = (s.strip() for s in line.split("=", 1))
        try:
            results[target] = expr.evaluate(rhs, values)
        except ExprError as e:
            raise ExecutionError(f"{what}.attrs.code: {e}") from None
    for c, m in node.outputs.items():
        arr = ctx.data[m.data]
        idx = _grid_indices(m, env, ctx, f"{what}.outputs.{c}")
        full = tuple(np.broadcast_to(i, grid) for i in idx)
        value = np.broadcast_to(results[c], grid).astype(arr.dtype)
        if m.reduce is not None:  # a reduction (D102): start from the identity, fold every write
            arr[full] = np.array(m.reduce.identity).astype(arr.dtype)
            m.reduce.ufunc.at(arr, full, value)
            continue
        flat = np.ravel_multi_index(full, arr.shape)
        if np.unique(flat).size != np.size(flat):
            raise ExecutionError(
                f"{what}.outputs.{c}: several iterations write one element of {m.data!r} "
                "(a write conflict; a reduction says so with a reduce on the memlet, D102)"
            )
        arr[full] = value


# =============================================================================
# accelerated
# =============================================================================


def _accel(node: Node, what: str) -> Any:
    from snax_forge.brm import BrmError, load_brm

    a = node.attrs
    try:
        instance = load_brm(a["brm"]).resolve(a["implementation"], a["params"])
        return instance, instance.accel_config()
    except BrmError as e:
        raise ExecutionError(f"{what}: {e}") from None


def named_rates(
    ports: Mapping[str, tuple[Memlet, int | str]], loops: list[tuple[str, int]], what: str
) -> dict[str, int]:
    """Each named rate's value over the firing ``loops`` (var, count), outermost first (D104).

    ``ports`` maps a port to its memlet and rate. For a named rate, the
    innermost loops its memlet does not use are the ones one beat folds over;
    the rate is the product of their counts. Two ports naming one rate must
    agree. Raises ExecutionError naming the port.
    """
    out: dict[str, int] = {}
    for port, (m, rate) in ports.items():
        if not isinstance(rate, str):
            continue
        used: set[str] = set()
        for d in m.subset:
            used |= dim_names(d)
        k = len(loops)
        while k and loops[k - 1][0] not in used:
            k -= 1
        skipped = [var for var, _ in loops[:k] if var not in used]
        if skipped:
            raise ExecutionError(
                f"{what}: port {port!r} ({m.data}{m.subset}) does not use loop {skipped[0]!r}, "
                "which is outside a loop it uses: a named rate folds innermost loops only"
            )
        value = int(np.prod([count for _, count in loops[k:]], dtype=int))
        if out.setdefault(rate, value) != value:
            raise ExecutionError(f"{what}: rate {rate!r} is {out[rate]} and {value} on two ports")
    return out


def _run_accelerated(node: Node, ctx: Context) -> None:
    what = f"node {node.id!r}"
    if node.body:
        raise ExecutionError(f"{what}: a nested accelerated block is not run yet")
    instance, cfg = _accel(node, what)
    brm_code = instance.brm.function.code
    if brm_code is not None and node.attrs["code"] != brm_code:
        raise ExecutionError(
            f"{what}.attrs.code: {node.attrs['code']!r}, but {node.attrs['brm']!r} "
            f"computes {brm_code!r}"
        )
    for side, conns, ports in (
        ("inputs", node.inputs, cfg.inputs),
        ("outputs", node.outputs, cfg.outputs),
    ):
        names = [p.name for p in ports]
        if sorted(conns) != sorted(names):
            raise ExecutionError(
                f"{what}.{side}: connectors {list(conns)}, the BRM's ports {names}"
            )
    for p in cfg.ports:
        m = (node.inputs | node.outputs)[p.name]
        want = instance.brm.port(p.name).dtype
        if ctx.graph.containers[m.data].dtype != want:
            raise ExecutionError(
                f"{what}: port {p.name!r} is {want}, container {m.data!r} is "
                f"{ctx.graph.containers[m.data].dtype}"
            )
    if any(loop.node.attrs.get("loop.kind") == "spatial" for loop in ctx.loops):
        raise ExecutionError(f"{what}: inside a spatial map; its lanes belong in the memlet")
    task = 0
    for loop in reversed(ctx.loops):
        if loop.node.attrs.get("loop.kind") != "temporal":
            break
        task += 1
    outer, inner = ctx.loops[: len(ctx.loops) - task], ctx.loops[len(ctx.loops) - task :]
    n = int(np.prod([len(loop.values) for loop in inner], dtype=int))
    memlets = node.inputs | node.outputs
    params = {"n": n} | named_rates(
        {p.name: (memlets[p.name], p.rate) for p in cfg.ports},
        [(loop.var, len(loop.values)) for loop in inner],
        what,
    )
    rate = {p.name: p.rate if isinstance(p.rate, int) else params[p.rate] for p in cfg.ports}
    for name, r in rate.items():
        if n % r:
            raise ExecutionError(f"{what}: n = {n} is not a multiple of port {name!r}'s rate {r}")
    written = {m.data: np.zeros(ctx.data[m.data].shape, bool) for m in node.outputs.values()}
    for point in itertools.product(*(loop.values for loop in outer)):
        state: dict[str, Any] = {}
        env0 = dict(ctx.symbols) | {lp.var: int(v) for lp, v in zip(outer, point, strict=True)}
        for k, beat in enumerate(itertools.product(*(loop.values for loop in inner))):
            env = env0 | {lp.var: int(v) for lp, v in zip(inner, beat, strict=True)}
            ins = {}
            for p in cfg.inputs:
                if k % rate[p.name]:
                    continue
                m = node.inputs[p.name]
                idx = _beat_indices(m, env, ctx, f"{what}.inputs.{p.name}")
                ins[p.name] = _lanes(ctx.data[m.data][idx], p, what)
            outs = cfg.fn(k, ins, state, params)
            due = [p for p in cfg.outputs if (k + 1) % rate[p.name] == 0]
            if sorted(outs) != sorted(p.name for p in due):
                raise ExecutionError(f"{what}: firing {k} gave {sorted(outs)}")
            for p in due:
                m = node.outputs[p.name]
                idx = _beat_indices(m, env, ctx, f"{what}.outputs.{p.name}")
                arr = ctx.data[m.data]
                region = arr[idx]
                beat_out = _lanes(np.asarray(outs[p.name]), p, what)
                if written[m.data][idx].any():
                    raise ExecutionError(
                        f"{what}.outputs.{p.name}: firing {k} writes an element twice"
                    )
                written[m.data][idx] = True
                arr[idx] = beat_out.reshape(region.shape).astype(arr.dtype)


def _lanes(x: np.ndarray, port: Any, what: str) -> np.ndarray:
    """A beat as the model sees it: one axis of ``lanes`` elements."""
    flat = np.asarray(x).reshape(-1)
    if flat.size != port.lanes:
        raise ExecutionError(f"{what}: port {port.name!r} has {port.lanes} lanes, got {flat.size}")
    return flat


register_executor("map", _run_map)
register_executor("tasklet", _run_tasklet)
register_executor("accelerated", _run_accelerated)
