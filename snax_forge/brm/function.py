"""A BRM's ``function``, run firing by firing: the accelerator kinds.

``function_of(instance, dtypes)`` builds the kind a BRM's function names (``register_kind``)
with its params resolved and the node's port dtypes, as an ``AccelFunction``: its ports
(``AccelPort``: name, direction, lanes, rate) and ``fn(k, ins, state, params) -> outs``, firing
``k``'s outputs from the inputs due at it, ``state`` living for one task. The reference executor
runs an accelerated node through it; timing is the implementation's, not the kind's. Kinds:
``simd`` (every rate 1; the inputs folded lane by lane at their dtype, or ``op(a, k)``, or
``axpby``, ``axpbypcz``, ``requant`` or ``threshold``), ``reduce`` (``a`` at rate 1, cast to the output's dtype and folded over T beats,
the lanes too when ``lanes_out`` is 1; at ``seed`` 1 a ``seed`` input at rate T), ``mac`` (``a *
b`` folded per lane over T beats; at ``seed`` 1 a ``seed`` input of its lanes at rate T) and ``lane_accumulate`` (``a`` folded per lane over T beats),
each with an op registered by name (``register_op``). A slot read from a scalar is unset in the
instance; the executor resolves it from the node's slot connector. ``function_of`` raises
``BrmError`` for an unregistered kind, params it rejects, or ports that are not the BRM's.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce as _fold
from typing import TYPE_CHECKING, Any

import numpy as np

from snax_forge import expr

from .brm import BrmError

if TYPE_CHECKING:
    from .instance import Instance

# fn(k, ins, state, params) -> outs, see the module doc.
AccelFn = Callable[
    [int, Mapping[str, np.ndarray], dict[str, Any], Mapping[str, int]],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class AccelPort:
    """One data port: ``rate``, an int or the name of a runtime param, is firings per beat; an
    output of a bank of ``rows`` gives ``rows`` beats per ``rate`` firings, one on each
    of the last ``rows``, and an input of one (a seeded bank's seed) takes them on the
    first ``rows``."""

    name: str
    direction: str  # "in" or "out"
    lanes: int = 1
    rate: int | str = 1
    rows: int = 1

    def __post_init__(self) -> None:
        if self.direction not in ("in", "out"):
            raise ValueError(f"port {self.name}: direction must be 'in' or 'out'")
        if self.lanes < 1:
            raise ValueError(f"port {self.name}: lanes must be >= 1")
        if isinstance(self.rate, int) and self.rate < 1:
            raise ValueError(f"port {self.name}: rate must be >= 1")


@dataclass(frozen=True)
class AccelFunction:
    """A kind built: its ports and the function of one firing."""

    ports: tuple[AccelPort, ...]
    fn: AccelFn
    kind: str

    def __post_init__(self) -> None:
        names = [p.name for p in self.ports]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate port names: {names}")
        if not self.inputs or not self.outputs:
            raise ValueError("need at least one input and one output port")

    @property
    def inputs(self) -> tuple[AccelPort, ...]:
        return tuple(p for p in self.ports if p.direction == "in")

    @property
    def outputs(self) -> tuple[AccelPort, ...]:
        return tuple(p for p in self.ports if p.direction == "out")

    def port(self, name: str) -> AccelPort:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"no port {name!r}")

    def rates(self, params: Mapping[str, int]) -> dict[str, int]:
        """Each port's rate in a task with runtime params ``params`` (``n``, named rates).

        Raises ValueError for a missing ``n`` or rate, a rate below 1, or an
        ``n`` that is not a multiple of every rate.
        """
        n = params.get("n")
        if n is None or n < 0:
            raise ValueError("need params['n'] >= 0 (firings)")
        rates = {}
        for p in self.ports:
            r = p.rate if isinstance(p.rate, int) else params.get(p.rate)
            if r is None:
                raise ValueError(f"port {p.name} needs parameter {p.rate!r}")
            if r < 1:
                raise ValueError(f"port {p.name} rate {r} must be >= 1")
            if n % r:
                raise ValueError(f"n = {n} is not a multiple of {p.name}'s rate {r}")
            rates[p.name] = r
        return rates

    def due(self, k: int, rates: Mapping[str, int], direction: str) -> list[str]:
        """The ports of ``direction`` that move a beat at firing ``k`` (CONTRACTS, "Block runtime
        model")."""
        if direction == "in":
            return [p.name for p in self.inputs if k % rates[p.name] < p.rows]
        return [p.name for p in self.outputs if k % rates[p.name] >= rates[p.name] - p.rows]


# =============================================================================
# Kinds
# =============================================================================


AXPBY = "axpby"  # simd's k1 * a + k2 * b
AXPBYPCZ = "axpbypcz"  # simd's (k1 * a + k2 * b + k3 * c) // d
REQUANT = "requant"  # simd's min(max(a // d, lo), hi)
THRESHOLD = "threshold"  # simd's hi * min(max(a - k + 1, 0), 1)
# simd's ops by name that are no registered op: (inputs, the slots they take)
NAMED = {AXPBY: (2, {"k1", "k2"}), AXPBYPCZ: (3, {"k1", "k2", "k3", "d"}),
         REQUANT: (1, {"d", "lo", "hi"}), THRESHOLD: (1, {"k", "hi"})}  # fmt: skip


def simd(
    lanes: int = 1,
    n_inputs: int = 2,
    op: Callable[[Any, Any], Any] | str = np.add,
    k: int | None = None,
    k1: int | None = None,
    k2: int | None = None,
    k3: int | None = None,
    d: int | None = None,
    lo: int | None = None,
    hi: int | None = None,
    dtypes: Mapping[str, str] | None = None,
    op_type: str | None = None,
) -> AccelFunction:
    """N inputs -> 1 output, one beat each per firing, ``op`` folding them left to right at
    the inputs' dtype, or at ``op_type`` when the kernel's casts set it, each input extended
    to it first (``dtypes`` is not read: the result is cast when written). One input takes
    the constant ``k`` as its second operand, ``op(a, k)``, k cast to a's dtype; two or more
    take none. ``op`` ``"axpby"`` (two inputs): ``k1 * a + k2 * b``, or ``k1 * a + b`` with
    ``k2`` unset, at the inputs' result type; ``"axpbypcz"`` (three) ``(k1 * a + k2 * b + k3 *
    c) // d``, an unset k one; ``"requant"`` (one) ``min(max(a // d, lo), hi)``;
    ``"threshold"`` (one) ``hi * min(max(a - k + 1, 0), 1)``, hi where a >= k (no wrap at
    a's type). d is a power of two: the hardware shifts. A slot read from a scalar is unset
    here (module doc): the firing needs it."""

    def at(x: Any) -> Any:
        return np.asarray(x) if op_type is None else np.asarray(x).astype(op_type)

    names = [chr(ord("a") + i) for i in range(n_inputs)]
    axpby = op == AXPBY
    named = op if isinstance(op, str) else None
    given = {n for n, v in (("k1", k1), ("k2", k2), ("k3", k3), ("d", d), ("lo", lo),
                            ("hi", hi)) if v is not None}  # fmt: skip
    if named is not None and k is not None:
        given.add("k")
    if named is not None and named not in NAMED:
        raise ValueError(f"unknown op {named!r}")
    if named is not None and n_inputs != NAMED[named][0]:
        raise ValueError(f"{named} takes {NAMED[named][0]} inputs, not {n_inputs}")
    if n_inputs != 1 and k is not None:
        raise ValueError(f"k = {k} is one input's constant; {n_inputs} inputs take none")
    if named is None and given:
        raise ValueError(f"{', '.join(sorted(given))}: the named ops' ({', '.join(NAMED)}), not "
                         "a registered op's")  # fmt: skip
    if named is not None and given - NAMED[named][1]:
        raise ValueError(f"{', '.join(sorted(given - NAMED[named][1]))}: not {named}'s")
    if d is not None and (d <= 0 or d & (d - 1)):
        raise ValueError(f"d = {d} is not a power of two: the unit divides by shifting")

    def slot(name: str, v: int | None, t: Any) -> Any:
        if v is None:
            raise ValueError(f"{name} has no value: a slot read from a scalar needs it")
        return np.asarray(v).astype(t)

    def fn(i, ins, state, params):
        if named == AXPBYPCZ:
            xs = [at(ins[n]) for n in names]
            t = np.result_type(*(x.dtype for x in xs))
            terms = [x.astype(t) if kv is None else np.asarray(kv).astype(t) * x.astype(t)
                     for x, kv in zip(xs, (k1, k2, k3), strict=True)]  # fmt: skip
            return {"out": (terms[0] + terms[1] + terms[2]) // slot("d", d, t)}
        if named == REQUANT:
            a = at(ins["a"])
            q = a // slot("d", d, a.dtype)
            return {
                "out": np.minimum(np.maximum(q, slot("lo", lo, a.dtype)), slot("hi", hi, a.dtype))
            }
        if named == THRESHOLD:
            a = at(ins["a"])
            one = np.asarray(1).astype(a.dtype)
            step = np.minimum(np.maximum(a - slot("k", k, a.dtype) + one, 0), one)
            return {"out": slot("hi", hi, a.dtype) * step}
        if axpby:
            if k1 is None:
                raise ValueError("k1 has no value: a slot read from a scalar needs it")
            a, b = at(ins["a"]), at(ins["b"])
            t = np.result_type(a.dtype, b.dtype)
            a, b = a.astype(t), b.astype(t)
            scaled = b if k2 is None else np.asarray(k2).astype(t) * b
            return {"out": np.asarray(k1).astype(t) * a + scaled}
        if n_inputs == 1:
            if k is None:
                raise ValueError("one input takes the constant k, op(a, k); k has no value")
            a = at(ins["a"])
            return {"out": op(a, np.asarray(k).astype(a.dtype))}
        return {"out": _fold(op, (at(ins[n]) for n in names))}

    ports = [AccelPort(n, "in", lanes) for n in names] + [AccelPort("out", "out", lanes)]
    return AccelFunction(tuple(ports), fn, "simd")


def reduce_kind(
    lanes: int = 1,
    lanes_out: int | None = None,
    op: Callable[[Any, Any], Any] = np.add,
    seed: int = 0,
    dtypes: Mapping[str, str] | None = None,
    op_type: str | None = None,
) -> AccelFunction:
    """T input beats -> 1 output beat, T the runtime param ``"T"``.

    ``lanes_out = lanes`` (default): each lane is folded over T beats.
    ``lanes_out = 1``: the lanes are folded too.
    Each input lane is cast to the output port's dtype (``dtypes["out"]``) before it is
    folded, so the tree and the accumulator run at that dtype; without ``dtypes``, at
    the input's. With ``seed`` 1 an input ``seed`` of ``lanes_out`` lanes at rate T: a job's
    first firing folds into it instead of its running sum (``acc_in = first ? seed : s``,
    ``s <- acc_in op tree(a)``), cast to the output's dtype too. ``op_type`` (a cast of the
    input in the kernel) changes nothing: the lanes are cast to the output's dtype anyway.
    """
    out_dtype = None if dtypes is None else dtypes.get("out")
    lanes_out = lanes if lanes_out is None else lanes_out
    if lanes_out not in (lanes, 1):
        raise ValueError("lanes_out must equal lanes or be 1")
    if seed not in (0, 1):
        raise ValueError("seed must be 0 or 1")

    def fn(k, ins, state, params):
        x = ins["a"] if out_dtype is None else np.asarray(ins["a"]).astype(out_dtype)
        if lanes_out == 1:
            x = _fold(op, (x[j : j + 1] for j in range(x.shape[0])))
        if "acc" in state:
            state["acc"] = op(state["acc"], x)
        elif seed:  # a job's first firing: the seed in place of the running sum
            s = ins["seed"] if out_dtype is None else np.asarray(ins["seed"]).astype(out_dtype)
            state["acc"] = op(s, x)
        else:
            state["acc"] = x.copy()
        if (k + 1) % params["T"] == 0:
            return {"out": state.pop("acc")}
        return {}

    ports = (AccelPort("a", "in", lanes, 1),
             *((AccelPort("seed", "in", lanes_out, "T"),) if seed else ()),
             AccelPort("out", "out", lanes_out, "T"))  # fmt: skip
    return AccelFunction(ports, fn, "reduce")


def mac_kind(
    lanes: int = 1,
    op: Callable[[Any, Any], Any] = np.add,
    seed: int = 0,
    dtypes: Mapping[str, str] | None = None,
    op_type: str | None = None,
    rows: int = 1,
) -> AccelFunction:
    """T beats of ``a`` and ``b`` -> 1 beat of ``out``, each lane on its own: ``a * b`` at the
    inputs' result type, or at ``op_type`` when the kernel's casts set it, cast to the output's
    dtype (``dtypes["out"]``), folded with ``op`` over the T beats; without ``dtypes``, at the
    product's. A ``b`` of one lane is a broadcast: every lane multiplies by it. With
    ``seed`` 1 an input ``seed`` of ``lanes`` lanes at rate T: a job's first firing
    folds into it instead of starting from the product, cast to the output's dtype. With
    ``rows`` R > 1 a bank: firing k adds into row k mod R, each row starting on a job's
    first R firings and given out on its last R, so a job of T firings gives R beats; seeded,
    each row starts from its own seed beat, read on those first R firings."""
    out_dtype = None if dtypes is None else dtypes.get("out")
    if seed not in (0, 1):
        raise ValueError("seed must be 0 or 1")
    if rows < 1:
        raise ValueError("rows must be >= 1")

    def fn(k, ins, state, params):
        a, b = np.asarray(ins["a"]), np.asarray(ins["b"])
        if op_type is not None:
            a, b = a.astype(op_type), b.astype(op_type)
        x = np.multiply(a, b)
        x = x if out_dtype is None else np.asarray(x).astype(out_dtype)
        row, t = k % rows, params["T"]
        if t % rows:
            raise ValueError(f"T = {t} is not a multiple of the bank's {rows} rows")
        if row in state:
            state[row] = op(state[row], x)
        elif seed:  # a job's first firing: the seed in place of a sum of nothing
            s = ins["seed"] if out_dtype is None else np.asarray(ins["seed"]).astype(out_dtype)
            state[row] = op(s, x)
        else:
            state[row] = x.copy()
        if k % t >= t - rows:  # a row is done when the job's last pass adds into it
            return {"out": state.pop(row)}
        return {}

    ports = (AccelPort("a", "in", lanes, 1), AccelPort("b", "in", lanes, 1),
             *((AccelPort("seed", "in", lanes, "T", rows),) if seed else ()),
             AccelPort("out", "out", lanes, "T", rows))  # fmt: skip
    return AccelFunction(ports, fn, "mac")


def lane_accumulate_kind(
    lanes: int = 1,
    op: Callable[[Any, Any], Any] = np.add,
    dtypes: Mapping[str, str] | None = None,
    op_type: str | None = None,
) -> AccelFunction:
    """T beats of ``a`` -> 1 beat of ``out``, each lane its own: ``a`` at its dtype cast to
    the output's (``dtypes["out"]``), folded with ``op`` over the T beats; without
    ``dtypes``, at ``a``'s; ``op_type`` changes nothing, as ``reduce``'s. The lanes are never
    folded together (``reduce``'s ``lanes_out`` 1 does that)."""
    out_dtype = None if dtypes is None else dtypes.get("out")

    def fn(k, ins, state, params):
        x = np.asarray(ins["a"])
        x = x if out_dtype is None else x.astype(out_dtype)
        state["acc"] = x.copy() if "acc" not in state else op(state["acc"], x)
        if (k + 1) % params["T"] == 0:
            return {"out": state.pop("acc")}
        return {}

    ports = (AccelPort("a", "in", lanes, 1), AccelPort("out", "out", lanes, "T"))
    return AccelFunction(ports, fn, "lane_accumulate")


# =============================================================================
# Registries: ops and kinds
# =============================================================================

OPS: dict[str, Callable[[Any, Any], Any]] = {}
KindFactory = Callable[..., AccelFunction]
KINDS: dict[str, KindFactory] = {}


def register_op(name: str, fn: Callable[[Any, Any], Any]) -> None:
    """Make ``fn`` usable as ``"op": name`` in a function's params."""
    if name in OPS:
        raise ValueError(f"op {name!r} already registered")
    OPS[name] = fn


def register_kind(kind: str, factory: KindFactory) -> None:
    """Make ``"accel": kind`` buildable; ``factory(**params)`` returns an AccelFunction."""
    if kind in KINDS:
        raise ValueError(f"accelerator kind {kind!r} already registered")
    KINDS[kind] = factory


for _name, _fn in (
    ("add", np.add),
    ("sub", np.subtract),  # folds left to right: a - b - c
    ("mul", np.multiply),
    ("min", np.minimum),
    ("max", np.maximum),
    ("and", np.bitwise_and),
    ("or", np.bitwise_or),
    ("xor", np.bitwise_xor),
):
    register_op(_name, _fn)


def _with_op(kind: Callable[..., AccelFunction]) -> KindFactory:
    """The kind with ``op`` given by name instead of as a function (default: add); ``dtypes``
    passes through."""

    def factory(**params: Any) -> AccelFunction:
        name = params.get("op", "add")
        if name in NAMED and kind is simd:  # simd's ops with slots of their own, by name
            return kind(**params)
        if name not in OPS:
            raise ValueError(f"unknown op {name!r} (registered: {sorted(OPS)})")
        return kind(**{**params, "op": OPS[name]})

    return factory


register_kind("simd", _with_op(simd))
register_kind("reduce", _with_op(reduce_kind))
register_kind("mac", _with_op(mac_kind))
register_kind("lane_accumulate", _with_op(lane_accumulate_kind))


def function_of(inst: Instance, dtypes: Mapping[str, str] | None = None,
                op_type: str | None = None) -> AccelFunction:  # fmt: skip
    """The instance's function, its params resolved and the node's port ``dtypes`` given, at
    ``op_type`` when the kernel's casts set it (the node's ``opType``); see the module doc.
    Raises BrmError."""
    what = f"brm {inst.brm.name!r} implementation {inst.implementation!r}"
    kind = inst.brm.function.accel
    if kind not in KINDS:
        raise BrmError(f"{what}: accelerator kind {kind!r} not registered ({sorted(KINDS)})")
    params = {k: inst.value(v) for k, v in inst.brm.function.params.items()
              if expr.names(v) <= set(inst.params)}  # an unset param is not evaluated  # fmt: skip
    try:
        built = KINDS[kind](**params, dtypes=dtypes, **({"op_type": op_type} if op_type else {}))
    except (TypeError, ValueError) as e:
        raise BrmError(f"{what}: kind {kind!r} rejects {params}: {e}") from None
    declared = [(p.name, p.direction, inst.lanes(p.name), p.rate) for p in inst.ports]
    ports = [(p.name, p.direction, p.lanes, p.rate) for p in built.ports]
    if declared != ports:
        raise BrmError(
            f"{what}: ports (name, direction, lanes, rate) differ from kind {kind!r}: "
            f"declared {declared}, built {ports}"
        )
    return built
