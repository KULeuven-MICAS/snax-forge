"""A BRM's ``function``, run firing by firing: the accelerator kinds (D19, D68).

A BRM's ``function`` names a kind registered here and gives the kind's
params as expressions over its design params; ``function_of(instance)``
builds the kind with them resolved. The reference executor (dfg/execute.py)
runs an accelerated node through it, so the kind says what a firing computes
and when each port moves a beat; timing (L, II, drain) is the
implementation's, not the kind's.

Ports (``AccelPort``): name, direction, lanes per beat and a rate: a port
moves one beat every ``rate`` firings, an int or the name of a runtime param
(e.g. ``"T"``). An input of rate r is taken at firings 0, r, 2r, ...; an
output of rate r is given at firings r-1, 2r-1, ... (C§10).

    fn(k, ins, state, params) -> outs

``k`` is the firing index in the task; ``ins`` maps every input due at ``k``
to its beat (a NumPy array, first axis = lanes); ``state`` lives for one
task (a reduction's partial sum); ``params`` are the task's runtime params.
``outs`` holds exactly the outputs due at ``k``.

Kinds: ``elementwise`` (every port rate 1, N inputs folded into one output,
lane by lane) and ``reduce`` (input ``a`` at rate 1, output ``out`` at rate
``"T"``, the lanes folded too when ``lanes_out`` is 1), each with ``op`` a
registered op by name (``add``, ``sub``, ``mul``, ``min``, ``max``, ``and``,
``or``, ``xor``). New ones with ``register_kind`` and ``register_op``.

``function_of`` refuses, as a BrmError, a kind that is not registered, params
the kind rejects, and a kind whose ports (name, direction, lanes, rate, in
order) are not the BRM's.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce as _fold
from typing import TYPE_CHECKING, Any

import numpy as np

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
    """One data port: ``rate``, an int or the name of a runtime param, is firings per beat."""

    name: str
    direction: str  # "in" or "out"
    lanes: int = 1
    rate: int | str = 1

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
        """The ports of ``direction`` that move a beat at firing ``k`` (module doc)."""
        if direction == "in":
            return [p.name for p in self.inputs if k % rates[p.name] == 0]
        return [p.name for p in self.outputs if (k + 1) % rates[p.name] == 0]


# =============================================================================
# Kinds
# =============================================================================


def elementwise(
    lanes: int = 1,
    n_inputs: int = 2,
    op: Callable[[Any, Any], Any] = np.add,
) -> AccelFunction:
    """N inputs -> 1 output, one beat each per firing, ``op`` folding them left to right."""
    names = [chr(ord("a") + i) for i in range(n_inputs)]

    def fn(k, ins, state, params):
        return {"out": _fold(op, (ins[n] for n in names))}

    ports = [AccelPort(n, "in", lanes) for n in names] + [AccelPort("out", "out", lanes)]
    return AccelFunction(tuple(ports), fn, "elementwise")


def reduce_kind(
    lanes: int = 1,
    lanes_out: int | None = None,
    op: Callable[[Any, Any], Any] = np.add,
) -> AccelFunction:
    """T input beats -> 1 output beat, T the runtime param ``"T"``.

    ``lanes_out = lanes`` (default): each lane is folded over T beats.
    ``lanes_out = 1``: the lanes are folded too.
    """
    lanes_out = lanes if lanes_out is None else lanes_out
    if lanes_out not in (lanes, 1):
        raise ValueError("lanes_out must equal lanes or be 1")

    def fn(k, ins, state, params):
        x = ins["a"]
        if lanes_out == 1:
            x = _fold(op, (x[j : j + 1] for j in range(x.shape[0])))
        state["acc"] = x.copy() if "acc" not in state else op(state["acc"], x)
        if (k + 1) % params["T"] == 0:
            return {"out": state.pop("acc")}
        return {}

    ports = (AccelPort("a", "in", lanes, 1), AccelPort("out", "out", lanes_out, "T"))
    return AccelFunction(ports, fn, "reduce")


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
    """The kind with ``op`` given by name instead of as a function (default: add)."""

    def factory(**params: Any) -> AccelFunction:
        name = params.get("op", "add")
        if name not in OPS:
            raise ValueError(f"unknown op {name!r} (registered: {sorted(OPS)})")
        return kind(**{**params, "op": OPS[name]})

    return factory


register_kind("elementwise", _with_op(elementwise))
register_kind("reduce", _with_op(reduce_kind))


def function_of(inst: Instance) -> AccelFunction:
    """The instance's function, its params resolved; see the module doc. Raises BrmError."""
    what = f"brm {inst.brm.name!r} implementation {inst.implementation!r}"
    kind = inst.brm.function.accel
    if kind not in KINDS:
        raise BrmError(f"{what}: accelerator kind {kind!r} not registered ({sorted(KINDS)})")
    params = {k: inst.value(v) for k, v in inst.brm.function.params.items()}
    try:
        built = KINDS[kind](**params)
    except (TypeError, ValueError) as e:
        raise BrmError(f"{what}: kind {kind!r} rejects {params}: {e}") from None
    declared = [(p.name, p.direction, inst.lanes(p.name), p.rate) for p in inst.brm.interface.ports]
    ports = [(p.name, p.direction, p.lanes, p.rate) for p in built.ports]
    if declared != ports:
        raise BrmError(
            f"{what}: ports (name, direction, lanes, rate) differ from kind {kind!r}: "
            f"declared {declared}, built {ports}"
        )
    return built
