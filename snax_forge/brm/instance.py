"""A BRM instance and its accelerator entry (BRM1, D68).

An instance is one BRM built one way: an implementation and a value for
every design param. It is what a design point will hold (DP1) and what
SNAX-LOWER turns into the accelerator entry of the cluster file (LOW1c):

    inst = brm.resolve("chisel_tiled_spatial", {"W": 4})
    inst.accel_entry()
    # ("elementwise", {"lanes": 4, "n_inputs": 2, "op": "add", "latency": 0, "ii": 1})

The entry's ``params`` are the function params resolved, in their order,
then the implementation's ``latency`` and ``ii`` (the BRM's
``initiation_interval``). The model is not changed: the cluster file keeps
naming the registered accelerator kind (D43, D51).

Resolving checks the values: an unknown implementation or param, a runtime
param given here (runtime params are the task's, CONTRACTS.md section 4), a
value of the wrong type, outside the param's ``values`` or outside the
implementation's ``supports``, and a design param with neither a value nor a
default are errors. It then builds the ``AccelConfig`` through the
registered kind and checks it against the BRM: the same ports in the same
order (names, directions, lanes, rates) and the same ``latency`` and ``ii``.
An instance that exists is therefore consistent with the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from snax_forge import expr
from snax_forge.snax_model.accel import AccelConfig
from snax_forge.snax_model.scenario import ACCEL_KINDS

from .brm import TYPES, BrmError
from .notation import NOTATIONS

if TYPE_CHECKING:
    from .brm import Brm, Implementation


@dataclass(frozen=True)
class Instance:
    """One BRM, one implementation, a value for every design param. Made by ``resolve``."""

    brm: Brm
    implementation: str
    params: dict[str, int | str]  # every design param, in declaration order

    @property
    def impl(self) -> Implementation:
        return self.brm.implementations[self.implementation]

    def value(self, v: expr.Value) -> Any:
        """A BRM value field evaluated with this instance's design params."""
        return expr.evaluate(v, self.params)

    @property
    def latency(self) -> int:
        return self.value(self.impl.timing.latency)

    @property
    def initiation_interval(self) -> int:
        return self.value(self.impl.timing.initiation_interval)

    def lanes(self, port: str) -> int:
        return self.value(self.brm.port(port).lanes)

    def accel_entry(self) -> tuple[str, dict[str, Any]]:
        """``(accel, params)`` of the accelerator entry in the cluster file."""
        params = {k: self.value(v) for k, v in self.brm.function.params.items()}
        params["latency"] = self.latency
        params["ii"] = self.initiation_interval
        return self.brm.function.accel, params

    def accel_config(self) -> AccelConfig:
        """The model's accelerator, built through the registered kind (D43)."""
        kind, params = self.accel_entry()
        if kind not in ACCEL_KINDS:
            raise BrmError(
                f"{self._what()}: accelerator kind {kind!r} not registered ({sorted(ACCEL_KINDS)})"
            )
        try:
            return ACCEL_KINDS[kind](**params)
        except (TypeError, ValueError) as e:
            raise BrmError(f"{self._what()}: kind {kind!r} rejects {params}: {e}") from None

    def _what(self) -> str:
        return f"brm {self.brm.name!r} implementation {self.implementation!r}"


def resolve(brm: Brm, implementation: str, params: Mapping[str, Any] | None = None) -> Instance:
    """The instance of ``brm`` built as ``implementation`` with ``params``; see the module doc."""
    what = f"brm {brm.name!r}"
    if implementation not in brm.implementations:
        raise BrmError(
            f"{what}: no implementation {implementation!r} ({list(brm.implementations)})"
        )
    what = f"{what} implementation {implementation!r}"
    impl = brm.implementations[implementation]
    given = dict(params or {})
    design, runtime = brm.params("design"), brm.params("runtime")
    for k in given:
        if k in runtime:
            raise BrmError(f"{what}: {k!r} is a runtime param, given per task, not here")
        if k not in design:
            raise BrmError(f"{what}: unknown param {k!r} (design params: {list(design)})")
    values: dict[str, int | str] = {}
    for k, p in design.items():
        v = given.get(k, p.default)
        if v is None:
            raise BrmError(f"{what}: no value for {k!r} and no default")
        if type(v) is not TYPES[p.type]:
            raise BrmError(f"{what}: {k} = {v!r} is not of type {p.type}")
        if p.values is not None and v not in p.values:
            raise BrmError(f"{what}: {k} = {v!r} not in the param's values {p.values}")
        if k in impl.supports and v not in impl.supports[k]:
            raise BrmError(f"{what}: {k} = {v!r} not supported ({impl.supports[k]})")
        values[k] = v
    inst = Instance(brm, implementation, values)
    _check_values(inst, what)
    _check_dataflow(inst, what)
    _check_model(inst, what)
    return inst


def _check_dataflow(inst: Instance, what: str) -> None:
    """The notation's own checks that need design-param values (e.g. spatial size = lanes)."""
    d = inst.brm.dataflow
    check = NOTATIONS[d.notation].check_instance
    if check is None:
        return
    for p in inst.brm.interface.ports:
        try:
            check(d.ports[p.name], p, inst)
        except (TypeError, ValueError) as e:
            raise BrmError(f"{what}: dataflow of port {p.name!r}: {e}") from None


def _check_values(inst: Instance, what: str) -> None:
    """Resolved lanes and timing are ints in range."""

    def num(v: expr.Value, field: str, low: int) -> None:
        try:
            x = inst.value(v)
        except expr.ExprError as e:
            raise BrmError(f"{what}: {field}: {e}") from None
        if type(x) is not int or x < low:
            raise BrmError(f"{what}: {field} = {v!r} gives {x!r}, need an int >= {low}")

    for p in inst.brm.interface.ports:
        num(p.lanes, f"port {p.name!r} lanes", 1)
    num(inst.impl.timing.latency, "latency", 0)
    num(inst.impl.timing.initiation_interval, "initiation_interval", 1)


def _check_model(inst: Instance, what: str) -> None:
    """The AccelConfig the registered kind builds is the accelerator the BRM declares."""
    cfg = inst.accel_config()
    declared = [(p.name, p.direction, inst.lanes(p.name), p.rate) for p in inst.brm.interface.ports]
    built = [(p.name, p.direction, p.lanes, p.rate) for p in cfg.ports]
    if declared != built:
        raise BrmError(
            f"{what}: ports (name, direction, lanes, rate) differ from kind "
            f"{inst.brm.function.accel!r}: declared {declared}, built {built}"
        )
    timing = (inst.latency, inst.initiation_interval)
    if (cfg.latency, cfg.ii) != timing:
        raise BrmError(
            f"{what}: kind {inst.brm.function.accel!r} builds latency {cfg.latency}, ii {cfg.ii}; "
            f"the implementation declares latency {timing[0]}, initiation_interval {timing[1]}"
        )
