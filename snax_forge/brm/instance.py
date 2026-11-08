"""A BRM instance: one BRM built one way (D68).

An instance is an implementation and a value for every design param. It is
what a design point holds, what SNAX-LOWER and the generator read, and what
the reference executor runs (its function, function.py):

    inst = brm.resolve("chisel_tiled_spatial", {"W": 4})

Resolving checks the values: an unknown implementation or param, a runtime
param given here (runtime params are the task's, CONTRACTS.md section 10), a
value of the wrong type, outside the param's ``values`` or outside the
implementation's ``supports`` are errors. A design param with neither a value
nor a default is unset: the instance leaves it out of its ``params``, and a
lanes or timing field that names it is an error; a function param or binding
arg that names it is not evaluated (the reader that needs it reads it only
where it applies). A slot read from a container (a kernel's scalar argument, written by the
program as a CSR) is unset and listed in ``read`` (``resolve``'s argument): it must be a
slot of the BRM's code. One code entry must fit the instance (brm.py's ``Function``: its
params, every slot its code names set or read, every other slot neither), so a one-stream
``simd`` without ``k``, neither baked nor read, is refused here; ``code`` is that entry's.
The instance's ``ports`` are the BRM's ports present at its
params (``Port.when``), in order: every reader after ``resolve`` takes them. It then checks a bound implementation against the
generator's datapaths (``gen.DATAPATHS``): its ``datapath`` must be registered
and the datapath's adapter must take the instance, so an instance that exists
can be generated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from snax_forge import expr

from .brm import TYPES, BrmError
from .notation import NOTATIONS

if TYPE_CHECKING:
    from .brm import Brm, Implementation, Port


@dataclass(frozen=True)
class Instance:
    """One BRM, one implementation, a value for every design param. Made by ``resolve``."""

    brm: Brm
    implementation: str
    params: dict[str, int | str]  # every design param that is set, in declaration order
    read: tuple[str, ...] = ()  # the slots read from a container (CSRs), in slot order

    @property
    def code(self) -> str | None:
        """The code of the entry that fits (brm.py's ``Function``); None for a null code."""
        return self.brm.code_for(self.params, self.read)

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

    @property
    def drain(self) -> int:
        return self.value(self.impl.timing.drain)

    @property
    def span(self) -> int:
        return self.value(self.impl.timing.span)

    @property
    def ports(self) -> list[Port]:
        """The BRM's ports present at these params (``Port.when``), in order."""
        return self.brm.present(self.params)

    def lanes(self, port: str) -> int:
        return self.value(self.brm.port(port).lanes)


def resolve(brm: Brm, implementation: str, params: Mapping[str, Any] | None = None,
            read: Iterable[str] = ()) -> Instance:  # fmt: skip
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
        if v is None:  # unset (module doc)
            continue
        if type(v) is not TYPES[p.type]:
            raise BrmError(f"{what}: {k} = {v!r} is not of type {p.type}")
        if p.values is not None and v not in p.values:
            raise BrmError(f"{what}: {k} = {v!r} not in the param's values {p.values}")
        if k in impl.supports and v not in impl.supports[k]:
            raise BrmError(f"{what}: {k} = {v!r} not supported ({impl.supports[k]})")
        values[k] = v
    slots = brm.all_slots()
    for k in read:
        if k not in slots:
            raise BrmError(f"{what}: {k!r} is read from a container, but it is no slot of the "
                           f"BRM's code ({slots})")  # fmt: skip
        if k in values:
            raise BrmError(f"{what}: {k} = {values[k]!r} is set, and read from a container too")
    inst = Instance(brm, implementation, values, tuple(s for s in slots if s in set(read)))
    _check_values(inst, what)
    _check_dataflow(inst, what)
    _check_datapath(inst, what)
    _check_code(inst, what)
    return inst


def _check_code(inst: Instance, what: str) -> None:
    """One code entry fits the instance (brm.py's ``Function``)."""
    b = inst.brm
    entries = b.function.entries()
    if not entries or inst.code is not None:
        return
    at = [e for e in entries if all(inst.params.get(k) == v for k, v in e.params.items())]
    has = [s for s in b.all_slots() if s in inst.params or s in inst.read]
    shown = ", ".join(f"{k} = {v!r}" for k, v in inst.params.items())
    if not at:
        raise BrmError(f"{what}: no entry of its code is at {shown}")
    wants = " or ".join(f"{e.code!r} with {' and '.join(b.slots(e)) or 'no slot'}" for e in at)
    raise BrmError(f"{what}: at {shown}, the code is {wants}; the slots set here (baked, or read "
                   f"from a container): {', '.join(has) or 'none'}")  # fmt: skip


def _check_dataflow(inst: Instance, what: str) -> None:
    """The notation's own checks that need design-param values (e.g. spatial size = lanes)."""
    d = inst.brm.dataflow
    check = NOTATIONS[d.notation].check_instance
    if check is None:
        return
    for p in inst.ports:
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

    for p in inst.ports:
        num(p.lanes, f"port {p.name!r} lanes", 1)
    num(inst.impl.timing.latency, "latency", 0)
    num(inst.impl.timing.initiation_interval, "initiation_interval", 1)
    num(inst.impl.timing.drain, "drain", 0)
    num(inst.impl.timing.span, "span", 1)


def _check_datapath(inst: Instance, what: str) -> None:
    """A bound implementation's datapath is one the generator builds and takes this
    instance."""
    if inst.impl.binding is None:
        return
    from snax_forge.gen.blocks import DATAPATHS, GenError  # gen imports brm

    datapath = inst.impl.binding["datapath"]
    if datapath not in DATAPATHS:
        raise BrmError(
            f"{what}: datapath {datapath!r} is not one the generator builds ({sorted(DATAPATHS)})"
        )
    try:
        DATAPATHS[datapath].fill(inst)
    except GenError as e:
        raise BrmError(f"{what}: datapath {datapath!r} refuses the instance: {e}") from None
