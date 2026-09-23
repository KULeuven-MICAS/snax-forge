"""The ``values`` of a configure step, per component type (D64).

A task names its component's type, which is the kind of the component's
register adapter (``streamer``, ``accel``, ``dma``), and gives its values in
that type's own terms rather than as registers:

    streamer  base, temporal_bounds, temporal_strides, spatial_strides (bytes);
              the spatial bounds are design-time and come from the cluster
    dma       direction ("l2_to_l1" / "l1_to_l2"), src and dst, each with
              base, bounds and strides (bytes, one wide beat per step)
    accel     its start parameters by name: n and any named rate (e.g. T)

``arg_of`` turns values into the start argument the adapter encodes, so the
registers are written exactly as ``RegisterMap.config_writes`` writes them
(D36, D45). Only the loops a task uses need to be given; the adapter pads
the rest (bound 1, stride 0). ``values_of`` is the inverse, for Python
generators that build task lists from start arguments. A new type is added
with ``register_values``, next to its adapter (principle 6).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from snax_forge.snax_model import DmaDescriptor, DmaPattern, StreamerRegs
from snax_forge.snax_model.ctrl import Block

ToArg = Callable[[Mapping[str, Any], Block], Any]
FromArg = Callable[[Any], dict[str, Any]]


def _exact(v: Any, keys: Sequence[str], what: str) -> None:
    """``v`` is a mapping with exactly ``keys``."""
    if not isinstance(v, Mapping):
        raise TypeError(f"{what} must be an object, got {v!r}")
    missing = [k for k in keys if k not in v]
    unknown = sorted(set(v) - set(keys))
    if missing or unknown:
        raise ValueError(f"{what}: missing {missing}, unknown {unknown} (expected {list(keys)})")


def _ints(v: Any, what: str) -> tuple[int, ...]:
    if not isinstance(v, (list, tuple)):
        raise TypeError(f"{what} must be a list, got {v!r}")
    return tuple(int(x) for x in v)


# -- streamer -----------------------------------------------------------------

_STREAMER = ("base", "temporal_bounds", "temporal_strides", "spatial_strides")


def _streamer_arg(v: Mapping[str, Any], block: Block) -> StreamerRegs:
    _exact(v, _STREAMER, "streamer values")
    return StreamerRegs(
        int(v["base"]),
        _ints(v["temporal_bounds"], "temporal_bounds"),
        _ints(v["temporal_strides"], "temporal_strides"),
        block.adapter.spatial_bounds,
        _ints(v["spatial_strides"], "spatial_strides"),
    )


def _streamer_values(arg: StreamerRegs) -> dict[str, Any]:
    return {
        "base": arg.base,
        "temporal_bounds": list(arg.temporal_bounds),
        "temporal_strides": list(arg.temporal_strides),
        "spatial_strides": list(arg.spatial_strides),
    }


# -- dma ----------------------------------------------------------------------

_DMA = ("direction", "src", "dst")
_PATTERN = ("base", "bounds", "strides")


def _pattern(v: Any, what: str) -> DmaPattern:
    _exact(v, _PATTERN, what)
    return DmaPattern(int(v["base"]), _ints(v["bounds"], f"{what} bounds"),
                      _ints(v["strides"], f"{what} strides"))  # fmt: skip


def _dma_arg(v: Mapping[str, Any], block: Block) -> DmaDescriptor:
    _exact(v, _DMA, "dma values")
    return DmaDescriptor(str(v["direction"]), _pattern(v["src"], "src"), _pattern(v["dst"], "dst"))


def _pattern_values(p: DmaPattern) -> dict[str, Any]:
    return {"base": p.base, "bounds": list(p.bounds), "strides": list(p.strides)}


def _dma_values(arg: DmaDescriptor) -> dict[str, Any]:
    return {"direction": arg.direction, "src": _pattern_values(arg.src),
            "dst": _pattern_values(arg.dst)}  # fmt: skip


# -- accel --------------------------------------------------------------------


def _accel_arg(v: Mapping[str, Any], block: Block) -> dict[str, int]:
    if not isinstance(v, Mapping):
        raise TypeError(f"accel values must be an object, got {v!r}")
    return {k: int(x) for k, x in v.items()}  # names are checked by the adapter


def _accel_values(arg: Mapping[str, int]) -> dict[str, Any]:
    return {k: int(x) for k, x in arg.items()}


# -- registry -----------------------------------------------------------------

_VALUES: dict[str, tuple[ToArg, FromArg]] = {
    "streamer": (_streamer_arg, _streamer_values),
    "dma": (_dma_arg, _dma_values),
    "accel": (_accel_arg, _accel_values),
}


def register_values(kind: str, to_arg: ToArg, from_arg: FromArg) -> None:
    """Add a component type: ``to_arg(values, block)`` and its inverse ``from_arg(arg)``."""
    _VALUES[kind] = (to_arg, from_arg)


def _codec(kind: str) -> tuple[ToArg, FromArg]:
    if kind not in _VALUES:
        raise ValueError(f"no values for component type {kind!r} (types: {list(_VALUES)})")
    return _VALUES[kind]


def arg_of(kind: str, values: Mapping[str, Any], block: Block) -> Any:
    """The start argument of ``block`` for a task's values.

    Raises TypeError for values of the wrong shape, ValueError for values the
    start argument rejects.
    """
    return _codec(kind)[0](values, block)


def values_of(kind: str, arg: Any) -> dict[str, Any]:
    """A start argument as a task's values (the inverse of ``arg_of``)."""
    return _codec(kind)[1](arg)
