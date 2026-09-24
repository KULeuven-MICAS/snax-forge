"""The affine dataflow notation (BRM2, D70): the first version of open item 1.

One nest per port says in which order the accelerator consumes or produces
its operand's elements, in logical indices. It describes only the
accelerator; where the elements live is the layout's business (SNAX-LOWER,
``snax_forge.lower.streams``).

    {
     "shape":  ["n*W"],                 the operand's extent per dimension
     "offset": [0],                     index of the first element (default 0s)
     "loops": [                         outermost first
       {"bound": "n", "strides": ["W"], "spatial": false},
       {"bound": "W", "strides": [1],   "spatial": true}
     ]
    }

Loop l runs a counter i_l over [0, bound_l) and moves index dimension d by
``strides_l[d]`` per step, so the element of one step is

    index[d] = offset[d] + sum_l i_l * strides_l[d]

which is affine by construction and maps one to one onto the streamer's
registers once a layout gives each dimension a byte stride. The temporal
loops come first: one step of them is one beat, the last temporal loop
innermost. The spatial loops come last and enumerate the lanes of a beat,
the last one fastest (lane dimension 0, as the streamer's spatial dimension
0, CONTRACTS.md section 3).

Every value is an int or an expression over the BRM's params
(snax_forge/expr.py).
Checked when the BRM is made: the keys, one stride per dimension, at least
one spatial loop and only at the end, and spatial bounds that use design
params only (lanes are design time, as the streamer's spatial bounds).
Temporal bounds, strides, shape and offset may use runtime params.
Checked by ``Brm.resolve``: the spatial bounds are ints >= 1 whose product
is the port's lanes. Checked by ``task_nest`` once the task's start
parameters are known: ``n`` is a multiple of the port's rate, the temporal
bounds give n / rate beats (D25), and every index lies inside the shape.

A stride may be 0 (the same element again; on a reader's innermost
temporal loop the streamer repeats the beat, D69) or negative. Whether the
ports agree on element positions is the BRM author's responsibility (D70).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np

from snax_forge import expr

from .notation import register_notation

if TYPE_CHECKING:
    from .brm import Brm, Port
    from .instance import Instance

NAME = "affine"
_NEST_KEYS = ("shape", "offset", "loops")
_LOOP_KEYS = ("bound", "strides", "spatial")


# =============================================================================
# Structure (no values yet)
# =============================================================================


def _keys(d: Any, required: tuple[str, ...], allowed: tuple[str, ...], what: str) -> None:
    if not isinstance(d, Mapping):
        raise TypeError(f"{what}: must be an object, got {d!r}")
    unknown = sorted(set(d) - set(allowed))
    if unknown:
        raise ValueError(f"{what}: unknown keys {unknown} (allowed: {list(allowed)})")
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"{what}: missing key {missing[0]!r}")


def _values(v: Any, what: str, allowed: set[str], length: int | None = None) -> None:
    if not isinstance(v, list) or not v:
        raise ValueError(f"{what}: must be a non-empty list")
    if length is not None and len(v) != length:
        raise ValueError(f"{what}: {len(v)} entries for {length} dimensions")
    for i, x in enumerate(v):
        _value(x, f"{what}[{i}]", allowed)


def _value(v: Any, what: str, allowed: set[str]) -> None:
    expr.check(v, what)
    unknown = sorted(expr.names(v) - allowed)
    if unknown:
        raise ValueError(f"{what}: {v!r} uses {unknown}, allowed here: {sorted(allowed)}")


def check(nest: Any, port: Port, brm: Brm) -> None:
    """The nest is well formed for ``port``; see the module doc."""
    _keys(nest, ("shape", "loops"), _NEST_KEYS, "nest")
    every = set(brm.interface.params)
    design = set(brm.params("design"))
    _values(nest["shape"], "shape", every)
    rank = len(nest["shape"])
    if "offset" in nest:
        _values(nest["offset"], "offset", every, rank)
    loops = nest["loops"]
    if not isinstance(loops, list) or not loops:
        raise ValueError("loops: must be a non-empty list")
    spatial_seen = False
    for i, loop in enumerate(loops):
        w = f"loops[{i}]"
        _keys(loop, ("bound", "strides"), _LOOP_KEYS, w)
        spatial = loop.get("spatial", False)
        if type(spatial) is not bool:
            raise ValueError(f"{w}.spatial: must be true or false")
        if spatial_seen and not spatial:
            raise ValueError(f"{w}: a temporal loop after a spatial one; spatial loops come last")
        spatial_seen = spatial_seen or spatial
        _value(loop["bound"], f"{w}.bound", design if spatial else every)
        _values(loop["strides"], f"{w}.strides", every, rank)
    if not spatial_seen:
        raise ValueError("needs at least one spatial loop (the lanes)")


def normalize(nest: Mapping[str, Any]) -> dict[str, Any]:
    """Every field written: the default offset and ``spatial`` flags filled in."""
    rank = len(nest["shape"])
    return {
        "shape": list(nest["shape"]),
        "offset": list(nest.get("offset", [0] * rank)),
        "loops": [
            {
                "bound": loop["bound"],
                "strides": list(loop["strides"]),
                "spatial": bool(loop.get("spatial", False)),
            }
            for loop in nest["loops"]
        ],
    }


def check_instance(nest: Any, port: Port, inst: Instance) -> None:
    """With the design params known: the spatial loops give exactly the port's lanes."""
    bounds = []
    for loop in normalize(nest)["loops"]:
        if loop["spatial"]:
            b = inst.value(loop["bound"])
            if type(b) is not int or b < 1:
                raise ValueError(f"spatial bound {loop['bound']!r} gives {b!r}, need an int >= 1")
            bounds.append(b)
    lanes = inst.lanes(port.name)
    if prod(bounds) != lanes:
        raise ValueError(f"spatial bounds {bounds} give {prod(bounds)} lanes, the port has {lanes}")


# =============================================================================
# Resolved nests
# =============================================================================


@dataclass(frozen=True)
class Nest:
    """A nest with every value resolved to an int. Loops are listed outermost first."""

    shape: tuple[int, ...]
    offset: tuple[int, ...]
    temporal: tuple[tuple[int, tuple[int, ...]], ...]  # (bound, strides), outermost first
    spatial: tuple[tuple[int, tuple[int, ...]], ...]  # (bound, strides), last = lane dim 0

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def n_beats(self) -> int:
        return prod(b for b, _ in self.temporal)

    @property
    def n_lanes(self) -> int:
        return prod(b for b, _ in self.spatial)

    def indices(self) -> np.ndarray:
        """Every element in order, shape ``[n_beats, n_lanes, rank]``."""
        return _enumerate(self.offset, self.temporal, self.spatial)

    def extent(self) -> tuple[tuple[int, int], ...]:
        """Per dimension the smallest and largest index the nest touches (it has beats)."""
        lo, hi = list(self.offset), list(self.offset)
        for b, strides in (*self.temporal, *self.spatial):
            for d, s in enumerate(strides):
                step = (b - 1) * s
                lo[d] += min(0, step)
                hi[d] += max(0, step)
        return tuple(zip(lo, hi, strict=True))


def _enumerate(offset, temporal, spatial) -> np.ndarray:
    rank = len(offset)
    tb = [b for b, _ in temporal]
    sb = [b for b, _ in spatial]
    n, lanes = prod(tb), prod(sb)
    out = np.zeros((n, lanes, rank), dtype=np.int64) + np.asarray(offset, dtype=np.int64)
    if n == 0:
        return out
    # Outermost first, so the last listed loop runs fastest (C order).
    ti = np.array(np.unravel_index(np.arange(n), tb)) if tb else np.zeros((0, n), np.int64)
    si = np.array(np.unravel_index(np.arange(lanes), sb))
    ts = np.array([s for _, s in temporal], dtype=np.int64).reshape(len(tb), rank)
    ss = np.array([s for _, s in spatial], dtype=np.int64).reshape(len(sb), rank)
    out += (ti.T @ ts)[:, None, :]
    out += (si.T @ ss)[None, :, :]
    return out


def resolve_nest(nest: Mapping[str, Any], env: Mapping[str, Any]) -> Nest:
    """The nest with every value taken from ``env`` (design params plus the task's)."""
    nest = normalize(nest)

    def ints(vs: list[Any], what: str, low: int | None = None) -> tuple[int, ...]:
        out = []
        for v in vs:
            x = expr.evaluate(v, env)
            if type(x) is not int or (low is not None and x < low):
                need = "an int" if low is None else f"an int >= {low}"
                raise ValueError(f"{what}: {v!r} gives {x!r}, need {need}")
            out.append(x)
        return tuple(out)

    temporal, spatial = [], []
    for i, loop in enumerate(nest["loops"]):
        low = 1 if loop["spatial"] else 0
        (bound,) = ints([loop["bound"]], f"loops[{i}].bound", low)
        entry = (bound, ints(loop["strides"], f"loops[{i}].strides"))
        (spatial if loop["spatial"] else temporal).append(entry)
    return Nest(
        shape=ints(nest["shape"], "shape", 1),
        offset=ints(nest["offset"], "offset"),
        temporal=tuple(temporal),
        spatial=tuple(spatial),
    )


def task_nest(inst: Instance, port: str, task: Mapping[str, int]) -> Nest:
    """The resolved nest of ``port`` for one task, whose start parameters are ``task``.

    ``task`` holds exactly the BRM's registers (``n`` and every named rate),
    the accelerator's ``values`` in the task list. Raises ValueError if ``n``
    is not a multiple of the port's rate, if the nest does not give n / rate
    beats, or if an index falls outside the shape.
    """
    brm = inst.brm
    if brm.dataflow.notation != NAME:
        raise ValueError(f"brm {brm.name!r}: dataflow notation is {brm.dataflow.notation!r}")
    if sorted(task) != sorted(brm.registers):
        raise ValueError(f"task values {sorted(task)} must be exactly {brm.registers}")
    p = brm.port(port)
    what = f"brm {brm.name!r} port {port!r}"
    env = {**inst.params, **task}
    rate = expr.evaluate(p.rate, env)
    n = env["n"]
    if type(rate) is not int or rate < 1 or n % rate:
        raise ValueError(f"{what}: n = {n} is not a multiple of the rate {rate}")
    try:
        nest = resolve_nest(brm.dataflow.ports[port], env)
    except (ValueError, expr.ExprError) as e:
        raise ValueError(f"{what}: {e}") from None
    if nest.n_beats != n // rate:
        raise ValueError(
            f"{what}: the nest gives {nest.n_beats} beats, need n / rate = {n // rate}"
        )
    if nest.n_beats:
        for d, ((lo, hi), size) in enumerate(zip(nest.extent(), nest.shape, strict=True)):
            if lo < 0 or hi >= size:
                raise ValueError(
                    f"{what}: indices of dimension {d} run over [{lo}, {hi}], "
                    f"outside the shape {list(nest.shape)}"
                )
    return nest


register_notation(NAME, check, check_instance, normalize)
