from __future__ import annotations

import inspect
import types as pytypes
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import dace
import numpy as np


@dataclass(frozen=True)
class KernelSpec:
    """
    Contract between a kernel module and the toolchain.
    """

    name: str
    func: Callable
    domain: str
    descriptors: dict[str, Any]  # arg name -> dace data descriptor
    make_inputs: Callable  # rng, **sizes -> dict of concrete arrays
    inout: tuple[str, ...]
    ref: Callable | None = None
    flops: Callable | None = None  # **sizes -> op count
    bytes_moved: Callable | None = None  # **sizes -> compulsory traffic
    tags: tuple[str, ...] = ()
    sweep_sizes: tuple = ()  # entries: int (n=) or dict of size kwargs
    # type parameters the function reads as globals (OP_TYPE in ``A.astype(OP_TYPE)``) -> their default
    types: Mapping[str, str] = field(default_factory=dict)
    # preset name -> {container or type parameter: dtype}, chosen with ``--set types=<name>``
    presets: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    # container -> (lo, hi), both included: what full-range data draws it from, inside
    # its dtype; the others are drawn over all of their dtype
    ranges: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.inout:
            raise ValueError(
                f"{self.name}: inout is empty — verification would "
                "compare nothing and pass vacuously."
            )
        params = set(inspect.signature(self.func).parameters)
        for label, names in (("inout", set(self.inout)), ("descriptors", set(self.descriptors))):
            unknown = names - params
            if unknown:
                raise ValueError(
                    f"{self.name}: {label} names {sorted(unknown)} are not "
                    f"parameters of {self.func.__name__}{tuple(sorted(params))}"
                )
        missing = params - set(self.descriptors)
        if missing:
            raise ValueError(f"{self.name}: no descriptor for {sorted(missing)}")
        if clash := sorted(set(self.types) & set(self.descriptors)):
            raise ValueError(f"{self.name}: type parameter {clash[0]!r} is also a container")
        for c, (lo, hi) in self.ranges.items():
            if c not in self.descriptors:
                raise ValueError(f"{self.name}: ranges names {c!r}, which is not a container")
            info = np.iinfo(self.descriptors[c].dtype.type)
            if not info.min <= lo <= hi <= info.max:
                raise ValueError(f"{self.name}: the range of {c}, [{lo}, {hi}], is not inside "
                                 f"its dtype's [{info.min}, {info.max}]")  # fmt: skip
        for name, preset in self.presets.items():
            if unknown := sorted(set(preset) - set(self.descriptors) - set(self.types)):
                raise ValueError(f"{self.name}: preset {name!r} names {unknown[0]!r}, neither a "
                                 "container nor a type parameter")  # fmt: skip

    def kernel_dtypes(self, preset: str | None, dtypes: Mapping[str, str],
                      what: str = "types") -> dict[str, str]:  # fmt: skip
        """The dtypes a recipe asks for: its preset's (``--set types=``), then each of
        ``dtypes`` (``--set dtypes.X=``) on top; an unknown preset raises ValueError naming the
        kernel's."""
        if preset is not None and preset not in self.presets:
            raise ValueError(f"{what}: {self.name} has no preset {preset!r} (presets: "
                             f"{sorted(self.presets) or 'none'})")  # fmt: skip
        return {**(self.presets[preset] if preset is not None else {}), **dtypes}

    def with_types(self, dtypes: Mapping[str, str], what: str = "dtypes") -> KernelSpec:
        """The spec with some containers' dtypes (``with_dtypes``) and type parameters
        replaced: the function's globals rebound (``OP_TYPE = np.int32``), so both DaCe and the
        reference compute at them."""
        if unknown := sorted(set(dtypes) - set(self.descriptors) - set(self.types)):
            raise ValueError(f"{what}: {unknown[0]!r} is not a container of {self.name} "
                             f"({sorted(self.descriptors)}) nor a type parameter "
                             f"({sorted(self.types) or 'none'})")  # fmt: skip
        params = {k: v for k, v in dtypes.items() if k in self.types}
        spec = self.with_dtypes({k: v for k, v in dtypes.items() if k not in self.types}, what)
        if not params:
            return spec
        f = self.func
        env = {**f.__globals__, **{k: getattr(np, v) for k, v in params.items()}}
        func = pytypes.FunctionType(f.__code__, env, f.__name__, f.__defaults__, f.__closure__)
        func.__module__, func.__qualname__ = f.__module__, f.__qualname__
        return replace(spec, func=func)

    def type_params(self, dtypes: Mapping[str, str]) -> dict[str, str]:
        """The type parameters ``dtypes`` set to other than their default."""
        return {k: v for k, v in dtypes.items() if k in self.types and self.types[k] != v}

    def with_dtypes(self, dtypes: dict[str, str], what: str = "dtypes") -> KernelSpec:
        """The spec with some containers' integer dtype replaced (a recipe's ``dtypes``):
        their descriptors, so DaCe builds the SDFG at those types; ``make_inputs`` is cast
        by its caller (``dfg.reference.kernel_inputs``). A name that is not a descriptor raises
        ValueError naming it."""
        unknown = sorted(set(dtypes) - set(self.descriptors))
        if unknown:
            raise ValueError(
                f"{what}: {unknown[0]!r} is not a container of {self.name} "
                f"({sorted(self.descriptors)})"
            )
        descs = dict(self.descriptors)
        for c, dt in dtypes.items():
            t = getattr(dace, dt)
            desc = descs[c]
            descs[c] = t[tuple(desc.shape)] if isinstance(desc, dace.data.Array) else t
        return replace(self, descriptors=descs)

    @property
    def reference(self) -> Callable:
        return self.ref if self.ref is not None else self.func

    def bind_symbols(self, args: dict) -> dict[str, int]:
        """Match symbolic dims against actual array shapes. Generic over rank."""
        out: dict[str, int] = {}
        for name, desc in self.descriptors.items():
            arr = args.get(name)
            if arr is None or not hasattr(desc, "shape"):
                continue
            # a scalar's descriptor has shape (1,), its argument ()
            for sym, dim in zip(desc.shape, np.shape(arr), strict=False):
                if isinstance(sym, dace.symbolic.symbol):
                    out[str(sym)] = int(dim)
        return out
