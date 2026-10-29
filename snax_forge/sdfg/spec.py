from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
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

    def with_dtypes(self, dtypes: dict[str, str], what: str = "dtypes") -> KernelSpec:
        """The spec with some containers' integer dtype replaced (a recipe's ``dtypes``, D113):
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
            for sym, dim in zip(desc.shape, np.shape(arr)):
                if isinstance(sym, dace.symbolic.symbol):
                    out[str(sym)] = int(dim)
        return out
