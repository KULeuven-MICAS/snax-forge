from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
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
    notes: str = ""
    tags: tuple[str, ...] = ()

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
