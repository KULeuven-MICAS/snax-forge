"""Reductions on a memlet: how the writes to one element combine.

A tasklet output whose memlet has a ``reduce`` (``Reduction``) folds the value of every
iteration of the maps around it into its element, starting from ``identity``, so the node
leaves the fold of exactly its own writes: what a DaCe ``Reduce`` and a hardware accumulator
do. Its stored form, and the ``seed`` (``Seed``) that may take the identity's place, are in
CONTRACTS, "SNAX-DFG"; a plain ``identity: null`` (DaCe's plain ``wcr``) is refused.
``map_seed`` rewrites the seed's subset with the memlet's, so every transform moves both.

An op is registered by name (``register_reduction``) with the NumPy ufunc that folds it and its
identity for a dtype. The registered ops (``add``, ``mul``, ``min``, ``max``) are associative and
commutative on integers, so a map with a ``reduce`` memlet may be split, tiled or run in
parallel. ``ReduceOp.order_free`` says so per op: a transform that changes the order of a fold's
writes (``interchange_map``) refuses an op registered with ``order_free=False``. A float op is
not registered: its result depends on the order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .kinds import DfgError, _wrap
from .subset import canonical_dim


@dataclass(frozen=True)
class ReduceOp:
    """A registered op: the ufunc that folds it, its identity for an integer dtype, and whether
    its result is the same in any order (associative and commutative)."""

    ufunc: np.ufunc
    identity: Callable[[np.dtype], int]
    order_free: bool = True


REDUCE_OPS: dict[str, ReduceOp] = {}


def register_reduction(name: str, ufunc: np.ufunc, identity: Callable[[np.dtype], int],
                       order_free: bool = True) -> None:  # fmt: skip
    """Make ``"reduce": {"op": name, ...}`` usable; ``order_free`` False for an op that is not
    associative and commutative, which a transform reordering its fold refuses."""
    if name in REDUCE_OPS:
        raise ValueError(f"reduce op {name!r} already registered")
    REDUCE_OPS[name] = ReduceOp(ufunc, identity, order_free)


register_reduction("add", np.add, lambda dt: 0)
register_reduction("mul", np.multiply, lambda dt: 1)
register_reduction("min", np.minimum, lambda dt: int(np.iinfo(dt).max))
register_reduction("max", np.maximum, lambda dt: int(np.iinfo(dt).min))


@dataclass(frozen=True)
class Seed:
    """The element a seeded fold starts from: a container and a subset, as a memlet's."""

    data: str
    subset: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.data, str) or not self.data:
            raise DfgError(f"reduce.seed.data: must be a container name, got {self.data!r}")
        if not isinstance(self.subset, (list, tuple)) or not self.subset:
            raise DfgError(f"reduce.seed.subset: must be a non-empty list, got {self.subset!r}")
        dims = tuple(_wrap(lambda d=d: canonical_dim(d, "reduce.seed.subset")) for d in self.subset)
        object.__setattr__(self, "subset", dims)

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data, "subset": list(self.subset)}

    @classmethod
    def from_dict(cls, d: Any, what: str) -> Seed:
        if not isinstance(d, Mapping) or set(d) != {"data", "subset"}:
            raise DfgError(f"{what}: must be {{data, subset}}, got {d!r}")
        return cls(d["data"], tuple(d["subset"]))


@dataclass(frozen=True)
class Reduction:
    """The ``reduce`` of a memlet: a registered op and the value the element starts from, its
    ``identity`` or a ``seed`` element (module doc), one of the two."""

    op: str
    identity: int | None = None
    seed: Seed | None = None

    def __post_init__(self) -> None:
        if self.op not in REDUCE_OPS:
            raise DfgError(f"reduce.op: {self.op!r} is not a registered op {sorted(REDUCE_OPS)}")
        if self.seed is not None:
            if not isinstance(self.seed, Seed):
                raise DfgError(f"reduce.seed: must be a Seed, got {self.seed!r}")
            if self.identity is not None:
                raise DfgError("reduce: identity and seed are mutually exclusive (a seeded fold "
                               "starts from its seed)")  # fmt: skip
            return
        if self.identity is None:
            raise DfgError(
                "reduce.identity: null (fold into the element's contents) is not accepted yet"
            )
        if type(self.identity) is not int:
            raise DfgError(f"reduce.identity: must be an int, got {self.identity!r}")

    @property
    def ufunc(self) -> np.ufunc:
        return REDUCE_OPS[self.op].ufunc

    def map_seed(self, fn: Callable[[Any], Any]) -> Reduction:
        """The reduction with ``fn`` on each dimension of its seed's subset (itself unseeded)."""
        if self.seed is None:
            return self
        return Reduction(
            self.op, None, Seed(self.seed.data, tuple(fn(d) for d in self.seed.subset))
        )

    def to_dict(self) -> dict[str, Any]:
        if self.seed is not None:
            return {"op": self.op, "seed": self.seed.to_dict()}
        return {"op": self.op, "identity": self.identity}

    @classmethod
    def from_dict(cls, d: Any, what: str) -> Reduction:
        if not isinstance(d, Mapping) or set(d) not in ({"op", "identity"}, {"op", "seed"}):
            raise DfgError(f"{what}: must be {{op, identity}} or {{op, seed}}, got {d!r}")
        try:
            if "seed" in d:
                return cls(d["op"], None, Seed.from_dict(d["seed"], "seed"))
            return cls(d["op"], d["identity"])
        except DfgError as e:
            raise DfgError(f"{what}.{e}") from None
