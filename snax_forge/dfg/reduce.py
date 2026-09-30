"""Reductions on a memlet (DFG3, D102): how many writes to one element combine.

A tasklet output whose memlet has a ``reduce`` does not overwrite its element:
every iteration of the maps around it folds its value in with the op. The
element is set to ``identity`` when the node starts, so the node leaves the
fold of exactly its own writes, whatever the element held before. That is
what a DaCe ``Reduce`` means, and what an accumulator does in hardware
(it starts from zero and writes the sum). DaCe calls it write-conflict
resolution (``wcr``); MLIR a reduction iterator with a combiner and an
initial fill. The SNAX-DFG names it after neither, so any importer writes
the same field:

    {"data": "out", "subset": [0], "reduce": {"op": "add", "identity": 0}}

The op is a name registered here (``register_reduction``, principle 6) with the
NumPy ufunc that folds it and the identity it has for a dtype. Every
registered op is associative and commutative on the integers it runs on
(D28), so the order the writes land in never changes the result: a map
with a ``reduce`` memlet can be split, tiled or run in parallel as any other.
A float op, whose result depends on the order, needs the order its BRM
defines and is not registered here.

A fold into what the element already holds (DaCe's plain ``wcr`` memlet,
``C[i] += ...`` in a loop) would be ``identity: null``; it is not accepted
yet (FE1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .kinds import DfgError


@dataclass(frozen=True)
class ReduceOp:
    """A registered op: the ufunc that folds it and its identity for an integer dtype."""

    ufunc: np.ufunc
    identity: Callable[[np.dtype], int]


REDUCE_OPS: dict[str, ReduceOp] = {}


def register_reduction(name: str, ufunc: np.ufunc, identity: Callable[[np.dtype], int]) -> None:
    """Make ``"reduce": {"op": name, ...}`` usable; the op must be associative and commutative."""
    if name in REDUCE_OPS:
        raise ValueError(f"reduce op {name!r} already registered")
    REDUCE_OPS[name] = ReduceOp(ufunc, identity)


register_reduction("add", np.add, lambda dt: 0)
register_reduction("mul", np.multiply, lambda dt: 1)
register_reduction("min", np.minimum, lambda dt: int(np.iinfo(dt).max))
register_reduction("max", np.maximum, lambda dt: int(np.iinfo(dt).min))


@dataclass(frozen=True)
class Reduction:
    """The ``reduce`` of a memlet: a registered op and the value the element starts from."""

    op: str
    identity: int

    def __post_init__(self) -> None:
        if self.op not in REDUCE_OPS:
            raise DfgError(f"reduce.op: {self.op!r} is not a registered op {sorted(REDUCE_OPS)}")
        if self.identity is None:
            raise DfgError(
                "reduce.identity: null (fold into the element's contents) is not accepted yet (FE1)"
            )
        if type(self.identity) is not int:
            raise DfgError(f"reduce.identity: must be an int, got {self.identity!r}")

    @property
    def ufunc(self) -> np.ufunc:
        return REDUCE_OPS[self.op].ufunc

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "identity": self.identity}

    @classmethod
    def from_dict(cls, d: Any, what: str) -> Reduction:
        if not isinstance(d, Mapping) or set(d) != {"op", "identity"}:
            raise DfgError(f"{what}: must be {{op, identity}}, got {d!r}")
        try:
            return cls(d["op"], d["identity"])
        except DfgError as e:
            raise DfgError(f"{what}.{e}") from None
