"""Write-conflict resolution on a memlet (DFG3, D102): how many writes to one element combine.

A tasklet output whose memlet has a ``wcr`` does not overwrite its element:
every iteration of the maps around it folds its value in with the op. The
element is set to ``identity`` when the node starts, so the node leaves the
fold of exactly its own writes, whatever the element held before. That is
what a DaCe ``Reduce`` means, and what an accumulator does in hardware
(it starts from zero and writes the sum):

    {"data": "out", "subset": [0], "wcr": {"op": "add", "identity": 0}}

The op is a name registered here (``register_wcr``, principle 6) with the
NumPy ufunc that folds it and the identity it has for a dtype. Every
registered op is associative and commutative on the integers it runs on
(D28), so the order the writes land in never changes the result: a map
with a ``wcr`` memlet can be split, tiled or run in parallel as any other.
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
class WcrOp:
    """A registered op: the ufunc that folds it and its identity for an integer dtype."""

    ufunc: np.ufunc
    identity: Callable[[np.dtype], int]


WCR_OPS: dict[str, WcrOp] = {}


def register_wcr(name: str, ufunc: np.ufunc, identity: Callable[[np.dtype], int]) -> None:
    """Make ``"wcr": {"op": name, ...}`` usable; the op must be associative and commutative."""
    if name in WCR_OPS:
        raise ValueError(f"wcr op {name!r} already registered")
    WCR_OPS[name] = WcrOp(ufunc, identity)


register_wcr("add", np.add, lambda dt: 0)
register_wcr("mul", np.multiply, lambda dt: 1)
register_wcr("min", np.minimum, lambda dt: int(np.iinfo(dt).max))
register_wcr("max", np.maximum, lambda dt: int(np.iinfo(dt).min))


@dataclass(frozen=True)
class Wcr:
    """The ``wcr`` of a memlet: a registered op and the value the element starts from."""

    op: str
    identity: int

    def __post_init__(self) -> None:
        if self.op not in WCR_OPS:
            raise DfgError(f"wcr.op: {self.op!r} is not a registered op {sorted(WCR_OPS)}")
        if self.identity is None:
            raise DfgError(
                "wcr.identity: null (fold into the element's contents) is not accepted yet (FE1)"
            )
        if type(self.identity) is not int:
            raise DfgError(f"wcr.identity: must be an int, got {self.identity!r}")

    @property
    def ufunc(self) -> np.ufunc:
        return WCR_OPS[self.op].ufunc

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "identity": self.identity}

    @classmethod
    def from_dict(cls, d: Any, what: str) -> Wcr:
        if not isinstance(d, Mapping) or set(d) != {"op", "identity"}:
            raise DfgError(f"{what}: must be {{op, identity}}, got {d!r}")
        try:
            return cls(d["op"], d["identity"])
        except DfgError as e:
            raise DfgError(f"{what}.{e}") from None
