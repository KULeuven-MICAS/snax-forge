"""Dataflow notations: how a BRM writes its per-port nests (D68, open item 1).

The dataflow part of a BRM names a notation and gives one nest per port:

    {"notation": "affine", "ports": {"a": nest, "b": nest, "out": nest}}

A nest describes only the accelerator: the order in which the port consumes
or produces its operand's elements, in logical indices. It holds no
addresses; SNAX-LOWER maps it through the buffer layout the design point
chooses (section 5.5).

What a nest looks like is the notation's business. A notation is registered
by name with ``register_notation(name, check)``; ``check(nest, port, brm)``
raises ValueError if the nest is not valid for that port of that BRM. The
first notation, ``affine``, comes with BRM2; M6 may add a second one without
changing the BRM structure (open item 1).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .brm import Brm, Port

NotationCheck = Callable[[Any, "Port", "Brm"], None]
NOTATIONS: dict[str, NotationCheck] = {}


def register_notation(name: str, check: NotationCheck) -> None:
    """Make ``"notation": name`` usable in a BRM's dataflow part."""
    if name in NOTATIONS:
        raise ValueError(f"notation {name!r} already registered")
    NOTATIONS[name] = check
