"""Dataflow notations: how a BRM writes its per-port nests (D68, D70, open item 1).

The dataflow part of a BRM names a notation and gives one nest per port:

    {"notation": "affine", "ports": {"a": nest, "b": nest, "out": nest}}

A nest describes only the accelerator: the order in which the port consumes
or produces its operand's elements, in logical indices. It holds no
addresses; SNAX-LOWER maps it through the buffer layout the design point
chooses (section 5.5). Whether the ports agree on element positions (lane j
of beat t on every port is the element the function expects) is the BRM
author's responsibility (D70).

What a nest looks like is the notation's business. A notation is registered
by name with ``register_notation(name, check, check_instance=None)``:

* ``check(nest, port, brm)`` raises ValueError (or TypeError) if the nest is not valid for
  that port of that BRM, before any param has a value;
* ``check_instance(nest, port, instance)``, optional, does the same once the
  design params have values (``Brm.resolve``);
* ``normalize(nest)``, optional, returns the nest with every field written
  (defaults filled in), which the BRM keeps once the nest is checked.

The first notation, ``affine``, is in affine.py (D70); M6 may add a second
one without changing the BRM structure (open item 1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .brm import Brm, Port
    from .instance import Instance

NotationCheck = Callable[[Any, "Port", "Brm"], None]
InstanceCheck = Callable[[Any, "Port", "Instance"], None]
Normalize = Callable[[Any], Any]


@dataclass(frozen=True)
class Notation:
    check: NotationCheck
    check_instance: InstanceCheck | None = None
    normalize: Normalize | None = None


NOTATIONS: dict[str, Notation] = {}


def register_notation(
    name: str,
    check: NotationCheck,
    check_instance: InstanceCheck | None = None,
    normalize: Normalize | None = None,
) -> None:
    """Make ``"notation": name`` usable in a BRM's dataflow part."""
    if name in NOTATIONS:
        raise ValueError(f"notation {name!r} already registered")
    NOTATIONS[name] = Notation(check, check_instance, normalize)
