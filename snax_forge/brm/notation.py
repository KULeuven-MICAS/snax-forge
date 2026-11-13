"""Dataflow notations: how a BRM writes its per-port nests (CONTRACTS, "Block runtime model").

A BRM's ``dataflow`` names a notation and gives one nest per port. A nest holds no addresses:
SNAX-LOWER maps it through the layout the design point chooses. A notation is registered with
``register_notation(name, check, check_instance=None, normalize=None)``: ``check(nest, port,
brm)`` raises ValueError or TypeError for a nest not valid on that port, before any param has a
value; ``check_instance(nest, port, instance)`` does the same once the design params have
values; ``normalize(nest)`` returns it with every field written. Two are registered: ``affine``
(affine.py) and ``stream`` (stream.py), whose nest is empty.
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
