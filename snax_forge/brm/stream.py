"""The stream dataflow notation (D70): a unit BRM's ports, ordered by the bound memlet.

A unit block (one function applied lane by lane, beat by beat) takes its
elements in whatever order the streamers bring them; the order is the bound
memlet's, the one the reference executor gathers each beat from. So a port
in this notation says nothing beyond the interface: its lanes, rate and
dtype are the interface's, and its nest is empty:

    {"notation": "stream", "ports": {"a": {}, "b": {}, "out": {}}}

Where a BRM uses it, ``bind`` checks no nest order (every other check
stays), SNAX-LOWER takes the streamer's values from the memlet alone, and
the design step takes a port's spatial bounds from the bound memlet's
ranges. The check refuses any key in a nest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .notation import register_notation

if TYPE_CHECKING:
    from .brm import Brm, Port

NAME = "stream"


def check(nest: Any, port: Port, brm: Brm) -> None:
    if not isinstance(nest, dict):
        raise TypeError(f"a stream nest is an empty object, got {nest!r}")
    if nest:
        raise ValueError(
            f"a stream nest has no keys (the bound memlet gives the order), got {sorted(nest)}"
        )


def memlet_ordered(brm: Brm) -> bool:
    """The BRM's ports take their order from the bound memlet (``stream``), not a nest."""
    return brm.dataflow.notation == NAME


register_notation(NAME, check)
