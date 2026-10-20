"""The BRM library: hand-written BRMs, one JSON file each (D68).

``library/<name>.json`` holds the BRM called ``name``; the file name and the
BRM's ``name`` must agree. Files are kept in the form ``Brm.to_json``
writes, every field included, so a hand edit that leaves that form shows up
in tests/brm/test_library.py. Generating BRMs from RTL sources is open item
27; until then every file here is written by hand.

Library BRMs:

    elementwise_add   c = a + b over W lanes (W = 4 by default); one Chisel
                      implementation, ElementwiseTiledSpatial (L = 0, II = 1)
    elementwise_mul   c = a * b, the same with op mul (ElementwiseTiledSpatial
                      supports it, bound as elementwise_add),
                      dot's multiplier and vecmul's
    accumulate        the sum of every lane of T beats of ``a``, one element
                      of ``out`` per T beats (pattern family ``reduce``):
                      the Chisel Accumulator (L = 1, II = 1, drain 0,
                      D133) twice: chisel_accumulator at 1 lane (vsum's)
                      and chisel_adder_tree at W lanes, summed by its tree
                      and accumulated
"""

from __future__ import annotations

from pathlib import Path

from .brm import Brm, BrmError

LIBRARY = Path(__file__).resolve().parent / "library"


def library_names() -> list[str]:
    """The names of every BRM in the library, sorted."""
    return sorted(p.stem for p in LIBRARY.glob("*.json"))


def load_brm(name: str) -> Brm:
    """The library BRM called ``name``."""
    path = LIBRARY / f"{name}.json"
    if not path.is_file():
        raise BrmError(f"no BRM {name!r} in the library ({library_names()})")
    brm = Brm.load(path)
    if brm.name != name:
        raise BrmError(f"{path.name}: holds BRM {brm.name!r}, the file name says {name!r}")
    return brm
