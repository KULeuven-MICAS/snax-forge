"""The BRM library: hand-written BRMs, one JSON file each (BRM3, D68).

``library/<name>.json`` holds the BRM called ``name``; the file name and the
BRM's ``name`` must agree. Files are kept in the form ``Brm.to_json``
writes, every field included, so a hand edit that leaves that form shows up
in tests/brm/test_library.py. Generating BRMs from RTL sources is open item
27; until then every file here is written by hand.

Library BRMs so far:

    elementwise_add   c = a + b over W lanes (W = 4 by default); one Chisel
                      implementation, ElementwiseTiledSpatial (L = 0, II = 1)
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
