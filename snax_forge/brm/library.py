"""The BRM library: hand-written BRMs, one JSON file each (D68).

``library/<name>.json`` holds the BRM called ``name``; the file name and the
BRM's ``name`` must agree. Files are kept in the form ``Brm.to_json``
writes, every field included, so a hand edit that leaves that form shows up
in tests/brm/test_library.py. Generating BRMs from RTL sources is open item
27; until then every file here is written by hand.

Library BRMs:

    simd              out = a <op> b over W lanes (W = 4 by default), op add
                      (``out = a + b``) or mul (``out = a * b``), one code per
                      op (a ``Code`` table, bind picks the op from the
                      tasklet's code); one Chisel implementation, the
                      datapath SimdLanes (L = 0, II = 1): vecadd's adder,
                      dot's multiplier and vecmul's
    accumulate        the sum of every lane of T beats of ``a``, one element
                      of ``out`` per T beats (pattern family ``reduce``):
                      chisel_adder_tree, the Chisel datapath TreeAccumulate
                      (L = 1, II = 1, drain 0, D133) at W lanes, summed by its
                      tree and accumulated (vsum's at W = 1); chisel_serial,
                      the Chisel datapath SerialAccumulate (L = 1, span W:
                      lane k in cycle k), every recipe named serial
    mac               out = a * b folded per lane over T beats (pattern family
                      ``lane_fold``, a fused nest; D149): the Chisel datapath
                      Mac with accumulate (L = 1, II = 1, span 1)
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
