"""SNAX-CHAR: the per-technology table of CHAR1's cost model (D154, C§20).

    load(tech) -> Table                                          snax_forge/char/<tech>.json
    lookup(table, datapath, params, lanes, depth=None) -> Lookup | None
    TECHS                                                        the technologies held here
    L1_ACCESS_LOADS, CTRL_COMB_PER_FLOP                          D154's two placeholders

``<tech>.json`` is a byte copy of snax-forge-experiments' ``results/char/<tech>/char.json``
(its X37), which names its own source commits: per catalogue datapath (``hw/chisel/blocks.json``)
entries of ``params`` (the catalogue's but ``lanes`` and ``depth``), a ``lanes_rule`` and the
measured ``points`` (``lanes``, LinkBuffer's ``depth``, ``comb_um2`` and ``e_comb_pj``: a fire's
comb energy), and the technology's ``flop`` (``area_um2``, ``e_load_fj``, ``e_hold_fj``) and
``ctrl.counter_factor``. ``tests/char/`` checks it against the catalogue.

**The lookup** of a block, its params as the generator fills them (``lanes`` and ``depth`` in
them are ignored) and its lanes (and depth): the entry whose params equal them, then

    measured      a point at these lanes (and depth), as is
    copy          the rule ``copy``: lanes × the lanes-1 point (LinkBuffer: at the same depth)
    interpolate   the rule ``interpolate``: linear between the two measured lanes around it;
                  a LinkBuffer's depth, when no point has it, is linear between the two
                  measured depths around it (at lanes 1, then copied), and found so

and None for anything else: no entry with these params (a dtype other than int64, another op),
lanes or depth outside the measured range (TreeAccumulate at lanes 64, SerialAccumulate at
lanes 1, a LinkBuffer deeper than 16). None is never a refusal: the estimate counts the block as
0 and lists it (C§19), and the design note ``char.missing`` names it (C§14).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from snax_forge.common import CodedError

HERE = Path(__file__).parent
TECHS = ("nangate45", "nangate15")
DEFAULT_TECH = "nangate45"
FOUND = ("measured", "copy", "interpolate")
# CHAR1's two placeholders until the streamers and OpenRAM's memories are characterized (D154;
# HWCOST, RAM1): a word a streamer reads or writes in L1 costs L1_ACCESS_LOADS × e_load per bit,
# and a control unit's comb area is CTRL_COMB_PER_FLOP × its flops' area (its comb energy taken
# as negligible). Neither is in the table, a byte copy of snax-forge-experiments' char.json.
L1_ACCESS_LOADS = 4
CTRL_COMB_PER_FLOP = 1.0
_IGNORED = ("lanes", "depth")


class CharError(CodedError):
    """A technology the tables do not hold; ``code`` ``char.<rule>``."""


@dataclass(frozen=True)
class Table:
    """One technology's table (module doc); energies in pJ, areas in µm²."""

    tech: str
    clock_ns: float
    source: dict[str, str]
    flop_um2: float
    e_load_pj: float
    e_hold_pj: float
    counter_factor: float
    blocks: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class Lookup:
    """A block's comb area and comb energy per fire, and how it was found (``FOUND``)."""

    comb_um2: float
    e_comb_pj: float
    found: str


@functools.cache
def load(tech: str) -> Table:
    """The table of ``tech`` (``snax_forge/char/<tech>.json``)."""
    path = HERE / f"{tech}.json"
    if tech not in TECHS or not path.is_file():
        raise CharError("char.tech", f"no table for technology {tech!r}: one of {list(TECHS)}")
    d = json.loads(path.read_text())
    return Table(d["tech"], d["clock_ns"], dict(d["source"]), d["flop"]["area_um2"],
                 d["flop"]["e_load_fj"] / 1000, d["flop"]["e_hold_fj"] / 1000,
                 d["ctrl"]["counter_factor"], d["blocks"])  # fmt: skip


def _between(x: int, pts: list[tuple[int, Lookup]]) -> Lookup | None:
    """Linear in ``x`` between the two points around it (``pts`` by key); None outside."""
    pts = sorted(pts, key=lambda p: p[0])
    for (a, pa), (b, pb) in pairwise(pts):
        if a < x < b:
            t = (x - a) / (b - a)
            return Lookup(pa.comb_um2 + t * (pb.comb_um2 - pa.comb_um2),
                          pa.e_comb_pj + t * (pb.e_comb_pj - pa.e_comb_pj), "interpolate")  # fmt: skip
    return None


def _at_depth(points: list[dict[str, Any]], lanes: int, depth: int | None) -> Lookup | None:
    """The point at ``lanes`` and ``depth``, measured or (a depth no point has) interpolated
    in depth at those lanes."""
    at = [p for p in points if p["lanes"] == lanes]
    for p in at:
        if p.get("depth") == depth:
            return Lookup(p["comb_um2"], p["e_comb_pj"], "measured")
    if depth is None:
        return None
    return _between(depth, [(p["depth"], Lookup(p["comb_um2"], p["e_comb_pj"], "measured"))
                            for p in at])  # fmt: skip


def lookup(table: Table, datapath: str, params: Mapping[str, Any], lanes: int,
           depth: int | None = None) -> Lookup | None:  # fmt: skip
    """A block's comb area and energy per fire on ``table`` (module doc), or None."""
    want = {k: v for k, v in params.items() if k not in _IGNORED}
    entry = next((e for e in table.blocks.get(datapath, []) if e["params"] == want), None)
    if entry is None:
        return None
    points = entry["points"]
    got = _at_depth(points, lanes, depth)
    if got is not None:
        return got
    if entry["lanes_rule"] == "copy":
        one = _at_depth(points, 1, depth)
        if one is None:
            return None
        found = "interpolate" if one.found == "interpolate" else "copy"
        return Lookup(lanes * one.comb_um2, lanes * one.e_comb_pj, found)
    if entry["lanes_rule"] == "interpolate" and depth is None:
        return _between(lanes, [(p["lanes"], Lookup(p["comb_um2"], p["e_comb_pj"], "measured"))
                                for p in points])  # fmt: skip
    return None


__all__ = [
    "CTRL_COMB_PER_FLOP",
    "DEFAULT_TECH",
    "FOUND",
    "L1_ACCESS_LOADS",
    "TECHS",
    "CharError",
    "Lookup",
    "Table",
    "load",
    "lookup",
]
