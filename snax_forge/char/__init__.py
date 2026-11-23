"""SNAX-CHAR: the per-technology table of the block cost model.

    load(tech) -> Table                                          snax_forge/char/<tech>.json
    lookup(table, datapath, params, lanes, depth=None) -> Lookup | None
    TECHS                                                        the technologies offered
    CTRL_COMB_PER_FLOP                                           a control unit's comb : flops

The table, its lookup (``FOUND``) and what None means are in CONTRACTS, "Cost model". A block is
looked up at its width pair (D176, ``pair_params``): its widest input and its output, a
SimdLanes' output at most its op's width, an unsigned type as the signed one of its width; a
packed output's packer by its lane type and d_out, a sliced input's slicer by its lane type, lanes
and d_in. A param the fill leaves None is not compared.
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
# The technologies offered: nangate45 only (snax-forge-experiments synthesizes nothing else).
TECHS = ("nangate45",)
DEFAULT_TECH = "nangate45"
FOUND = ("measured", "copy", "interpolate", "slope", "width", "unseeded")
# A control unit's comb area is CTRL_COMB_PER_FLOP × its flops' area, a rough early estimate (its
# comb energy taken as negligible); not in the table, a byte copy of snax-forge-experiments'
# char.json. The L1 side is tables.py's tables.
CTRL_COMB_PER_FLOP = 1.0
_IGNORED = ("lanes", "depth")
WORD_BITS = 64


class CharError(CodedError):
    """A technology the tables do not hold; ``code`` ``char.<rule>``."""


@dataclass(frozen=True)
class Table:
    """One technology's table (CONTRACTS, "Cost model"); energies in pJ, areas in µm².
    ``e_load_pj`` is a flop bit's energy in a cycle it loads; ``e_ctrl_pj`` a control unit's per
    bit and cycle, the table's ``counter_factor`` × ``e_hold_fj``."""

    tech: str
    clock_ns: float
    source: dict[str, str]
    flop_um2: float
    e_load_pj: float
    e_ctrl_pj: float
    blocks: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class Lookup:
    """A block's comb area and comb energy per fire, and how it was found (``FOUND``);
    ``load_bits`` the bits that load on a fire where not all of them do (a packer's or a
    slicer's), else None."""

    comb_um2: float
    e_comb_pj: float
    found: str
    load_bits: float | None = None


@functools.cache
def load(tech: str) -> Table:
    """The table of ``tech`` (``snax_forge/char/<tech>.json``)."""
    path = HERE / f"{tech}.json"
    if tech not in TECHS or not path.is_file():
        raise CharError("char.tech", f"no table for technology {tech!r}: one of {list(TECHS)}")
    d = json.loads(path.read_text())
    return Table(d["tech"], d["clock_ns"], dict(d["source"]), d["flop"]["area_um2"],
                 d["flop"]["e_load_fj"] / 1000,
                 d["ctrl"]["counter_factor"] * d["flop"]["e_hold_fj"] / 1000, d["blocks"])  # fmt: skip


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


def _width(dtype: str) -> int:
    return int(dtype.removeprefix("u").removeprefix("int"))


def _signed(w: int) -> str:
    return f"int{w}"


def pair_params(datapath: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """``params`` as the table keys them (D176): lanes, depth and None values dropped; a
    SimdLanes' and a Mac's inputs at their widest input, the op and output at the pair's output
    (a SimdLanes' the smaller of its op and output: the low bits of a sum or product need only
    the operands' low bits, a wider output only copies the sign); the folds' and a LinkBuffer's
    types signed; a packer's ``d_out``, ``lanes`` and ``dtype`` as its ``inLanes``,
    ``laneType`` and ``words``, a slicer's ``d_in``, ``lanes`` and ``dtype`` as its ``words``,
    ``laneType`` and ``outLanes``."""
    p = {k: v for k, v in params.items() if k not in _IGNORED and v is not None}
    if datapath == "FifoPacker" and "d_out" in p:
        w = _width(p["dtype"])
        return {"inLanes": params["lanes"], "laneType": _signed(w),
                "words": p["d_out"] * params["lanes"] * w // WORD_BITS}  # fmt: skip
    if datapath == "FifoSlicer" and "d_in" in p:
        w = _width(p["dtype"])
        return {"words": p["d_in"] * params["lanes"] * w // WORD_BITS, "laneType": _signed(w),
                "outLanes": params["lanes"]}  # fmt: skip
    if datapath in ("SimdLanes", "Mac") and "inTypes" in p and "outType" in p:
        ins = list(p["inTypes"])
        w_in, w_out = max(_width(t) for t in ins), _width(p["outType"])
        if datapath == "SimdLanes" and "opType" in p:
            w_out = min(w_out, _width(p["opType"]))
        p["inTypes"] = [_signed(w_in)] * len(ins)
        p["opType"] = p["outType"] = _signed(w_out)
    for k in ("inType", "outType", "dtype", "laneType"):
        if k in p and isinstance(p[k], str):
            p[k] = _signed(_width(p[k]))
    return p


def _scaled(got: Lookup | None, k: float, found: str | None = None) -> Lookup | None:
    if got is None:
        return None
    return Lookup(k * got.comb_um2, k * got.e_comb_pj, found or got.found, got.load_bits)


def _over_lanes(entry: dict[str, Any], lanes: int, depth: int | None) -> Lookup | None:
    """An entry's point at ``lanes`` (and ``depth``): measured, else its ``lanes_rule``."""
    points = entry["points"]
    got = _at_depth(points, lanes, depth)
    if got is not None:
        return got
    rule = entry["lanes_rule"]
    if rule == "copy":
        w0 = min(p["lanes"] for p in points)
        one = _at_depth(points, w0, depth)
        if one is None:
            return None
        found = "interpolate" if one.found == "interpolate" else "copy"
        return _scaled(one, lanes / w0, found)
    if rule == "interpolate" and depth is None:
        return _between(lanes, [(p["lanes"], Lookup(p["comb_um2"], p["e_comb_pj"], "measured"))
                                for p in points])  # fmt: skip
    if rule == "slope" and depth is None:
        near = min(points, key=lambda p: (abs(p["lanes"] - lanes), p["lanes"]))
        d, s = lanes - near["lanes"], entry["slope"]
        return Lookup(near["comb_um2"] + d * s["comb_um2"], near["e_comb_pj"] + d * s["e_comb_pj"],
                      "slope")  # fmt: skip
    return None


def lookup(table: Table, datapath: str, params: Mapping[str, Any], lanes: int,
           depth: int | None = None) -> Lookup | None:  # fmt: skip
    """A block's comb area and energy per fire on ``table`` (CONTRACTS, "Cost model"), or None."""
    want = pair_params(datapath, params)
    entries = table.blocks.get(datapath, [])

    def key(e: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in e["params"].items() if v is not None}

    keyed = any("seed" in e["params"] for e in entries)
    if want.get("seed") is False and not keyed:  # measured before it took a seed (a Mac, D187)
        want = {k: v for k, v in want.items() if k != "seed"}
    if "rows" in want and not any("rows" in e["params"] for e in entries):
        # NB (D191): measured before it had a bank; a bank's comb is its one row's, a lower bound
        want = {k: v for k, v in want.items() if k != "rows"}
    entry = next((e for e in entries if key(e) == want), None)
    if datapath in ("FifoPacker", "FifoSlicer"):
        if entry is None:
            return None
        p = entry["points"][0]
        return Lookup(p["comb_um2"], p["e_comb_pj"], "measured", p["load_bits"])
    if entry is None and datapath == "LinkBuffer" and "dtype" in want:
        # a narrower dtype: the int64 entry by the width (D176)
        k64 = {**want, "dtype": "int64"}
        base = next((e for e in entries if key(e) == k64), None)
        if base is None:
            return None
        return _scaled(_over_lanes(base, lanes, depth), _width(want["dtype"]) / 64, "width")
    if entry is None and want.get("seed") is True:
        # a seeded fold: the unseeded entry, the same adder and register, the first beat's mux
        # choosing the seed for the register instead of 0 for the sum; a block measured before
        # it took a seed (a Mac, D187) has no seed key in any entry
        unseeded = {**want, "seed": False} if keyed else {k: v for k, v in want.items()
                                                          if k != "seed"}  # fmt: skip
        base = next((e for e in entries if key(e) == unseeded), None)
        if base is None:
            return None
        return _scaled(_over_lanes(base, lanes, depth), 1.0, "unseeded")
    if entry is None:
        return None
    return _over_lanes(entry, lanes, depth)


__all__ = [
    "CTRL_COMB_PER_FLOP",
    "DEFAULT_TECH",
    "FOUND",
    "TECHS",
    "CharError",
    "Lookup",
    "Table",
    "load",
    "lookup",
    "pair_params",
]
