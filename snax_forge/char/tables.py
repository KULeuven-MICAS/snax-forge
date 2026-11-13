"""HWCOST's tables: the streamers, the TCDM crossbar and the L1 banks.

    load_costs(tech) -> Costs | None                       snax_forge/char/<tech>/, None without
    streamer_cost(costs, depth, loops, channels) -> Cost   one reader (a writer priced as one)
    xbar_cost(costs, inputs, banks, read_only, write_only, form) -> Cost    the crossbar
    bank(costs, depth, width) -> Bank | None               one SRAM bank, a lookup
    Grid.at(field, **q) -> (value, extrapolated axes)      the rule both interpolated tables use

The four files of ``<tech>/`` are copied from snax-forge-experiments by ``pixi run
import-costs``, their commit and sha256 in ``<tech>/SOURCE``. The tables, the interpolation and
extrapolation rule, the crossbar's two forms (``XBAR_FORMS``, factor the default) and the banks'
lookup are in CONTRACTS, "Cost model"; char/README.md walks through them.
"""

from __future__ import annotations

import bisect
import csv
import functools
import itertools
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
LOG = logging.getLogger(__name__)
FILES = ("ram_table.csv", "streamer.csv", "xbar.csv", "xbar_kinds.json")
XBAR_FORMS = ("factor", "crosspoint")  # X71: the default first
READER_AXES = ("depth", "loops", "channels")
XBAR_AXES = ("inputs", "banks")
SCALES: dict[str, Callable[[float], float]] = {"lin": lambda x: x, "inv": lambda x: 1.0 / x}
XBAR_SCALES = {"e_access_pj": {"inputs": "inv"}}  # X54


@dataclass(frozen=True)
class Grid:
    """A rectilinear grid: ``axes`` (name -> sorted grid values), per field {coordinates:
    value}, and per field the scale of an axis (``"lin"`` unless ``scales[field][axis]``)."""

    axes: dict[str, tuple[int, ...]]
    fields: dict[str, dict[tuple[int, ...], float]]
    scales: dict[str, dict[str, str]] = field(default_factory=dict)

    def at(self, name: str, **q: float) -> tuple[float, tuple[str, ...]]:
        """``name`` at ``q`` (one value per axis) and the axes it was extrapolated along
        (module doc); a ``KeyError`` when a corner the query needs has no value."""
        if set(q) != set(self.axes):
            raise ValueError(f"query {q}: give exactly {list(self.axes)}")
        spans, outside = [], []
        for axis, vals in self.axes.items():
            x = q[axis]
            i = bisect.bisect_left(vals, x)
            if i < len(vals) and vals[i] == x:
                spans.append(((vals[i], 1.0),))
                continue
            if x < vals[0]:
                lo, hi = vals[0], vals[1]
                outside.append(axis)
            elif x > vals[-1]:
                lo, hi = vals[-2], vals[-1]
                outside.append(axis)
            else:
                lo, hi = vals[i - 1], vals[i]
            g = SCALES[self.scales.get(name, {}).get(axis, "lin")]
            w = (g(x) - g(lo)) / (g(hi) - g(lo))
            spans.append(((lo, 1.0 - w), (hi, w)))
        total = 0.0
        for corner in itertools.product(*spans):
            key = tuple(c for c, _ in corner)
            weight = 1.0
            for _, w in corner:
                weight *= w
            v = self.fields[name].get(key)
            if v is None:
                raise KeyError(f"{name} has no value at {dict(zip(self.axes, key, strict=True))}")
            total += weight * v
        return (max(total, 0.0), tuple(outside)) if outside else (total, ())


@dataclass(frozen=True)
class Cost:
    """An area (µm²) and an energy (pJ per beat for a streamer, per access for the crossbar),
    and the axes the query lay outside the grid on (empty when inside)."""

    area_um2: float
    energy_pj: float
    extrapolated: tuple[str, ...] = ()


@dataclass(frozen=True)
class Bank:
    """One SRAM bank of ``depth`` words of ``width`` bits: its area and energy per access."""

    depth: int
    width: int
    area_um2: float
    e_rd_pj: float
    e_wr_pj: float


@dataclass(frozen=True)
class XbarKinds:
    """``xbar_kinds.json``'s two forms (module doc): the factor form's ``f_ro`` and ``f_wo``, the
    crosspoint form's coefficients (µm²) and the grid they were fitted on."""

    f_ro: float
    f_wo: float
    c_rw: float
    c_ro: float
    c_wo: float
    d: float
    e: float
    g: float
    inputs: tuple[int, ...]
    banks: tuple[int, ...]


@dataclass(frozen=True)
class Costs:
    """One technology's tables and where they came from (``SOURCE``'s fields)."""

    tech: str
    source: dict[str, str]
    streamer: Grid
    xbar: Grid
    banks: dict[tuple[int, int], Bank]
    kinds: XbarKinds


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _grid(rows: list[dict[str, str]], axes: tuple[str, ...], names: tuple[str, ...],
          scales: Mapping[str, dict[str, str]] | None = None) -> Grid:  # fmt: skip
    """A table's rows as a Grid; a row whose status is not ok gives no values."""
    ax = {a: tuple(sorted({int(r[a]) for r in rows})) for a in axes}
    fv: dict[str, dict[tuple[int, ...], float]] = {n: {} for n in names}
    for r in rows:
        if r.get("status", "ok") != "ok":
            continue
        key = tuple(int(r[a]) for a in axes)
        for n in names:
            if r[n] not in ("", None):
                fv[n][key] = float(r[n])
    return Grid(ax, fv, dict(scales or {}))


def _complete_xbar(g: Grid) -> Grid:
    """The crossbar's energy where inputs > banks: each bank count's value at inputs = banks."""
    e = dict(g.fields["e_access_pj"])
    for i, b in itertools.product(g.axes["inputs"], g.axes["banks"]):
        if i > b and (i, b) not in e and (b, b) in e:
            e[(i, b)] = e[(b, b)]
    return Grid(g.axes, {**g.fields, "e_access_pj": e}, g.scales)


def read_kinds(path: Path) -> XbarKinds:
    """``xbar_kinds.json`` (report/xbar_kinds.py in snax-forge-experiments, X71)."""
    d = json.loads(path.read_text())
    f, c = d["factor"], d["crosspoint"]
    return XbarKinds(f["f_ro"], f["f_wo"], c["c_rw"], c["c_ro"], c["c_wo"], c["d"], c["e"], c["g"],
                     tuple(c["grid"]["inputs"]), tuple(c["grid"]["banks"]))  # fmt: skip


def read_source(path: Path) -> dict[str, str]:
    """``SOURCE``'s ``key: value`` lines (``#`` lines are comments)."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


@functools.cache
def load_costs(tech: str) -> Costs | None:
    """The tables of ``tech`` (``snax_forge/char/<tech>/``), None when it has none."""
    d = HERE / tech
    if not all((d / f).is_file() for f in FILES):
        return None
    banks = {(int(r["depth"]), int(r["width"])):
             Bank(int(r["depth"]), int(r["width"]), float(r["area_um2"]), float(r["e_rd_pj"]),
                  float(r["e_wr_pj"]))
             for r in _rows(d / "ram_table.csv")}  # fmt: skip
    streamer = _grid(_rows(d / "streamer.csv"), READER_AXES, ("area_um2", "e_beat_pj"))
    xbar = _complete_xbar(_grid(_rows(d / "xbar.csv"), XBAR_AXES, ("area_um2", "e_access_pj"),
                                XBAR_SCALES))  # fmt: skip
    return Costs(tech, read_source(d / "SOURCE"), streamer, xbar, banks,
                 read_kinds(d / "xbar_kinds.json"))  # fmt: skip


def _warn(what: str, q: Mapping[str, float], axes: tuple[str, ...]) -> None:
    if axes:
        LOG.warning("%s at %s: outside the table's grid on %s, extrapolated",
                    what, ", ".join(f"{k} {v}" for k, v in q.items()), ", ".join(axes))  # fmt: skip


def streamer_cost(costs: Costs, depth: int, loops: int, channels: int) -> Cost:
    """One reader's (or writer's, X52) area and energy per beat."""
    q = {"depth": depth, "loops": loops, "channels": channels}
    area, out_a = costs.streamer.at("area_um2", **q)
    e, out_e = costs.streamer.at("e_beat_pj", **q)
    out = tuple(dict.fromkeys(out_a + out_e))
    _warn("streamer", q, out)
    return Cost(area, e, out)


def xbar_cost(costs: Costs, inputs: int, banks: int, read_only: int = 0, write_only: int = 0,
              form: str = "factor") -> Cost:  # fmt: skip
    """The crossbar's area and energy per access, ``read_only`` and ``write_only`` of its
    ``inputs`` priced as such in ``form`` (module doc); with none, the table's area."""
    if form not in XBAR_FORMS:
        raise ValueError(f"crossbar form {form!r}: one of {', '.join(XBAR_FORMS)}")
    if read_only < 0 or write_only < 0 or read_only + write_only > inputs:
        raise ValueError(f"{read_only} read-only and {write_only} write-only of {inputs} inputs")
    q = {"inputs": inputs, "banks": banks}
    e, out_e = costs.xbar.at("e_access_pj", **q)
    k = costs.kinds
    if form == "factor":
        table, out_a = costs.xbar.at("area_um2", **q)
        area = table * (1 - k.f_ro * read_only / inputs - k.f_wo * write_only / inputs)
    else:
        rw = inputs - read_only - write_only
        area = max(0.0, banks * (k.c_rw * rw + k.c_ro * read_only + k.c_wo * write_only)
                   + k.d * inputs + k.e * banks + k.g)  # fmt: skip
        out_a = tuple(a for a, v in q.items()
                      if not min(getattr(k, a)) <= v <= max(getattr(k, a)))  # fmt: skip
    out = tuple(dict.fromkeys(out_a + out_e))
    _warn("crossbar", q, out)
    return Cost(area, e, out)


def bank(costs: Costs, depth: int, width: int) -> Bank | None:
    """The bank of ``depth`` × ``width`` bits, None off the table's grid (never interpolated)."""
    return costs.banks.get((depth, width))
