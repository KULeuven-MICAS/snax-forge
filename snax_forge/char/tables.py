"""The cost tables: the streamers, the TCDM crossbar, the L1 banks, the cluster's fixed parts, the
DMA engine and the terms fitted at gate level.

    load_costs(tech) -> Costs | None                       snax_forge/char/<tech>/, None without
    streamer_cost(costs, depth, loops, channels) -> Cost   one reader (a writer priced as one)
    streamer_refusal(costs, depth, loops, channels) -> float   a refused request-cycle's share of a lane's grant
    xbar_cost(costs, inputs, banks, read_only, write_only, form) -> Cost    the crossbar
    bank(costs, depth, width) -> Bank | None               one SRAM bank, a lookup
    Grid.at(field, **q) -> (value, extrapolated axes)      the rule both interpolated tables use

The seven files of ``<tech>/`` are copied from snax-forge-experiments by ``pixi run
import-costs``, their commit and sha256 in ``<tech>/SOURCE``. Every table carries leakage
beside area (``leak_uw``); ``cluster_fixed.json`` gives the core, the DMA and the rest of the
cluster (``Fixed``), ``dma_copy.json`` the DMA engine's energy per burst, ``model_fit.json`` the
refusal fractions, the idle floor and the transfer beat (``Terms``). The tables, the interpolation and
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
FILES = ("ram_table.csv", "streamer.csv", "xbar.csv", "xbar_kinds.json", "cluster_fixed.json",
         "dma_copy.json", "model_fit.json")  # fmt: skip
# The cluster's fixed parts in cluster_fixed.json, in the order shown
FIXED_PARTS = ("core", "dma", "rest")
# The crossbar's input-kind forms (CONTRACTS, "Cost model"), the default first
XBAR_FORMS = ("factor", "crosspoint")
READER_AXES = ("depth", "loops", "channels")
XBAR_AXES = ("inputs", "banks")
SCALES: dict[str, Callable[[float], float]] = {"lin": lambda x: x, "inv": lambda x: 1.0 / x}
# The crossbar's energy interpolated in 1/inputs (CONTRACTS, "Cost model")
XBAR_SCALES = {"e_access_pj": {"inputs": "inv"}}


@dataclass(frozen=True)
class Grid:
    """A rectilinear grid: ``axes`` (name -> sorted grid values), per field {coordinates:
    value}, and per field the scale of an axis (``"lin"`` unless ``scales[field][axis]``)."""

    axes: dict[str, tuple[int, ...]]
    fields: dict[str, dict[tuple[int, ...], float]]
    scales: dict[str, dict[str, str]] = field(default_factory=dict)

    def at(self, name: str, **q: float) -> tuple[float, tuple[str, ...]]:
        """``name`` at ``q`` (one value per axis) and the axes it was extrapolated along
        (CONTRACTS, "Cost model"); a ``KeyError`` when a corner the query needs has no value."""
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
    the axes the query lay outside the grid on (empty when inside), and the leakage (µW)."""

    area_um2: float
    energy_pj: float
    extrapolated: tuple[str, ...] = ()
    leak_uw: float = 0.0


@dataclass(frozen=True)
class Bank:
    """One SRAM bank of ``depth`` words of ``width`` bits: its area, energy per access and
    leakage (µW)."""

    depth: int
    width: int
    area_um2: float
    e_rd_pj: float
    e_wr_pj: float
    leak_uw: float = 0.0


@dataclass(frozen=True)
class FixedPart:
    """One of the cluster's fixed parts (``FIXED_PARTS``): its area (µm²), its leakage (µW) and
    its idle floor, the dynamic energy it spends per cycle whatever the accelerators do (pJ)."""

    area_um2: float
    leak_uw: float
    idle_pj_cycle: float


@dataclass(frozen=True)
class Terms:
    """``model_fit.json``'s terms (CONTRACTS, "Cost model"): a refused request's fraction of a granted
    one in the streamers and the crossbar; the idle floor a cycle (pJ) of the streamers per mover
    or channel, of the crossbar as a constant plus a cost per input, of the DMA; the DMA engine's
    energy per L2 <-> L1 beat. The defaults price nothing new: a refusal as a grant, no floor."""

    r_strm: float = 1.0
    r_xbar: float = 1.0
    floor_strm_pj: float = 0.0
    floor_xbar_pj: float = 0.0
    floor_xbar_input_pj: float = 0.0
    floor_dma_pj: float = 0.0
    e_l2_pj: float = 0.0

    def floor(self, streamer_units: int, xbar_inputs: int) -> dict[str, float]:
        """pJ a cycle per part: the streamers', the crossbar's and the DMA's."""
        return {"streamers": self.floor_strm_pj * streamer_units,
                "crossbar": self.floor_xbar_pj + self.floor_xbar_input_pj * xbar_inputs,
                "dma": self.floor_dma_pj}  # fmt: skip


@dataclass(frozen=True)
class XbarKinds:
    """``xbar_kinds.json``'s forms (CONTRACTS, "Cost model"): the area's factor form's ``f_ro`` and
    ``f_wo``, its crosspoint form's coefficients (µm²) and the grid they were fitted on, and the
    energy's factors per bank count, ``(banks, g_ro, g_wo)`` in order of banks."""

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
    energy: tuple[tuple[int, float, float], ...]

    def energy_factors(self, banks: int) -> tuple[float, float]:
        """``(g_ro, g_wo)`` at ``banks``: linear in banks between the measured bank counts, the
        nearest one's outside them."""
        bs = [b for b, _, _ in self.energy]
        lo = max([i for i, b in enumerate(bs) if b <= banks], default=0)
        hi = min([i for i, b in enumerate(bs) if b >= banks], default=len(bs) - 1)
        (b0, r0, w0), (b1, r1, w1) = self.energy[lo], self.energy[hi]
        t = 0.0 if b1 == b0 else (banks - b0) / (b1 - b0)
        return r0 + t * (r1 - r0), w0 + t * (w1 - w0)


@dataclass(frozen=True)
class Costs:
    """One technology's tables and where they came from (``SOURCE``'s fields)."""

    tech: str
    source: dict[str, str]
    streamer: Grid
    xbar: Grid
    banks: dict[tuple[int, int], Bank]
    kinds: XbarKinds
    fixed: dict[str, FixedPart] = field(default_factory=dict)
    dma_burst_pj: float = 0.0
    terms: Terms = field(default_factory=Terms)


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
            if r.get(n) not in ("", None):
                fv[n][key] = float(r[n])
    return Grid(ax, fv, dict(scales or {}))


def _complete_xbar(g: Grid) -> Grid:
    """The crossbar's energy where inputs > banks: each bank count's value at inputs = banks."""
    e = dict(g.fields["e_access_pj"])
    for i, b in itertools.product(g.axes["inputs"], g.axes["banks"]):
        if i > b and (i, b) not in e and (b, b) in e:
            e[(i, b)] = e[(b, b)]
    return Grid(g.axes, {**g.fields, "e_access_pj": e}, g.scales)


def read_fixed(path: Path) -> dict[str, FixedPart]:
    """``cluster_fixed.json``'s parts, which snax-forge-experiments' report/cluster_fixed.py
    writes."""
    parts = json.loads(path.read_text())["parts"]
    return {k: FixedPart(parts[k]["area_um2"], parts[k]["leak_uw"], parts[k]["idle_pj_cycle"])
            for k in FIXED_PARTS}  # fmt: skip


def read_dma_burst(path: Path) -> float:
    """``dma_copy.json``'s ``engine.e_burst_pj``: the DMA engine's energy per burst of a copy."""
    return float(json.loads(path.read_text())["engine"]["e_burst_pj"])


def read_terms(path: Path) -> Terms:
    """``model_fit.json``, which snax-forge-experiments' report/model_fit.py writes."""
    d = json.loads(path.read_text())
    f, r = d["floor"], d["refusal"]
    return Terms(r["r_strm"], r["r_xbar"], f["streamers"]["k_pj"], f["crossbar"]["a_pj"],
                 f["crossbar"]["b_pj"], f["dma"]["pj"], d["l2"]["e_l2_pj"])  # fmt: skip


def read_kinds(path: Path) -> XbarKinds:
    """``xbar_kinds.json``, which snax-forge-experiments' report/xbar_kinds.py writes."""
    d = json.loads(path.read_text())
    f, c = d["factor"], d["crosspoint"]
    energy = tuple(
        sorted((int(b), e["g_ro"], e["g_wo"]) for b, e in d["energy"]["by_banks"].items())
    )
    return XbarKinds(f["f_ro"], f["f_wo"], c["c_rw"], c["c_ro"], c["c_wo"], c["d"], c["e"], c["g"],
                     tuple(c["grid"]["inputs"]), tuple(c["grid"]["banks"]), energy)  # fmt: skip


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
                  float(r["e_wr_pj"]), float(r["leak_uw"]))
             for r in _rows(d / "ram_table.csv")}  # fmt: skip
    streamer = _grid(
        _rows(d / "streamer.csv"), READER_AXES, ("area_um2", "e_beat_pj", "leak_uw", "r_refused")
    )
    xbar = _complete_xbar(_grid(_rows(d / "xbar.csv"), XBAR_AXES,
                                ("area_um2", "e_access_pj", "leak_uw"), XBAR_SCALES))  # fmt: skip
    return Costs(tech, read_source(d / "SOURCE"), streamer, xbar, banks,
                 read_kinds(d / "xbar_kinds.json"), read_fixed(d / "cluster_fixed.json"),
                 read_dma_burst(d / "dma_copy.json"), read_terms(d / "model_fit.json"))  # fmt: skip


def _warn(what: str, q: Mapping[str, float], axes: tuple[str, ...]) -> None:
    if axes:
        LOG.warning("%s at %s: outside the table's grid on %s, extrapolated",
                    what, ", ".join(f"{k} {v}" for k, v in q.items()), ", ".join(axes))  # fmt: skip


def streamer_cost(costs: Costs, depth: int, loops: int, channels: int) -> Cost:
    """One reader's (or writer's, priced as a reader) area and energy per beat."""
    q = {"depth": depth, "loops": loops, "channels": channels}
    area, out_a = costs.streamer.at("area_um2", **q)
    e, out_e = costs.streamer.at("e_beat_pj", **q)
    leak, _ = costs.streamer.at("leak_uw", **q)
    out = tuple(dict.fromkeys(out_a + out_e))
    _warn("streamer", q, out)
    return Cost(area, e, out, leak)


def streamer_refusal(costs: Costs, depth: int, loops: int, channels: int) -> float:
    """A reader's refused request-cycle as a share of its lane's granted price (``r_refused``,
    measured per reader on the blocks), the readers' mean ``Terms.r_strm`` where the table has none."""
    if not costs.streamer.fields.get("r_refused"):
        return costs.terms.r_strm
    try:
        return costs.streamer.at("r_refused", depth=depth, loops=loops, channels=channels)[0]
    except KeyError:
        return costs.terms.r_strm


def xbar_cost(costs: Costs, inputs: int, banks: int, read_only: int = 0, write_only: int = 0,
              form: str = "factor") -> Cost:  # fmt: skip
    """The crossbar's area and energy per access, ``read_only`` and ``write_only`` of its
    ``inputs`` priced as such in ``form`` (CONTRACTS, "Cost model"); with none, the table's area.
    Its leakage follows its area: the table's leakage per µm² at its inputs and banks."""
    if form not in XBAR_FORMS:
        raise ValueError(f"crossbar form {form!r}: one of {', '.join(XBAR_FORMS)}")
    if read_only < 0 or write_only < 0 or read_only + write_only > inputs:
        raise ValueError(f"{read_only} read-only and {write_only} write-only of {inputs} inputs")
    q = {"inputs": inputs, "banks": banks}
    e, out_e = costs.xbar.at("e_access_pj", **q)
    table_area, _ = costs.xbar.at("area_um2", **q)
    table_leak, _ = costs.xbar.at("leak_uw", **q)
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
    return Cost(area, e, out, area * table_leak / table_area if table_area else 0.0)


def bank(costs: Costs, depth: int, width: int) -> Bank | None:
    """The bank of ``depth`` × ``width`` bits, None off the table's grid (never interpolated)."""
    return costs.banks.get((depth, width))
