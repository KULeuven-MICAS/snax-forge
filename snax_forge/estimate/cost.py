"""The cost of a design (CONTRACTS, "Cost model"): the block model of its groups, and its
streamers, crossbar and L1 banks.

``block_cost`` and ``group_cost`` price a group's units and buffer links (``Block``) and its
control unit from a technology's table; ``streamer_cost`` prices one port's streamer
(``StreamerCounts``; a shared mover once, on its first user), ``xbar_cost`` the TCDM crossbar
(its read-only and write-only inputs from ``xbar_kinds``; the factor form by default,
``form="crosspoint"`` the option), ``memory_cost`` the L1 banks, and ``l1_cost`` an
accelerator's words through the crossbar and the banks (``L1Cost``). ``PARTS`` are the five
parts a design's cost is shown in and ``shares`` each one's share of the totals. The fields keep
full precision; what a reader sees is rounded by ``sig3`` (an area, 3 significant figures) and
``dec3`` (an energy, 3 decimal places) and headed by ``heading`` and ``CAVEAT``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import floor, log10
from typing import Any

from snax_forge.char import CTRL_COMB_PER_FLOP, Table, lookup
from snax_forge.char.tables import Costs, bank
from snax_forge.char.tables import streamer_cost as _streamer
from snax_forge.char.tables import xbar_cost as _xbar

CAVEAT = ("First-order: aimed at the large costs (about 30 % of synthesis there; small blocks "
          "can be much further off); the control unit's comb area (equal to its flops') is a "
          "rough estimate; the streamers are priced per beat and the crossbar per access; the csr, "
          "the cores, the I-cache and the DMA are left out.")  # fmt: skip


def caveat() -> str:
    """The line every shown cost carries (``CAVEAT``)."""
    return CAVEAT


L1_LABEL = "L1 accesses (crossbar and banks)"  # its row in every cost table
# The parts a design's cost is shown in, in order
PARTS = ("datapath", "control", "streamers", "crossbar", "memory")


def heading(tech: str) -> str:
    """What a shown cost is headed with."""
    return f"Cost estimate (first-order) on {tech}"


def sig3(x: float | None) -> str | None:
    """``x`` to 3 significant figures, thousands split by a space: 26.3, 63 200, 0.135; None
    stays None."""
    if x is None:
        return None
    r = float(f"{x:.3g}")
    if r == 0:
        return "0"
    decimals = max(0, 2 - floor(log10(abs(r))))
    return f"{r:,.{decimals}f}".replace(",", " ")


def dec3(x: float | None) -> str | None:
    """An energy as shown: 3 decimal places in its unit, thousands split by a space (26.285,
    97.259, 0.135, 1 234.500); None stays None. Display precision only: the model is
    first-order (``CAVEAT``)."""
    if x is None:
        return None
    return f"{x:,.3f}".replace(",", " ")


@dataclass(frozen=True)
class Block:
    """A unit or buffer link to cost: its catalogue datapath (None when the generator does not
    fill it), its params as the generator fills them (lanes and depth included), its fires."""

    datapath: str | None
    params: Mapping[str, Any]
    fires: int


@dataclass
class BlockCost:
    """One unit's or buffer link's cost (CONTRACTS, "Cost model"); areas µm², energies pJ,
    ``task_nj`` nJ."""

    datapath: str | None
    fires: int
    found: str | None
    comb_um2: float
    flops_um2: float
    e_fire_pj: float
    e_hold_pj: float
    task_nj: float

    @property
    def total_um2(self) -> float:
        return self.comb_um2 + self.flops_um2

    def to_dict(self) -> dict[str, Any]:
        return {"datapath": self.datapath, "fires": self.fires, "found": self.found,
                "area": {"comb": self.comb_um2, "flops": self.flops_um2, "total": self.total_um2},
                "energy": {"fire": self.e_fire_pj, "hold": self.e_hold_pj,
                           "task": self.task_nj}}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BlockCost:
        a, e = d["area"], d["energy"]
        return cls(d["datapath"], d["fires"], d["found"], a["comb"], a["flops"], e["fire"],
                   e["hold"], e["task"])  # fmt: skip


@dataclass
class GroupCost:
    """One group's cost: its units' and buffer links' (by container), its control unit's, and
    the totals of all three."""

    units: dict[str, BlockCost] = field(default_factory=dict)
    links: dict[str, BlockCost] = field(default_factory=dict)
    ctrl_bits: int = 0
    ctrl_flops_um2: float = 0.0
    ctrl_nj: float = 0.0

    @property
    def ctrl_comb_um2(self) -> float:
        """The control unit's comb area: equal to its flops' (a placeholder)."""
        return CTRL_COMB_PER_FLOP * self.ctrl_flops_um2

    @property
    def ctrl_um2(self) -> float:
        """The control unit's area: its flops' and its comb's."""
        return self.ctrl_flops_um2 + self.ctrl_comb_um2

    @property
    def area(self) -> float:
        """µm²: units, buffer links and the control unit (flops and comb)."""
        return sum(b.total_um2 for b in (*self.units.values(), *self.links.values())) + (
            self.ctrl_um2)  # fmt: skip

    @property
    def energy(self) -> float:
        """nJ per run of the accelerator: units, buffer links and the control unit."""
        return sum(b.task_nj for b in (*self.units.values(), *self.links.values())) + (
            self.ctrl_nj)  # fmt: skip

    @property
    def missing(self) -> list[str]:
        """The units and links not characterized (``found`` None)."""
        return [k for k, b in (*self.units.items(), *self.links.items()) if b.found is None]

    def to_dict(self) -> dict[str, Any]:
        return {"units": {k: b.to_dict() for k, b in self.units.items()},
                "links": {k: b.to_dict() for k, b in self.links.items()},
                "ctrl": {"bits": self.ctrl_bits,
                         "area": {"flops": self.ctrl_flops_um2, "comb": self.ctrl_comb_um2,
                                  "total": self.ctrl_um2}, "energy": self.ctrl_nj},
                "area": self.area, "energy": self.energy}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GroupCost:
        c = d["ctrl"]
        area = c["area"]["flops"] if isinstance(c["area"], dict) else c["area"]  # older: flops
        return cls({k: BlockCost.from_dict(b) for k, b in d["units"].items()},
                   {k: BlockCost.from_dict(b) for k, b in d["links"].items()}, c["bits"],
                   area, c["energy"])  # fmt: skip


@dataclass(frozen=True)
class StreamerCounts:
    """One port's streamer, what the tables price it by: its ``name`` (``<instance>_<port>``), its
    ``accelerator``, whether it writes, its FIFO ``depth``, ``loops`` and
    ``channels``, and the 64-bit ``words`` it moves in L1 in one run over its ``beats``. On a
    shared mover ``mover`` names it (``reader0``), ``mover_channels`` are its channels,
    which its area is priced at, and only its first user carries that area (``area``); the
    depth and loops are the mover's, the channels the port's, which its energy per beat is
    priced at."""

    name: str
    accelerator: str
    write: bool
    depth: int
    loops: int
    channels: int
    words: int
    beats: int
    mover: str | None = None
    mover_channels: int | None = None
    area: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "accelerator": self.accelerator, "write": self.write,
             "depth": self.depth, "loops": self.loops, "channels": self.channels,
             "words": self.words, "beats": self.beats}  # fmt: skip
        if self.mover is not None:
            d |= {"mover": self.mover, "mover_channels": self.mover_channels, "area": self.area}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StreamerCounts:
        return cls(d["name"], d["accelerator"], d["write"], d["depth"], d["loops"],
                   d["channels"], d["words"], d["beats"], d.get("mover"),
                   d.get("mover_channels"), d.get("area", True))  # fmt: skip


@dataclass(frozen=True)
class StreamerCost:
    """A port's streamer priced: its counts, its area (µm²), its energy per beat (pJ) and per
    run (nJ), and the axes its point lay outside the table's grid on."""

    counts: StreamerCounts
    area_um2: float
    e_beat_pj: float
    energy_nj: float
    extrapolated: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return self.counts.to_dict() | {"area": self.area_um2, "e_beat": self.e_beat_pj,
                                        "energy": self.energy_nj,
                                        "extrapolated": list(self.extrapolated)}  # fmt: skip


def streamer_cost(costs: Costs, c: StreamerCounts) -> StreamerCost:
    """A streamer's area and energy per run on ``costs`` (CONTRACTS, "Cost model"); on a shared
    mover its area at the mover's channels, on its first user only, its energy at the port's."""
    got = _streamer(costs, c.depth, c.loops, c.channels)
    area = got.area_um2
    extrapolated = got.extrapolated
    if c.mover_channels is not None:
        mover = _streamer(costs, c.depth, c.loops, c.mover_channels)
        area = mover.area_um2 if c.area else 0.0
        extrapolated = tuple(dict.fromkeys((*got.extrapolated, *mover.extrapolated)))
    return StreamerCost(c, area, got.energy_pj, c.beats * got.energy_pj / 1000, extrapolated)


@dataclass(frozen=True)
class XbarCost:
    """The TCDM crossbar priced: its ``inputs`` and ``banks``, the ``accesses`` it serves in one
    run, its area (µm²), its energy per access (pJ) and per run (nJ), how many of its inputs
    only read or only write, and the ``form`` its area was priced in (CONTRACTS, "Cost model")."""

    inputs: int
    banks: int
    accesses: int
    area_um2: float
    e_access_pj: float
    energy_nj: float
    extrapolated: tuple[str, ...] = ()
    read_only: int = 0
    write_only: int = 0
    form: str = "factor"

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "banks": self.banks, "accesses": self.accesses,
                "area": self.area_um2, "e_access": self.e_access_pj, "energy": self.energy_nj,
                "extrapolated": list(self.extrapolated), "read_only": self.read_only,
                "write_only": self.write_only, "form": self.form}  # fmt: skip


def xbar_kinds(counts: list[StreamerCounts]) -> tuple[int, int]:
    """The crossbar inputs the streamers take that only read and only write (CONTRACTS, "Cost
    model"): a reader's channels and a writer's, a shared mover's counted once at its own channels
    on its first user, as check.py's ``crossbar_inputs`` counts them."""
    ro = wo = 0
    for c in counts:
        if c.mover is not None and not c.area:
            continue
        n = c.mover_channels if c.mover is not None else c.channels
        if c.write:
            wo += n
        else:
            ro += n
    return ro, wo


def xbar_cost(costs: Costs, inputs: int, banks: int, accesses: int, read_only: int = 0,
              write_only: int = 0, form: str = "factor") -> XbarCost:  # fmt: skip
    """The crossbar's area and energy per run on ``costs`` (CONTRACTS, "Cost model"): its area at
    its inputs, ``read_only`` and ``write_only`` of them priced as such in ``form``, its accesses at
    one price per bank count, a full crossbar's (inputs = banks)."""
    area = _xbar(costs, inputs, banks, read_only, write_only, form)
    price = _xbar(costs, banks, banks)
    out = tuple(dict.fromkeys(area.extrapolated + price.extrapolated))
    return XbarCost(inputs, banks, accesses, area.area_um2, price.energy_pj,
                    accesses * price.energy_pj / 1000, out, read_only, write_only, form)  # fmt: skip


@dataclass(frozen=True)
class MemoryCost:
    """The L1 banks priced: ``n`` banks of ``depth`` words of ``width`` bits, the words read and
    written in one run, the area of all banks (µm²), a bank's energy per read and write (pJ)
    and the run's (nJ); ``found`` False when the table has no such bank (all costs 0)."""

    n: int
    depth: int
    width: int
    reads: int
    writes: int
    area_um2: float
    e_rd_pj: float
    e_wr_pj: float
    found: bool = True

    @property
    def energy_nj(self) -> float:
        return (self.reads * self.e_rd_pj + self.writes * self.e_wr_pj) / 1000

    def to_dict(self) -> dict[str, Any]:
        return {"banks": self.n, "depth": self.depth, "width": self.width,
                "reads": self.reads, "writes": self.writes, "area": self.area_um2,
                "e_rd": self.e_rd_pj, "e_wr": self.e_wr_pj, "energy": self.energy_nj,
                "found": self.found}  # fmt: skip


def memory_cost(costs: Costs, n: int, depth: int, width: int, reads: int,
                writes: int) -> MemoryCost:  # fmt: skip
    """The banks' area and energy per run on ``costs`` (CONTRACTS, "Cost model"); a bank the table
    lacks is costed 0 and marked."""
    b = bank(costs, depth, width)
    if b is None:
        return MemoryCost(n, depth, width, reads, writes, 0.0, 0.0, 0.0, False)
    return MemoryCost(n, depth, width, reads, writes, n * b.area_um2, b.e_rd_pj, b.e_wr_pj)


@dataclass
class L1Cost:
    """One accelerator's words in L1 (module doc): the 64-bit words its streamers read and
    write there, and their energy in nJ, the crossbar's per access and the banks' per read and
    write."""

    words: int
    energy_nj: float
    reads: int = 0
    writes: int = 0
    xbar_nj: float = 0.0
    banks_nj: float = 0.0

    @property
    def bits(self) -> int:
        return 64 * self.words

    def to_dict(self) -> dict[str, Any]:
        return {"words": self.words, "bits": self.bits, "reads": self.reads,
                "writes": self.writes, "xbar": self.xbar_nj, "banks": self.banks_nj,
                "energy": self.energy_nj}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> L1Cost:
        return cls(d["words"], d["energy"], d.get("reads", 0), d.get("writes", 0),
                   d.get("xbar", 0.0), d.get("banks", 0.0))  # fmt: skip


def l1_cost(xbar: XbarCost, memory: MemoryCost, reads: int, writes: int) -> L1Cost:
    """An accelerator's words through the crossbar and the banks."""
    x = (reads + writes) * xbar.e_access_pj / 1000
    b = (reads * memory.e_rd_pj + writes * memory.e_wr_pj) / 1000
    return L1Cost(reads + writes, x + b, reads, writes, x, b)


@dataclass(frozen=True)
class Part:
    """One part of a design's cost (``PARTS``): its area (µm²) and energy per run (nJ), None
    where not costed."""

    area_um2: float | None
    energy_nj: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"area": self.area_um2, "energy": self.energy_nj}


def shares(parts: Mapping[str, Part]) -> dict[str, tuple[float | None, float | None]]:
    """Each part's share of the total area and of the total energy (0..1), None for a part
    not costed or a total of 0."""
    area = sum(p.area_um2 or 0.0 for p in parts.values())
    energy = sum(p.energy_nj or 0.0 for p in parts.values())

    def of(x: float | None, whole: float) -> float | None:
        return None if x is None or not whole else x / whole

    return {k: (of(p.area_um2, area), of(p.energy_nj, energy)) for k, p in parts.items()}


def block_cost(table: Table, block: Block, bits: int, c_acc: int) -> BlockCost:
    """One block's cost on ``table`` (CONTRACTS, "Cost model")."""
    p = block.params
    got = None
    if block.datapath is not None:
        got = lookup(table, block.datapath, p, p.get("lanes", 1), p.get("depth"))
    comb, e_comb = (got.comb_um2, got.e_comb_pj) if got is not None else (0.0, 0.0)
    e_fire = e_comb + bits * (table.e_load_pj - table.e_hold_pj)
    hold = bits * table.e_hold_pj
    return BlockCost(block.datapath, block.fires, got.found if got is not None else None, comb,
                     bits * table.flop_um2, e_fire, hold,
                     (block.fires * e_fire + c_acc * hold) / 1000)  # fmt: skip


def group_cost(table: Table, units: Mapping[str, Block], links: Mapping[str, Block],
               bits: Mapping[str, int], ctrl_bits: int | None, c_acc: int) -> GroupCost:  # fmt: skip
    """A group's cost on ``table``: ``units`` by name and buffer ``links`` by container, their
    ``bits`` the estimate's, ``ctrl_bits`` its control unit's (CONTRACTS, "Cost model")."""
    n = ctrl_bits or 0
    return GroupCost({k: block_cost(table, b, bits.get(k, 0), c_acc) for k, b in units.items()},
                     {k: block_cost(table, b, bits.get(k, 0), c_acc) for k, b in links.items()},
                     n, n * table.flop_um2,
                     table.counter_factor * n * c_acc * table.e_hold_pj / 1000)  # fmt: skip
