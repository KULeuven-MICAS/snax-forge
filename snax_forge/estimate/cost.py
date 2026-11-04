"""The cost of a bound graph's groups: CHAR1's first-order model on a technology's table (D154,
C§20).

    Block(datapath, params, fires)                 a unit or a buffer link to cost
    block_cost(table, block, bits, c_acc) -> BlockCost
    group_cost(table, blocks, bits, ctrl_bits, c_acc) -> GroupCost
    l1_cost(table, words) -> L1Cost                an accelerator's L1 accesses (placeholder)
    sig3(x) -> str                                 an area as shown: 3 significant figures
    dec3(x) -> str                                 an energy as shown: 3 decimal places
    heading(tech), CAVEAT                          what every shown cost is headed with

**The model**, per unit and per buffer link over one run of the accelerator, with n_fire its
fires, n_bits its register bits (the estimate's ``bits``), C_acc the accelerator's cycles
(``accelerator_cycles``: no clock gating, so every flop holds through every group's cycles) and
e_comb, A_comb its comb energy per fire and area from the table (snax_forge/char/):

    e_fire = e_comb + n_bits (e_load − e_hold)          pJ per fire
    hold   = n_bits e_hold                              pJ per cycle
    task   = n_fire e_fire + C_acc hold                 (nJ)
    area   = A_comb + n_bits A_flop                     µm²

and per group its control unit, n_bits^ctrl its ``ctrl_bits`` and f_ctrl the table's counter
factor: energy f_ctrl n_bits^ctrl C_acc e_hold, area its flops' n_bits^ctrl A_flop plus an equal
comb area (``CTRL_COMB_PER_FLOP``, a placeholder, D154; its comb energy taken as negligible); the
csr is left out (D152). n_fire is a unit's firings times its span (SerialAccumulate reads span
beats per firing); a buffer link loads once per beat of its producer, its producer's firings. A
block the table does not hold (``lookup`` None) or the generator does not fill has ``found``
None and its comb part counted as 0; its flops are still counted, the description gives them.

**L1 accesses** (a placeholder until the streamers and the memories are characterized, D154):
per accelerator, every 64-bit word its streamers read or write in L1 costs ``L1_ACCESS_LOADS``
(4) × e_load per bit, energy 4 e_load · 64 · words; a link's container is not in L1, and the
DMA's traffic, the same for every variant, is not counted. It adds to the accelerator's energy,
not to a group's.

**Shown, not stored.** The fields keep full precision; everything a reader sees (``design.md``,
the viewer and the editor) is rounded for display only, an area by ``sig3`` (3 significant figures) and an energy by
``dec3`` (3 decimal places in its unit, so two close designs can be told apart), and headed by
``heading`` with ``CAVEAT``: the model is first-order, aimed at about 20 % of synthesis on the
large costs (C§20).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import floor, log10
from typing import Any

from snax_forge.char import CTRL_COMB_PER_FLOP, L1_ACCESS_LOADS, Table, lookup

CAVEAT = ("First-order: aimed at the large costs (about 20 % of synthesis there; small blocks "
          "can be much further off); the L1 accesses (4 × e_load per bit) and the control unit's "
          "comb area (equal to its flops') are placeholders, the streamers' own logic and the csr "
          "are left out.")  # fmt: skip
L1_LABEL = "L1 accesses (placeholder, 4 × e_load per bit)"  # its row in every cost table


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
    """One unit's or buffer link's cost (module doc); areas µm², energies pJ, ``task_nj`` nJ."""

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
        """The control unit's comb area: equal to its flops' (a placeholder, D154)."""
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


@dataclass
class L1Cost:
    """One accelerator's L1 accesses (module doc): the 64-bit words its streamers read and write
    in L1, and their energy in nJ."""

    words: int
    energy_nj: float

    @property
    def bits(self) -> int:
        return 64 * self.words

    def to_dict(self) -> dict[str, Any]:
        return {"words": self.words, "bits": self.bits, "energy": self.energy_nj}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> L1Cost:
        return cls(d["words"], d["energy"])


def l1_cost(table: Table, words: int) -> L1Cost:
    """An accelerator's L1 accesses on ``table``: ``L1_ACCESS_LOADS`` × e_load per bit."""
    return L1Cost(words, L1_ACCESS_LOADS * table.e_load_pj * 64 * words / 1000)


def block_cost(table: Table, block: Block, bits: int, c_acc: int) -> BlockCost:
    """One block's cost on ``table`` (module doc)."""
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
    ``bits`` the estimate's, ``ctrl_bits`` its control unit's (module doc)."""
    n = ctrl_bits or 0
    return GroupCost({k: block_cost(table, b, bits.get(k, 0), c_acc) for k, b in units.items()},
                     {k: block_cost(table, b, bits.get(k, 0), c_acc) for k, b in links.items()},
                     n, n * table.flop_um2,
                     table.counter_factor * n * c_acc * table.e_hold_pj / 1000)  # fmt: skip
