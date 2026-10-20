"""The profile of a flow run (C§7, D137).

The totals of one SNAX-GVSoC run, read off the counters its components print at the end of
each run or of the simulation (snax-gvsoc's ``summary`` lines, its ``docs/EVENTS.md``); the
converter renames them and never recounts (D38). The static fields (names, ports, FIFO
depths) come from the hjson files and the design point, never from the events.

Maps are keyed by C§7 names (``acc``, ``acc_a``, ``dma``, ``core0``, ...), in the cluster's
order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AccelProfile:
    cycles: dict[str, int]  # busy, stall_in, stall_out, flush over its runs; idle the rest
    firings: int
    utilisation: float  # busy / the window


@dataclass
class FifoProfile:
    name: str
    depth: int
    max: list[int]  # per lane, the highest count over the runs


@dataclass
class StreamerProfile:
    write: bool
    cycles: dict[str, int]  # busy, stall_xbar, stall_fifo over its runs; idle the rest
    ports: list[str]
    fifo: FifoProfile


@dataclass
class WaitProfile:
    block: str  # the polled block, or "barrier"
    first: int
    last: int


@dataclass
class CoreProfile:
    cycles: dict[str, int] | None  # run, poll, barrier, sleep in the window; None at trace off
    csr_writes: int
    csr_reads: int
    waits: list[WaitProfile] = field(default_factory=list)


@dataclass
class DmaProfile:
    transfers: int
    busy: int  # the cycles DMA_BUSY counts
    beats_read: int
    beats_written: int
    bytes_l2_to_l1: int
    bytes_l1_to_l2: int


@dataclass
class PortProfile:
    grants: int
    stalls: int
    stalls_dma: int  # the part of stalls lost to the DMA's superbank
    whole_run: bool = False  # a core's or AXI's input: counted from reset, boot included


@dataclass
class BankProfile:
    """Per L1 bank, from the SNAX inputs (streamers) and the DMA only."""

    reads: list[int]
    writes: list[int]
    conflicts: list[int]


@dataclass
class Profile:
    total_cycles: int  # the program's window, its `forge cycles:` line
    accelerators: dict[str, AccelProfile] = field(default_factory=dict)
    streamers: dict[str, StreamerProfile] = field(default_factory=dict)
    cores: dict[str, CoreProfile] = field(default_factory=dict)
    dmas: dict[str, DmaProfile] = field(default_factory=dict)
    ports: dict[str, PortProfile] = field(default_factory=dict)
    banks: BankProfile | None = None
    functional_check: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Profile:
        banks = d.get("banks")
        return cls(
            total_cycles=d["total_cycles"],
            accelerators={k: AccelProfile(**v) for k, v in d.get("accelerators", {}).items()},
            streamers={
                k: StreamerProfile(
                    v["write"], dict(v["cycles"]), list(v["ports"]), FifoProfile(**v["fifo"])
                )
                for k, v in d.get("streamers", {}).items()
            },
            cores={
                k: CoreProfile(
                    v["cycles"],
                    v["csr_writes"],
                    v["csr_reads"],
                    [WaitProfile(**w) for w in v.get("waits", [])],
                )
                for k, v in d.get("cores", {}).items()
            },
            dmas={k: DmaProfile(**v) for k, v in d.get("dmas", {}).items()},
            ports={k: PortProfile(**v) for k, v in d.get("ports", {}).items()},
            banks=None if banks is None else BankProfile(**banks),
            functional_check=d.get("functional_check"),
        )
