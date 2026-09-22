"""SNAX-MODEL: kernel-agnostic, cycle-level model of the SNAX cluster."""

from .accel import (
    AccelConfig,
    Accelerator,
    AccelPort,
    elementwise_stub,
    reduce_stub,
)
from .cluster import Cluster
from .mem import (
    AddressMap,
    BankConflictError,
    BankReq,
    BankResp,
    L1Config,
    L1Memory,
    WordInterleaved,
)
from .sched import (
    Component,
    Phase,
    Scheduler,
    SimulationError,
    SimulationTimeout,
    Stateful,
)
from .streamer import (
    Fifo,
    Streamer,
    StreamerConfig,
    StreamerRegs,
    address_stream,
    lane_offsets,
)
from .xbar import HoldViolation, Port, Xbar

__all__ = [
    "AccelConfig",
    "AccelPort",
    "Accelerator",
    "AddressMap",
    "BankConflictError",
    "BankReq",
    "BankResp",
    "Cluster",
    "Component",
    "Fifo",
    "HoldViolation",
    "L1Config",
    "L1Memory",
    "Phase",
    "Port",
    "Scheduler",
    "SimulationError",
    "SimulationTimeout",
    "Stateful",
    "Streamer",
    "StreamerConfig",
    "StreamerRegs",
    "WordInterleaved",
    "Xbar",
    "address_stream",
    "elementwise_stub",
    "lane_offsets",
    "reduce_stub",
]
