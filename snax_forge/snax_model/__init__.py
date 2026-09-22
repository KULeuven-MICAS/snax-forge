"""SNAX-MODEL: kernel-agnostic, cycle-level model of the SNAX cluster."""

from .accel import (
    AccelConfig,
    Accelerator,
    AccelPort,
    elementwise_stub,
    reduce_stub,
)
from .cluster import Cluster
from .dma import Dma, DmaConfig, DmaDescriptor, DmaPattern, check_descriptor
from .l2 import L2AccessError, L2Config, L2Memory, L2Resp
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
    "Dma",
    "DmaConfig",
    "DmaDescriptor",
    "DmaPattern",
    "Fifo",
    "HoldViolation",
    "L1Config",
    "L1Memory",
    "L2AccessError",
    "L2Config",
    "L2Memory",
    "L2Resp",
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
    "check_descriptor",
    "elementwise_stub",
    "lane_offsets",
    "reduce_stub",
]
