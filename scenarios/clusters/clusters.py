"""The checked-in clusters: alu4.json, red4.json and mul1.json (MOD9, D41).

scenarios/make.py writes one file per entry of ``CLUSTERS``. The builders
move into SNAX-LOWER with LOW1c, which derives a cluster file from a design
point and BRMs (D53, open item 24).
"""

from __future__ import annotations

from collections.abc import Callable

from common import LANES

from snax_forge.snax_model import (
    ControllerConfig,
    DmaConfig,
    L1Config,
    L2Config,
    StreamerConfig,
)
from snax_forge.snax_model.scenario import ClusterConfig, ComponentSpec, RegisterMapSpec

# Controller costs of every scenario: one cycle per csr_write and csr_read on
# every block kind, a poll every 4 cycles. Declared defaults, not measured
# (D51, open item 10). test_profile.VECADD_CFG keeps its own non-default costs
# (DMA writes and reads 2) to exercise the D37 formulas; test_scenario runs the
# hand-built vecadd with these instead.
CTL = ControllerConfig(write_cost=1, read_cost=1, poll_interval=4)


def _streamer(name: str, write: bool, lanes: int, temporal_dims: int = 1) -> ComponentSpec:
    cfg = StreamerConfig(write=write, n_ports=lanes, fifo_depth=2, temporal_dims=temporal_dims)
    return ComponentSpec(name, "streamer", cfg.to_dict())


def alu4() -> ClusterConfig:
    """The cluster of test_profile's run_vecadd, in its registration order."""
    return ClusterConfig(
        l1=L1Config(n_banks=16, rows=64, read_latency=1),
        l2=L2Config(size_bytes=1 << 15, read_latency=1),
        components=[
            ComponentSpec("xbar", "xbar", {"check_hold": True}),
            ComponentSpec("dma", "dma", DmaConfig().to_dict()),
            _streamer("ra", False, LANES),
            _streamer("rb", False, LANES),
            _streamer("wr", True, LANES),
            ComponentSpec(
                "acc",
                "accel",
                accel="elementwise",
                params={"lanes": LANES, "n_inputs": 2, "op": "add", "latency": 0, "ii": 1},
                attach={"a": "ra", "b": "rb", "out": "wr"},
            ),
            ComponentSpec("ctl", "controller", CTL.to_dict()),
        ],
        register_map=RegisterMapSpec(blocks=["dma", "ra", "rb", "wr", "acc"]),
    )


def red4() -> ClusterConfig:
    """A reduce over 4 lanes into one lane (lanes_out = 1), L1 only."""
    return ClusterConfig(
        l1=L1Config(n_banks=16, rows=64, read_latency=1),
        components=[
            ComponentSpec("xbar", "xbar", {"check_hold": True}),
            _streamer("ra", False, LANES),
            _streamer("wr", True, 1),
            ComponentSpec(
                "acc",
                "accel",
                accel="reduce",
                params={"lanes": LANES, "lanes_out": 1, "op": "add", "latency": 1, "ii": 1},
                attach={"in": "ra", "out": "wr"},
            ),
            ComponentSpec("ctl", "controller", CTL.to_dict()),
        ],
        register_map=RegisterMapSpec(blocks=["ra", "wr", "acc"]),
    )


# mul1: a multi-cycle multiplier with 1-lane streamers, for fmul.
MUL_BANKS = 32
MUL_LATENCY = 5
MUL_II = 5


def mul1() -> ClusterConfig:
    """One multiplier (1 lane, L = II = 5) between readers ra, rb and writer wr, and a DMA.

    32 banks = 4 superbanks of 8, so one tile's buffers fit in two
    superbanks and the DMA can work on the other two (fmul). The streamers
    have 2 temporal loops, to walk a buffer that lives in one superbank.
    """
    L, II = MUL_LATENCY, MUL_II
    return ClusterConfig(
        l1=L1Config(n_banks=MUL_BANKS, rows=32, read_latency=1),
        l2=L2Config(size_bytes=1 << 15, read_latency=1),
        components=[
            ComponentSpec("xbar", "xbar", {"check_hold": True}),
            ComponentSpec("dma", "dma", DmaConfig().to_dict()),
            _streamer("ra", False, 1, temporal_dims=2),
            _streamer("rb", False, 1, temporal_dims=2),
            _streamer("wr", True, 1, temporal_dims=2),
            ComponentSpec(
                "acc",
                "accel",
                accel="elementwise",
                params={"lanes": 1, "n_inputs": 2, "op": "mul", "latency": L, "ii": II},
                attach={"a": "ra", "b": "rb", "out": "wr"},
            ),
            ComponentSpec("ctl", "controller", CTL.to_dict()),
        ],
        register_map=RegisterMapSpec(blocks=["dma", "ra", "rb", "wr", "acc"]),
    )


# File name stem -> builder, in the order make.py writes them.
CLUSTERS: dict[str, Callable[[], ClusterConfig]] = {"alu4": alu4, "red4": red4, "mul1": mul1}
