"""The checked-in clusters: alu4.json, red4.json and mul1.json (MOD9, D41, D88).

Each is a platform plus its accelerators, assembled by SNAX-LOWER's cluster
builder (snax_forge/lower/cluster.py, LOW1c), the same code that derives a
design point's cluster file, so the checked-in clusters and the derived ones
cannot drift apart:

    alu4   platforms/small16.json with the elementwise_add BRM at W = 4
           (instance ``acc``): exactly the cluster of vecadd's design point
    red4   an L1-only platform with the accumulate BRM at W = 4, its
           chisel_adder_tree implementation (instance ``acc``, BRM4, D103)
    mul1   32 banks and 2-loop streamers with a 1-lane ``elementwise`` mul
           stub (no BRM for mul yet)

Streamers are named after the accelerator port they serve, ``<instance>_<port>``
(D75): alu4's and mul1's ``acc_a``, ``acc_b``, ``acc_out``, red4's ``acc_a``,
``acc_out``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from snax_forge.brm import load_brm
from snax_forge.design import Platform, StreamerOptions
from snax_forge.lower import cluster_config, cluster_of, stub
from snax_forge.snax_model import ControllerConfig, L1Config, L2Config
from snax_forge.snax_model.scenario import ClusterConfig

PLATFORMS = Path(__file__).resolve().parents[2] / "platforms"
LANES = 4  # alu4 and red4

# Controller costs of every scenario: one cycle per csr_write and csr_read on
# every block kind, a poll every 4 cycles. Declared defaults, not measured
# (D51, open item 10). test_profile.VECADD_CFG keeps its own non-default costs
# (DMA writes and reads 2) to exercise the D37 formulas; test_scenario runs the
# hand-built vecadd (helpers.mod7_vecadd) with these instead.
# platforms/small16.json holds the same.
CTL = ControllerConfig(write_cost=1, read_cost=1, poll_interval=4)


def alu4() -> ClusterConfig:
    """small16 with elementwise_add at W = 4: vecadd's cluster (the MOD7 vecadd's)."""
    inst = load_brm("elementwise_add").resolve("chisel_tiled_spatial", {"W": LANES})
    return cluster_of(Platform.load(PLATFORMS / "small16.json"), {"acc": inst})


def red4() -> ClusterConfig:
    """accumulate over 4 lanes into one lane, L1 only: the reduce stub's entry, from the BRM."""
    pf = Platform(
        "red4",
        l1=L1Config(n_banks=16, rows=64, read_latency=1),
        controller=CTL,
        default=StreamerOptions(fifo_depth=2),
    )
    inst = load_brm("accumulate").resolve("chisel_adder_tree", {"W": LANES})
    return cluster_of(pf, {"acc": inst})


# mul1: a multi-cycle multiplier with 1-lane streamers, for fmul.
MUL_BANKS = 32
MUL_LATENCY = 5
MUL_II = 5


def mul1() -> ClusterConfig:
    """One multiplier (1 lane, L = II = 5), its streamers acc_a, acc_b, acc_out, and a DMA.

    32 banks = 4 superbanks of 8, so one tile's buffers fit in two
    superbanks and the DMA can work on the other two (fmul). The streamers
    have 2 temporal loops, to walk a buffer that lives in one superbank.
    """
    pf = Platform(
        "mul1",
        l1=L1Config(n_banks=MUL_BANKS, rows=32, read_latency=1),
        l2=L2Config(size_bytes=1 << 15, read_latency=1),
        controller=CTL,
        default=StreamerOptions(temporal_dims=2, fifo_depth=2),
    )
    params = {"lanes": 1, "n_inputs": 2, "op": "mul", "latency": MUL_LATENCY, "ii": MUL_II}
    ports = [("a", False, 1), ("b", False, 1), ("out", True, 1)]
    return cluster_config(pf, [stub(pf, "acc", "elementwise", params, ports)])


# File name stem -> builder, in the order make.py writes them.
CLUSTERS: dict[str, Callable[[], ClusterConfig]] = {"alu4": alu4, "red4": red4, "mul1": mul1}
