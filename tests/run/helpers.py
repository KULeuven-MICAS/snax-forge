"""Shared snippets for the run tests: a cluster's three hjson files and design point, as the
generator writes them for vecadd on snax_alu (cut to what run/ reads), and GVSoC header lines."""

from __future__ import annotations

import copy
from typing import Any

PATH = "/chip/soc/cluster_0"

CLUSTER: dict[str, Any] = {
    "cluster": {
        "cluster_base_addr": 0x10000000,
        "data_width": 64,
        "tcdm": {"size": 128, "banks": 32},
        "hives": [{"cores": [{"$ref": "#/acc_core"}, {"$ref": "#/dma_core"}]}],
    },
    "acc_core": {
        "xdma": False,
        "snax_acc_cfg": [{
            "snax_acc_name": "snax_alu",
            "snax_streamer_cfg": {"$ref": "snax_alu_streamer.hjson"},
            "snax_datapath_cfg": {"$ref": "snax_alu_acc.hjson"},
        }],
    },
    "dma_core": {"xdma": True},
}  # fmt: skip

STREAMER: dict[str, Any] = {
    "data_reader_params": {"spatial_bounds": [[4], [4]], "temporal_dim": [1, 1],
                           "num_channel": [4, 4], "fifo_depth": [8, 8]},
    "data_writer_params": {"spatial_bounds": [[4]], "temporal_dim": [1], "num_channel": [4],
                           "fifo_depth": [8]},
}  # fmt: skip

UNIT: dict[str, Any] = {
    "unit": "SnaxUnit", "inputs": ["a", "b"], "outputs": ["out"], "lanes": 4, "data_width": 64,
    "op": 0, "signed": True, "control": "counted", "wait": "lockstep", "latency": 0, "ii": 1,
    "fields": ["loopCount"],
}  # fmt: skip

POINT: dict[str, Any] = {
    "name": "vecadd",
    "platform": {"accelerator_slot": "snax_alu"},
    "streamers": {
        "acc_a": {"instance": "acc", "port": "a", "write": False},
        "acc_b": {"instance": "acc", "port": "b", "write": False},
        "acc_out": {"instance": "acc", "port": "out", "write": True},
    },
}


def files() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fresh copies of the cluster file, the streamer file, the datapath file, the point."""
    return tuple(copy.deepcopy(x) for x in (CLUSTER, STREAMER, UNIT, POINT))  # type: ignore[return-value]


def line(path: str, trace: str, text: str, cycle: int = 0) -> str:
    """One events.log line as GVSoC writes it, ANSI colours and padding included."""
    full = f"{PATH}/{path}/{trace}"
    return f"{cycle * 100000}: {cycle}: [\x1b[34m{full:<80}\x1b[0m] {text}"
