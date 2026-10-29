"""The slot's CSR numbers: the streamer's, from its template, then the accelerator's, from
its description (accelerator.py's defines), as snax-gvsoc's SnaxAccelerator and the RTL number
them.

    movers(template) -> [mover]           readers first, named as the headers name them
    mover_csr_count(spatial_bounds, temporal_dim) -> int
    streamer_csrs(template) -> int        snax_streamer.nb_csrs
    channels(template) -> int             snax_cfg.streamer_channels
    csr_layout(template, acc) -> CsrLayout

The generator writes them into the headers and the manifest; ``run/`` reads them back to name
the CSR writes of a run (``run/cluster.py``, ``run/beats.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from snax_forge.lower.c import mover_name

from .accelerator import Accelerator
from .blocks import GenError

# snax-gvsoc snax/snax_accelerator.py: the first SNAX CSR (CSR_SNAX_BEGIN) and the
# last one an accelerator may use (SNAX_CSR_FIRST, SNAX_ACC_CSR_LAST): the core
# sends out up to 0x5FF, but snax_cluster's hw/snitch/src/csr_snax_def.sv
# reserves 0x5FE (CSR_SNAX_READ_TASK_READY_QUEUE) and 0x5FF
# (CSR_SNAX_WRITE_TASK_DONE_QUEUE), which hw/snax_util/src/snax_intf_translator.sv
# routes to the cluster's top-level CSR port, not to the accelerator
SNAX_CSR_FIRST = 0x3C0
SNAX_ACC_CSR_LAST = 0x5FD


@dataclass(frozen=True)
class CsrLayout:
    """The slot's CSRs: the streamer's range, then the accelerator's, each by its define
    (accelerator.py) -> its number."""

    streamer_first: int
    streamer_last: int
    accelerator: dict[str, int]

    @property
    def first(self) -> int:
        """The accelerator's first CSR."""
        return self.streamer_last + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "streamer": {"first": self.streamer_first, "last": self.streamer_last},
            "accelerator": dict(self.accelerator),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CsrLayout:
        return cls(d["streamer"]["first"], d["streamer"]["last"], dict(d["accelerator"]))


def movers(template: dict[str, Any]) -> list[dict[str, Any]]:
    """The template's movers, readers first, named as the headers and the program name them
    (lower/c.py's ``mover_name``)."""
    out = []
    for group, write in (("data_reader_params", False), ("data_writer_params", True)):
        params = template.get(group, {})
        for i in range(len(params.get("num_channel", []))):
            out.append({"name": mover_name(write, i), **{k: v[i] for k, v in params.items()}})
    return out


def mover_csr_count(spatial_bounds: Sequence[int], temporal_dim: int) -> int:
    """A mover's CSRs: the pointer (2), the spatial strides, the temporal bounds and strides."""
    return 2 + len(spatial_bounds) + 2 * temporal_dim


def streamer_csrs(template: dict[str, Any]) -> int:
    """snax_streamer.nb_csrs: per mover its CSRs (``mover_csr_count``); then start; then busy
    and the counter."""
    rw = sum(mover_csr_count(m["spatial_bounds"], m["temporal_dim"]) for m in movers(template))
    return rw + 1 + 2


def channels(template: dict[str, Any]) -> int:
    """The streamer's TCDM ports: num_channel over its movers (snax_cfg.streamer_channels)."""
    return sum(m["num_channel"] for m in movers(template))


def csr_layout(template: dict[str, Any], acc: Accelerator) -> CsrLayout:
    """The streamer's CSRs from ``SNAX_CSR_FIRST``, then the accelerator's; ``gen.csrs`` past
    ``SNAX_ACC_CSR_LAST``."""
    first = SNAX_CSR_FIRST + streamer_csrs(template)
    nums = {name: first + offset for name, offset in acc.defines().items()}
    last = max(nums.values())
    if last > SNAX_ACC_CSR_LAST:
        raise GenError(
            "gen.csrs", f"the accelerator's last CSR {last} is past {SNAX_ACC_CSR_LAST}, the "
            "last an accelerator may use (0x5FE and 0x5FF are snax_cluster's task queues)",
        )  # fmt: skip
    return CsrLayout(SNAX_CSR_FIRST, first - 1, nums)
