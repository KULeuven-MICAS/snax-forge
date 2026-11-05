"""The entry's CSR numbers (D157, C§16): per streamer set, its streamer's, from its template,
then its accelerator's own, from the description (accelerator.py's defines), packed from 960,
as snax-gvsoc's SnaxAccelerator and snax_cluster's ``snax_csr_range_demux`` number them.

    movers(template) -> [mover]           readers first, named as the headers name them
    mover_csr_count(spatial_bounds, temporal_dim) -> int
    streamer_csrs(template) -> int        snax_streamer.nb_csrs
    channels(template) -> int             snax_cfg.streamer_channels
    csr_layout(templates, desc) -> CsrLayout

With s_k and c_k set k's streamer and own CSRs, b_0 = 960 and b_{k+1} = b_k + s_k + c_k:
streamer k at [b_k, b_k + s_k), its own at [b_k + s_k, b_{k+1}); a CSR at offset o from the
shell's first (accelerator.json's, contiguous over the sets) of set k is b_k + s_k + o - o_k,
o_k the own CSRs of the sets before it. ``gen.csrs`` when b_K - 1 is past ``SNAX_ACC_CSR_LAST``.
The generator writes them into the headers and the manifest; ``run/`` reads them back to name
the CSR writes of a run (``run/cluster.py``, ``run/beats.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from snax_forge.lower.c import mover_name

from .accelerator import Description
from .blocks import GenError

# snax-gvsoc snax/snax_accelerator.py: the first SNAX CSR (CSR_SNAX_BEGIN) and the
# last one an accelerator may use (SNAX_CSR_FIRST, SNAX_ACC_CSR_LAST): the core
# sends out up to 0x5FF, but snax_cluster's hw/snitch/src/csr_snax_def.sv
# reserves 0x5FE (CSR_SNAX_READ_TASK_READY_QUEUE) and 0x5FF
# (CSR_SNAX_WRITE_TASK_DONE_QUEUE), which hw/snax_util/src/snax_intf_translator.sv
# routes to the cluster's top-level CSR port, not to the accelerator
SNAX_CSR_FIRST = 0x3C0
SNAX_ACC_CSR_LAST = 0x5FD
# The trace paths: the compute core's accelerator, named snax_acc<core> by
# pulp/snitch/snitch_cluster/snitch_cluster.py
ACC_TRACE = "snax_acc0"


@dataclass(frozen=True)
class StreamerRange:
    """One streamer set's streamer: its name, first and last CSR and its trace path."""

    name: str
    first: int
    last: int
    trace: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "first": self.first, "last": self.last, "trace": self.trace}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StreamerRange:
        return cls(d["name"], d["first"], d["last"], d["trace"])


@dataclass(frozen=True)
class CsrLayout:
    """The entry's CSRs: each set's streamer range, in set order, and every accelerator CSR by
    its define (accelerator.py) -> its number."""

    streamers: list[StreamerRange]
    accelerator: dict[str, int]

    def streamer(self, name: str) -> StreamerRange:
        """Set ``name``'s streamer range."""
        return next(s for s in self.streamers if s.name == name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "streamers": [s.to_dict() for s in self.streamers],
            "accelerator": dict(self.accelerator),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CsrLayout:
        return cls([StreamerRange.from_dict(s) for s in d["streamers"]], dict(d["accelerator"]))


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


def csr_layout(templates: Mapping[str, dict[str, Any]], desc: Description) -> CsrLayout:
    """Each set's streamer from ``SNAX_CSR_FIRST``, then its accelerator's own CSRs, set after
    set (module doc); ``templates`` each set's streamer template by its name, ``gen.csrs`` past
    ``SNAX_ACC_CSR_LAST``."""
    base, shell, ranges, nums = SNAX_CSR_FIRST, 0, [], {}
    for acc in desc.accelerators:
        s = streamer_csrs(templates[acc.name])
        ranges.append(
            StreamerRange(acc.name, base, base + s - 1, f"{ACC_TRACE}/streamer_{acc.name}")
        )
        own = acc.defines()
        nums |= {name: base + s + offset - shell for name, offset in own.items()}
        base += s + len(own)
        shell += len(own)
    if base - 1 > SNAX_ACC_CSR_LAST:
        raise GenError(
            "gen.csrs", f"the entry's last CSR {base - 1} is past {SNAX_ACC_CSR_LAST}, the last an "
            "accelerator may use (0x5FE and 0x5FF are snax_cluster's task queues): every streamer "
            "set's streamer and own CSRs, packed from 960",
        )  # fmt: skip
    return CsrLayout(ranges, nums)
