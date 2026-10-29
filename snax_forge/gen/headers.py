"""The generated C headers: what a program needs to drive the accelerator,
and the runtime's view of the cluster.

``generate`` writes them to ``out/gvsoc/<design>/include/`` and lists them
in the manifest:

    streamer_csr_addr_map.h   the streamer's CSRs, with the name, defines,
                              values and order of the header snax_cluster's
                              Chisel streamer writes (``Streamer.scala``,
                              ``writeCsrAddrMap`` at the pin of snax-gvsoc
                              ``sw/README.md``): per mover, readers then
                              writers (``READER_<i>``, ``WRITER_<j>``),
                              ``BASE_PTR_<M>_LOW`` / ``_HIGH``, then for the
                              spatial strides, the temporal bounds and the
                              temporal strides a ``_BASE_<M>``, a ``_NUM_<M>``
                              and one define per index; then
                              ``STREAMER_START_CSR``, ``STREAMER_BUSY_CSR``,
                              ``STREAMER_PERFORMANCE_COUNTER_CSR``
    <name>_csr.h              the accelerator's CSRs (accelerator.py, D146),
                              each by ``Accelerator.define``, in CSR order:
                              ``<ACC>_<RW|RO>_[<GROUP>_][<UNIT>_]<NAME>``;
                              vecadd's ``ACC_RW_START``, ``ACC_RW_JOBS``,
                              ``ACC_RO_STATUS``, ``ACC_RO_COUNTER``
    snitch_cluster_cfg.h      the runtime's core count and first hart
                              (``CFG_CLUSTER_NR_CORES``, the cores over the
                              hives, ``$ref``s resolved, as clustergen
                              counts them; ``CFG_CLUSTER_BASE_HARTID``)
    snitch_cluster_addrmap.h  the runtime's address map: the L1's base and
                              size, the cluster's size and address width,
                              the peripherals and the zero memory after the
                              L1, the CLINT

The two cluster headers are what snax_cluster's templates
``target/snitch_cluster/sw/runtime/common/snitch_cluster_cfg.h.tpl`` and
``snitch_cluster_addrmap.h.tpl`` (at the pin of snax-gvsoc ``sw/README.md``)
render from the cluster file, the generated one: below the header comment,
text-equal but for the templates' copyright and blank lines. The runtime
puts its stacks at the L1's end (``start.S``), so a program and its runtime
built against them fit the design point's L1 (D126). A
``cluster_base_offset`` that is not a power of two is a ``GenError``
(``gen.platform``): ``CLUSTER_ADDRWIDTH`` is its log2; a missing or
non-integer key is ``gen.source``.

Only the comments differ from the Chisel header. Streamer.scala's optional
per-mover CSRs (address remap, channel and byte enables) and its datapath
extensions are not in snax-gvsoc's streamer model, so they never appear.
The one-dimensional case is checked against the Chisel header snaxgen
writes for snax-gvsoc (``sw/generated/snax_forge/``); more dimensions follow
Streamer.scala's loops, read, not compared with a generated file.
Neither header has an include guard, as the Chisel one has none.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from snax_forge.common import Missing, dig, generated_header
from snax_forge.design import DesignError
from snax_forge.design.hjson_platform import cores

from .blocks import GenError

STREAMER_HEADER = "streamer_csr_addr_map.h"
CFG_HEADER = "snitch_cluster_cfg.h"
ADDRMAP_HEADER = "snitch_cluster_addrmap.h"


def streamer_header(movers: list[dict[str, Any]], first: int, point_path: str) -> tuple[str, int]:
    """The text of streamer_csr_addr_map.h for ``movers`` (gvsoc.movers) from CSR ``first``,
    and the streamer's last CSR."""
    lines = [generated_header(point_path, "The streamer's CSRs, as snax_cluster's Streamer.scala.")]
    csr = first
    for m in movers:
        tag = m["name"]
        lines.append(f"// CSR Map for {tag}")
        lines.append(f"#define BASE_PTR_{tag}_LOW {csr}")
        lines.append(f"#define BASE_PTR_{tag}_HIGH {csr + 1}")
        csr += 2
        for kind, n in (("S_STRIDE", len(m["spatial_bounds"])), ("T_BOUND", m["temporal_dim"]),
                        ("T_STRIDE", m["temporal_dim"])):  # fmt: skip
            lines.append(f"#define {kind}_BASE_{tag} {csr}")
            lines.append(f"#define {kind}_NUM_{tag} {n}")
            for i in range(n):
                lines.append(f"#define {kind}_{tag}_{i} {csr}")
                csr += 1
    lines += [
        "// Status register",
        f"#define STREAMER_START_CSR {csr}",
        "// Read only CSRs",
        f"#define STREAMER_BUSY_CSR {csr + 1}",
        f"#define STREAMER_PERFORMANCE_COUNTER_CSR {csr + 2}",
    ]
    return "\n".join(lines) + "\n", csr + 2


def accelerator_header(acc: Any, first: int, point_path: str) -> str:
    """The text of <name>_csr.h for the description ``acc`` (accelerator.Accelerator), its
    first CSR at ``first``."""
    what = (f"Accelerator {acc.name}'s CSRs (accelerator.json, D146): per group START, JOBS, "
            "its units' R/W, STATUS, COUNTER.")  # fmt: skip
    lines = [generated_header(point_path, what)]
    lines += [f"#define {name} {first + offset}" for name, offset in acc.defines().items()]
    return "\n".join(lines) + "\n"


def _int(d: Mapping[str, Any], path: str) -> int:
    """The integer at the dotted ``path`` of the cluster file ``d``; ``gen.source`` if not one."""
    try:
        node = dig(d, path)
    except Missing:
        node = None
    if isinstance(node, bool) or not isinstance(node, int):
        raise GenError("gen.source", f"the cluster file's {path} is {node!r}, not an integer")
    return node


def cluster_headers(cluster: Mapping[str, Any], point_path: str) -> dict[str, str]:
    """The texts of snitch_cluster_cfg.h and snitch_cluster_addrmap.h by file name, for the
    cluster file ``cluster`` as written."""
    try:
        nr_cores = len(cores(cluster))
    except DesignError as e:
        raise GenError("gen.source", str(e)) from None
    offset = _int(cluster, "cluster.cluster_base_offset")
    if offset <= 0 or offset & (offset - 1):
        raise GenError(
            "gen.platform", f"cluster.cluster_base_offset {offset} ({offset:#x}) is not a power "
            "of two: CLUSTER_ADDRWIDTH is its log2",
        )  # fmt: skip
    base = _int(cluster, "cluster.cluster_base_addr")
    tpl = "as snax_cluster's {}.tpl renders it from the cluster file."
    cfg = [
        generated_header(
            point_path, f"The runtime's core count and first hart, {tpl.format(CFG_HEADER)}"
        ),
        f"#define CFG_CLUSTER_NR_CORES {nr_cores}",
        f"#define CFG_CLUSTER_BASE_HARTID {_int(cluster, 'cluster.cluster_base_hartid')}",
    ]
    kib = {k: hex(_int(cluster, f"cluster.{k}") * 1024)
           for k in ("tcdm.size", "cluster_periph_size", "zero_mem_size")}  # fmt: skip
    addrmap = [
        generated_header(point_path, f"The runtime's address map, {tpl.format(ADDRMAP_HEADER)}"),
        f"#define CLUSTER_TCDM_BASE_ADDR {hex(base)}",
        f"#define CLUSTER_TCDM_SIZE {kib['tcdm.size']}",
        f"#define CLUSTER_SIZE {hex(offset)}",
        f"#define CLUSTER_BASE_ADDR {base}",
        f"#define CLUSTER_ADDRWIDTH {offset.bit_length() - 1}",
        "#define CLUSTER_PERIPH_BASE_ADDR (CLUSTER_TCDM_BASE_ADDR + CLUSTER_TCDM_SIZE)",
        (
            "#define CLUSTER_ZERO_MEM_START_ADDR (CLUSTER_PERIPH_BASE_ADDR + "
            f"{kib['cluster_periph_size']})"
        ),
        f"#define CLUSTER_ZERO_MEM_END_ADDR (CLUSTER_ZERO_MEM_START_ADDR + {kib['zero_mem_size']})",
        f"#define CLINT_BASE_ADDR {hex(_int(cluster, 'peripherals.clint.address'))}",
    ]
    return {CFG_HEADER: "\n".join(cfg) + "\n", ADDRMAP_HEADER: "\n".join(addrmap) + "\n"}
