"""A run's trace level and sources, as SNAX-GVSoC is told them (C§7, D137).

    parse_trace("stall:acc,acc_a")      -> ("stall", ["acc", "acc_a"])
    sources_for(cluster, level, given)  -> the sources, given or by default
    gvsoc_options(cluster, level, sources, vcd) -> GvsocOptions

A level is which traces are on (snax-gvsoc ``docs/EVENTS.md``), cumulative: ``off`` the
``summary`` lines, ``task`` and the ``events`` lines, ``beat`` and the ``beats`` lines,
``stall`` and the ``stalls`` lines, ``finegrained`` and the cores' ``insn`` lines. The sources
are C§7 names (the accelerator ``acc``, its movers, ``dma``, ``core<i>``, ``axi``); they say
whose beat, stall and instruction lines are written, never the summary's or events'. By
default, at ``beat`` and ``stall``, the accelerator, its movers and the DMA; at
``finegrained`` also every core, for its ``insn`` lines only (its crossbar lines are written
when it is named).

``gvsoc_options`` maps them onto SNAX-GVSoC: the trace names for ``gvsoc/Makefile``'s
``TRACES`` (one ``--trace`` regex each: ``summary`` and ``events`` whole, then per component,
the unit's ``beats`` for ``acc``, the streamer's ``beats`` (its ``class`` lines) and ``stalls``
for a mover, the crossbar's
``beats`` and ``stalls`` for its inputs, the AXI backend's ``beats`` for ``dma``, ``pe<i>/insn``
for a core), the crossbar's ``trace_inputs`` parameter (the named inputs by name: a mover's
lanes ``snax_acc<i>.streamer.<mover>.lane<l>``, ``core<i>``, ``axi``, and ``dma``), and with
``vcd`` gvrun's ``--vcd`` with one ``--event`` regex per source's components. A level or
source this run does not have is a RunError (``sources.<rule>``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cluster import DMA, Cluster
from .events import RunError
from .trace import LEVELS

XBAR_PATH = "chip/soc/cluster_0/tcdm/snax_xbar"
AXI = "axi"
# level -> the traces it adds
LEVEL_TRACES = {"off": "summary", "task": "events", "beat": "beats", "stall": "stalls",
                "finegrained": "insn"}  # fmt: skip


def parse_trace(text: str) -> tuple[str, list[str] | None]:
    """``LEVEL[:SOURCES]``: the level, and the sources (joined by ``,``) or None."""
    level, sep, rest = text.partition(":")
    if level not in LEVELS:
        raise RunError("sources.level", f"trace level {level!r}: one of {', '.join(LEVELS)}")
    if not sep:
        return level, None
    names = [s for s in rest.split(",") if s]
    if level in ("off", "task") or not names:
        raise RunError("sources.given", f"--trace {text}: sources go with beat, stall or "
                       "finegrained, as LEVEL:NAME,NAME")  # fmt: skip
    return level, names


def all_sources(cluster: Cluster) -> list[str]:
    """Every source a run can name, in the cluster's order, then AXI."""
    return [*cluster.sources, AXI]


def sources_for(cluster: Cluster, level: str, given: list[str] | None) -> list[str]:
    """The sources of a run (module doc), in the cluster's order."""
    if LEVELS.index(level) < LEVELS.index("beat"):
        return []
    known = all_sources(cluster)
    if given is not None:
        unknown = [s for s in given if s not in known]
        if unknown:
            raise RunError("sources.unknown", f"sources {unknown}: not of this run (its "
                           f"sources are {', '.join(known)})")  # fmt: skip
        return [s for s in known if s in given]
    a = cluster.accelerator
    out = [a.name, *(m.name for m in a.movers), DMA]
    if level == "finegrained":
        out += cluster.cores
    return out


@dataclass
class GvsocOptions:
    """What gvsoc/Makefile's run gets: TRACES, and gvrun options for GVRUN_ARGS."""

    traces: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)


def gvsoc_options(cluster: Cluster, level: str, sources: list[str], vcd: bool,
                  given: bool = False) -> GvsocOptions:  # fmt: skip
    """The traces, the crossbar's inputs and the VCD's events of a run (module doc); ``given``
    says the sources were named, so a core's crossbar lines are written too."""
    on = LEVELS[: LEVELS.index(level) + 1]
    a = cluster.accelerator
    acc = f"snax_acc{a.core}"
    movers = [m for m in a.movers if m.name in sources]
    cores = [c for c in cluster.cores if c in sources]
    xbar_cores = cores if given or level != "finegrained" else []
    inputs = [f"{acc}.streamer.{m.gvsoc}.lane{lane}" for m in movers for lane in range(m.lanes)]
    inputs += xbar_cores + ([AXI] if AXI in sources else []) + ([DMA] if DMA in sources else [])
    o = GvsocOptions(traces=["summary"] + (["events"] if "task" in on else []))
    if "beat" in on:
        if a.name in sources:
            o.traces.append(f"{acc}/unit/beats")
        if movers:
            o.traces.append(f"{acc}/streamer/beats")
        if inputs:
            o.traces.append("tcdm/snax_xbar/beats")
        if DMA in sources:
            o.traces += ["idma/axi_read/beats", "idma/axi_write/beats"]
    if "stall" in on:
        if movers:
            o.traces.append(f"{acc}/streamer/stalls")
        if [i for i in inputs if i != DMA]:
            o.traces.append("tcdm/snax_xbar/stalls")
    if "finegrained" in on:
        o.traces += [f"{c.replace('core', 'pe')}/insn" for c in cores]
    if "beat" in on:
        o.args += ["--param", f"{XBAR_PATH}/trace_inputs={','.join(inputs)}"]
    if vcd:
        events = []
        if a.name in sources:
            events.append(f"{acc}/unit/.*")
        if movers:
            events.append(f"{acc}/streamer/.*")
        if inputs:
            events.append("tcdm/snax_xbar/.*")
        if DMA in sources:
            events.append("idma/fe/dma_busy")
        events += [f"{c.replace('core', 'pe')}/.*" for c in cores]
        o.args += ["--vcd", *(f"--event=.*/{e}" for e in events)]
    return o
