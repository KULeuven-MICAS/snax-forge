"""A flow run's measurements (C§7, D137): SNAX-GVSoC's events converted into the profile and the
trace the reports and the viewer read.

* events.py: ``run/events.log`` read, line by line, in snax-gvsoc's grammar (its
  ``docs/EVENTS.md``);
* cluster.py: the cluster as the run is named: the accelerator, its movers and CSRs, the
  cores, the DMA, the L1 and its word-interleaved map, from the generator's hjson files and
  the design point;
* regions.py: where each container lives in each memory: L1 from the design point's layouts,
  L2 from the program's ELF (elf.py, its symbols as run.json records them);
* profile.py: the profile, the totals of one run, from the components' own counters;
* trace.py: the trace, one event per line, in the program's window;
* beats.py: the converter's beat, stall and instruction part and its invariants (convert.py
  the rest);
* sources.py: a trace level and its sources, as SNAX-GVSoC's traces and options;
* outputs.py: a run directory written and read back.
"""

from .cluster import Accelerator, Cluster, Csrs, Mover, cluster_from, load_cluster
from .events import GRAMMAR, Line, RunError, parse_line, read_events
from .outputs import RunOutputs, read_run, write_run
from .profile import (
    AccelProfile,
    BankProfile,
    CoreProfile,
    DmaProfile,
    FifoProfile,
    PortProfile,
    Profile,
    StreamerProfile,
    WaitProfile,
)
from .regions import Region, region_at, regions_of
from .trace import KINDS, LEVELS, Trace

__all__ = [
    "GRAMMAR",
    "KINDS",
    "LEVELS",
    "AccelProfile",
    "Accelerator",
    "BankProfile",
    "Cluster",
    "CoreProfile",
    "Csrs",
    "DmaProfile",
    "FifoProfile",
    "Line",
    "Mover",
    "PortProfile",
    "Profile",
    "Region",
    "RunError",
    "RunOutputs",
    "StreamerProfile",
    "Trace",
    "WaitProfile",
    "cluster_from",
    "load_cluster",
    "parse_line",
    "read_events",
    "read_run",
    "region_at",
    "regions_of",
    "write_run",
]
