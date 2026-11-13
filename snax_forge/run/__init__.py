"""A flow run's measurements: SNAX-GVSoC's events converted into the profile and the trace the
reports and the viewer read (CONTRACTS, "Profile and trace").

* events.py: ``run/events.log`` read in snax-gvsoc's grammar (its ``docs/EVENTS.md``);
* cluster.py: the cluster as the run names it, from the generator's files and the design point;
* regions.py: where each container lives in each memory (L2 from the ELF, elf.py);
* profile.py: the profile, the totals of one run, from the components' own counters;
* trace.py: the trace, one event per line, in the program's window;
* convert.py and beats.py: the converter and its invariants;
* cores.py: a core's cycles classed barrier > poll > sleep > run, as spans;
* sources.py: a trace level and its sources, as SNAX-GVSoC's traces and options;
* outputs.py: a run directory written and read back.
"""

from .cluster import Accelerator, Cluster, Csrs, Mover, Unit, cluster_from, load_cluster
from .cores import CORE_CLASSES, core_counts, core_spans, merge_spans, start_mode
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
from .regions import Padding, Region, padding_of, region_at, regions_of
from .trace import KINDS, LEVELS, Trace, at_least

__all__ = [
    "CORE_CLASSES",
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
    "Padding",
    "PortProfile",
    "Profile",
    "Region",
    "RunError",
    "RunOutputs",
    "StreamerProfile",
    "Trace",
    "Unit",
    "WaitProfile",
    "at_least",
    "cluster_from",
    "core_counts",
    "core_spans",
    "load_cluster",
    "merge_spans",
    "padding_of",
    "parse_line",
    "read_events",
    "read_run",
    "region_at",
    "regions_of",
    "start_mode",
    "write_run",
]
