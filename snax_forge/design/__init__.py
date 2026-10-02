"""SNAX-DESIGN: a bound graph on a platform, checked, towards the design point (D74, D84, D85).

Built so far, DP1a: the platform file (platform.py): the SNAX cluster
around the accelerators, with its streamer shell, ``--set`` changes and
working copies; the shell resolved against a bound graph (streamers.py,
one streamer ``<instance>_<port>`` per accelerator port); and the design
checks (check.py), which report every problem of a platform, a graph and
their pairing with the change that fixes it (problems.py). DP1b: the
memory plan (memory.py: registered residency, layout and placement passes,
contiguous by default, pins, and the context later policies read) and the
design point (point.py), what SNAX-LOWER reads. The command line is
``python -m snax_forge.design`` (pixi ``design``).
"""

from .check import CHECKS, STAGES, Design, check, load, register_check, run_checks
from .memory import (
    MEMORY_PASSES,
    MemoryContext,
    MemoryPlan,
    MemorySpec,
    plan,
    register_memory_pass,
)
from .platform import Platform, StreamerOptions, parse_value
from .point import DesignPoint, design_name
from .problems import DesignError, Problem, report
from .streamers import Streamer, instances, nest_spatial_bounds, resolve

__all__ = [
    "CHECKS",
    "MEMORY_PASSES",
    "STAGES",
    "Design",
    "DesignError",
    "DesignPoint",
    "MemoryContext",
    "MemoryPlan",
    "MemorySpec",
    "Platform",
    "Problem",
    "Streamer",
    "StreamerOptions",
    "check",
    "design_name",
    "instances",
    "load",
    "nest_spatial_bounds",
    "parse_value",
    "plan",
    "register_check",
    "register_memory_pass",
    "report",
    "resolve",
    "run_checks",
]
