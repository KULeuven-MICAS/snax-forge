"""SNAX-DESIGN: a bound graph on a platform, checked, towards the design point (D74, D84, D85).

Built so far, DP1a: the platform file (platform.py): the SNAX cluster
around the accelerators, with its streamer shell, ``--set`` changes and
working copies; the shell resolved against a bound graph (streamers.py,
one streamer ``<instance>_<port>`` per accelerator port); and the design
checks (check.py), which report every problem of a platform, a graph and
their pairing with the change that fixes it (problems.py). The command
line is ``python -m snax_forge.design`` (pixi ``design``). DP1b adds the
memory plan and the design point.
"""

from .check import CHECKS, STAGES, Design, check, load, register_check, run_checks
from .platform import Platform, StreamerOptions, parse_value
from .problems import DesignError, Problem, report
from .streamers import Streamer, instances, nest_spatial_bounds, resolve

__all__ = [
    "CHECKS",
    "STAGES",
    "Design",
    "DesignError",
    "Platform",
    "Problem",
    "Streamer",
    "StreamerOptions",
    "check",
    "instances",
    "load",
    "nest_spatial_bounds",
    "parse_value",
    "register_check",
    "report",
    "resolve",
    "run_checks",
]
