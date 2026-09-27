"""SNAX-LOWER: design point -> cluster file and control program (ARCHITECTURE.md 5.5).

Built so far, LOW1b (D45, D63, D64): an ordered task list (tasks.py) is
lowered to the plain command list the model runs (commands.py), through the
model's register adapters. BRM2 (D70): a BRM port's nest is mapped through a
buffer layout (layout.py, the memory plan's form since DP1b) onto a streamer's values
(streams.py). LOW1c (D53, D88): a design point's cluster file (cluster.py),
the platform with one streamer per accelerator port and each accelerator's
entry; the checked-in clusters are built through it too. The design point
-> task list step (LOW1a) comes next. The command line is
``python -m snax_forge.lower cluster`` (pixi ``lower``).
"""

from .cluster import Accel, accel_of, cluster_config, cluster_file, cluster_of, stub
from .commands import lower_program, upstream
from .layout import Layout, LayoutError
from .program import Program
from .streams import StreamError, streamer_values
from .tasks import Configure, Read, Start, Sync, TaskList, TaskListError, Tasks, step_from_dict
from .values import arg_of, register_values, values_of

__all__ = [
    "Accel",
    "Configure",
    "Layout",
    "LayoutError",
    "Program",
    "Read",
    "Start",
    "StreamError",
    "Sync",
    "TaskList",
    "TaskListError",
    "Tasks",
    "accel_of",
    "arg_of",
    "cluster_config",
    "cluster_file",
    "cluster_of",
    "lower_program",
    "register_values",
    "step_from_dict",
    "streamer_values",
    "stub",
    "upstream",
    "values_of",
]
