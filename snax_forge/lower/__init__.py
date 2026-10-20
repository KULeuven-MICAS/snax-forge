"""SNAX-LOWER: design point -> cluster file and control program (ARCHITECTURE.md 5.5).

An ordered task list (tasks.py, D45, D64) is lowered to the plain command
list the model runs (commands.py), through the model's register adapters. A
BRM port's nest is mapped through a buffer layout (layout.py, the memory
plan's form) onto a streamer's values (streams.py, D70). A design point's
cluster file (cluster.py, D53, D88): the platform with one streamer per
accelerator port and each accelerator's entry; the checked-in clusters are
built through it too. A design point's task list (derive.py, D89): streamer
values from memlets through the memory plan's layouts, checked against each
BRM's nest; named rates in the accelerator task, so accelerators chain through
L1 with one wait between them; containers the DMA moves padded to whole beats
(D105). The task list as C on snRuntime (c.py, D135), the second backend
beside the command list. The command line is
``python -m snax_forge.lower cluster | tasks`` (pixi ``lower``).
"""

from .c import CError, main_c
from .cluster import Accel, accel_of, cluster_config, cluster_file, cluster_of, stub
from .commands import PlacedWait, lower_program, placed_waits, upstream
from .derive import LowerError, memlet_values, task_list
from .layout import Layout, LayoutError, dma_side
from .program import Program
from .streams import StreamError, streamer_values
from .tasks import Configure, Read, Start, Sync, TaskList, TaskListError, Tasks, step_from_dict
from .values import arg_of, register_values, values_of

__all__ = [
    "Accel",
    "CError",
    "Configure",
    "Layout",
    "LayoutError",
    "LowerError",
    "PlacedWait",
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
    "dma_side",
    "lower_program",
    "main_c",
    "memlet_values",
    "placed_waits",
    "register_values",
    "step_from_dict",
    "streamer_values",
    "stub",
    "task_list",
    "upstream",
    "values_of",
]
