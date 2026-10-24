"""SNAX-LOWER: design point -> task list -> program (ARCHITECTURE.md 5.5).

An ordered task list (tasks.py, D64) says which component runs which task
with which values; ``check_values`` checks it against the design point, and
waits.py places the waits correctness needs. A BRM port's nest is mapped
through a buffer layout (layout.py, the memory plan's form) onto a
streamer's values (streams.py, D70). A design point's task list (derive.py,
D64): streamer values from memlets through the memory plan's layouts,
checked against each BRM's nest; named rates in the accelerator task, so
accelerators chain through L1 with one wait between them; containers the DMA
moves padded to whole beats (D105). The task list as C on snRuntime (c.py,
D135). The command line is ``python -m snax_forge.lower tasks`` (pixi
``lower``).
"""

from .c import CError, main_c
from .derive import LowerError, memlet_values, task_list
from .layout import Layout, LayoutError, dma_side
from .streams import StreamError, streamer_values
from .tasks import (
    Configure,
    Read,
    Start,
    Sync,
    TaskList,
    TaskListError,
    check_values,
    components,
    step_from_dict,
)
from .waits import PlacedWait, placed_waits, upstream

__all__ = [
    "CError",
    "Configure",
    "Layout",
    "LayoutError",
    "LowerError",
    "PlacedWait",
    "Read",
    "Start",
    "StreamError",
    "Sync",
    "TaskList",
    "TaskListError",
    "check_values",
    "components",
    "dma_side",
    "main_c",
    "memlet_values",
    "placed_waits",
    "step_from_dict",
    "streamer_values",
    "task_list",
    "upstream",
]
