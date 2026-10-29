"""SNAX-GEN: generated accelerators from a design point.

A design point's accelerator as one description, ``accelerator.json``
(accelerator.py, D145), each datapath an adapter registered by name
(blocks.py); its cluster on SNAX-GVSoC as three description files (D134):
the cluster, the streamer and ``accelerator.json`` (gvsoc.py), and the C
headers a program drives it with (headers.py). The program (D135): ``main.c``
from the task list (SNAX-LOWER's C backend, ``lower/c.py``), and its data,
``data.h`` (data.py). What is written names the design point by its path from
snax-forge's root and is byte-identical wherever it is written (gvsoc.py,
"Stable paths"). The command line is ``python -m snax_forge.gen`` (pixi
``gen-gvsoc``). The accelerator's RTL spec is rtl.py (pixi ``gen-rtl``);
bundle.py (pixi ``bundle``) writes both, with the RTL's file list, for
snax_cluster.
"""

from .accelerator import ACCELERATOR_JSON, Accelerator, Csr, Group, Link, Unit, describe
from .blocks import DATAPATHS, ELEMENTWISE_OP, Fill, GenError, Port, register_datapath
from .csrs import CsrLayout
from .gvsoc import AccTask, Manifest, generate

__all__ = [
    "ACCELERATOR_JSON",
    "DATAPATHS",
    "ELEMENTWISE_OP",
    "AccTask",
    "Accelerator",
    "Csr",
    "CsrLayout",
    "Fill",
    "GenError",
    "Group",
    "Link",
    "Manifest",
    "Port",
    "Unit",
    "describe",
    "generate",
    "register_datapath",
]
