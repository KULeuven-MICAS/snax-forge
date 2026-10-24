"""SNAX-GEN: generated accelerators from a design point.

A design point's cluster on SNAX-GVSoC as three hjson files (D134):
the cluster, the streamer and the datapath block (gvsoc.py), each Chisel
block an adapter registered by name (blocks.py), and the C headers a program
drives it with (headers.py). The program (D135): ``main.c`` from the task
list (SNAX-LOWER's C backend, ``lower/c.py``), and its data, ``data.h``
(data.py). What is written names the design point by its path from
snax-forge's root and is byte-identical wherever it is written (gvsoc.py,
"Stable paths"). The command line is ``python -m snax_forge.gen`` (pixi ``gen-gvsoc``).
The accelerator's RTL spec is rtl.py (pixi ``gen-rtl``); bundle.py (pixi
``bundle``) writes both, with the RTL's file list, for snax_cluster.
"""

from .blocks import BLOCKS, ELEMENTWISE_OP, GenError, Unit, register_block
from .gvsoc import AccTask, CsrLayout, Manifest, generate

__all__ = [
    "BLOCKS",
    "ELEMENTWISE_OP",
    "AccTask",
    "CsrLayout",
    "GenError",
    "Manifest",
    "Unit",
    "generate",
    "register_block",
]
