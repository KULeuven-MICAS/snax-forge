"""SNAX-GEN: generated accelerators from a design point (M14).

Built so far, GEN3 step 3 (D128, D132): a design point's accelerator on
SNAX-GVSoC, as Python on snax-gvsoc's installed components (gvsoc.py),
each Chisel block an adapter registered by name (blocks.py). The command
line is ``python -m snax_forge.gen`` (pixi ``gen-gvsoc``).
"""

from .blocks import BLOCKS, ELEMENTWISE_OP, GenError, Unit, register_block
from .gvsoc import IMPORTS, PREFIX, AccTask, CsrLayout, Manifest, check_install, generate

__all__ = [
    "BLOCKS",
    "ELEMENTWISE_OP",
    "IMPORTS",
    "PREFIX",
    "AccTask",
    "CsrLayout",
    "GenError",
    "Manifest",
    "Unit",
    "check_install",
    "generate",
    "register_block",
]
