"""SNAX-GEN: generated accelerators from a design point.

The accelerators' one description, ``accelerator.json`` (accelerator.py), each datapath an
adapter registered by name (blocks.py); the cluster on SNAX-GVSoC, its headers, ``data.h`` and
``main.c`` (gvsoc.py, headers.py, data.py, with lower/c.py), ``python -m snax_forge.gen``
(``pixi run gen-gvsoc``); the accelerator's RTL (rtl.py, ``gen-rtl``); and the bundle for
snax_cluster (bundle.py, ``bundle``). One design point gives the same bytes wherever written.
"""

from .accelerator import (
    ACCELERATOR_JSON,
    Accelerator,
    Csr,
    Description,
    Group,
    Link,
    Unit,
    describe,
)
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
    "Description",
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
