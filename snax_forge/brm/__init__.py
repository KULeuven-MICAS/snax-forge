"""SNAX-BRM: block runtime models, the user's accelerators as plain data (ARCHITECTURE.md 5.3).

A BRM is a hand-written JSON file (D68) with a shared part (interface,
function, dataflow, pattern) and a map of implementations (source, supports,
timing, binding). ``Brm.resolve`` builds an instance from an implementation
and design-param values; its ``accel_entry`` is the accelerator entry of the
cluster file, checked against the model's registered kind. The ``affine``
dataflow notation (D70) gives nests that ``task_nest`` resolves and
enumerates for one task; SNAX-LOWER maps them onto streamer values
(``snax_forge.lower.streams``). The library of hand-written BRMs
(library.py, ``load_brm``) holds ``elementwise_add``, ``elementwise_mul``
and ``accumulate``, a reduction named by its pattern and kind. ``drain`` is
a third timing number (D103), and port names can appear in code. An
implementation's ``binding`` (binding.py, D129) has its ``control`` and
``wait`` kinds registered.
"""

from snax_forge.expr import ExprError

from . import affine  # registers the "affine" notation
from .affine import Nest, resolve_nest, task_nest
from .binding import CONTROLS, WAITS, register_control, register_wait
from .brm import (
    Brm,
    BrmError,
    Dataflow,
    Function,
    Implementation,
    Interface,
    Param,
    Pattern,
    Port,
    Timing,
)
from .instance import Instance, resolve
from .library import LIBRARY, library_names, load_brm
from .notation import NOTATIONS, register_notation

__all__ = [
    "CONTROLS",
    "LIBRARY",
    "NOTATIONS",
    "WAITS",
    "Brm",
    "BrmError",
    "Dataflow",
    "ExprError",
    "Function",
    "Implementation",
    "Instance",
    "Interface",
    "Nest",
    "Param",
    "Pattern",
    "Port",
    "Timing",
    "affine",
    "library_names",
    "load_brm",
    "register_control",
    "register_notation",
    "register_wait",
    "resolve",
    "resolve_nest",
    "task_nest",
]
