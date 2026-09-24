"""SNAX-BRM: block runtime models, the user's accelerators as plain data (ARCHITECTURE.md 5.3).

Built so far, BRM1 (D68): a BRM is a hand-written JSON file with a shared
part (interface, function, dataflow, pattern) and a map of implementations
(source, supports, timing, binding). ``Brm.resolve`` builds an instance from
an implementation and design-param values; its ``accel_entry`` is the
accelerator entry of the cluster file, checked against the model's
registered kind. BRM2 (D70): the ``affine`` dataflow notation, whose nests
``task_nest`` resolves and enumerates for one task; SNAX-LOWER maps them
onto streamer values (``snax_forge.lower.streams``). BRM3: the library of
hand-written BRMs (library.py, ``load_brm``), starting with
``elementwise_add``.
"""

from snax_forge.expr import ExprError

from . import affine  # registers the "affine" notation (D70)
from .affine import Nest, resolve_nest, task_nest
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
    "LIBRARY",
    "NOTATIONS",
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
    "register_notation",
    "resolve",
    "resolve_nest",
    "task_nest",
]
