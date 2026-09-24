"""SNAX-BRM: block runtime models, the user's accelerators as plain data (ARCHITECTURE.md 5.3).

Built so far, BRM1 (D68): a BRM is a hand-written JSON file with a shared
part (interface, function, dataflow, pattern) and a map of implementations
(source, supports, timing, binding). ``Brm.resolve`` builds an instance from
an implementation and design-param values; its ``accel_entry`` is the
accelerator entry of the cluster file, checked against the model's
registered kind. The affine notation (BRM2) and the library's first BRM
(BRM3) come later.
"""

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
from .expr import ExprError
from .instance import Instance, resolve
from .notation import NOTATIONS, register_notation

__all__ = [
    "NOTATIONS",
    "Brm",
    "BrmError",
    "Dataflow",
    "ExprError",
    "Function",
    "Implementation",
    "Instance",
    "Interface",
    "Param",
    "Pattern",
    "Port",
    "Timing",
    "register_notation",
    "resolve",
]
