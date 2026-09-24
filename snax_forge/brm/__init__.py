"""SNAX-BRM: block runtime models, the user's accelerators as plain data (ARCHITECTURE.md 5.3).

Built so far, BRM1's format (D68): a BRM is a hand-written JSON file with a
shared part (interface, function, dataflow, pattern) and a map of
implementations (source, supports, timing, binding). The link to the
accelerator entry of the cluster file, the affine notation (BRM2) and the
library's first BRM (BRM3) come later.
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
from .notation import NOTATIONS, register_notation

__all__ = [
    "NOTATIONS",
    "Brm",
    "BrmError",
    "Dataflow",
    "ExprError",
    "Function",
    "Implementation",
    "Interface",
    "Param",
    "Pattern",
    "Port",
    "Timing",
    "register_notation",
]
