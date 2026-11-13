"""SNAX-BRM: block runtime models, the user's accelerators as plain data (ARCHITECTURE, "SNAX-BRM").

The format is CONTRACTS, "Block runtime model". brm.py holds a BRM, its shared part and its
implementations; ``Brm.resolve`` builds an instance (instance.py), checked against the
generator's blocks. function.py has the accelerator kinds the reference executor runs
(``function_of``); notation.py, affine.py and stream.py the dataflow notations; binding.py an
implementation's binding; dtypes.py a port's dtypes rule (``check_dtypes``); library.py the
hand-written BRMs (``load_brm``).
"""

from snax_forge.expr import ExprError

from . import affine, stream  # register the "affine" and "stream" notations
from .affine import Nest, resolve_nest, task_nest
from .binding import CONTROLLERS, register_controller
from .brm import (
    SEED,
    Brm,
    BrmError,
    Code,
    Dataflow,
    Function,
    Implementation,
    Interface,
    Param,
    Pattern,
    Port,
    Timing,
)
from .dtypes import (
    INT_DTYPES,
    accepted,
    check_dtypes,
    fits,
    output_dtypes,
    port_dtypes,
    result_type,
)
from .function import KINDS, OPS, AccelFunction, AccelPort, function_of, register_kind, register_op
from .instance import Instance, resolve
from .library import LIBRARY, library_names, load_brm
from .notation import NOTATIONS, register_notation
from .stream import memlet_ordered

__all__ = [
    "CONTROLLERS",
    "INT_DTYPES",
    "KINDS",
    "LIBRARY",
    "NOTATIONS",
    "OPS",
    "SEED",
    "AccelFunction",
    "AccelPort",
    "Brm",
    "BrmError",
    "Code",
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
    "accepted",
    "affine",
    "check_dtypes",
    "fits",
    "function_of",
    "library_names",
    "load_brm",
    "memlet_ordered",
    "output_dtypes",
    "port_dtypes",
    "register_controller",
    "register_kind",
    "register_notation",
    "register_op",
    "resolve",
    "resolve_nest",
    "result_type",
    "stream",
    "task_nest",
]
