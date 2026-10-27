"""SNAX-BRM: block runtime models, the user's accelerators as plain data (ARCHITECTURE.md 5.3).

A BRM is a hand-written JSON file (D68) with a shared part (interface,
function, dataflow, pattern) and a map of implementations (source, supports,
timing, binding). ``Brm.resolve`` builds an instance from an implementation
and design-param values, checked against the generator's blocks; its
function (function.py, ``function_of``) is a registered accelerator kind the
reference executor runs. Two dataflow notations (D70): ``affine`` gives
nests that ``task_nest`` resolves and enumerates for one task, which
SNAX-LOWER maps onto streamer values (``snax_forge.lower.streams``);
``stream``, for unit BRMs, leaves the order to the bound memlet
(``memlet_ordered``). The library of hand-written BRMs
(library.py, ``load_brm``) holds ``elementwise_add``, ``elementwise_mul``
and ``accumulate``, a reduction named by its pattern and kind. ``drain`` is
a third timing number (D103), and port names can appear in code. An
implementation's ``binding`` (binding.py, D129) has its ``control`` and
``wait`` kinds registered. A port's ``dtypes`` are a rule (dtypes.py,
``check_dtypes``, D143): a node's containers give the dtypes.
"""

from snax_forge.expr import ExprError

from . import affine, stream  # register the "affine" and "stream" notations
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
from .dtypes import INT_DTYPES, accepted, check_dtypes, output_dtypes, port_dtypes
from .function import KINDS, OPS, AccelFunction, AccelPort, function_of, register_kind, register_op
from .instance import Instance, resolve
from .library import LIBRARY, library_names, load_brm
from .notation import NOTATIONS, register_notation
from .stream import memlet_ordered

__all__ = [
    "CONTROLS",
    "INT_DTYPES",
    "KINDS",
    "LIBRARY",
    "NOTATIONS",
    "OPS",
    "WAITS",
    "AccelFunction",
    "AccelPort",
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
    "accepted",
    "affine",
    "check_dtypes",
    "function_of",
    "library_names",
    "load_brm",
    "memlet_ordered",
    "output_dtypes",
    "port_dtypes",
    "register_control",
    "register_kind",
    "register_notation",
    "register_op",
    "register_wait",
    "resolve",
    "resolve_nest",
    "stream",
    "task_nest",
]
