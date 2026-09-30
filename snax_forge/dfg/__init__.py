"""SNAX-DFG: the workload as SNAX-FORGE's own dataflow graph (ARCHITECTURE.md 5.1).

Built so far, DFG1 (D71, D77): the ``.snaxdfg`` format. A graph holds
symbols, containers and an ordered body of nodes; a node is ``id``,
``kind``, ``inputs``, ``outputs``, ``attrs`` and, for a scope, ``body``.
Kinds are registered (kinds.py: ``map``, ``tasklet``, ``accelerated``);
memlet subsets and map ranges use the shared expression grammar
(snax_forge/expr.py) with the dimension notation of subset.py.

IMP1 (D78): import_sdfg.py maps a simplified DaCe SDFG onto the format
(``import_sdfg``, ``import_kernel``; it imports DaCe, so it is not loaded
here). REF1 (D79): execute.py runs a graph in NumPy (``execute``), an
accelerated node through its BRM's function. The command line is
``python -m snax_forge.dfg import | check``.

DFG3 (D102): reduce.py, a reduction on a tasklet's output memlet
(``Reduction``, ``register_reduction``), so a reduction is a map like any other; the
importer turns DaCe's ``Reduce`` into one and the executor folds it.
"""

from .execute import EXECUTORS, ExecutionError, execute, register_executor
from .graph import Container, Graph, Memlet, Node
from .kinds import KINDS, LOOP_KINDS, DfgError, Kind, Scope, register_kind
from .reduce import REDUCE_OPS, Reduction, register_reduction
from .subset import canonical_dim, dim_names, format_dim, is_range, parse_dim

__all__ = [
    "EXECUTORS",
    "KINDS",
    "LOOP_KINDS",
    "REDUCE_OPS",
    "Container",
    "DfgError",
    "ExecutionError",
    "Graph",
    "Kind",
    "Memlet",
    "Node",
    "Reduction",
    "Scope",
    "canonical_dim",
    "dim_names",
    "execute",
    "format_dim",
    "is_range",
    "parse_dim",
    "register_executor",
    "register_kind",
    "register_reduction",
]
