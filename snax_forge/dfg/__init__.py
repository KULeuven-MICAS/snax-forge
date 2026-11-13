"""SNAX-DFG: the workload as SNAX-FORGE's own dataflow graph (ARCHITECTURE.md 5.1).

The format is CONTRACTS, "SNAX-DFG". graph.py holds a graph, kinds.py the registered node kinds
(``map``, ``tasklet``, ``accelerated``, ``link``), subset.py the dimension notation over the
shared expression grammar (snax_forge/expr.py) and reduce.py the fold on a tasklet's output
memlet. import_sdfg.py maps a simplified DaCe SDFG onto the format (``import_sdfg``,
``import_kernel``; it imports DaCe, so it is not loaded here); execute.py runs a graph in NumPy
(``execute``); loops.py names each map's variable after its dimension. The command line is
``python -m snax_forge.dfg import | check``.
"""

from .execute import EXECUTORS, ExecutionError, execute, named_rates, register_executor
from .graph import Container, Graph, Memlet, Node
from .kinds import KINDS, LOOP_KINDS, DfgError, Kind, Scope, register_kind
from .reduce import REDUCE_OPS, Reduction, Seed, register_reduction
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
    "Seed",
    "canonical_dim",
    "dim_names",
    "execute",
    "format_dim",
    "is_range",
    "named_rates",
    "parse_dim",
    "register_executor",
    "register_kind",
    "register_reduction",
]
