"""SNAX-DFG: the workload as SNAX-FORGE's own dataflow graph (ARCHITECTURE.md 5.1).

Built so far, DFG1 (D71, D77): the ``.snaxdfg`` format. A graph holds
symbols, containers and an ordered body of nodes; a node is ``id``,
``kind``, ``inputs``, ``outputs``, ``attrs`` and, for a scope, ``body``.
Kinds are registered (kinds.py: ``map``, ``tasklet``, ``accelerated``);
memlet subsets and map ranges use the shared expression grammar
(snax_forge/expr.py) with the dimension notation of subset.py.

Next in this package: the SDFG importer (IMP1) and the reference executor
(REF1).
"""

from .graph import Container, Graph, Memlet, Node
from .kinds import KINDS, LOOP_KINDS, DfgError, Kind, Scope, register_kind
from .subset import canonical_dim, dim_names, format_dim, is_range, parse_dim

__all__ = [
    "KINDS",
    "LOOP_KINDS",
    "Container",
    "DfgError",
    "Graph",
    "Kind",
    "Memlet",
    "Node",
    "Scope",
    "canonical_dim",
    "dim_names",
    "format_dim",
    "is_range",
    "parse_dim",
    "register_kind",
]
