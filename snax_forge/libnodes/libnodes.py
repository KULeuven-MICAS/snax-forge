"""Library nodes, their expansions, and the transformations that raise to them.

Detection lives in snax_forge.patterns and never mutates. This module is the
other half: it consumes a match and replaces the scope with one library node
whose properties ARE the descriptor. That equality is the point -- there is no
second schema to keep in sync, because there is only one.

Not to be confused with transforms/ at the repo root, which holds recipes:
declarative INPUTS to the toolchain. This is machinery.

One node, two patterns
----------------------

FlatElementwise and TiledElementwise both raise to SnaxVectorOp. They describe
one piece of hardware -- W lanes fed T times -- and differ only in how the
graph spelled the split, so giving them separate node types would hand an RTL
backend two input formats for one datapath. The descriptor's `shape` section
carries (W, T); the expansion reads it back to rebuild whichever scope
structure was raised.

Every node registers a `pure` expansion, so a raised SDFG still compiles and
runs on the CPU. Without it the raise would cost us the golden reference at
exactly the moment we most want to check it, and recipes could not run with
verify_each=True.

Serialization caveat: a .sdfg holding a SnaxVectorOp only deserializes in an
interpreter that can import this module. Fine for our own pipeline; worth
knowing before assuming the files are portable.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import dace
from dace import library, properties
from dace.sdfg import nodes as dn
from dace.transformation.optimizer import Optimizer
from dace.transformation.transformation import ExpandTransformation

from snax_forge.patterns.patterns import ElementwisePattern, FlatElementwise, TiledElementwise

# ---------------------------------------------------------------------------
# SnaxVectorOp
# ---------------------------------------------------------------------------


@library.expansion
class ExpandSnaxVectorPure(ExpandTransformation):
    """Rebuild the scope the node was raised from.

    Not an approximation of it: the same maps, the same unroll declarations,
    the same tasklet, the same per-iteration memlets. Expanding a raised node
    therefore shows exactly what was compressed into it, which is the only
    honest way to document what the node means.

    It also gives a testable round-trip. Because the reconstruction is
    structural, the originating pattern matches it again after expand +
    simplify -- so `expand(raise(G))` can be checked against `G` rather than
    trusted. One scope or two is handled by the same loop, so a flat raise
    round-trips as a flat scope and a tiled one as a tiled scope.

    "pure" in the DaCe sense means the expansion uses plain SDFG constructs
    and needs no external library. It does not mean simplified.
    """

    environments: ClassVar[list] = []

    @staticmethod
    def expansion(node: SnaxVectorOp, parent_state, parent_sdfg):
        in_edges = {e.dst_conn: e for e in parent_state.in_edges(node)}
        out_edges = {e.src_conn: e for e in parent_state.out_edges(node)}
        desc = node.descriptor

        sdfg = dace.SDFG(f"{node.label}_expanded")
        state = sdfg.add_state("main")

        # A nested SDFG's arrays must be named after the connectors, so the
        # connectors must NOT carry the tasklet's own variable names -- codegen
        # would emit `int a = a[i];` and the array would shadow itself. Hence
        # array-derived connector names, and port_map to get back to the names
        # the tasklet code was written with.
        for conn, edge in {**in_edges, **out_edges}.items():
            d = parent_sdfg.arrays[edge.data.data]
            sdfg.add_array(conn, d.shape, d.dtype)

        # Outermost first. A flat raise has one entry here, a tiled raise two;
        # nothing below needs to know which.
        #
        # The schedule has to be carried across. A reconstructed map left at
        # Default is inferred CPU_Multicore when it lands at the top of the
        # nested SDFG, and DaCe refuses to unroll an OpenMP map -- so dropping
        # it turns a valid spatial raise into a codegen error.
        entries, exits = [], []
        for scope in (desc["temporal"], desc["spatial"]):
            if scope is None:
                continue
            entry, exit_ = state.add_map(
                scope["label"],
                {scope["params"][0]: scope["range"]},
                schedule=dace.ScheduleType[scope["schedule"]],
                unroll=scope["unroll"],
            )
            entries.append(entry)
            exits.append(exit_)

        tasklet = state.add_tasklet(
            desc["datapath"]["label"],
            set(desc["datapath"]["in_connectors"]),
            set(desc["datapath"]["out_connectors"]),
            node.code,
        )

        # add_memlet_path threads the edge through every scope and creates the
        # map connectors on the way, which is what keeps this short enough to
        # trust for either arity.
        for conn in in_edges:
            state.add_memlet_path(
                state.add_read(conn),
                *entries,
                tasklet,
                dst_conn=node.port_map[conn],
                memlet=dace.Memlet(f"{conn}[{node.port_subset[conn]}]"),
            )
        for conn in out_edges:
            state.add_memlet_path(
                tasklet,
                *reversed(exits),
                state.add_write(conn),
                src_conn=node.port_map[conn],
                memlet=dace.Memlet(f"{conn}[{node.port_subset[conn]}]"),
            )
        return sdfg


@library.node
class SnaxVectorOp(dn.LibraryNode):
    """An elementwise datapath of `lanes` width, fed `trips` times.

    Every property came out of an ElementwisePattern's describe(). None of
    them is a knob for a search: the split was fixed when the recipe declared
    which map is unrolled.
    """

    implementations: ClassVar[dict[str, type[ExpandTransformation]]] = {
        "pure": ExpandSnaxVectorPure
    }
    default_implementation = "pure"

    # -- datapath ----------------------------------------------------------
    code = properties.Property(dtype=str, default="", desc="tasklet body")

    # -- allocation: lanes * trips == elements -----------------------------
    # loop | spatial | tiled_spatial. Each is a different datapath, so it is
    # stored rather than re-derived: a backend selecting an implementation
    # reads one field instead of inspecting two nullable scope dicts.
    variant = properties.Property(dtype=str, default="loop", desc="loop | spatial | tiled_spatial")
    lanes = properties.Property(dtype=int, default=1, desc="spatially replicated lanes")
    trips = properties.Property(dtype=str, default="1", desc="times the datapath is fed")
    elements = properties.Property(dtype=str, default="1", desc="total iteration space")
    # False means the width did not resolve to an integer -- an unrolled map
    # over a symbolic bound. Not buildable until the symbol is specialized.
    bounded = properties.Property(dtype=bool, default=True, desc="lanes is a concrete width")
    ragged = properties.Property(
        dtype=bool, default=False, desc="clamped bound -> partial-tile predication"
    )

    # -- ports -------------------------------------------------------------
    port_map = properties.DictProperty(
        key_type=str, value_type=str, default={}, desc="connector -> datapath variable"
    )
    port_subset = properties.DictProperty(
        key_type=str, value_type=str, default={}, desc="connector -> per-iteration subset"
    )
    # Held as a string, not a DictProperty. A DictProperty with value_type
    # `object` writes fine and then fails to read back, because deserializing
    # calls object(value) and object() takes no arguments -- the saved graph
    # degrades to an UnregisteredLibraryNode with no warning worth the name.
    descriptor_json = properties.Property(
        dtype=str, default="{}", desc="full pattern descriptor, JSON-encoded"
    )

    def __init__(
        self,
        name: str,
        descriptor: dict[str, Any] | None = None,
        port_map: dict[str, str] | None = None,
        port_subset: dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.port_map = dict(port_map or {})
        self.port_subset = dict(port_subset or {})
        if descriptor:
            self.descriptor_json = json.dumps(descriptor)
            shape = descriptor["shape"]
            self.variant = shape["variant"]
            self.code = descriptor["datapath"]["code"]
            # An unbounded width has no integer to store; 0 stands for
            # "unresolved", and `bounded` is what says so.
            self.lanes = int(shape["lanes"]) if shape["bounded"] else 0
            self.trips = shape["trips"]
            self.elements = shape["elements"]
            self.bounded = bool(shape["bounded"])
            spatial = descriptor["spatial"]
            self.ragged = bool(spatial["ragged"]) if spatial else False

    @property
    def descriptor(self) -> dict[str, Any]:
        """The descriptor this node was raised from."""
        return json.loads(self.descriptor_json)


# ---------------------------------------------------------------------------
# Raising
# ---------------------------------------------------------------------------


def _node_label(variant: str, tasklet_label: str) -> str:
    """`snax_vector_<variant>_<op>`, e.g. snax_vector_tiled_spatial_Add.

    DaCe-generated tasklets are labelled `_Add_`, so the edges are trimmed to
    avoid `snax_vector_loop__Add_`. The result must stay a valid identifier:
    the expansion builds an SDFG named after it, and SDFG names are checked.
    """
    op = tasklet_label.strip("_") or "op"
    return f"snax_vector_{variant}_{op}"


class RaiseToSnaxVector(ElementwisePattern):
    """Shared apply() for every elementwise raise.

    Subclasses pair this with a pattern class, so the shape and the predicate
    are stated once in snax_forge.patterns and only the rewrite lives here.
    """

    def apply(self, graph, sdfg):
        state = graph
        descriptor = self.describe(state)
        reads, writes = self.ports(state)
        entries, exits = self.scopes(state)

        # Connectors are named after the ARRAY, not the datapath variable --
        # see ExpandSnaxVectorPure for why they cannot be the variable names,
        # and note that prefixing those would give `___in1` on a DaCe-generated
        # tasklet. Array names are unique and survive into RTL port names.
        both = (*reads.items(), *writes.items())
        node = SnaxVectorOp(
            _node_label(descriptor["shape"]["variant"], self.tasklet.label),
            descriptor=descriptor,
            port_map={f"_{a}": p["var"] for a, p in both},
            port_subset={f"_{a}": p["subset"] for a, p in both},
        )
        state.add_node(node)

        for arr in reads:
            node.add_in_connector(f"_{arr}")
            state.add_edge(
                state.add_read(arr),
                None,
                node,
                f"_{arr}",
                dace.Memlet.from_array(arr, sdfg.arrays[arr]),
            )
        for arr in writes:
            node.add_out_connector(f"_{arr}")
            state.add_edge(
                node,
                f"_{arr}",
                state.add_write(arr),
                None,
                dace.Memlet.from_array(arr, sdfg.arrays[arr]),
            )

        # Drop the scope, then the access nodes it left stranded. simplify()
        # would collect them anyway, but doing it here keeps the saved graph
        # readable without one.
        scope = [*entries, *exits, self.tasklet]
        stranded = {e.src for n in scope for e in state.in_edges(n)}
        stranded |= {e.dst for n in scope for e in state.out_edges(n)}
        state.remove_nodes_from(scope)
        for n in stranded:
            if n is not node and isinstance(n, dn.AccessNode) and state.degree(n) == 0:
                state.remove_node(n)

        return node


class RaiseFlatVector(RaiseToSnaxVector, FlatElementwise):
    """Raise an untiled elementwise scope to a SnaxVectorOp."""

    pattern_name = "raise_flat_vector"


class RaiseTiledVector(RaiseToSnaxVector, TiledElementwise):
    """Raise a tiled elementwise scope to a SnaxVectorOp."""

    pattern_name = "raise_tiled_vector"


#: Raise transformations, most specific first. Order matters only if a graph
#: could match both, which the top-level requirement rules out.
RAISERS: tuple[type[RaiseToSnaxVector], ...] = (RaiseTiledVector, RaiseFlatVector)


def raise_vector_ops(sdfg: dace.SDFG, patterns=None) -> int:
    """Apply every elementwise raise that fits. Returns the count.

    One match is applied per search, and the search is re-run afterwards. A
    DaCe match stores node INDICES, and removing the raised scope renumbers
    the state -- so a batch of matches collected up front goes stale after the
    first apply, and the second one binds a MapEntry role to whatever now sits
    at that index. With a single scope per state nothing goes wrong, which is
    exactly why it is worth not relying on.

    Terminates because a raised scope is not re-matchable, so each pass
    strictly reduces the number of matches; the bound is belt-and-braces
    against a pattern that ever matched its own output.
    """
    total = 0
    limit = sum(len(s.nodes()) for s in sdfg.states())
    for raiser in patterns or RAISERS:
        while total < limit:
            match = next(iter(Optimizer(sdfg).get_pattern_matches(patterns=[raiser])), None)
            if match is None:
                break
            match.apply(sdfg.node(match.state_id), sdfg)
            total += 1
    return total
