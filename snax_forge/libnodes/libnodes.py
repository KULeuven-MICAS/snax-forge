"""Library nodes, their expansions, and the transformations that raise to them.

Detection lives in snax_forge.patterns and never mutates. This module is the
other half: it consumes a match and replaces the scope with one library node
whose properties ARE the descriptor. That equality is the point -- there is no
second schema to keep in sync, because there is only one.

Not to be confused with transforms/ at the repo root, which holds recipes:
declarative INPUTS to the toolchain. This is machinery.

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

from snax_forge.patterns.patterns import TiledElementwise

# ---------------------------------------------------------------------------
# SnaxVectorOp
# ---------------------------------------------------------------------------


@library.expansion
class ExpandSnaxVectorPure(ExpandTransformation):
    """Rebuild the scope the node was raised from.

    Not an approximation of it: the same two maps, the same unroll
    declaration, the same tasklet, the same per-iteration memlets. Expanding a
    raised node therefore shows you exactly what was compressed into it, which
    is the only honest way to document what the node means.

    It also gives a testable round-trip. Because the reconstruction is
    structural, TiledElementwise matches it again after expand + simplify --
    so `expand(raise(G))` can be checked against `G`, rather than trusted.

    "pure" in the DaCe sense means the expansion uses plain SDFG constructs
    and needs no external library. It does not mean simplified.
    """

    environments: ClassVar[list] = []

    @staticmethod
    def expansion(node: SnaxVectorOp, parent_state, parent_sdfg):
        in_edges = {e.dst_conn: e for e in parent_state.in_edges(node)}
        out_edges = {e.src_conn: e for e in parent_state.out_edges(node)}
        desc = node.descriptor
        temporal, spatial = desc["temporal"], desc["spatial"]

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

        # The two map scopes, restored with their original parameters and
        # ranges. The inner one carries the unroll declaration -- without it
        # this would be two sequential loops, not a datapath.
        outer_entry, outer_exit = state.add_map(
            temporal["label"], {temporal["params"][0]: temporal["range"]}
        )
        inner_entry, inner_exit = state.add_map(
            spatial["label"],
            {spatial["params"][0]: spatial["range"]},
            unroll=spatial["unroll"],
        )

        tasklet = state.add_tasklet(
            desc["datapath"]["label"],
            set(desc["datapath"]["in_connectors"]),
            set(desc["datapath"]["out_connectors"]),
            node.code,
        )

        # add_memlet_path threads the edge through both scopes and creates the
        # map connectors on the way, which is what makes this reconstruction
        # short enough to trust.
        for conn in in_edges:
            state.add_memlet_path(
                state.add_read(conn),
                outer_entry,
                inner_entry,
                tasklet,
                dst_conn=node.port_map[conn],
                memlet=dace.Memlet(f"{conn}[{node.port_subset[conn]}]"),
            )
        for conn in out_edges:
            state.add_memlet_path(
                tasklet,
                inner_exit,
                outer_exit,
                state.add_write(conn),
                src_conn=node.port_map[conn],
                memlet=dace.Memlet(f"{conn}[{node.port_subset[conn]}]"),
            )
        return sdfg


@library.node
class SnaxVectorOp(dn.LibraryNode):
    """An elementwise datapath of `lanes` width, fed by streamers.

    Every property came out of TiledElementwise.describe(). None of them is a
    knob for a search: the width was fixed when the recipe declared the inner
    map unrolled.
    """

    implementations: ClassVar[dict[str, type[ExpandTransformation]]] = {
        "pure": ExpandSnaxVectorPure
    }
    default_implementation = "pure"

    # -- datapath ----------------------------------------------------------
    code = properties.Property(dtype=str, default="", desc="tasklet body")
    spatial_param = properties.Property(dtype=str, default="i", desc="lane index")

    # -- allocation --------------------------------------------------------
    lanes = properties.Property(dtype=int, default=1, desc="spatially replicated lanes")
    stride = properties.Property(dtype=int, default=1, desc="outer map stride")
    trips = properties.Property(dtype=str, default="1", desc="outer trip count")
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
            self.code = descriptor["datapath"]["code"]
            self.spatial_param = descriptor["spatial"]["params"][0]
            self.lanes = int(descriptor["spatial"]["lanes"])
            self.ragged = bool(descriptor["spatial"]["ragged"])
            self.stride = int(descriptor["temporal"]["step_value"])
            self.trips = descriptor["temporal"]["trip_count"]

    @property
    def descriptor(self) -> dict[str, Any]:
        """The descriptor this node was raised from."""
        return json.loads(self.descriptor_json)


# ---------------------------------------------------------------------------
# Raising
# ---------------------------------------------------------------------------


class SnaxForgeVectorExpand(TiledElementwise):
    """Replace a matched tiled elementwise scope with one SnaxVectorOp.

    Subclasses the detector so the shape and the predicate are stated once.
    Only apply() differs: the detector's is inert, this one rewrites.

    Usable directly as a recipe Step, which is the intended route -- running it
    under verify_each=True checks bit-exactness at the moment the scope is
    swapped for a node.
    """

    pattern_name = "snax_forge_vector_expand"

    def apply(self, graph, sdfg):
        state = graph
        descriptor = self.describe(state)
        reads, writes = self.ports(state)

        # Connectors are named after the ARRAY, not the datapath variable --
        # see ExpandSnaxVectorPure for why they cannot be the variable names,
        # and note that prefixing those would give `___in1` on a DaCe-generated
        # tasklet. Array names are unique and survive into RTL port names.
        both = (*reads.items(), *writes.items())
        node = SnaxVectorOp(
            f"snax_vector_{self.tasklet.label}",
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
        scope = [
            self.outer_entry,
            self.inner_entry,
            self.tasklet,
            self.inner_exit,
            self.outer_exit,
        ]
        stranded = {e.src for n in scope for e in state.in_edges(n)}
        stranded |= {e.dst for n in scope for e in state.out_edges(n)}
        state.remove_nodes_from(scope)
        for n in stranded:
            if n is not node and isinstance(n, dn.AccessNode) and state.degree(n) == 0:
                state.remove_node(n)

        return node


def raise_vector_ops(sdfg: dace.SDFG) -> int:
    """Apply SnaxForgeVectorExpand everywhere it fits. Returns the count.

    Matches are collected before any rewriting starts: applying invalidates
    the iterator, and a raised scope is not re-matchable, so there is no fixed
    point to chase. Callable form, so it can also be used as a bare recipe
    Step alongside set_map_property.
    """
    matches = list(Optimizer(sdfg).get_pattern_matches(patterns=[SnaxForgeVectorExpand]))
    for match in matches:
        match.apply(sdfg.node(match.state_id), sdfg)
    return len(matches)