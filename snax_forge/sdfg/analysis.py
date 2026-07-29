"""Structural extraction and data-movement accounting for an SDFG.

Three extractors, each returning a JSON-serialisable dict and printing nothing:

    extract_arrays   -- data descriptors: role, layout, footprint
    extract_compute  -- map scopes, tasklets, library nodes
    extract_memlets  -- every edge: endpoints, subset, volume, traffic

`extract_sdfg` runs all three plus interstate edges and rolls up totals.
Each has a matching `print_*` for human consumption; `print_sdfg` prints all.

Symbolic values are stored as strings so results survive a JSON round-trip.
When `symbols` is supplied, resolved integers appear alongside under *_value
keys, so a census record keeps both the parametric form and the concrete
instantiation.
"""

from __future__ import annotations

import ast
from collections import Counter

import dace
from dace import nodes as dn
from dace.sdfg import propagation
from dace.symbolic import SymExpr

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sym(expr, symbols: dict | None) -> int | None:
    """Symbolic expression -> int when resolvable, else None.

    Returns None rather than raising: data-dependent sizes are legal and
    should surface as unknown, not as a crash.
    """
    try:
        return int(dace.symbolic.evaluate(expr, symbols or {}))
    except Exception:  # noqa: BLE001
        return None


def _enum(value) -> str:
    """EnumProperty -> bare name, e.g. ScheduleType.Default -> 'Default'."""
    return str(value).split(".")[-1]


def _num(value) -> str:
    """Format a possibly-missing number for a table cell."""
    return "-" if value is None else f"{value:,}"


def _node_ref(node) -> dict:
    """Identify a graph node: {'type': ..., 'label': ...}.

    AccessNodes are labelled by the array they hold, map nodes by the map's
    label, everything else by its own label.
    """
    if isinstance(node, dn.AccessNode):
        return {"type": "AccessNode", "label": node.data}
    if isinstance(node, (dn.MapEntry, dn.MapExit)):
        return {"type": type(node).__name__, "label": node.map.label}
    if isinstance(node, dn.LibraryNode):
        return {"type": type(node).__name__, "label": node.label}
    return {"type": type(node).__name__, "label": getattr(node, "label", str(node))}


def _node_str(ref: dict, connector: str | None = None) -> str:
    s = f"{ref['type']}({ref['label']})"
    return s if connector is None else f"{s}[{connector}]"


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


def _array_roles(sdfg: dace.SDFG) -> dict[str, str]:
    """name -> 'in' | 'out' | 'inout' | 'transient' | 'unused'.

    DaCe computes the read and write sets itself. Transient wins, because a
    transient is internal regardless of how it is accessed.
    """
    reads, writes = sdfg.read_and_write_sets()
    roles = {}
    for name, d in sdfg.arrays.items():
        if d.transient:
            roles[name] = "transient"
        elif name in reads and name in writes:
            roles[name] = "inout"
        elif name in reads:
            roles[name] = "in"
        elif name in writes:
            roles[name] = "out"
        else:
            roles[name] = "unused"
    return roles


def _access_node_stats(sdfg: dace.SDFG) -> tuple[dict, dict]:
    """name -> how many AccessNodes reference it, and in which states."""
    counts: dict[str, int] = {}
    states: dict[str, set] = {}
    for st in sdfg.states():
        for n in st.nodes():
            if isinstance(n, dn.AccessNode):
                counts[n.data] = counts.get(n.data, 0) + 1
                states.setdefault(n.data, set()).add(st.label)
    return counts, states


def extract_arrays(sdfg: dace.SDFG, symbols: dict | None = None) -> dict:
    """Every data descriptor, with role, layout and allocated footprint."""
    roles = _array_roles(sdfg)
    counts, states = _access_node_stats(sdfg)

    arrays, external, transient = {}, 0, 0
    for name, d in sdfg.arrays.items():
        nbytes = _sym(d.total_size * d.dtype.bytes, symbols)
        entry = {
            "role": roles[name],
            "kind": type(d).__name__,  # Array | Scalar | View | Reference
            "transient": bool(d.transient),
            "dtype": str(d.dtype.as_numpy_dtype()),
            "bytes_per_element": d.dtype.bytes,
            "shape": [str(x) for x in d.shape],
            "strides": [str(x) for x in d.strides],
            "total_size": str(d.total_size),
            "storage": _enum(d.storage),
            "lifetime": _enum(d.lifetime),
            "access_nodes": counts.get(name, 0),
            "states": sorted(states.get(name, ())),
        }
        if symbols:
            entry["shape_value"] = [_sym(x, symbols) for x in d.shape]
            entry["total_size_value"] = _sym(d.total_size, symbols)
            entry["bytes"] = nbytes
        arrays[name] = entry

        if nbytes is not None:
            if d.transient:
                transient += nbytes
            else:
                external += nbytes

    return {
        "arrays": arrays,
        "arglist": list(sdfg.arglist()),
        "top_level_transients": sorted(sdfg.top_level_transients()),
        "shared_transients": sorted(sdfg.shared_transients()),
        # Allocation footprint, not traffic -- see extract_memlets for traffic.
        "allocated_external_bytes": external if symbols else None,
        "allocated_transient_bytes": transient if symbols else None,
    }


def print_arrays(data: dict) -> None:
    """Human-readable table. Accepts extract_arrays() or extract_sdfg() output."""
    a_data = data.get("arrays_section", data)
    print(f"  arglist={a_data['arglist']}")
    hdr = (
        f"  {'name':10} {'role':9} {'kind':7} {'shape':14} {'strides':12} "
        f"{'dtype':7} {'total':>10} {'bytes':>12} {'storage':9} {'nodes':>5}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name, a in a_data["arrays"].items():
        shape = a.get("shape_value") or a["shape"]
        total = a.get("total_size_value", a["total_size"])
        print(
            f"  {name:10} {a['role']:9} {a['kind']:7} "
            f"{tuple(shape)!s:14} {tuple(a['strides'])!s:12} "
            f"{a['dtype']:7} {total!s:>10} {_num(a.get('bytes')):>12} "
            f"{a['storage']:9} {a['access_nodes']:>5}"
        )

    if a_data["allocated_external_bytes"] is not None:
        print(
            f"  allocated: external={a_data['allocated_external_bytes']:,} B   "
            f"transient={a_data['allocated_transient_bytes']:,} B"
        )
    print(
        f"  top-level transients: {a_data['top_level_transients']}   "
        f"shared: {a_data['shared_transients']}"
    )


# ---------------------------------------------------------------------------
# Compute: maps, tasklets, library nodes
# ---------------------------------------------------------------------------

BINOPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.LShift: "shl",
    ast.RShift: "shr",
    ast.BitAnd: "and",
    ast.BitOr: "or",
    ast.BitXor: "xor",
    ast.USub: "neg",
}


def tasklet_signature(tasklet: dn.Tasklet) -> dict:
    """Operation histogram and expression-DAG shape for one tasklet.

    This is the datapath signature: which functional units are needed, how
    many, and how deep the combinational chain is. `calls` catches implicit
    casts such as dace.int64(x), which widen beyond the array dtype.
    """
    ops, calls, consts, depth = Counter(), Counter(), 0, 0
    trees = tasklet.code.code
    trees = trees if isinstance(trees, list) else [trees]

    def walk(node, d=0):
        nonlocal consts, depth
        depth = max(depth, d)
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            ops[BINOPS.get(type(node.op), type(node.op).__name__)] += 1
        elif isinstance(node, ast.Call):
            calls[getattr(node.func, "id", getattr(node.func, "attr", "?"))] += 1
        elif isinstance(node, ast.Constant):
            consts += 1
        for child in ast.iter_child_nodes(node):
            walk(child, d + 1)

    for tree in trees:
        walk(tree)

    return {
        "ops": dict(ops),
        "calls": dict(calls),
        "constants": consts,
        "ast_depth": depth,
        "total_ops": sum(ops.values()),
    }


def extract_compute(sdfg: dace.SDFG) -> dict:
    """Map scopes, tasklets and library nodes."""
    maps, tasklets, libnodes, nested = [], [], [], []
    lib_base = set(dn.LibraryNode.__properties__)

    for st in sdfg.states():
        for n in st.nodes():
            if isinstance(n, dn.MapEntry):
                maps.append(
                    {
                        "label": n.map.label,
                        "state": st.label,
                        "params": list(n.map.params),
                        "range": str(n.map.range),
                        "schedule": _enum(n.map.schedule),
                        "unroll": n.map.unroll,
                        "collapse": n.map.collapse,
                        "top_level": st.entry_node(n) is None,
                        # SymExpr only exists when main != approx, so its presence
                        # means a clamped bound -> partial-tile predication needed.
                        "ragged": any(isinstance(e, SymExpr) for _, e, _ in n.map.range.ndrange()),
                    }
                )
            elif isinstance(n, dn.Tasklet):
                tasklets.append(
                    {
                        "label": n.label,
                        "state": st.label,
                        "language": n.code.language.name,
                        "code": n.code.as_string,
                        "in_connectors": {k: str(v) for k, v in n.in_connectors.items()},
                        "out_connectors": {k: str(v) for k, v in n.out_connectors.items()},
                        "side_effects": n.side_effects,
                        "in_scope": st.entry_node(n) is not None,
                        **tasklet_signature(n),
                    }
                )
            elif isinstance(n, dn.LibraryNode):
                libnodes.append(
                    {
                        "type": type(n).__name__,
                        "state": st.label,
                        "implementation": n.implementation,  # None = unexpanded
                        "available": sorted(type(n).implementations),
                        "has_pure": "pure" in type(n).implementations,
                        "schedule": _enum(n.schedule),
                        "in_connectors": list(n.in_connectors),
                        "out_connectors": list(n.out_connectors),
                        "properties": {
                            p: str(getattr(n, p))
                            for p in type(n).__properties__
                            if p not in lib_base
                        },
                    }
                )
            elif isinstance(n, dn.NestedSDFG):
                nested.append(
                    {
                        "label": n.label,
                        "state": st.label,
                        "inner_states": len(list(n.sdfg.states())),
                    }
                )

    return {
        "maps": maps,
        "tasklets": tasklets,
        "library_nodes": libnodes,
        "nested_sdfgs": nested,
        "map_scopes": sum(1 for m in maps if m["top_level"]),
        "op_histogram": dict(sum((Counter(t["ops"]) for t in tasklets), Counter())),
    }


def print_compute(data: dict) -> None:
    """Accepts extract_compute() or extract_sdfg() output."""
    c = data.get("compute", data)
    for m in c["maps"]:
        tag = "top" if m["top_level"] else "inner"
        rag = "  RAGGED" if m["ragged"] else ""
        print(
            f"  Map     {m['label']:22} [{tag:5}] range={m['range']} "
            f"sched={m['schedule']} unroll={m['unroll']}{rag}"
        )
    for t in c["tasklets"]:
        print(
            f"  Tasklet {t['label']:22} in={list(t['in_connectors'])} "
            f"out={list(t['out_connectors'])}"
        )
        print(f"      code : {t['code'].replace(chr(10), ' ; ')[:76]}")
        print(f"      ops={t['ops']} calls={t['calls']} depth={t['ast_depth']}")
    for lb in c["library_nodes"]:
        star = "  <- has 'pure'" if lb["has_pure"] else ""
        print(
            f"  LibNode {lb['type']:22} impl={lb['implementation']} "
            f"available={lb['available']}{star}"
        )
        if lb["properties"]:
            print(f"      properties: {lb['properties']}")
    for ns in c["nested_sdfgs"]:
        print(f"  Nested  {ns['label']:22} inner_states={ns['inner_states']}")
    print(f"  map_scopes={c['map_scopes']}  ops={c['op_histogram']}")


# ---------------------------------------------------------------------------
# Memlets and traffic
# ---------------------------------------------------------------------------


def extract_memlets(sdfg: dace.SDFG, symbols: dict | None = None) -> dict:
    """Every memlet edge: endpoints, connectors, subset, volume, flags.

    `role` is the key field. Only AccessNode-incident edges carry whole-scope
    volume; edges inside a scope are per-iteration views of the SAME traffic
    and are marked scope-internal so totals do not double-count.

    subset is the ADDRESS PATTERN (what a streamer descriptor generates);
    volume is the ACCESS COUNT (how many transfers). Neither derives from the
    other -- a union subset with a larger volume means overlapping reads.
    """
    propagation.propagate_states(sdfg)  # fills state.executions for loops
    rows = []

    for st in sdfg.states():
        execs = _sym(st.executions, symbols)
        execs = 1 if execs is None else execs
        for e in st.edges():
            m = e.data
            if m.data is None:
                continue
            desc = sdfg.arrays[m.data]
            vol = _sym(m.volume, symbols)
            nbytes = None if vol is None else vol * desc.dtype.bytes
            src_is_an = isinstance(e.src, dn.AccessNode) and e.src.data == m.data
            dst_is_an = isinstance(e.dst, dn.AccessNode) and e.dst.data == m.data

            rows.append(
                {
                    "state": st.label,
                    "state_executions": execs,
                    "dynamic_executions": bool(st.dynamic_executions),
                    "src": _node_ref(e.src),
                    "src_connector": e.src_conn,
                    "dst": _node_ref(e.dst),
                    "dst_connector": e.dst_conn,
                    "data": m.data,
                    "transient": bool(desc.transient),
                    "subset": str(m.subset),
                    "other_subset": None if m.other_subset is None else str(m.other_subset),
                    "volume": str(m.volume),
                    "volume_value": vol,
                    "bytes_value": nbytes,
                    "total_bytes": None if nbytes is None else nbytes * execs,
                    "wcr": None if m.wcr is None else str(m.wcr),
                    "dynamic": bool(m.dynamic),
                    "role": "read" if src_is_an else ("write" if dst_is_an else "scope-internal"),
                }
            )

    return {"memlets": rows, **movement_from_memlets(rows)}


def movement_from_memlets(rows: list[dict]) -> dict:
    """Reduce memlet rows to per-array traffic. Single source of truth."""
    per: dict[str, dict] = {}
    for r in rows:
        if r["role"] == "scope-internal" or r["total_bytes"] is None:
            continue
        entry = per.setdefault(
            r["data"], {"transient": r["transient"], "read_bytes": 0, "write_bytes": 0}
        )
        entry[f"{r['role']}_bytes"] += r["total_bytes"]

    return {
        "traffic": per,
        # LOGICAL accesses, not compulsory DRAM traffic: a 3-point stencil
        # reads each element three times, which a cache would partly absorb.
        "traffic_external_bytes": sum(
            v["read_bytes"] + v["write_bytes"] for v in per.values() if not v["transient"]
        ),
        "traffic_transient_bytes": sum(
            v["read_bytes"] + v["write_bytes"] for v in per.values() if v["transient"]
        ),
    }


def print_memlets(data: dict, only_traffic: bool = False) -> None:
    """only_traffic hides scope-internal rows (per-iteration views of the
    same data). Keep them when you need the indexing expression that drives
    a streamer stride, or the connector that fixes tasklet operand order."""
    m_data = data.get("memlets_section", data)
    hdr = (
        f"  {'state':18} {'src':30} {'dst':30} {'data':9} {'subset':16} "
        f"{'volume':>10} {'execs':>6} {'bytes':>13} {'role':15} flags"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in m_data["memlets"]:
        if only_traffic and r["role"] == "scope-internal":
            continue
        flags = " ".join(
            f
            for f, on in (
                ("WCR", r["wcr"]),
                ("DYN", r["dynamic"]),
                ("DYNEXEC", r["dynamic_executions"]),
                ("transient", r["transient"]),
            )
            if on
        )
        vol = r["volume_value"] if r["volume_value"] is not None else r["volume"]
        print(
            f"  {r['state'][:18]:18} "
            f"{_node_str(r['src'], r['src_connector'])[:30]:30} "
            f"{_node_str(r['dst'], r['dst_connector'])[:30]:30} "
            f"{r['data']:9} {r['subset'][:16]:16} {vol!s:>10} "
            f"{r['state_executions']:>6} {_num(r['total_bytes']):>13} "
            f"{r['role']:15} {flags}"
        )

    print(
        f"  traffic: external={m_data['traffic_external_bytes']:,} B   "
        f"transient={m_data['traffic_transient_bytes']:,} B"
    )


# ---------------------------------------------------------------------------
# Interstate edges and whole-SDFG extraction
# ---------------------------------------------------------------------------


def extract_interstate(sdfg: dace.SDFG) -> list[dict]:
    """Control-flow edges between states: conditions and symbol assignments."""
    return [
        {
            "src": e.src.label,
            "dst": e.dst.label,
            "condition": e.data.condition.as_string,
            "is_unconditional": e.data.is_unconditional(),
            "assignments": {k: str(v) for k, v in e.data.assignments.items()},
        }
        for e in sdfg.edges()
    ]


def _analysis_warnings(sdfg: dace.SDFG, arrays: dict, compute: dict, memlets: dict) -> list[str]:
    """Conditions under which the numbers above stop being exact."""
    w = []
    refs = [n for n, a in arrays["arrays"].items() if a["kind"] == "Reference"]
    if refs:
        w.append(f"Reference descriptors detach access from data: {refs}")
    if compute["nested_sdfgs"]:
        w.append(
            f"{len(compute['nested_sdfgs'])} nested SDFG(s) not traversed; "
            "inline via simplify() before extraction"
        )
    if compute["library_nodes"]:
        w.append(
            f"{len(compute['library_nodes'])} unexpanded library node(s); "
            "expand_library_nodes() then simplify() to surface map scopes"
        )
    if any(r["dynamic"] for r in memlets["memlets"]):
        w.append("dynamic memlet(s): volume is not statically known")
    if any(r["dynamic_executions"] for r in memlets["memlets"]):
        w.append("dynamic state execution count: traffic totals are estimates")
    if any(r["volume_value"] is None for r in memlets["memlets"]):
        w.append("some memlet volumes unresolved; excluded from totals")
    return w


def extract_sdfg(sdfg: dace.SDFG, symbols: dict | None = None) -> dict:
    """Arrays, compute and memlets in one JSON-serialisable record."""
    arrays = extract_arrays(sdfg, symbols)
    compute = extract_compute(sdfg)
    memlets = extract_memlets(sdfg, symbols)

    return {
        "sdfg": sdfg.name,
        "dace": dace.__version__,
        "symbols": dict(symbols) if symbols else None,
        "free_symbols": sorted(str(x) for x in sdfg.free_symbols),
        "states": [st.label for st in sdfg.states()],
        "arrays_section": arrays,
        "compute": compute,
        "memlets_section": memlets,
        "interstate": extract_interstate(sdfg),
        "totals": {
            "states": len(list(sdfg.states())),
            "map_scopes": compute["map_scopes"],
            "tasklets": len(compute["tasklets"]),
            "library_nodes": len(compute["library_nodes"]),
            "nested_sdfgs": len(compute["nested_sdfgs"]),
            "memlets": len(memlets["memlets"]),
            "allocated_transient_bytes": arrays["allocated_transient_bytes"],
            "traffic_external_bytes": memlets["traffic_external_bytes"],
            "traffic_transient_bytes": memlets["traffic_transient_bytes"],
        },
        "warnings": _analysis_warnings(sdfg, arrays, compute, memlets),
    }


def print_sdfg(data: dict, only_traffic: bool = True) -> None:
    """Full human-readable report from extract_sdfg() output."""
    print(
        f"SDFG {data['sdfg']}  (dace {data['dace']})  "
        f"free_symbols={data['free_symbols']}"
        + (f"  symbols={data['symbols']}" if data["symbols"] else "")
    )
    print(f"  states: {data['states']}")

    print("\n[arrays]")
    print_arrays(data)

    print("\n[compute]")
    print_compute(data)

    print("\n[memlets]")
    print_memlets(data, only_traffic=only_traffic)

    if data["interstate"]:
        print("\n[interstate]")
        for e in data["interstate"]:
            cond = "" if e["is_unconditional"] else f"  if {e['condition']}"
            asg = f"  assign={e['assignments']}" if e["assignments"] else ""
            print(f"  {e['src']} -> {e['dst']}{cond}{asg}")

    print(f"\n[totals] {data['totals']}")
    for warning in data["warnings"]:
        print(f"  WARNING: {warning}")
