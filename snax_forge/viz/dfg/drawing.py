"""The DFG page's drawing of one step (D185): boxes with ports, edges and outlines, as a flat
graph the page lays out with ELK. Loops are a chain of boxes from the outermost down; an edge
from a container passes through exactly the loops its subset uses, a loop it skips on the way
in marked ``↻``; a link's reader nest is drawn on its writer's chain when their loops match, a
fork's other readers on its first's (D189), their read one edge that splits at the unit; a
group is drawn as one box holding its hardware and the loops that drive only it, one loop chain
per configuration, and a block whose boxes all land in groups is named on them instead;
hardware is keyed by what it is (``unit:<instance>``, ``link:<from>.<port>><to>.<port>``,
``slicer:<instance>.<port>``), so a unit shared between tasks is one box, a temporal mux in front
of its inputs per task. Python decides every box, edge, text and hint; the page measures and
places them.

Entry point: ``drawing(g, tops, placed, blocks)``; CONTRACTS, "Recipe" (its viewer's API).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from snax_forge.brm import SEED
from snax_forge.design.streamers import accelerator_of, group_of
from snax_forge.dfg import Graph, Memlet, Node
from snax_forge.dfg.graph import SCALAR, TRANSPOSE
from snax_forge.sandbox.blocks import BLOCK
from snax_forge.sandbox.share import SHARED

LEAVES = ("tasklet", "accelerated", "transpose", "block")


def _api() -> Any:
    from snax_forge.viz.dfg import api  # the texts live there; imported late (api imports us)

    return api


def _maps_to(top: Node) -> list[tuple[Node, list[Node]]]:
    """Every node under ``top`` with the maps above it inside ``top``, outermost first."""
    out: list[tuple[Node, list[Node]]] = []

    def go(n: Node, maps: list[Node]) -> None:
        out.append((n, maps))
        for c in n.body or []:
            go(c, [*maps, n] if n.kind == "map" else maps)

    go(top, [])
    return out


def _reads(g: Graph, n: Node) -> list[tuple[str, Memlet]]:
    """A leaf's reads as drawn: its inputs, a seeded tasklet's seed, no CSR slot."""
    api = _api()
    csr = api.csr_connectors(g, n)
    out = [(c, m) for c, m in n.inputs.items() if c not in csr]
    seed = api.seed_memlet(n) if n.kind == "tasklet" else None
    if seed is not None:
        out.append((SEED, seed))
    return out


def _over(n: Node, maps: list[Node]) -> list[str]:
    """The loops a fold accumulates over: its firing loops its output subset does not use."""
    names: set[str] = set()
    for m in n.outputs.values():
        names |= m.names
    return [lp.attrs["var"] for lp in maps if lp.attrs["var"] not in names]


def _task_names(nodes: list[Node], top_of: dict[str, str]) -> dict[str, str]:
    """The task each node of a shared instance runs as: the group ``share`` renamed it from
    (``sandbox.shared``), else its group; the nest's id when those do not tell them apart."""
    tasks = {n.id: n.attrs[SHARED]["group"] if SHARED in n.attrs else group_of(n) for n in nodes}
    if len(set(tasks.values())) < len(nodes):
        tasks = {n.id: top_of[n.id] for n in nodes}
    return tasks


class _Drawing:
    def __init__(self, g: Graph, placed: dict[str, Any]) -> None:
        self.g = g
        self.placed = placed
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.wires: set[tuple[Any, ...]] = set()
        self.version = dict.fromkeys(g.containers, 0)

    def box(self, id_: str, **kw: Any) -> dict[str, Any]:
        if id_ not in self.nodes:
            self.nodes[id_] = {"id": id_, **kw}
        return self.nodes[id_]

    def container(self, c: str, first: bool = False) -> str:
        api = _api()
        key = f"{c}@{self.version[c]}"
        if key not in self.nodes:
            d = api._container(self.g, c, self.placed.get(c))
            self.box(
                key,
                kind="container",
                sel=c,
                **{k: d[k] for k in ("name", "text", "dtype", "badge", "transient", "scalar")},
                **({"share": d["share"]} if "share" in d else {}),
            )
            if first:
                self.nodes[key]["layer"] = "first"
        return key

    def edge(self, a: str, ap: str | None, b: str, bp: str | None, *, once: bool = False,
             **kw: Any) -> dict[str, Any] | None:  # fmt: skip
        """An edge from box ``a`` (port ``ap``) to box ``b`` (port ``bp``); ``once`` a wire of the
        hardware, drawn once whatever uses it."""
        if once:
            key = (a, ap, b, bp)
            if key in self.wires:
                return None
            self.wires.add(key)
        base = f"{a}{'.' + ap if ap else ''}->{b}{'.' + bp if bp else ''}"
        ids = {e["id"] for e in self.edges}
        eid, k = base, 1
        while eid in ids:
            k += 1
            eid = f"{base}#{k}"
        e = {"id": eid, "from": a, "from_port": ap, "to": b, "to_port": bp, "style": "data",
             "task": None, "dt": None, "skip": [], **kw}  # fmt: skip
        parts = [p for p in (e.pop("mark", None), e["dt"], "broadcast" if e.pop("broadcast", False)
                             else None) if p]  # fmt: skip
        if e["skip"]:
            parts.append("↻ " + ", ".join(e["skip"]))
        e["label"] = " · ".join(parts) or None
        self.edges.append(e)
        return e


def drawing(g: Graph, tops: Sequence[Node], placed: dict[str, Any] | None = None,
            blocks: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:  # fmt: skip
    """The step's drawing (module doc): ``{"nodes", "edges", "outlines", "tasks"}``; ``tops`` the
    top-level nodes as drawn (api's ``fold``), ``placed`` the estimate's containers (their
    badges), ``blocks`` api's blocks list (the open ones become outlines)."""
    api = _api()
    d = _Drawing(g, placed or {})
    scalars = {c for c, cont in g.containers.items() if cont.attrs.get(SCALAR)}
    local = {c for c, cont in g.containers.items() if "loop.local" in cont.attrs}
    linked: dict[str, Node] = {n.attrs["container"]: n for n, _ in g.walk() if n.kind == "link"}
    forked = {r for n, _ in g.walk() if n.kind == "fork" for r in n.attrs["readers"]}  # D189
    nested = [_maps_to(t) for t in tops]
    top_of = {n.id: t.id for t, under in zip(tops, nested, strict=True) for n, _ in under}
    maps_of = {n.id: maps for under in nested for n, maps in under}
    nodes = {n.id: n for under in nested for n, _ in under}

    # the instances drawn, their nodes, the tasks of a shared one
    by_inst: dict[str, list[Node]] = {}
    for under in nested:
        for n, _ in under:
            if n.kind == "accelerated":
                by_inst.setdefault(n.attrs["instance"], []).append(n)
    task_of: dict[str, str] = {}
    shared: dict[str, list[str]] = {}
    for inst, ns in by_inst.items():
        if len(ns) > 1:
            t = _task_names(ns, top_of)
            task_of |= t
            shared[inst] = list(dict.fromkeys(t[n.id] for n in ns))
    all_tasks = list(dict.fromkeys(t for ts in shared.values() for t in ts))
    top_task: dict[str, str | None] = {}
    for t, under in zip(tops, nested, strict=True):
        mine = {task_of[n.id] for n, _ in under if n.id in task_of}
        top_task[t.id] = mine.pop() if len(mine) == 1 else None

    # a link's reader nest drawn on its writer's chain when their loops match level by level
    alias: dict[str, str] = {}

    def find(m: str) -> str:
        while m in alias:
            m = alias[m]
        return m

    also: dict[str, list[str]] = {}
    writer_of = {m.data: n for under in nested for n, _ in under for m in n.outputs.values()}
    readers_of: dict[str, Node] = {}
    for under in nested:
        for n, _ in under:
            for m in n.inputs.values():
                readers_of.setdefault(m.data, n)
    for c, ln in linked.items():
        w, r = writer_of.get(c), readers_of.get(c)
        if w is None or r is None or top_of[w.id] == top_of[r.id]:
            continue
        cw, cr = maps_of[w.id], maps_of[r.id]
        linear = all(len(mp.body or []) == 1 for mp in cr)
        if (
            cw
            and len(cw) == len(cr)
            and linear
            and all(
                (a.attrs["range"], a.attrs.get("loop.kind"))
                == (b.attrs["range"], b.attrs.get("loop.kind"))
                for a, b in zip(cw, cr, strict=True)
            )
        ):
            for a, b in zip(cw, cr, strict=True):
                if find(b.id) != find(a.id):
                    alias[b.id] = find(a.id)
                    also.setdefault(find(a.id), []).append(b.id)

    # a fork's other readers' nests drawn on its first reader's chain (D189): each loop onto the
    # first's of the same range and kind, else level by level; their read of the container one
    # edge, from the first's innermost loop it passes
    fork_tail: dict[
        str, tuple[str, str]
    ] = {}  # "<instance>.<port>" -> (the first's node, the data)
    for f, _ in g.walk():
        if f.kind != "fork":
            continue
        reader = {}
        for under in nested:
            for n, _ in under:
                if n.kind == "accelerated":
                    for p, m in n.inputs.items():
                        if m.data == f.attrs["container"]:
                            reader[f"{n.attrs['instance']}.{p}"] = n
        first = reader.get(f.attrs["readers"][0])
        if first is None:
            continue
        c0 = maps_of[first.id]
        for r in f.attrs["readers"][1:]:
            n = reader.get(r)
            if n is None or top_of[n.id] == top_of[first.id]:
                continue
            ci = maps_of[n.id]
            if not c0 or len(ci) != len(c0) or not all(len(mp.body or []) == 1 for mp in ci):
                continue
            key = [(mp.attrs["range"], mp.attrs.get("loop.kind")) for mp in c0]
            pairs = [(key.index(k), b) for b in ci
                     if key.count(k := (b.attrs["range"], b.attrs.get("loop.kind"))) == 1]  # fmt: skip
            if len(pairs) != len(ci) or len({i for i, _ in pairs}) != len(ci):
                if key != [(mp.attrs["range"], mp.attrs.get("loop.kind")) for mp in ci]:
                    continue
                pairs = list(enumerate(ci))
            for i, b in pairs:
                if find(b.id) != find(c0[i].id):
                    alias[b.id] = find(c0[i].id)
                    also.setdefault(find(c0[i].id), []).append(b.id)
            fork_tail[r] = (first.id, f.attrs["container"])
    depth = {find(mp.id): i for n in nodes.values() for i, mp in enumerate(maps_of[n.id])
             if mp.id not in alias}  # fmt: skip

    # hardware: a unit's ports, its slicers, packers, temporal muxes
    def unit_ports(inst: str) -> tuple[list[str], list[str]]:
        ins: list[str] = []
        outs: list[str] = []
        for n in by_inst[inst]:
            ins += [c for c, _ in _reads(g, n) if c not in ins]
            outs += [c for c in n.outputs if c not in outs]
        return ins, outs

    def unit_box(n: Node) -> str:
        inst = n.attrs["instance"]
        bid = f"unit:{inst}"
        if bid not in d.nodes:
            ins, outs = unit_ports(inst)
            ns = by_inst[inst]
            box = d.box(bid, kind="unit", text=api.compact_text(g, ns[0]), sel=ns[0].id,
                        nodes=[x.id for x in ns], ins=ins, outs=outs)  # fmt: skip
            if api.folds(n):
                box["fold"] = True
                overs = [", ".join(_over(x, maps_of[x.id])) for x in ns]
                if any(overs):
                    box["sub"] = "over " + " | ".join(dict.fromkeys(overs))
            if inst in shared:
                box["chips"] = [[x.id, task_of[x.id]] for x in ns]
        return bid

    def slicer_of(n: Node, port: str, data: str) -> str | None:
        sl = api.slicer_node(g, n, port, data)
        if sl is None:
            return None
        bid = f"slicer:{n.attrs['instance']}.{port}"
        d.box(bid, kind="slicer", text=sl["text"], sel=bid, ins=["in"], outs=["out"],
              unit=n.attrs["instance"], port=port, node=n.id, data=data)  # fmt: skip
        return bid

    def packer_of(n: Node, port: str, data: str) -> str | None:
        pk = api.packer_node(g, n, port, data)
        if pk is None:
            return None
        bid = f"packer:{n.attrs['instance']}.{port}"
        d.box(bid, kind="packer", text=pk["text"], sel=bid, ins=["in"], outs=["out"],
              unit=n.attrs["instance"], port=port, node=n.id, data=data)  # fmt: skip
        return bid

    def into(n: Node, port: str, data: str, task: str | None) -> tuple[str, str]:
        """Where a read from outside the hardware enters it: the unit's port, through its slicer,
        through a temporal mux per task when the unit is shared."""
        if n.kind != "accelerated":
            return n.id, port
        inst = n.attrs["instance"]
        ub = unit_box(n)
        sl = slicer_of(n, port, data)
        if inst in shared:
            tasks = shared[inst]
            if sl is not None:
                mid = f"mux:{inst}.{port}"
                mux_ins, mux_outs, wire_to = [f"{port}·{t}" for t in tasks], [port], (sl, "in")
            else:
                mid = f"mux:{inst}"
                outer = [p for p in unit_ports(inst)[0] if slicer_any(inst, p) is None
                         and not linked_port(inst, p)]  # fmt: skip
                mux_ins = [f"{p}·{t}" for t in tasks for p in outer]
                mux_outs, wire_to = outer, (ub, port)
            d.box(mid, kind="mux", text="temporal mux", sel=mid, ins=mux_ins, outs=mux_outs,
                  port_tasks={p: p.split("·", 1)[1] for p in mux_ins}, unit=inst,
                  width_of=ub if sl is None else None)  # fmt: skip
            d.edge(mid, port, *wire_to, once=True, style="wire")
            if sl is not None:
                d.edge(sl, "out", ub, port, once=True, style="seed" if port == SEED else "wire",
                       **bcast(n, port))  # fmt: skip
            return mid, f"{port}·{task_of[n.id]}"
        if sl is not None:
            d.edge(sl, "out", ub, port, once=True, style="seed" if port == SEED else "wire",
                   **bcast(n, port))  # fmt: skip
            return sl, "in"
        return ub, port

    def bcast(n: Node, port: str) -> dict[str, Any]:
        """A broadcast input's wire into its unit (D186): labelled ``<port> · broadcast``."""
        return {"mark": port, "broadcast": True} if api.broadcast_port(g, n, port) else {}

    def slicer_any(inst: str, port: str) -> str | None:
        for x in by_inst[inst]:
            for c, m in _reads(g, x):
                if c == port and m.data not in linked and m.data not in local:
                    return slicer_of(x, port, m.data)
        return None

    def linked_port(inst: str, port: str) -> bool:
        return any(c == port and m.data in linked for x in by_inst[inst] for c, m in _reads(g, x))

    def link_box(ln: Node, w: Node, wp: str, r: Node, rp: str) -> str:
        bid = f"link:{w.attrs['instance']}.{wp}>{r.attrs['instance']}.{rp}"
        if bid not in d.nodes:
            box = d.box(bid, kind="link", text=api.compact_text(g, ln), sel=ln.id, ins=["in"],
                        outs=["out"], links=[])  # fmt: skip
            box["depth"] = ln.attrs["depth"]
        box = d.nodes[bid]
        if ln.id not in box["links"]:
            box["links"].append(ln.id)
            if (
                w.id in task_of
            ):  # used by several tasks in turn: its FIFO, the tasks' links as chips
                box.setdefault("chips", []).append([ln.id, task_of[w.id]])
                if len(box["links"]) > 1:
                    depth = ln.attrs["depth"]
                    box["text"] = "wire" if depth == 0 else f"FIFO {depth}"
        return bid

    def leaf_box(n: Node) -> str:
        if n.kind == "accelerated":
            return unit_box(n)
        if n.kind == "transpose":
            return d.box(n.id, kind="copy", text=api.compact_text(g, n), sel=n.id,
                         ins=list(n.inputs), outs=list(n.outputs))["id"]  # fmt: skip
        if n.kind == "block":
            return d.box(n.id, kind="block", text=api.block_text(n.attrs["block"]), sel=n.id, block=n.id,
                         ins=list(n.inputs), outs=list(n.outputs))["id"]  # fmt: skip
        box = d.box(n.id, kind="tasklet", text=api.compact_text(g, n), sel=n.id,
                    ins=[c for c, _ in _reads(g, n)], outs=list(n.outputs))  # fmt: skip
        if api.folds(n):
            box["fold"] = True
            over = _over(n, maps_of[n.id])
            if over:
                box["sub"] = "over " + ", ".join(over)
        return n.id

    def loop_box(mp: Node) -> str:
        bid = find(mp.id)
        if bid not in d.nodes:
            src = nodes.get(bid, mp)
            its = api._iterations(g, src.attrs["range"])
            kind = src.attrs.get("loop.kind")
            box = d.box(bid, kind="loop", text=f"{bid} · {kind or 'untagged'}",
                        sub=f"{src.attrs['var']} · {'?' if its is None else its}", loop=kind,
                        sel=bid)  # fmt: skip
            merged = src.attrs.get("view.merged") or []  # a block's firing loop over its nests
            if len(merged) > 1:
                box["chips"] = [[m, None] for m in merged if m != bid]
        return bid

    def dtype_changes(n: Node, data: str) -> str | None:
        """The output's dtype when it is not one of the leaf's inputs' (vmul's int8 to int32)."""
        dt = g.containers[data].dtype
        ins = {g.containers[m.data].dtype for _, m in _reads(g, n) if m.data in g.containers}
        return dt if ins and dt not in ins else None

    tree: list[
        tuple[str, str, str | None]
    ] = []  # (map box, child box, child port), for the empty hops
    local_writer: dict[str, tuple[str, str]] = {}
    for t, under in zip(tops, nested, strict=True):
        task = top_task[t.id]
        written_here: dict[str, list[tuple[str, str]]] = {}
        pending: dict[str, list[tuple[Node, str, Memlet]]] = {}
        for n, maps in under:
            if n.kind == "map":
                loop_box(n)
                if maps and n.id not in alias:
                    tree.append((find(maps[-1].id), find(n.id), None))
                continue
            if n.kind == "link" or n.kind not in LEAVES:
                continue
            nb = leaf_box(n)
            if maps and maps[-1].id not in alias:
                tree.append((find(maps[-1].id), nb, None))
            chain = [find(mp.id) for mp in maps]
            for c, m in _reads(g, n):
                port = c
                if m.data in local:
                    if m.data in local_writer:
                        a, ap = local_writer[m.data]
                        d.edge(a, ap, *into(n, port, m.data, task), style="local", task=task,
                               data=m.data, sub=api.memlet_text(m), node=n.id, port=port,
                               mark=m.data)  # fmt: skip
                    continue
                if m.data in linked:
                    ln = linked[m.data]
                    w = writer_of.get(m.data)
                    wp = (
                        next((p for p, x in w.outputs.items() if x.data == m.data), "out")
                        if w
                        else "out"
                    )
                    lb = (
                        link_box(ln, w, wp, n, port)
                        if w is not None and n.kind == "accelerated"
                        else ln.id
                    )
                    d.edge(lb, "out", unit_box(n) if n.kind == "accelerated" else n.id, port, once=True,
                           style="link", data=m.data, sub=api.memlet_text(m), node=n.id, port=port)  # fmt: skip
                    continue
                if m.data in written_here:
                    for a, ap in written_here[m.data]:
                        d.edge(a, ap, *into(n, port, m.data, task), task=task, data=m.data,
                               sub=api.memlet_text(m), node=n.id, port=port)  # fmt: skip
                    continue
                src = d.container(m.data, first=d.version[m.data] == 0)
                passed = [i for i, mp in enumerate(maps) if mp.attrs["var"] in m.names]
                hops: list[
                    tuple[str, list[str], int | None]
                ] = []  # (target box, skipped outer loops, depth)
                prev = -1
                for i in passed:
                    hops.append((chain[i], [maps[j].attrs["var"] for j in range(prev + 1, i)], i))
                    prev = i
                if not passed:
                    skipped_all = [mp.attrs["var"] for mp in maps]
                else:
                    skipped_all = []
                seeded = port == SEED
                perm = n.attrs.get(TRANSPOSE, {}).get(c)
                a, ap = src, None
                tail = fork_tail.get(f"{n.attrs.get('instance')}.{port}")
                if tail is not None and tail[1] == m.data:  # the first reader's edge carries it
                    hops = []
                    if passed:
                        a = max((chain[i] for i in passed), key=lambda b: depth.get(b, 0))
                for b, skipped, i in hops:  # the subset as known at the loop's entry
                    d.edge(a, ap, b, None, task=task, skip=skipped, data=m.data,
                           sub=api.memlet_text(api._projected(g, m, _vars_before(maps, i))),
                           node=n.id, port=port)  # fmt: skip
                    a, ap = b, None
                tgt, tp = into(n, port, m.data, task)
                fork = n.kind == "accelerated" and f"{n.attrs['instance']}.{port}" in forked
                d.edge(a, ap, tgt, tp, task=task, skip=skipped_all, data=m.data,
                       style="seed" if seeded else "fork" if fork else "data",
                       sub=api.memlet_text(m), node=n.id, port=port,
                       mark=(api.transpose_label(m.data, perm) if perm is not None
                             else "fork" if fork else None),
                       transpose=perm is not None,
                       broadcast=(m.data in scalars and not m.names)
                       or (tgt.startswith("unit:") and api.broadcast_port(g, n, port)))  # fmt: skip
            for c, m in n.outputs.items():
                if m.data in local:
                    local_writer[m.data] = (nb, c)
                    continue
                if m.data in linked:
                    ln = linked[m.data]
                    r = readers_of.get(m.data)
                    rp = (
                        next((p for p, x in r.inputs.items() if x.data == m.data), "in")
                        if r
                        else "in"
                    )
                    lb = (
                        link_box(ln, n, c, r, rp)
                        if r is not None and n.kind == "accelerated"
                        else ln.id
                    )
                    d.edge(nb, c, lb, "in", once=True, style="link", dt=dtype_changes(n, m.data),
                           data=m.data, sub=api.memlet_text(m), node=n.id, port=c)  # fmt: skip
                    continue
                written_here.setdefault(m.data, []).append((nb, c))
                pending.setdefault(m.data, []).append((n, c, m))
        for c, ws in pending.items():  # the containers this nest wrote: their next versions
            d.version[c] += 1
            out = d.container(c)
            for n, port, m in ws:
                nb = leaf_box(n)
                a, ap = nb, port
                dt = dtype_changes(n, c)
                if n.kind == "accelerated":
                    pk = packer_of(n, port, c)
                    if pk is not None:
                        d.edge(a, ap, pk, "in", once=True, style="wire", dt=dt, data=c,
                               sub=api.memlet_text(m), node=n.id, port=port)  # fmt: skip
                        a, ap, dt = pk, "out", None
                    inst = n.attrs["instance"]
                    if inst in shared:
                        tasks = shared[inst]
                        dm = f"demux:{inst}.{port}"
                        outs = [f"{port}·{t}" for t in tasks]
                        d.box(dm, kind="demux", text="temporal demux", sel=dm, ins=[port], outs=outs,
                              port_tasks={p: p.split("·", 1)[1] for p in outs}, unit=inst,
                              width_of=pk)  # fmt: skip
                        d.edge(a, ap, dm, port, once=True, style="wire", dt=dt, data=c)
                        a, ap, dt = dm, f"{port}·{task_of[n.id]}", None
                d.edge(a, ap, out, None, task=task, dt=dt, data=c, sub=api.memlet_text(m),
                       node=n.id, port=port)  # fmt: skip
    # empty hops: a loop that drives a child no data passes through (not on a link's merged chain)
    for a, b, bp in tree:
        unit = b.split(":", 1)[1] if b.startswith("unit:") else None
        reach = {b} | {k for k, x in d.nodes.items() if unit is not None and x.get("unit") == unit}
        if not any(e["from"] == a and e["to"] in reach for e in d.edges):
            task = next((e["task"] for e in d.edges if e["to"] == a), None)
            d.edge(a, None, b, bp, style="empty", task=task)
    for bid, maps in also.items():
        d.nodes[bid]["chips"] = [[m, None] for m in maps]
    used = {e[k].split("@")[0] for e in d.edges for k in ("from", "to") if "@" in e[k]}
    csr_read = {n.inputs[c].data for n, _ in g.walk() for c in api.csr_connectors(g, n)}
    for c in g.containers:  # what no node uses: a box of its own in the inputs' row
        if c not in used and c not in linked and c not in local and c not in csr_read:
            d.container(c, first=True)
    _rows(d, nodes, alias)
    drives: dict[str, set[str | None]] = {}  # a loop box: the groups of the leaves under it
    for n in nodes.values():
        if n.kind in LEAVES:
            for mp in maps_of[n.id]:
                drives.setdefault(find(mp.id), set()).add(
                    group_of(n) if n.kind == "accelerated" else None)  # fmt: skip
    outlines = _outlines(g, d, by_inst, shared, blocks, nodes, drives)
    for n in d.nodes.values():  # a mux's or packer's outline is its unit's
        if n["kind"] in ("mux", "demux", "slicer", "packer") and "parent" not in n:
            n["parent"] = d.nodes[f"unit:{n['unit']}"].get("parent")
    return {
        "nodes": list(d.nodes.values()),
        "edges": d.edges,
        "outlines": outlines,
        "tasks": all_tasks,
    }


def _vars_before(maps: list[Node], i: int | None) -> set[str]:
    """The loop variables known at the entry of loop ``i``: those of the loops above it."""
    return {mp.attrs["var"] for mp in maps[: i or 0]}


def _rows(d: _Drawing, nodes: dict[str, Node], alias: dict[str, str]) -> None:
    """The row of each loop the kernel's inputs feed (CONTRACTS, "Recipe"): every
    outermost loop that reads only inputs, or what a copy made of them, on one row, each next
    level of its chain on the next."""
    kind = {k: n["kind"] for k, n in d.nodes.items()}
    preds: dict[str, list[str]] = {k: [] for k in d.nodes}
    for e in d.edges:
        if e["style"] != "empty":
            preds[e["to"]].append(e["from"])
    child: dict[str, list[str]] = {}
    parent: dict[str, str] = {}
    for k in d.nodes:
        n = nodes.get(k)
        if n is not None and n.kind == "map":
            for c in n.body or []:
                if c.kind == "map" and c.id not in alias:
                    child.setdefault(k, []).append(c.id)
                    parent[c.id] = k
    rank: dict[str, int] = {}

    def rank_of(k: str) -> int:
        if k not in rank:
            rank[k] = 0
            rank[k] = max((rank_of(p) + 1 for p in preds[k]), default=0)
        return rank[k]

    clean: dict[str, bool] = {}

    def only_inputs(k: str) -> bool:
        if k not in clean:
            clean[k] = True
            clean[k] = all(kind[p] in ("container", "copy") and only_inputs(p) for p in preds[k])
        return clean[k]

    front = [
        k for k, n in d.nodes.items() if n["kind"] == "loop" and k not in parent and only_inputs(k)
    ]
    if not front:
        return
    base = max(rank_of(k) for k in front)

    def walk(k: str, depth: int) -> None:
        d.nodes[k]["row"] = base + depth
        for c in child.get(k, []):
            walk(c, depth + 1)

    for k in front:
        walk(k, 0)


def _outlines(g: Graph, d: _Drawing, by_inst: dict[str, list[Node]], shared: dict[str, list[str]],
              blocks: Sequence[dict[str, Any]], nodes: dict[str, Node],
              drives: dict[str, set[str | None]]) -> list[dict[str, Any]]:  # fmt: skip
    """Groups and accelerators around their hardware, open blocks around their nests. Every
    group is drawn, one box per configuration the program writes: its units, links, slicers
    and packers, and the loops that drive only it (``drives``, a loop box's groups, None for a
    leaf that is not a unit). An accelerator is drawn when it holds more than one group, or one
    group of another name. An open block whose boxes all land in groups is not drawn: each
    group it fed names it."""
    groups: dict[str, dict[str, Any]] = {}
    for inst, ns in by_inst.items():
        n = ns[0]
        grp = groups.setdefault(
            group_of(n), {"acc": accelerator_of(n), "units": [], "links": False}
        )
        grp["units"].append(inst)
    for n in d.nodes.values():
        if n["kind"] == "link":
            inst = n["id"].split(":", 1)[1].split(".", 1)[0]
            for grp in groups.values():
                if inst in grp["units"]:
                    grp["links"] = True
    out: list[dict[str, Any]] = []
    drawn_grp = set(groups)
    accs: dict[str, list[str]] = {}
    for k, grp in groups.items():
        accs.setdefault(grp["acc"], []).append(k)
    # open blocks: their outlines, innermost of a node its parent
    open_ = [b for b in blocks if not b["collapsed"] and not b["hidden"] and b["foldable"]]
    opened = {b["name"] for b in open_}
    block_parent = {
        b["name"]: next((p for p in _ancestors(blocks, b) if p in opened), None) for b in open_
    }

    def innermost(n: Node) -> str | None:
        names = [b["name"] for b in n.attrs.get(BLOCK) or [] if b["name"] in opened]
        return names[-1] if names else None

    for b in open_:
        out.append({"id": f"block:{b['name']}", "kind": "block", "text": f"▾ {b['text']}",
                    "sel": b["name"], "block": b["name"], "parent": f"block:{block_parent[b['name']]}"
                    if block_parent[b["name"]] else None})  # fmt: skip
    acc_outline: dict[str, str | None] = {}
    for acc, ks in accs.items():
        drawn = [k for k in ks if k in drawn_grp]
        if len(ks) > 1 or (ks and ks[0] != acc):
            inner = {innermost(x) for k in ks for u in groups[k]["units"] for x in by_inst[u]}
            out.append({"id": f"acc:{acc}", "kind": "accelerator", "text": f"accelerator {acc}", "sel": acc,
                        "parent": f"block:{next(iter(inner))}" if len(inner) == 1 and None not in inner
                        else None})  # fmt: skip
            acc_outline[acc] = f"acc:{acc}"
        else:
            acc_outline[acc] = None
        for k in drawn:
            tasks = sorted({t for u in groups[k]["units"] for t in shared.get(u, [])})
            text = f"group {k}" + (f" · shared by {', '.join(tasks)}" if tasks else "")
            inner = {innermost(x) for u in groups[k]["units"] for x in by_inst[u]}
            par = acc_outline[acc] or (
                f"block:{next(iter(inner))}" if len(inner) == 1 and None not in inner else None
            )
            out.append({"id": f"grp:{k}", "kind": "group", "text": text, "sel": k, "parent": par})
    # every box's outline
    for bid, box in d.nodes.items():
        if box["kind"] == "unit":
            inst = bid.split(":", 1)[1]
            k = group_of(by_inst[inst][0])
            acc = groups[k]["acc"]
            if k in drawn_grp:
                box["parent"] = f"grp:{k}"
            elif acc_outline[acc]:
                box["parent"] = acc_outline[acc]
            else:
                inner = {innermost(x) for x in by_inst[inst]}
                if len(inner) == 1 and None not in inner:
                    box["parent"] = f"block:{next(iter(inner))}"
        elif box["kind"] == "link":
            inst = bid.split(":", 1)[1].split(".", 1)[0]
            box["parent"] = d.nodes[f"unit:{inst}"].get("parent")
        elif box["kind"] in ("loop", "tasklet", "copy", "block"):
            n = nodes.get(bid)
            blk = innermost(n) if n is not None else None
            if blk is not None:
                box["parent"] = f"block:{blk}"
            mine = drives.get(bid, set()) if box["kind"] == "loop" else set()
            if len(mine) == 1 and None not in mine:
                box["parent"] = f"grp:{next(iter(mine))}"
    # folded blocks one group spans (a fork across lifted blocks, D189): its box around them
    folded: dict[str, list[str]] = {}
    for bid, box in d.nodes.items():
        n = nodes.get(bid)
        if box["kind"] != "block" or n is None:
            continue
        name = n.attrs["block"]["name"]
        mine = {group_of(x) for x, _ in g.walk() if x.kind == "accelerated"
                and name in [b["name"] for b in x.attrs.get(BLOCK) or []]}  # fmt: skip
        if len(mine) == 1:
            folded.setdefault(mine.pop(), []).append(bid)
    for k, bids in folded.items():
        if len(bids) > 1 and f"grp:{k}" not in {o["id"] for o in out}:
            ups = {d.nodes[b].get("parent") for b in bids}
            out.append({"id": f"grp:{k}", "kind": "group", "text": f"group {k}", "sel": k,
                        "parent": ups.pop() if len(ups) == 1 else None})  # fmt: skip
            for b in bids:
                d.nodes[b]["parent"] = f"grp:{k}"
    # an open block none of whose boxes stayed in it: named on the groups it fed instead
    for b in sorted(open_, key=lambda b: -len(_ancestors(blocks, b))):  # innermost first
        oid = f"block:{b['name']}"
        content = [bid for bid, box in d.nodes.items() if box["kind"] in
                   ("loop", "tasklet", "copy", "block", "unit")
                   and (n := nodes.get(bid) or (by_inst.get(bid.split(":", 1)[1], [None])[0]
                                                if bid.startswith("unit:") else None)) is not None
                   and innermost(n) == b["name"]]  # fmt: skip
        if not content or any(d.nodes[x].get("parent") == oid for x in content):
            continue
        if any(o["parent"] == oid and o["kind"] not in ("group", "accelerator") for o in out):
            continue
        fed = {d.nodes[x].get("parent") for x in content}
        up = next(o["parent"] for o in out if o["id"] == oid)
        out = [o for o in out if o["id"] != oid]
        for o in out:
            if o["parent"] == oid:
                o["parent"] = up
            if o["id"] in fed and o["kind"] == "group":
                if b["name"] != o["sel"]:
                    sep = ", " if "· from " in o["text"] else " · from "
                    o["text"] += sep + b["name"]
                o.setdefault("block", b["name"])
        for box in d.nodes.values():
            if box.get("parent") == oid:
                box["parent"] = up
    return out


def _ancestors(blocks: Sequence[dict[str, Any]], b: dict[str, Any]) -> list[str]:
    by = {x["name"]: x for x in blocks}
    out = []
    p = b["parent"]
    while p is not None:
        out.append(p)
        p = by[p]["parent"]
    return out
