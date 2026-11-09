"""Blocks: a kernel found inside a bigger graph, and a recipe applied there (REUSE1, open item
117).

A block is a region of the graph that equals a smaller kernel's import (its pattern, ``P``):
dot inside each row of a mat-vec, a mat-vec inside each half of mvt. Recipes are written on
their own kernel's ids (dot_3 on dot's ``mult_map``, ``mult``, ``tmp0``); ``apply_recipe`` runs
one on a region of another graph, so a design is built from recipes of recipes
(``matvec/matvec_dot3`` applies ``dot/dot_3`` on the rows) and nothing is written twice.

**The match** (``match``). A region ``R`` matches the pattern ``P`` of a kernel when, under a
renaming of maps, variables, containers and symbols:

- R1 P's nests (each a chain of single-child maps down to one tasklet) correspond to chains
  of R, in P's order, each later one found through the containers the earlier ones write;
- R2 each chain of R may sit under maps of its own (its outer maps: any range, tag or split),
  which become that chain's firing loops;
- R3 below them, R's chain equals P's level by level, P's ranges under the symbol renaming σ
  (dot's ``0:N`` is mvt's ``0:N`` in half 1, ``0:M`` in half 2), untagged and unsplit;
- R4 the tasklets' code is equal, and a fold has the same op and start;
- R5 each memlet of R, without its dimensions indexed by the chain's outer variables only,
  equals P's under the renaming, P's size-1 dimensions at a constant (``out[0]``) dropped; a
  container R reads without them is reused across the outer loops;
- R6 P's transients correspond to transients only R writes and reads, written over what is
  read (the outer ranges that index them equal on the writer and the reader);
- R7 the container renaming is one to one;
- R8 a container may correspond with its dimensions permuted (half 2 of mvt reads ``A[m, n]``
  under the loops (n, m): matvec_acc's A through π = (1, 0)); the transpose mark (D164) gates
  nothing;
- R9 the chains of R are untouched since import: nothing in them split, tagged, fused, bound,
  linked or seeded, and in no block of the same kernel yet.

``match(g, kernel, heads)`` is the match at given chain heads, ``matches(g, kernel)`` every
non-overlapping one, in the graph's order.

**Blocks** are records on nodes: ``sandbox.block``, the path of blocks a node is in, outer
first, each ``{"name", "kernel", "heads"}`` (the region's chain heads), with ``recipe`` and
``params`` once a recipe is applied. ``lift(kernel)`` makes a block of every region that
matches the kernel's pattern (none overlapping; a region lies inside a block or holds it
whole, in either order: lift dot then matvec, or matvec then dot), named after the kernel (``dot``, ``dot_1``, ...); ``unlift(name)`` takes its record away.

**apply_recipe(recipe, region, block, params, name)**: ``recipe`` as ``category/name``
(``dot/dot_3``), on the region whose chains start at ``region`` (map ids, in P's nest order) or
on the lifted ``block``; ``params`` the sub-recipe's params (ints: run.py evaluates an
expression over the outer recipe's params, ``{"W": "W"}``); ``name`` the block's (default the
kernel's, made unique). The sub-recipe's steps run on the outer graph one by one, each step's
names (maps, nodes, containers) renamed onto the region's: the match gives the first, and the
names each step creates are paired by running the same steps on the pattern side by side
(``mult_map_s`` is ``tmp0_mult_map_1_s``). The names a sub-recipe gives its units, groups and
accelerators (``mul``, ``sum``) are taken as they are, ``_1``, ``_2``, ... on a clash (the
second half's ``mul_1``); a group or accelerator named after the sub-recipe's kernel (dot_3's
group ``dot``) takes the block's name. The sub-recipe's symbols are the region's (σ); each param a
symbol names is set to that value. Every node of the region after the expansion, and every
node a sub-step made (a link), gets the block in its path, with the recipe and its params. The
whole step is one step of the outer recipe, checked against the reference once (run.py); every
inner step runs through its transform's own checks.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from typing import Any

from snax_forge import expr
from snax_forge.common import fresh
from snax_forge.dfg import DfgError, Graph, Memlet, Node
from snax_forge.dfg.subset import dim_names, map_range, parse_dim
from snax_forge.expr import ExprError

from .recipe import RECIPE_ROOT, Recipe, SandboxError
from .transforms import _copy, _result, register_transform

BLOCK = "sandbox.block"
# the attrs that say a map or a tasklet was transformed since import (R9)
TOUCHED = ("loop.kind", "loop.split", "loop.fused", "loop.split_reduce", "sandbox.seeded")


@functools.cache
def pattern(kernel: str) -> Graph:
    """The kernel's import, its symbols unbound: the pattern its recipes are written on."""
    from snax_forge.dfg.import_sdfg import import_kernel

    return import_kernel(kernel)


def recipe_path(ref: str) -> Any:
    """``category/name`` (``dot/dot_3``) as ``recipes/<category>/<name>.json``; a bare name is
    looked up across the folders (recipe.py's ``find_recipe``)."""
    from .recipe import find_recipe

    if "/" not in ref:
        return find_recipe(ref)
    p = RECIPE_ROOT / f"{ref}.json"
    if not p.is_file():
        raise SandboxError(f"no recipe {ref!r} ({p} is not a file)")
    return p


# -- the match ---------------------------------------------------------------------------------


def chain(n: Node) -> list[Node] | None:
    """``n`` and its single children down to one tasklet (maps, then the tasklet); None when a
    map holds anything else."""
    out = [n]
    while n.kind == "map":
        if len(n.body or []) != 1:
            return None
        n = n.body[0]
        out.append(n)
    return out if n.kind == "tasklet" else None


@dataclass
class Match:
    """A region matching a kernel's pattern (module doc)."""

    kernel: str
    heads: list[str]  # the region's chain heads, in P's nest order
    ids: dict[str, str] = field(default_factory=dict)  # P node id -> R node id
    containers: dict[str, str] = field(default_factory=dict)  # P container -> R container
    symbols: dict[str, Any] = field(default_factory=dict)  # P symbol -> R expression
    perms: dict[str, list[int]] = field(default_factory=dict)  # R container -> π (not identity)
    outer: dict[str, list[str]] = field(default_factory=dict)  # R head -> its outer map ids
    nodes: list[str] = field(default_factory=list)  # every node id of the region's chains


class _NoMatch(Exception):
    pass


def _outer_of(g: Graph, node_id: str) -> list[Node]:
    for n, outer in g.walk():
        if n.id == node_id:
            return [m for m in outer if m.kind == "map"]
    raise _NoMatch(f"no node {node_id}")


def _blocks_of(n: Node) -> list[dict[str, Any]]:
    return list(n.attrs.get(BLOCK, []))


def _same(a: Any, b: Any) -> bool:
    return str(expr.canonical(a)) == str(expr.canonical(b))


def _level(pm: Node, rm: Node, m: Match, var: dict[str, str]) -> None:
    """One level of a chain (R3): ranges under σ, untouched, the variables paired."""
    if any(k in rm.attrs for k in TOUCHED):
        raise _NoMatch(f"{rm.id} was transformed since import")
    pb, pe, ps = parse_dim(pm.attrs["range"], pm.id)
    rb, re, rs = parse_dim(rm.attrs["range"], rm.id)
    if not (_same(pb, rb) and ps == rs):
        raise _NoMatch(f"{rm.id}'s range {rm.attrs['range']} is not {pm.attrs['range']}")
    names = expr.names(pe)
    if len(names) == 1 and _same(pe, next(iter(names))):  # P's end a symbol: σ
        (s,) = names
        if s in m.symbols and not _same(m.symbols[s], re):
            raise _NoMatch(f"{s} is {m.symbols[s]} and {re}")
        m.symbols[s] = re
    elif not _same(expr.substitute(pe, m.symbols), re):
        raise _NoMatch(f"{rm.id}'s range {rm.attrs['range']} is not {pm.attrs['range']}")
    var[pm.attrs["var"]] = rm.attrs["var"]


def _memlet(p: Memlet, r: Memlet, var: dict[str, str], outer_vars: set[str], pg: Graph,
            m: Match, g: Graph) -> None:  # fmt: skip
    """One memlet (R5, R7, R8): R's without its outer dimensions equals P's renamed, its
    container one to one, its dimensions possibly permuted."""
    if (p.reduce is None) != (r.reduce is None):
        raise _NoMatch(f"{r.data}: a fold on one side only")
    if p.reduce is not None and (
        (p.reduce.op, p.reduce.identity) != (r.reduce.op, r.reduce.identity) or r.reduce.seed
    ):
        raise _NoMatch(f"{r.data}: another fold")
    shape = pg.containers[p.data].shape
    pd = [d for k, d in enumerate(p.subset)
          if not (expr.constant(d) is not None and shape[k] == 1)]  # fmt: skip
    pd = [expr.canonical(expr.substitute(d, var)) for d in pd]
    rd = []
    rshape = g.containers[r.data].shape
    for k, d in enumerate(r.subset):
        names = dim_names(d)
        if names and names <= outer_vars:
            continue  # indexed by the outer loops only: R2
        if expr.constant(d) is not None and rshape[k] == 1:
            continue  # a size-1 dimension at a constant, as P's
        rd.append(expr.canonical(d))
    if len(rd) != len(pd):
        raise _NoMatch(f"{r.data}: {len(rd)} inner dimensions, the pattern's {p.data} {len(pd)}")
    perm = []
    left = list(range(len(rd)))
    for d in pd:
        k = next((i for i in left if _same(rd[i], d)), None)
        if k is None:
            raise _NoMatch(f"{r.data}{r.subset}: no dimension for {p.data}'s {d}")
        left.remove(k)
        perm.append(k)
    if perm != sorted(perm):
        m.perms[r.data] = perm
    if m.containers.get(p.data, r.data) != r.data:
        raise _NoMatch(f"{p.data} is {m.containers[p.data]} and {r.data}")
    if r.data in m.containers.values() and m.containers.get(p.data) != r.data:
        raise _NoMatch(f"{r.data} stands for two of the pattern's containers")
    m.containers[p.data] = r.data


def _chain(pc: list[Node], rc: list[Node], g: Graph, pg: Graph, m: Match,
           outer: list[Node]) -> None:  # fmt: skip
    """One chain of R against one of P (R3, R4, R5, R9)."""
    if len(pc) != len(rc):
        raise _NoMatch(f"{rc[0].id}: {len(rc) - 1} maps, the pattern's {len(pc) - 1}")
    var: dict[str, str] = {}
    for pm, rm in zip(pc[:-1], rc[:-1], strict=True):
        _level(pm, rm, m, var)
    pt, rt = pc[-1], rc[-1]
    if any(k in rt.attrs for k in TOUCHED):
        raise _NoMatch(f"{rt.id} was transformed since import")
    try:
        same = expr.canonical_code(pt.attrs["code"], pt.id) == expr.canonical_code(
            rt.attrs["code"], rt.id
        )
    except ExprError:
        same = False
    if not same or set(pt.inputs) != set(rt.inputs) or set(pt.outputs) != set(rt.outputs):
        raise _NoMatch(f"{rt.id}'s code is not {pt.id}'s")
    outer_vars = set()
    for o in outer:
        outer_vars.add(o.attrs["var"])
    for c in pt.inputs:
        _memlet(pt.inputs[c], rt.inputs[c], var, outer_vars, pg, m, g)
    for c in pt.outputs:
        _memlet(pt.outputs[c], rt.outputs[c], var, outer_vars, pg, m, g)
    for pn, rn in zip(pc, rc, strict=True):
        m.ids[pn.id] = rn.id
    m.nodes += [n.id for n in rc]
    m.outer[rc[0].id] = [o.id for o in outer]


def _transients(g: Graph, pg: Graph, m: Match) -> None:
    """R6: P's transients are R's, written and read inside R only, over the same elements."""
    region = set(m.nodes)
    for pc, rc in m.containers.items():
        if not pg.containers[pc].transient:
            continue
        if not g.containers[rc].transient:
            raise _NoMatch(f"{rc} is not a transient as the pattern's {pc} is")
        uses = []
        for n, outer in g.walk():
            for side, ms in (("w", n.outputs), ("r", n.inputs)):
                for mm in ms.values():
                    if mm.data == rc:
                        if n.id not in region:
                            raise _NoMatch(f"{rc} is used outside the region, by {n.id}")
                        uses.append((side, mm, [o for o in outer if o.kind == "map"]))
        env = dict(g.bound_symbols)
        spans = {}
        for side, mm, outer in uses:
            ranges = {o.attrs["var"]: o.attrs["range"] for o in outer}
            for k, d in enumerate(mm.subset):
                names = dim_names(d)
                if len(names) == 1 and next(iter(names)) in ranges:
                    try:
                        r = map_range(ranges[next(iter(names))], env, rc)
                    except ExprError:
                        continue
                    spans.setdefault(k, {}).setdefault(side, r)
        for k, s in spans.items():
            if "w" in s and "r" in s and s["w"] != s["r"]:
                raise _NoMatch(f"{rc}'s dimension {k}: written over {s['w']}, read over {s['r']}")


def match(g: Graph, kernel: str, heads: list[str], partial: bool = False,
          block: str | None = None) -> Match:  # fmt: skip
    """The match of ``kernel``'s pattern at the region whose chains start at ``heads`` (in P's
    nest order); SandboxError naming the rule that fails. ``partial``: the first nests only,
    R6 not checked yet (the search's way in). ``block``: the lifted block at ``heads``, the one
    block of ``kernel`` the region may be in."""
    pg = pattern(kernel)
    pcs = [chain(n) for n in pg.body]
    if any(c is None for c in pcs):
        raise SandboxError(f"{kernel}: its import is not a sequence of nests")
    if len(heads) != len(pcs) and not (partial and len(heads) < len(pcs)):
        raise SandboxError(f"{kernel}: {len(pcs)} nests, the region has {len(heads)}")
    m = Match(kernel, list(heads))
    try:
        for pc, h in zip(pcs, heads, strict=False):
            rc = chain(g.node(h))
            if rc is None:
                raise _NoMatch(f"{h} is not a chain of maps down to one tasklet")
            for n in rc:
                if any(b["kernel"] == kernel and b["name"] != block for b in _blocks_of(n)):
                    raise _NoMatch(f"{n.id} is in a {kernel} block already")
            _chain(pc, rc, g, pg, m, _outer_of(g, h))
        if not partial:
            _transients(g, pg, m)
    except (_NoMatch, DfgError, ExprError, KeyError) as e:
        raise SandboxError(f"no {kernel} at {', '.join(heads)}: {e}") from None
    return m


def matches(g: Graph, kernel: str) -> list[Match]:
    """Every region of ``g`` matching ``kernel``'s pattern, none overlapping, in the graph's
    order: each later chain found through the containers the earlier ones write."""
    pg = pattern(kernel)
    pcs = [chain(n) for n in pg.body]
    if any(c is None for c in pcs):
        return []
    chains = [(n.id, chain(n)) for n, _ in g.walk() if n.kind == "map"]
    chains = [(h, c) for h, c in chains if c is not None]
    used: set[str] = set()
    found = []

    def extend(heads: list[str]) -> list[str] | None:
        k = len(heads)
        if k == len(pcs):
            return heads
        try:
            mm = match(g, kernel, heads, partial=True)
        except SandboxError:
            return None
        for h, c in chains:
            if len(c) != len(pcs[k]) or h in heads or any(x.id in used for x in c):
                continue
            if True:  # linked to an earlier chain through a container it writes
                written = {mm.containers[o.data] for pc in pcs[:k] for o in pc[-1].outputs.values()
                           if o.data in mm.containers}  # fmt: skip
                if not {x.data for x in c[-1].inputs.values()} & written:
                    continue
            try:
                match(g, kernel, [*heads, h], partial=k + 1 < len(pcs))
            except SandboxError:
                continue
            got = extend([*heads, h])
            if got is not None:
                return got
        return None

    for h, c in chains:
        if any(x.id in used for x in c) or len(c) != len(pcs[0]):
            continue
        heads = extend([h])
        if heads is None:
            continue
        try:
            mm = match(g, kernel, heads)
        except SandboxError:
            continue
        used |= set(mm.nodes)
        found.append(mm)
    return found


# -- lift and unlift ---------------------------------------------------------------------------


def _block_names(g: Graph) -> set[str]:
    return {b["name"] for n, _ in g.walk() for b in _blocks_of(n)}


def _taken(g: Graph) -> set[str]:
    """Every name a block, instance, group or accelerator has."""
    out = _block_names(g)
    for n, _ in g.walk():
        if n.kind == "accelerated":
            out |= {n.attrs[k] for k in ("instance", "group", "accelerator") if k in n.attrs}
    return out


def _region_nodes(g: Graph, m: Match) -> list[Node]:
    """The region's nodes: its chains, and every node under them."""
    out = []
    ids = set(m.nodes)
    for n, outer in g.walk():
        if n.id in ids or any(o.id in m.heads for o in outer):
            out.append(n)
    return out


def _nest(g: Graph, nodes: list[Node], what: str) -> set[str]:
    """The blocks a region lies in (the rest it touches it holds whole); refused on a block it
    holds part of."""
    ids = {n.id for n in nodes}
    holders = set()
    for name in {b["name"] for n in nodes for b in _blocks_of(n)}:
        mine = {n.id for n, _ in g.walk() if any(b["name"] == name for b in _blocks_of(n))}
        if ids <= mine:
            holders.add(name)
        elif not mine <= ids:
            raise SandboxError(f"{what}: the region holds part of block {name} only")
    return holders


def _mark(nodes: list[Node], entry: dict[str, Any], holders: set[str]) -> None:
    """``entry`` added to each node's block path: after the blocks holding the region, before
    those it holds."""
    for n in nodes:
        path = [b for b in _blocks_of(n) if b["name"] != entry["name"]]
        n.attrs[BLOCK] = ([b for b in path if b["name"] in holders] + [dict(entry)]
                          + [b for b in path if b["name"] not in holders])  # fmt: skip


def lift(g: Graph, kernel: str) -> Graph:
    """Every region matching ``kernel``'s pattern made a block (module doc)."""
    found = matches(g, kernel)
    if not found:
        raise SandboxError(f"lift {kernel}: no region of the graph matches {kernel}'s pattern")
    g = _copy(g)
    taken = _taken(g)
    for m in found:
        nodes = _region_nodes(g, m)
        holders = _nest(g, nodes, f"lift {kernel}")
        _mark(nodes, {"name": fresh(kernel, taken), "kernel": kernel, "heads": m.heads}, holders)
    return _result(g)


def unlift(g: Graph, name: str) -> Graph:
    """Block ``name``'s record taken away; refused once a recipe is applied there."""
    nodes = [n for n, _ in g.walk() if any(b["name"] == name for b in _blocks_of(n))]
    if not nodes:
        raise SandboxError(f"unlift {name!r}: no block of that name ({sorted(_block_names(g))})")
    g = _copy(g)
    for n, _ in g.walk():
        path = _blocks_of(n)
        for b in path:
            if b["name"] == name and "recipe" in b:
                raise SandboxError(f"unlift {name!r}: {b['recipe']} is applied there (unapply)")
        if any(b["name"] == name for b in path):
            rest = [b for b in path if b["name"] != name]
            if rest:
                n.attrs[BLOCK] = rest
            else:
                del n.attrs[BLOCK]
    return _result(g)


# -- apply_recipe -------------------------------------------------------------------------------

# a sub-step's params that name a unit, a group or an accelerator: renamed to a free name
NAMED = ("instance", "group", "accelerator", "name")


def _ids(g: Graph) -> list[str]:
    return [n.id for n, _ in g.walk()]


def _renamed(v: Any, ren: dict[str, str]) -> Any:
    if isinstance(v, str):
        return ren.get(v, v)
    if isinstance(v, list):
        return [_renamed(x, ren) for x in v]
    if isinstance(v, dict):
        return {k: _renamed(x, ren) for k, x in v.items()}
    return v


def _pair(before: list[str], after: list[str], rbefore: list[str], rafter: list[str],
          what: str) -> dict[str, str]:  # fmt: skip
    """The ids a step made on the pattern side paired with those it made on the region, in the
    graph's order."""
    new_p = [x for x in after if x not in set(before)]
    new_r = [x for x in rafter if x not in set(rbefore)]
    if len(new_p) != len(new_r):
        raise SandboxError(f"{what}: made {new_r} on the region, {new_p} on the pattern")
    return dict(zip(new_p, new_r, strict=True))


def _fuse_outer(g: Graph, first: str, second: str) -> dict[str, str]:
    """A sub-recipe's ``fuse_map`` on two of the region's nests: the nests themselves when one
    comes right after the other in one body, else their outer maps, the innermost pair that
    does (matvec's ``mult_map`` and ``sum_map`` for dot's ``mult_map`` and ``sum_map``: the
    fusion takes the rows' loop with it)."""
    from .compose import _fuse_next

    try:
        _fuse_next(g, first)
        return {}
    except SandboxError:
        pass
    a = [*_outer_of(g, first), g.node(first)]
    b = [*_outer_of(g, second), g.node(second)]
    for k in range(min(len(a), len(b)) - 1, -1, -1):
        try:
            if _fuse_next(g, a[k].id)["second"] == b[k].id:
                return {"first": a[k].id, "second": b[k].id}
        except SandboxError:
            continue
    return {}


def apply_recipe(g: Graph, recipe: str, region: list[str] | None = None,
                 block: str | None = None, params: dict[str, Any] | None = None,
                 name: str | None = None) -> Graph:  # fmt: skip
    """``recipe``'s steps on a region of ``g`` (module doc)."""
    from .run import step_params
    from .transforms import TRANSFORMS

    what = f"apply_recipe {recipe!r}"
    sub = Recipe.load(recipe_path(recipe))
    if block is not None:
        mine = [b for n, _ in g.walk() for b in _blocks_of(n) if b["name"] == block]
        if not mine:
            raise SandboxError(f"{what}: no block {block!r}")
        if mine[0]["kernel"] != sub.kernel:
            raise SandboxError(f"{what}: block {block} is a {mine[0]['kernel']}, the recipe is "
                               f"{sub.kernel}'s")  # fmt: skip
        if "recipe" in mine[0]:
            raise SandboxError(f"{what}: block {block} has {mine[0]['recipe']} already")
        region, name = list(mine[0]["heads"]), block
    if not region:
        raise SandboxError(f"{what}: give region (the chain heads) or block")
    m = match(g, sub.kernel, list(region), block=block)
    env = dict(g.bound_symbols)
    sigma = {}
    for s, v in m.symbols.items():
        try:
            sigma[s] = expr.evaluate(v, env)
        except ExprError as e:
            raise SandboxError(f"{what}: {s} = {v} has no value: {e}") from None
    values = dict(sub.params)
    for s, v in sub.symbols.items():  # a param a symbol names takes the region's value
        if isinstance(v, str) and expr.is_name(v) and s in sigma:
            values[v] = sigma[s]
    for k, v in (params or {}).items():
        if k not in sub.params:
            raise SandboxError(f"{what}: {sub.name} has no param {k!r} ({list(sub.params)})")
        values[k] = v
    sub = replace(sub, params=values, symbols={s: sigma[s] for s in sub.symbols if s in sigma})
    # the pattern side, its symbols the region's
    pd = pattern(sub.kernel).to_dict()
    pd["symbols"] = {s: sigma.get(s) for s in pd["symbols"]}
    pg = Graph.from_dict(pd)
    taken = _taken(g)
    ren = dict(m.ids) | dict(m.containers)
    units: dict[str, str] = {}  # the sub-recipe's unit, group and accelerator names
    if block is None:
        name = fresh(name or sub.kernel, taken)
    holders = _nest(g, _region_nodes(g, m), what)  # the rest, and blocks made, are inside
    out, made = g, []
    for i in range(len(sub.steps)):
        tname, sp = step_params(sub, i)
        spec = TRANSFORMS[tname]
        for k in NAMED:  # a unit's, group's or accelerator's name: free in the outer graph
            v = sp.get(k)
            if isinstance(v, str) and v not in units:
                # a group or accelerator named after the kernel (dot_3's group dot): the block's
                units[v] = (
                    name if k in ("group", "accelerator") and v == sub.kernel else fresh(v, taken)
                )
        for v in sp.get("units") or []:
            if v not in units:
                units[v] = fresh(v, taken)
        rp = {}
        for k, v in sp.items():
            if k in ("brm", "implementation", "kind", "recipe"):
                rp[k] = v
            elif k in NAMED or k == "units":  # their own namespace: dot's tasklet sum, unit sum
                rp[k] = _renamed(v, units)
            else:
                rp[k] = _renamed(v, ren)
        if tname == "fuse_map":
            rp |= _fuse_outer(out, rp["first"], rp["second"])
        step_what = f"{what} step {i + 1} ({tname})"
        pb, rb = _ids(pg), _ids(out)
        pcb, rcb = list(pg.containers), list(out.containers)
        try:
            pg2 = spec.fn(pg, **sp)
            out2 = spec.fn(out, **rp)
        except (SandboxError, DfgError) as e:
            raise SandboxError(f"{step_what}: {e}") from None
        ren |= _pair(pb, _ids(pg2), rb, _ids(out2), step_what)
        ren |= _pair(pcb, list(pg2.containers), rcb, list(out2.containers), step_what)
        made += [x for x in _ids(out2) if x not in set(rb)]
        pg, out = pg2, out2
    out = _copy(out)
    ids = set(m.nodes) | set(made)
    heads = set(region)
    entry = {"name": name, "kernel": sub.kernel, "heads": list(region), "recipe": recipe,
             "params": dict(params or {})}  # fmt: skip
    for n, outer in out.walk():
        if n.id not in ids and not any(o.id in heads for o in outer):
            continue
        path = [b for b in _blocks_of(n) if b["name"] != name]
        n.attrs[BLOCK] = ([b for b in path if b["name"] in holders] + [entry]
                          + [b for b in path if b["name"] not in holders])  # fmt: skip
    return _result(out)


register_transform("lift", lift)
register_transform("unlift", unlift)
register_transform("apply_recipe", apply_recipe)
