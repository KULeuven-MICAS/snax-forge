"""Blocks: a kernel found inside a bigger graph, and a recipe applied there.

A block is a region that equals a smaller kernel's import, its ``pattern`` P (dot in each row of
a mat-vec, a mat-vec in each half of mvt). ``match(g, kernel, heads)`` and ``matches(g, kernel)``
find a region R under a one-to-one renaming (R7) of maps, variables, containers and symbols:
P's nests are chains of R in P's order (R1), each under outer maps of its own (R2: its firing
loops, any range, tag or split); below them equal level by level (R3: P's ranges under the
symbol renaming, untagged and unsplit), with equal code and folds (R4); memlets equal once the
dimensions only the outer variables index are left out (R5); P's transients R's transients
(R6); a container may correspond permuted (R8: half 2 of mvt reads ``A`` through π = (1, 0));
and R untouched since import (R9).

``lift``, ``unlift``, ``name_block``, ``apply_recipe`` and ``unapply`` are the transforms of
CONTRACTS, "Recipe". ``apply_recipe`` runs the sub-recipe's steps on the region one by one, each
step's names renamed onto the region's by running the same steps on the pattern side by side;
the whole is one step, checked against the reference once. ``unapply`` reads the record
(``sandbox.applied``) on the first top-level node the recipe made. ``reusable``,
``lift_choices`` and ``recipes_for`` give the kernels and recipes the sites offer.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from pathlib import Path
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
    looked up across the folders (recipe.py's ``find_recipe``); a path to a ``.json`` is that
    file."""
    from .recipe import find_recipe

    if ref.endswith(".json") and Path(ref).is_file():  # a file of its own (a test's)
        return Path(ref)
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
            # linked to an earlier chain through a container it writes
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


@functools.cache
def reusable() -> tuple[str, ...]:
    """The kernels ``lift`` offers: every kernel a recipe under ``recipes/`` is written on."""
    import json

    out = set()
    for f in sorted(RECIPE_ROOT.glob("*/*.json")):
        try:
            out.add(json.loads(f.read_text())["kernel"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return tuple(sorted(out))


_FOUND: dict[int, tuple[Graph, dict[str, list[Match]]]] = {}


def _found(g: Graph) -> dict[str, list[Match]]:
    """Every reusable kernel's matches in ``g`` (not ``g``'s own kernel), worked out once per
    graph (the editor asks on every node of a step)."""
    hit = _FOUND.get(id(g))
    if hit is not None and hit[0] is g:
        return hit[1]
    out = {}
    for k in reusable():
        if k == g.name:
            continue
        try:
            got = matches(g, k)
        except (SandboxError, DfgError, KeyError):
            continue
        if got:
            out[k] = got
    if len(_FOUND) > 32:
        _FOUND.clear()
    _FOUND[id(g)] = (g, out)
    return out


def _lift_choices(g: Graph, at: str) -> list[tuple[str, Match]]:
    """The kernels whose pattern matches a region holding node ``at``, with that match;
    refused (the site's and the transform's message) when none does."""
    out = []
    for k, found in _found(g).items():
        for m in found:
            if at in {n.id for n in _region_nodes(g, m)}:
                out.append((k, m))
    if not out:
        raise SandboxError(f"lift at {at!r}: no kernel of the recipes matches a region here "
                           f"({', '.join(reusable())})")  # fmt: skip
    return out


def _lift_site(g: Graph, selected: str) -> dict[str, Any]:
    """A node in a region some recipe kernel's pattern matches: ``at`` it, ``kernel`` the
    first such (the editor offers the others and every region of the kernel, ``at`` empty)."""
    return {"at": selected, "kernel": _lift_choices(g, selected)[0][0]}


def lift_choices(g: Graph, at: str) -> list[str]:
    """The kernels ``lift`` takes at node ``at`` (the editor's choice)."""
    return list(dict.fromkeys(k for k, _ in _lift_choices(g, at)))


def lift(g: Graph, kernel: str, at: str | None = None) -> Graph:
    """The regions matching ``kernel``'s pattern made blocks (module doc): every one, or with
    ``at`` the one holding node ``at``."""
    if at:
        found = [m for k, m in _lift_choices(g, at) if k == kernel]
        if not found:
            raise SandboxError(f"lift at {at!r}: {kernel} does not match here (here: "
                               f"{', '.join(lift_choices(g, at))})")  # fmt: skip
    else:
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


def _block(g: Graph, selected: str, what: str, applied: bool) -> dict[str, Any]:
    """The block ``selected`` names: a block's name, or a node's innermost block; refused
    (one message for the site and the transform) when there is none, or when it has a recipe
    and ``applied`` is False, or none and ``applied`` is True."""
    rec = None
    for n, _ in g.walk():
        for b in _blocks_of(n):
            if b["name"] == selected:
                rec = b
        if rec is None and n.id == selected and _blocks_of(n):
            rec = _blocks_of(n)[-1]
        if rec is not None:
            break
    if rec is None:
        raise SandboxError(f"{what} {selected!r}: neither a block nor a node in one (lift)")
    if "recipe" in rec and not applied:
        raise SandboxError(f"{what} {selected!r}: block {rec['name']} has {rec['recipe']} "
                           "applied (unapply)")  # fmt: skip
    if "recipe" not in rec and applied:
        raise SandboxError(f"{what} {selected!r}: block {rec['name']} has no recipe applied")
    return rec


def _unlift_site(g: Graph, selected: str) -> dict[str, Any]:
    return {"name": _block(g, selected, "unlift", applied=False)["name"]}


def unlift(g: Graph, name: str) -> Graph:
    """Block ``name``'s record taken away (``name`` as the site takes it: the block, or a node
    in it); refused once a recipe is applied there."""
    name = _unlift_site(g, name)["name"]
    g = _copy(g)
    for n, _ in g.walk():
        path = _blocks_of(n)
        if any(b["name"] == name for b in path):
            rest = [b for b in path if b["name"] != name]
            if rest:
                n.attrs[BLOCK] = rest
            else:
                del n.attrs[BLOCK]
    return _result(g)


def recipes_for(kernel: str) -> list[str]:
    """The recipes under ``recipes/`` written on ``kernel``, as ``category/name``, those in the
    kernel's own folder first."""
    import json

    out = []
    for f in sorted(RECIPE_ROOT.glob("*/*.json")):
        try:
            if json.loads(f.read_text()).get("kernel") == kernel:
                out.append(f"{f.parent.name}/{f.stem}")
        except (OSError, ValueError, AttributeError):
            continue
    return sorted(out, key=lambda r: r.split("/")[0] != kernel)  # the kernel's own folder first


def _apply_site(g: Graph, selected: str) -> dict[str, Any]:
    """A lifted block with no recipe yet (or a node in it): ``block`` it, ``recipe`` the
    first recipe of its kernel."""
    rec = _block(g, selected, "apply_recipe", applied=False)
    if rec["kernel"] is None:
        raise SandboxError(f"apply_recipe {selected!r}: block {rec['name']} was named by hand "
                           "(name_block): no kernel to apply a recipe of")  # fmt: skip
    found = recipes_for(rec["kernel"])
    if not found:
        raise SandboxError(f"apply_recipe {selected!r}: no recipe under recipes/ is written on "
                           f"{rec['kernel']}")  # fmt: skip
    return {"block": rec["name"], "recipe": found[0]}


# -- apply_recipe -------------------------------------------------------------------------------

# a sub-step's params that name a unit, a group or an accelerator: renamed to a free name
NAMED = ("instance", "group", "accelerator")
# a sub-step's params that name a block: renamed onto the blocks made in the outer graph
BLOCK_PARAMS = {"apply_recipe": "block", "unapply": "block", "unlift": "name"}


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
    sub, m, region, block, name, pg = _bound_recipe(g, recipe, region, block, params, name, what)
    taken = _taken(g)
    ren = dict(m.ids) | dict(m.containers)
    units: dict[str, str] = {}  # the sub-recipe's unit, group and accelerator names
    blocks: dict[str, str] = {}  # the blocks its steps make, the pattern's -> the outer graph's
    if block is None:
        name = fresh(name or sub.kernel, taken)
    holders = _nest(g, _region_nodes(g, m), what)  # the rest, and blocks made, are inside
    out, made = g, []
    for i in range(len(sub.steps)):
        tname, sp = step_params(sub, i)
        spec = TRANSFORMS[tname]
        rp = _outer_step(tname, sp, units, blocks, ren, taken, name, sub.kernel)
        if tname == "fuse_map":
            rp |= _fuse_outer(out, rp["first"], rp["second"])
        if tname == "lift" and not rp.get("at"):  # every region of the pattern: here, the one
            found = matches(pg, rp["kernel"])
            if len(found) != 1:
                raise SandboxError(f"{what} step {i + 1} (lift): {len(found)} regions of "
                                   f"{rp['kernel']} in {sub.kernel}; give at")  # fmt: skip
            rp["at"] = ren.get(found[0].heads[0], found[0].heads[0])
        step_what = f"{what} step {i + 1} ({tname})"
        pb, rb = _ids(pg), _ids(out)
        pcb, rcb = list(pg.containers), list(out.containers)
        pbb, rbb = _block_order(pg), _block_order(out)
        try:
            pg2 = spec.fn(pg, **sp)
            out2 = spec.fn(out, **rp)
        except (SandboxError, DfgError) as e:
            raise SandboxError(f"{step_what}: {e}") from None
        ren |= _pair(pb, _ids(pg2), rb, _ids(out2), step_what)
        ren |= _pair(pcb, list(pg2.containers), rcb, list(out2.containers), step_what)
        new_blocks = _pair(pbb, _block_order(pg2), rbb, _block_order(out2), step_what)
        blocks |= new_blocks
        taken |= set(new_blocks.values())
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
    _keep_before(g, out, {n.id for n in _region_nodes(g, m)}, name)
    return _result(out)


def _bound_recipe(g: Graph, recipe: str, region: list[str] | None, block: str | None,
                  params: dict[str, Any] | None, name: str | None,
                  what: str) -> tuple[Recipe, Any, list[str], str | None, str | None, Graph]:  # fmt: skip
    """The sub-recipe on its region of ``g``: its params and symbols the region's, the match,
    the region's heads, the block and its name; and the pattern graph at the region's
    symbols."""
    if block is not None:  # the site's checks first, its messages
        _apply_site(g, block)
    sub = Recipe.load(recipe_path(recipe))
    if block is not None:
        rec = _block(g, block, "apply_recipe", applied=False)
        block = rec["name"]
        if rec["kernel"] != sub.kernel:
            raise SandboxError(f"{what}: block {block} is a {rec['kernel']}, the recipe is "
                               f"{sub.kernel}'s")  # fmt: skip
        region, name = list(rec["heads"]), block
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
    return sub, m, list(region), block, name, pg


def _outer_step(tname: str, sp: dict[str, Any], units: dict[str, str], blocks: dict[str, str],
                ren: dict[str, str], taken: set[str], name: str | None,
                kernel: str) -> dict[str, Any]:  # fmt: skip
    """A sub-recipe step's params for the outer graph: unit, group and accelerator names free
    there (``units`` grows), the blocks its steps made, the nodes and containers renamed."""
    for k in NAMED:  # a unit's, group's or accelerator's name: free in the outer graph
        v = sp.get(k)
        if isinstance(v, str) and v not in units:
            # a group or accelerator named after the kernel (dot_3's group dot): the block's
            units[v] = name if k in ("group", "accelerator") and v == kernel else fresh(v, taken)
    for v in sp.get("units") or []:
        if v not in units:
            units[v] = fresh(v, taken)
    rp = {}
    for k, v in sp.items():
        if k in ("brm", "implementation", "kind", "recipe", "params"):
            rp[k] = v
        elif k in NAMED or k == "units":  # their own namespace: dot's tasklet sum, unit sum
            rp[k] = _renamed(v, units)
        elif BLOCK_PARAMS.get(tname) == k:  # a block the sub-recipe made (lift, name_block)
            rp[k] = blocks.get(v, v)
        elif k == "name" and tname in ("name_block", "apply_recipe") and isinstance(v, str):
            rp[k] = blocks.setdefault(v, fresh(v, taken))  # a block it names: free here
        else:
            rp[k] = _renamed(v, ren)
    return rp


# -- unapply: the graph before a recipe was applied ----------------------------------------------

APPLIED = "sandbox.applied"


def _block_order(g: Graph) -> list[str]:
    return list(dict.fromkeys(b["name"] for n, _ in g.walk() for b in _blocks_of(n)))


def _tops_of(g: Graph, ids: set[str]) -> list[int]:
    """The indices of the top-level nodes holding any of ``ids``."""
    return [k for k, t in enumerate(g.body) if any(n.id in ids for n, _ in _walk_top(t))]


def _walk_top(t: Node) -> list[tuple[Node, None]]:
    out = [(t, None)]
    for c in t.body or []:
        out += _walk_top(c)
    return out


def _digest(tops: list[Node], containers: dict[str, Any]) -> str:
    """The top-level nodes and containers a block's recipe made, as unapply checks them: no
    step after it may have changed them (``sandbox.applied`` itself left out)."""
    import hashlib
    import json

    def strip(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if k != APPLIED}
        if isinstance(d, list):
            return [strip(v) for v in d]
        return d

    text = json.dumps([strip(t.to_dict()) for t in tops] + [containers], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _block_ids(g: Graph, name: str) -> set[str]:
    return {n.id for n, _ in g.walk() if any(b["name"] == name for b in _blocks_of(n))}


def _keep_before(g: Graph, out: Graph, region: set[str], name: str) -> None:
    """What ``unapply`` gives back, kept on the first top-level node holding block ``name``
    after the recipe: the top-level nodes that held its region before (and their place), the
    containers the recipe added and those it changed as they were, and the digest of what it
    made."""
    before = _tops_of(g, region)
    after = _tops_of(out, _block_ids(out, name))
    added = [c for c in out.containers if c not in g.containers]
    changed = {c: g.containers[c].to_dict() for c in g.containers  # or removed (a fuse)
               if c not in out.containers or g.containers[c] != out.containers[c]}  # fmt: skip
    mine = {c: out.containers[c].to_dict() for c in [*added, *changed] if c in out.containers}
    keep = {"tops": [g.body[k].to_dict() for k in before], "added": added, "changed": changed,
            "order": list(g.containers),
            "digest": _digest([out.body[k] for k in after], mine)}  # fmt: skip
    first = out.body[after[0]]
    first.attrs[APPLIED] = {**first.attrs.get(APPLIED, {}), name: keep}


def _unapply_site(g: Graph, selected: str) -> dict[str, Any]:
    """A block a recipe was applied on (or a node in it), not inside another such block."""
    rec = _block(g, selected, "unapply", applied=True)
    for n, _ in g.walk():
        path = _blocks_of(n)
        names = [b["name"] for b in path]
        if rec["name"] in names:
            outer = [b for b in path[: names.index(rec["name"])] if "recipe" in b]
            if outer:
                raise SandboxError(f"unapply {selected!r}: block {rec['name']} is inside "
                                   f"{outer[-1]['name']}, which {outer[-1]['recipe']} made "
                                   f"(unapply {outer[-1]['name']})")  # fmt: skip
            break
    return {"block": rec["name"]}


def unapply(g: Graph, block: str) -> Graph:
    """The graph as it was before ``apply_recipe`` made block ``block`` (as the site takes it:
    the block, or a node in it): its top-level nodes put back, the containers the recipe added
    removed and those it changed restored; a block the recipe applied on a lifted block stays
    lifted. Refused when a step after it changed what it made (undo those first)."""
    name = _unapply_site(g, block)["block"]
    keep = next((n.attrs[APPLIED][name] for n, _ in g.walk() if name in n.attrs.get(APPLIED, {})),
                None)  # fmt: skip
    after = _tops_of(g, _block_ids(g, name))
    mine = {c: g.containers[c].to_dict() for c in [*(keep or {}).get("added", []),
                                                   *(keep or {}).get("changed", {})]
            if c in g.containers}  # fmt: skip
    if keep is None or _digest([g.body[k] for k in after], mine) != keep["digest"]:
        raise SandboxError(f"unapply {block!r}: a step after its recipe changed block {name} "
                           "(undo it first)")  # fmt: skip
    from snax_forge.dfg.graph import Container

    g = _copy(g)
    at = after[0]
    for k in reversed(after):
        del g.body[k]
    g.body[at:at] = [Node.from_dict(d, f"unapply {name}") for d in keep["tops"]]
    for c in keep["added"]:
        g.containers.pop(c, None)
    order = list(keep.get("order") or g.containers)
    for c, d in keep["changed"].items():
        g.containers[c] = Container.from_dict(d, f"unapply {name} container {c}")
    g.containers = {c: g.containers[c] for c in [*order, *g.containers] if c in g.containers}
    return _result(g)


# -- name_block: a region named by hand -----------------------------------------------------------


def _name_multi(g: Graph, selected: list[str]) -> dict[str, Any]:
    """Nodes, the heads of the region (each with everything under it): ``nodes`` them, ``name``
    a free block name (``block``, ``block_1``, ...)."""
    _region_by_hand(g, list(selected), "name_block")
    return {"nodes": list(selected), "name": fresh("block", _taken(g))}


def _name_site(g: Graph, selected: str) -> dict[str, Any]:
    return _name_multi(g, [selected])


def _region_by_hand(g: Graph, nodes: list[str], what: str) -> list[Node]:
    ids = {n.id for n, _ in g.walk()}
    for x in nodes:
        if x not in ids:
            raise SandboxError(f"{what} {x!r}: not a node of the graph")
    out = [n for n, outer in g.walk() if n.id in nodes or any(o.id in nodes for o in outer)]
    _nest(g, out, f"{what} {nodes[0]!r}")
    return out


def name_block(g: Graph, nodes: list[str], name: str) -> Graph:
    """The region whose heads are ``nodes`` (each with everything under it) made a block named
    ``name``, by hand: no kernel and no recipe, drawn and collapsed as any block (the
    bottom-up entry: transforms by hand, then the block named)."""
    if not isinstance(nodes, list) or not nodes:
        raise SandboxError(f"name_block: nodes must be a non-empty list, got {nodes!r}")
    region = _region_by_hand(g, nodes, "name_block")
    if not isinstance(name, str) or not name.isidentifier():
        raise SandboxError(f"name_block: {name!r} is not a name")
    if name in _taken(g):
        raise SandboxError(f"name_block: {name!r} is taken (a block, unit, group or accelerator)")
    g = _copy(g)
    region = _region_by_hand(g, nodes, "name_block")
    _mark(region, {"name": name, "kernel": None, "heads": list(nodes)},
          _nest(g, region, f"name_block {nodes[0]!r}"))  # fmt: skip
    return _result(g)


register_transform("lift", lift, site=_lift_site)
register_transform("unapply", unapply, site=_unapply_site)
register_transform("name_block", name_block, site=_name_site, multi=_name_multi)
register_transform("unlift", unlift, site=_unlift_site)
register_transform("apply_recipe", apply_recipe, site=_apply_site)
