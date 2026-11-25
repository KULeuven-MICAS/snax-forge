"""A scale moved past the sum it feeds: alpha · sum_j (a_j x_j) for sum_j ((alpha · a_j) x_j).

``factor_out(container)`` and ``unfactor_out(container)``, as CONTRACTS, "Recipe" defines them.
``container`` is a kernel scalar argument k read by one scale nest: an elementwise ``out = k * a``
whose transient t is read by one product ``t * x`` summed by one fold (the product's own
reduce, or a nest summing it) into s. The scale nest and t go, the product reads a, the
fold writes a new transient ``<s>_sum``, and a nest over s's elements just after the fold's
writes ``s = k * <s>_sum``, so s keeps its meaning. Exact on integers when every type from the
scale to s is at least s's width (mod 2**w); refused on floats. The new scale's tasklet keeps
what went (``sandbox.factored``), for ``unfactor_out``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.brm.dtypes import bits
from snax_forge.common import fresh
from snax_forge.dfg import Container, Graph, Memlet, Node, loops
from snax_forge.dfg.graph import FACTORED, SCALAR
from snax_forge.expr import ExprError

from .recipe import SandboxError
from .seed import _indices, _nest_root, _shape, _valid
from .transforms import _copy, _locate, _result, register_transform

_PRODUCT = "out = a * b"


def _reader(g: Graph, container: str, what: str, *, factored: bool = False) -> Node:
    """The one tasklet reading the kernel scalar ``container``: its scale nest, or with
    ``factored`` the scale factor_out made."""
    c = g.containers.get(container)
    if c is None or not c.attrs.get(SCALAR):
        raise SandboxError(f"{what}: {container!r} is not a kernel scalar argument; the step "
                           "takes the scalar a nest scales by (gesummv's alpha)")  # fmt: skip
    readers = [n for n, _ in g.walk() if any(m.data == container for m in n.inputs.values())]
    bound = [n.id for n in readers if n.kind == "accelerated"]
    if bound:
        raise SandboxError(f"{what}: {container} is read by the bound unit {bound[0]} (unbind "
                           "first)")  # fmt: skip
    if factored:
        readers = [n for n in readers if FACTORED in n.attrs]
        if len(readers) != 1:
            raise SandboxError(f"{what}: {container} is read by {len(readers)} scales "
                               "factor_out made, not one")  # fmt: skip
    elif len(readers) != 1:
        raise SandboxError(f"{what}: {container} is read by {[n.id for n in readers]}; "
                           "factor_out moves a scalar that scales one nest")  # fmt: skip
    return readers[0]


def _stripped(t: Node, what: str) -> str | None:
    try:
        return expr.strip_casts(t.attrs["code"], what)[0]
    except ExprError:
        return None


def _product(t: Node, what: str) -> dict[str, str] | None:
    """``t``'s code as ``out = a * b`` without its casts: ``{a, b, out}`` the connectors, or
    None when it is not a product of its two inputs into its one output."""
    code = _stripped(t, what)
    if code is None or len(t.inputs) != 2 or len(t.outputs) != 1:
        return None
    ((out, _),) = t.outputs.items()
    a, b = t.inputs
    if expr.unify(f"{out} = {a} * {b}", code, ()) is None:
        return None
    return {"a": a, "b": b, "out": out}


def _out_bits(t: Node, dtypes: dict[str, str], what: str) -> tuple[int, str]:
    """The width and dtype ``t``'s code computes its output at, its inputs at ``dtypes``."""
    values = {c: np.zeros(1, dtype=d) for c, d in dtypes.items()}
    ((target, rhs),) = expr.statements(t.attrs["code"], f"{what}: {t.id}.attrs.code")
    del target
    try:
        dt = np.asarray(expr.evaluate(rhs, values)).dtype
    except ExprError as e:
        raise SandboxError(f"{what}: {t.id}: {e}") from None
    return bits(dt.name), dt.name


def _names(g: Graph, c: str, *keep: Node) -> list[str]:
    """Every node naming container ``c`` (a memlet, a link or a fork) but ``keep``."""
    out = []
    for n, _ in g.walk():
        if any(n is k for k in keep):
            continue
        if any(m.data == c for m in n.memlets()) or n.attrs.get("container") == c:
            out.append(n.id)
    return out


def _check(g: Graph, container: str, what: str) -> dict[str, Any]:
    s = _reader(g, container, what)
    prod = _product(s, what)
    scalars = [c for c, m in s.inputs.items() if g.containers[m.data].attrs.get(SCALAR)]
    if prod is None or len(scalars) != 1:
        raise SandboxError(f"{what}: {s.id} computes {s.attrs['code']!r}; factor_out takes a scale "
                           f"nest out = {container} * a")  # fmt: skip
    (ck,) = scalars
    (ca,) = [c for c in s.inputs if c != ck]
    ((cout, om),) = s.outputs.items()
    a, t = s.inputs[ca].data, om.data
    if om.reduce is not None or not g.containers[t].transient:
        raise SandboxError(f"{what}: {s.id} writes {t}, not a transient elementwise; factor_out "
                           "drops the scaled container")  # fmt: skip
    _, _, s_outer = _locate(g, s.id)
    shape = _shape(g, t, what)
    if _shape(g, a, what) != shape or s.inputs[ca].subset != om.subset:
        raise SandboxError(f"{what}: {s.id} reads {a} at another index than it writes {t}; "
                           "factor_out takes t = k * a element by element")  # fmt: skip
    written = _valid(_indices(g, s_outer, om, what), shape, s_outer)
    if len(written) != int(np.prod(shape)) or len({tuple(i) for i in written}) != len(written):
        raise SandboxError(f"{what}: {s.id} writes part of {t} only, or an element twice")
    readers = [n for n, _ in g.walk() if any(m.data == t for m in n.inputs.values())]
    others = _names(g, t, s, *readers)
    if others or len(readers) != 1:
        raise SandboxError(f"{what}: {t} is named by {[*(r.id for r in readers), *others]}; "
                           "factor_out takes a scaled transient read by one product")  # fmt: skip
    (p,) = readers
    if p.kind == "accelerated":
        raise SandboxError(f"{what}: {t}'s reader {p.id} is bound (unbind first)")
    pp = _product(p, what) if p.kind == "tasklet" else None
    cts = [c for c, m in p.inputs.items() if m.data == t]
    if pp is None or len(cts) != 1:
        raise SandboxError(f"{what}: {t}'s reader {p.id} must compute t * x, one read of t")
    (ct,) = cts
    ((pout, pm),) = p.outputs.items()
    if pm.reduce is not None:
        f, fout, fm = p, pout, pm
    else:
        prod_c = pm.data
        fs = [n for n, _ in g.walk() if any(m.data == prod_c for m in n.inputs.values())]
        rest = _names(g, prod_c, p, *fs)
        if rest or len(fs) != 1 or fs[0].kind != "tasklet":
            raise SandboxError(f"{what}: {p.id}'s product {prod_c} must be read by one fold "
                               f"that sums it, not {[*(n.id for n in fs), *rest]}")  # fmt: skip
        (f,) = fs
        ((fout, fm),) = f.outputs.items()
        code = _stripped(f, what)
        if (
            len(f.inputs) != 1
            or code is None
            or expr.unify(f"{fout} = {next(iter(f.inputs))}", code, ()) is None
        ):
            raise SandboxError(f"{what}: {f.id} computes {f.attrs['code']!r}, not a sum of "
                               f"{prod_c}")  # fmt: skip
    if fm.reduce is None or fm.reduce.op != "add":
        raise SandboxError(f"{what}: {f.id} does not sum (a reduce add); the scale moves past a "
                           "sum only")  # fmt: skip
    if fm.reduce.seed is not None:
        raise SandboxError(f"{what}: {f.id} is seeded from {fm.reduce.seed.data} (unfuse_seed "
                           "first)")  # fmt: skip
    out = fm.data
    if _names(
        g, out, f, *[n for n, _ in g.walk() if any(m.data == out for m in n.inputs.values())]
    ):
        raise SandboxError(f"{what}: {out} has another writer than {f.id}")
    _, _, f_outer = _locate(g, f.id)
    oshape = _shape(g, out, what)
    summed = {tuple(i) for i in _valid(_indices(g, f_outer, fm, what), oshape, f_outer)}
    if summed != {tuple(i) for i in np.ndindex(*oshape)}:
        raise SandboxError(f"{what}: {f.id} writes part of {out} only")
    if any(np.issubdtype(np.dtype(g.containers[c].dtype), np.floating) for c in (a, t, out)):
        raise SandboxError(f"{what}: a float type; moving the scale reorders the arithmetic, "
                           "exact on integers only")  # fmt: skip
    w = bits(g.containers[out].dtype)

    def dtype(c: str) -> str:
        return g.containers[c].dtype

    s_dt = {c: dtype(m.data) for c, m in s.inputs.items()}
    p_dt = {c: dtype(m.data) for c, m in p.inputs.items()}
    widths = [
        (f"{s.id}'s scale", _out_bits(s, s_dt, what)),
        (t, (bits(dtype(t)), dtype(t))),
        (f"{p.id}'s product", _out_bits(p, p_dt, what)),
        (f"{p.id}'s product of {a}", _out_bits(p, p_dt | {ct: dtype(a)}, what)),
        (pm.data, (bits(dtype(pm.data)), dtype(pm.data))),
        (f"the scale of {out}", _out_bits(s, s_dt | {ca: dtype(out)}, what)),
    ]
    for name, (b, d) in widths:
        if b < w:
            raise SandboxError(f"{what}: {name} is {d}, narrower than {out}'s "
                               f"{g.containers[out].dtype}; with the scale after the sum the "
                               f"bits would differ (cast the input, as x.astype)")  # fmt: skip
    return {"scale": s, "ck": ck, "ca": ca, "cout": cout, "a": a, "t": t, "product": p, "ct": ct,
            "fold": f, "fout": fout, "out": out}  # fmt: skip


def _site(g: Graph, selected: str) -> dict[str, Any]:
    """factor_out's site: a kernel scalar whose scale nest passes the checks."""
    _check(g, selected, f"factor_out {selected!r}")
    return {"container": selected}


def factor_out(g: Graph, container: str) -> Graph:
    """The scale by ``container`` moved past the sum it feeds (module doc)."""
    what = f"factor_out {container!r}"
    _check(g, container, what)
    g = _copy(g)
    found = _check(g, container, what)
    s, p, f = found["scale"], found["product"], found["fold"]
    ck, ca, cout, a, t, out = (found[k] for k in ("ck", "ca", "cout", "a", "t", "out"))
    root, parent, body = _nest_root(g, s)
    k = next(i for i, n in enumerate(body) if n is root)
    names = list(g.containers)
    taken_ids = {n.id for n, _ in g.walk()}
    summed = fresh(f"{out}_sum", set(g.containers) | set(g.symbols))
    record = {
        "container": t,
        "definition": g.containers[t].to_dict(),
        "index": names.index(t),
        "nest": root.to_dict(),
        "parent": parent.id if parent is not None else None,
        "after": body[k - 1].id if k else None,
        "product": p.id,
        "connector": found["ct"],
        "memlet": p.inputs[found["ct"]].to_dict(),
        "fold": f.id,
        "output": out,
    }
    p.inputs[found["ct"]] = Memlet(a, list(p.inputs[found["ct"]].subset))
    fm = f.outputs[found["fout"]]
    f.outputs[found["fout"]] = Memlet(summed, list(fm.subset), fm.reduce)
    del body[k]
    items = [(n, c) for n, c in g.containers.items() if n != t]
    at = next(i for i, (n, _) in enumerate(items) if n == out) + 1
    items.insert(at, (summed, Container(list(g.containers[out].shape), g.containers[out].dtype,
                                        transient=True)))  # fmt: skip
    g.containers = dict(items)
    shape = g.containers[out].shape
    dims = [d if isinstance(d, str) and d.isidentifier() else None for d in fm.subset]
    if len(set(dims)) != len(dims) or None in dims:
        dims = [f"i{d}" for d in range(len(shape))]
    sub = [str(d) for d in dims]
    tasklet = Node(fresh(f"{out}_scale", taken_ids), "tasklet",
                   {ck: Memlet(s.inputs[ck].data, list(s.inputs[ck].subset)),
                    ca: Memlet(summed, sub)},
                   {cout: Memlet(out, sub)},
                   {"code": f"{cout} = {ck} * {ca}", FACTORED: record})  # fmt: skip
    nest: Node = tasklet
    ids = [fresh(f"{out}_scale_map", taken_ids) for _ in dims]
    for mid, v, n in reversed(list(zip(ids, dims, shape, strict=True))):
        nest = Node(mid, "map", attrs={"var": v, "range": f"0:{n}", loops.DIM: v, loops.PIECE: 1},
                    body=[nest])  # fmt: skip
    froot, _, fbody = _nest_root(g, f)
    fbody.insert(next(i for i, n in enumerate(fbody) if n is froot) + 1, nest)
    return _result(g)


def _ids(n: Node) -> list[str]:
    return [n.id, *(i for c in n.body or [] for i in _ids(c))]


def _unfactor_check(g: Graph, container: str, what: str) -> tuple[Node, dict[str, Any]]:
    t = _reader(g, container, what, factored=True)
    rec = t.attrs[FACTORED]
    if rec["container"] in g.containers:
        raise SandboxError(f"{what}: the graph has a container {rec['container']!r} again")
    ids = {n.id for n, _ in g.walk()}
    nest = Node.from_dict(rec["nest"], what)
    back = [i for i in _ids(nest) if i in ids]
    if back:
        raise SandboxError(f"{what}: the graph has a node {back[0]!r} again")
    for key in ("product", "fold"):
        if rec[key] not in ids:
            raise SandboxError(f"{what}: the {key} {rec[key]!r} is gone (bound or renamed; undo "
                               "that first)")  # fmt: skip
    if rec["parent"] is not None and rec["parent"] not in ids:
        raise SandboxError(f"{what}: the map {rec['parent']!r} that held the scale is gone")
    summed = next(m.data for c, m in t.inputs.items() if m.data in g.containers
                  and not g.containers[m.data].attrs.get(SCALAR))  # fmt: skip
    fold = g.node(rec["fold"])
    others = _names(g, summed, t, fold)
    if others:
        raise SandboxError(f"{what}: {summed} is named by {others} too (undo that first)")
    return t, rec


def _unsite(g: Graph, selected: str) -> dict[str, Any]:
    """unfactor_out's site: a kernel scalar read by a scale factor_out made."""
    _unfactor_check(g, selected, f"unfactor_out {selected!r}")
    return {"container": selected}


def unfactor_out(g: Graph, container: str) -> Graph:
    """The inverse of factor_out: the scale nest and its container back, the fold writing s."""
    what = f"unfactor_out {container!r}"
    _unfactor_check(g, container, what)
    g = _copy(g)
    t, rec = _unfactor_check(g, container, what)
    fold, product = g.node(rec["fold"]), g.node(rec["product"])
    ((fout, fm),) = fold.outputs.items()
    summed = fm.data
    fold.outputs[fout] = Memlet(rec["output"], list(fm.subset), fm.reduce)
    product.inputs[rec["connector"]] = Memlet.from_dict(rec["memlet"], what)
    root, _, body = _nest_root(g, t)
    del body[next(i for i, n in enumerate(body) if n is root)]
    items = [(n, c) for n, c in g.containers.items() if n != summed]
    items.insert(min(rec["index"], len(items)),
                 (rec["container"], Container.from_dict(rec["definition"], what)))  # fmt: skip
    g.containers = dict(items)
    body = g.body if rec["parent"] is None else (g.node(rec["parent"]).body or [])
    ids = [n.id for n in body]
    if rec["after"] in ids:
        at = ids.index(rec["after"]) + 1
    elif rec["after"] is None:
        at = 0
    else:  # just before the nest holding the product
        proot, _, _ = _nest_root(g, product)
        at = next(i for i, n in enumerate(body) if n is proot)
    body.insert(at, Node.from_dict(rec["nest"], what))
    return _result(g)


register_transform("factor_out", factor_out, site=_site)
register_transform("unfactor_out", unfactor_out, site=_unsite)
