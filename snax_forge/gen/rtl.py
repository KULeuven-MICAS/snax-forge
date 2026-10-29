"""A design point -> its accelerator's RTL, from ``accelerator.json`` (D130, D145, C§17).

    accelerator_spec(design_point, root=None) -> Accelerator
    write_accelerator(design_point, out_dir, root=None) -> Accelerator
    vectors(design_point, root=None) -> dict
    write_fixture(design_point, out_dir, root=None) -> list[str]
    python -m snax_forge.gen.rtl DESIGN_POINT [OUT_DIR] [--fixture DIR]    (pixi gen-rtl)

``accelerator.json`` is the boundary between the Python side and ``hw/chisel/``: the one
description of the accelerator (accelerator.py), the same bytes gen-gvsoc writes, which
``Emit --spec`` reads to build ``<tag>_shell_wrapper`` (its ``<tag>_csr``, ``<tag>_ctrl`` and
``<tag>_datapath``, D147). Python resolves and checks; Scala only constructs.

**The checks**, against the datapath catalogue ``hw/chisel/blocks.json`` (the signature of each
Chisel datapath, equal to ``Emit --describe``), each a ``GenError``: first accelerator.py's
(``gen.instances``, ``gen.binding``, ``gen.datapath``, ``gen.controller``, ``gen.drain``,
``gen.timing``, the datapath's adapter's and a reader's link checks), then per unit, in this
order: the params by the fill rule, the ports and pins whose ``when`` holds, the ports, the
timing

    gen.catalogue    the datapath is not in the catalogue, or a fill rule is unknown
    gen.params       a param the fill rule gives that is not of its type or not among its values
                     (a dtype, an op, a bool, lanes below min)
    gen.port_map     the unit's datapath ports are not the catalogue's live ports, each once, or
                     a direction differs (each port's ``dir``)
    gen.timing       the unit's latency is not the catalogue's (an int, or an ``expr.py``
                     expression over the params, a bool as 0 or 1), its span not the
                     catalogue's, or its ii not 1

and per link of depth >= 1 its ``LinkBuffer``'s params (``link.depth``, and the ends'
``link.lanes`` and ``link.dtype``), ``gen.params``.

**Ports and pins** (C§17): a port is ``"in"`` / ``"out"`` or ``{"dir": ..., "when": {param:
value, ...}}``, a pin a name or ``{"name": ..., "when": {...}}``; one exists only when every
param its ``when`` lists has that value (``live``), one without ``when`` always.

and, from the command line, ``gen.elaborate``: ``Emit --spec`` failed or did not write a file
its ``emitted.json`` lists.

**The fill rule** (the catalogue's ``fill``): ``inputs.dtype`` the input ports' dtypes in order
(a ``dtypes`` param) or the first input's (a ``dtype``), ``outputs.dtype`` the output's,
``inputs.lanes`` the first input's lanes, ``unit.opType`` and ``unit.params.<name>`` the
unit's; a buffer's ``link.depth``, ``link.lanes`` and ``link.dtype``. Inputs and outputs are
the unit's ports by their ``dir``, in the BRM's order.

**The fixtures** (``write_fixture``): ``hw/chisel/src/test/resources/rtl/<design>/`` holds
``accelerator.json``, the design's streamer hjson (gvsoc.py's) and, for a design of one
accelerator task, ``vectors.json``, which the Chisel specs drive the wrapper with (dot 1's two
groups, a task each, have none: its spec drives each group itself): per outer port, in
``accelerator.json``'s ``inputs`` then ``outputs`` order, its BRM port, its data.h array (an
input, or the output's golden) and how many beats a task moves (lane i of beat t is element
t * lanes + i, modulo the array's size: matvec's vector is read again for every row), and per
task its CSR values (JOBS, JOB_BEATS: lower/c.py's ``job_writes``), the head unit's firings
and the counter with an always-ready streamer, the working in ``working``: one unit's worked
from the nest controller's rule (``counter``), a linked group's C§17's Cycles for its shape
(``group_counter``; ``gen.cycles`` for a shape C§17 does not state).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from snax_forge import expr
from snax_forge.common import cwd_relative, repo_root, to_json
from snax_forge.design.streamers import instance_nodes
from snax_forge.lower.c import job_values, job_writes
from snax_forge.lower.derive import task_list

from .accelerator import ACCELERATOR_JSON, Accelerator, Group, Link, Unit, units
from .blocks import GenError
from .data import arrays
from .gvsoc import ROOT, cli, generate, load_point, out_dir

OUT = ROOT / "out" / "rtl"
CATALOGUE = repo_root() / "hw" / "chisel" / "blocks.json"
VECTORS_JSON = "vectors.json"


# =============================================================================
# The checks against the catalogue
# =============================================================================


def catalogue() -> dict[str, Any]:
    """The datapath catalogue, ``hw/chisel/blocks.json``."""
    return json.loads(CATALOGUE.read_text())


def _ports(unit: Unit, direction: str) -> list[Any]:
    return [p for p in unit.ports.values() if p.dir == direction]


def _input_dtypes(unit: Unit, kind: str) -> Any:
    """The inputs' dtypes for a ``dtypes`` param, else the first input's."""
    dtypes = [p.dtype for p in _ports(unit, "in")]
    return dtypes if kind == "dtypes" else dtypes[0]


# The catalogue's fill rules (module doc): rule -> its value for a unit, given the param's type;
# ``unit.params.<key>`` is the unit's param of that name
FILL_RULES: dict[str, Callable[[Unit, str], Any]] = {
    "inputs.dtype": _input_dtypes,
    "outputs.dtype": lambda u, kind: _ports(u, "out")[0].dtype,
    "inputs.lanes": lambda u, kind: _ports(u, "in")[0].lanes,
    "unit.opType": lambda u, kind: u.op_type,
}
PARAMS_RULE = "unit.params."


def fill(acc: Accelerator, unit: Unit, sig: Mapping[str, Any]) -> dict[str, Any]:
    """The datapath's params of ``unit`` by the catalogue's fill rule (``FILL_RULES``)."""
    out = {}
    for name, rule in sig["fill"].items():
        if rule in FILL_RULES:
            out[name] = FILL_RULES[rule](unit, sig["params"][name]["type"])
        elif rule.startswith(PARAMS_RULE):
            out[name] = unit.params.get(rule.removeprefix(PARAMS_RULE))
        else:
            raise GenError(
                "gen.catalogue", f"{unit.datapath}.{name}: the fill rule {rule!r} is unknown"
            )
    return out


def link_params(acc: Accelerator, link: Link, sig: Mapping[str, Any]) -> dict[str, Any]:
    """A buffer's params by the catalogue's fill rule: its depth, and its ends' lanes and dtype
    (equal at both ends, a reader's check)."""
    unit, port = link.from_.split(".", 1)
    end = next(u for g in acc.groups for u in g.units if u.name == unit).ports[port]
    have = {"link.depth": link.depth, "link.lanes": end.lanes, "link.dtype": end.dtype}
    out = {}
    for name, rule in sig["fill"].items():
        if rule not in have:
            raise GenError("gen.catalogue", f"LinkBuffer.{name}: the fill rule {rule!r} is unknown")
        out[name] = have[rule]
    return out


def live(entries: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """The catalogue's ``ports`` (name -> dir) or ``pins`` (a list) whose ``when`` holds on
    ``params``: ports as name -> ``in`` / ``out``, pins as name -> None (module doc)."""
    items = entries.items() if isinstance(entries, Mapping) else ((p, p) for p in entries)
    out = {}
    for key, e in items:
        when = e.get("when", {}) if isinstance(e, Mapping) else {}
        if all(k in params and params[k] == v for k, v in when.items()):
            if isinstance(entries, Mapping):
                out[key] = e["dir"] if isinstance(e, Mapping) else e
            else:
                out[e["name"] if isinstance(e, Mapping) else e] = None
    return out


def value(v: int | str, params: Mapping[str, Any]) -> int:
    """A catalogue latency or span: an int, or an ``expr.py`` expression over the params (a bool
    as 0 or 1)."""
    if isinstance(v, int):
        return v
    env = {k: int(x) for k, x in params.items() if isinstance(x, (bool, int))}
    return expr.evaluate(v, env)


def check_params(datapath: str, params: Mapping[str, Any], sig: Mapping[str, Any]) -> None:
    """``params`` (``fill``'s) against the datapath's catalogue params ``sig``."""
    for k, t in sig.items():
        v, what = params[k], f"{datapath}.{k}"
        if t["type"] == "int":
            if type(v) is not int or v < t["min"]:
                raise GenError("gen.params", f"{what}: {v!r} is not an int of at least {t['min']}")
        elif t["type"] == "bool":
            if type(v) is not bool:
                raise GenError("gen.params", f"{what}: {v!r} is not a bool")
        elif t["type"] in ("dtype", "dtypes", "op"):
            vs = v if t["type"] == "dtypes" else [v]
            if not isinstance(vs, list) or not vs or not set(vs) <= set(t["values"]):
                raise GenError("gen.params", f"{what}: {v!r} must be among {list(t['values'])}")
        else:
            raise GenError("gen.params", f"{what}: the catalogue's type {t['type']!r} is unknown")


def check_unit(acc: Accelerator, unit: Unit, cat: Mapping[str, Any]) -> None:
    """One unit against the catalogue; see the module doc."""
    if unit.datapath not in cat:
        raise GenError("gen.catalogue", f"unit {unit.name}: datapath {unit.datapath!r} is not in "
                       f"the catalogue {sorted(cat)}")  # fmt: skip
    sig = cat[unit.datapath]
    params = fill(acc, unit, sig)
    check_params(unit.datapath, params, sig["params"])
    ports = live(sig["ports"], params)
    used = [p.port for p in unit.ports.values()]
    if sorted(used) != sorted(ports):
        raise GenError("gen.port_map", f"unit {unit.name}: ports onto {used}; {unit.datapath}'s "
                       f"are {list(ports)} here, each once")  # fmt: skip
    for brm, p in unit.ports.items():
        if p.dir != ports[p.port]:
            raise GenError("gen.port_map", f"unit {unit.name}: BRM port {brm} ({p.dir}) onto "
                           f"{p.port} ({ports[p.port]})")  # fmt: skip
    latency, span = value(sig["latency"], params), value(sig["span"], params)
    if (unit.latency, unit.ii, unit.span) != (latency, 1, span):
        raise GenError("gen.timing", f"unit {unit.name}: latency {unit.latency}, ii {unit.ii}, "
                       f"span {unit.span}; {unit.datapath} here is latency {latency}, ii 1, "
                       f"span {span}")  # fmt: skip


def check_link(acc: Accelerator, link: Link, cat: Mapping[str, Any]) -> None:
    """A link of depth >= 1 against the catalogue's ``LinkBuffer``; a wire is no datapath."""
    if link.depth == 0:
        return
    if "LinkBuffer" not in cat:
        raise GenError("gen.catalogue", f"link {link.name}: no LinkBuffer in the catalogue")
    sig = cat["LinkBuffer"]
    check_params("LinkBuffer", link_params(acc, link, sig), sig["params"])


def accelerator_spec(design_point: str | Path, root: str | Path | None = None,
                     cat: Mapping[str, Any] | None = None) -> Accelerator:  # fmt: skip
    """The design point's ``accelerator.json`` (accelerator.py), checked against the catalogue."""
    _, _, acc = load_point(design_point, ROOT if root is None else Path(root))
    cat = catalogue() if cat is None else cat
    for g in acc.groups:
        for u in g.units:
            check_unit(acc, u, cat)
        for k in g.links:
            check_link(acc, k, cat)
    return acc


def write_accelerator(design_point: str | Path, out_dir: str | Path,
                      root: str | Path | None = None) -> Accelerator:  # fmt: skip
    """Write ``accelerator.json`` to ``out_dir``, checked; the same bytes as gen-gvsoc's."""
    acc = accelerator_spec(design_point, root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / ACCELERATOR_JSON).write_text(acc.to_json())
    return acc


# =============================================================================
# The fixtures: vectors.json
# =============================================================================


def counter(unit: Unit, n: int) -> tuple[int, str]:
    """The counter a task of ``n`` firings gives with an always-ready streamer, and the working."""
    shell = ("The group's COUNTER is reset by the accepted start and counts each busy cycle; busy "
             "rises the cycle after the start")  # fmt: skip
    if unit.latency == 0:  # lockstep
        return n, (
            f"{shell}. L 0, II 1, inputs always valid and the output always ready: a firing every "
            f"busy cycle, the {n}th the run's last: {n} busy cycles"
        )
    last = (n - 1) * unit.ii + 1
    busy = last + unit.latency
    return busy, (
        f"{shell}. L {unit.latency}, II {unit.ii}, inputs always valid and the output always ready: "
        f"firing k (from 0) in busy cycle 1 + {unit.ii} * k, the last ({n - 1}) in {last}, its "
        f"result pushed {unit.latency} later, the run's last cycle (a sum before it leaves while the "
        f"next input is taken): {busy} busy cycles"
    )


# C§17's Cycles of a linked group by its shape (head's datapath, last's, the link's depth):
# J jobs of T beats (the group's one JOB_BEATS), W the last unit's span
CYCLES = {
    ("ElementwiseLanes", "TreeAccumulate", 1): ("C2 = JT + 2", lambda j, t, w: j * t + 2),
    ("ElementwiseLanes", "TreeAccumulate", 0): ("C3 = JT + 1", lambda j, t, w: j * t + 1),
    ("Mac", "TreeAccumulate", 0): ("C4 = JT + 2", lambda j, t, w: j * t + 2),
    ("Mac", "SerialAccumulate", 0): ("C5 = J(T + W - 1) + 2", lambda j, t, w: j * (t + w - 1) + 2),
    ("Mac", "SerialAccumulate", 1): ("C6 = (J - 1) max(T, W) + T + W + 2",
                                     lambda j, t, w: (j - 1) * max(t, w) + t + w + 2),
}  # fmt: skip


def group_counter(group: Group, csrs: Mapping[str, int]) -> tuple[int, str]:
    """The counter of a linked group's task (``csrs`` its JOBS and JOB_BEATS) with an always-ready
    streamer, from C§17's Cycles, and the working; ``gen.cycles`` for a shape C§17 does not
    state."""
    head, last = group.units[0], group.units[-1]
    shape = (head.datapath, last.datapath, *(k.depth for k in group.links))
    if shape not in CYCLES:
        raise GenError("gen.cycles", f"group {group.name}: C§17 states no cycles for {shape}")
    rule, f = CYCLES[shape]
    j, t, w = csrs["JOBS"], csrs.get("JOB_BEATS", 1), last.span
    return f(j, t, w), (
        f"The group's COUNTER counts its busy cycles. {head.datapath} -> {last.datapath}, a link "
        f"of depth {shape[2]}, inputs always valid and the output always ready: C§17's Cycles "
        f"{rule} at J {j}, T {t}, W {w}"
    )


def vectors(design_point: str | Path, root: str | Path | None = None) -> dict:
    """``vectors.json``: the data and tasks the Chisel specs drive the wrapper with; see the
    module doc."""
    shown, point, acc = load_point(design_point, ROOT if root is None else Path(root))
    tasks = task_list(point).accel_steps()
    if len(tasks) != 1:
        raise GenError("gen.instances", f"{len(tasks)} accelerator tasks; vectors.json has one")
    (task,) = tasks
    found = units(point)[1]
    (group,) = acc.groups
    linked = len(group.units) > 1
    data = {a.name: a for a in arrays(point)}
    ports = []
    for end in [*acc.inputs, *acc.outputs]:
        name, port = end.split(".", 1)
        inst, node = found[name], instance_nodes(point.graph)[name]
        values = task.values[name] if linked else task.values
        (p,) = (x for x in inst.brm.interface.ports if x.name == port)
        c = (node.inputs if p.direction == "in" else node.outputs)[p.name].data
        array = c if p.direction == "in" else f"{c}_golden"
        rate = 1 if p.rate == 1 else values[p.rate]
        ports.append({"port": p.name, "array": array, "lanes": inst.lanes(p.name),
                      "beats": values["n"] // rate, "values": list(data[array].values)})  # fmt: skip
    if linked:
        csrs = {k: v for _, k, v in job_writes(found, task.values, acc.fixed_folds)}
        n = task.values[group.units[0].name]["n"]
        value, working = group_counter(group, csrs)
    else:
        csrs = job_values(found[task.component], task.values)
        n = task.values["n"]
        value, working = counter(group.units[0], n)
    return {
        "design_point": shown,
        "ports": ports,
        "tasks": [{"name": task.task_name, "csrs": csrs, "firings": n, "counter": value,
                   "working": working}],
    }  # fmt: skip


def write_fixture(design_point: str | Path, out_dir: str | Path,
                  root: str | Path | None = None) -> list[str]:  # fmt: skip
    """A fixture of hw/chisel's specs: ``accelerator.json``, the streamer hjson and, for one
    accelerator task, ``vectors.json`` in ``out_dir``; the files written."""
    out = Path(out_dir)
    acc = write_accelerator(design_point, out, root)
    streamer = f"{acc.tag}_streamer.hjson"
    with tempfile.TemporaryDirectory() as gen:
        generate(design_point, gen, root)
        (out / streamer).write_text((Path(gen) / streamer).read_text())
    files = [ACCELERATOR_JSON, streamer]
    _, point, acc = load_point(design_point, ROOT if root is None else Path(root))
    if len(task_list(point).accel_steps()) == 1:
        (out / VECTORS_JSON).write_text(to_json(vectors(design_point, root)))
        files.append(VECTORS_JSON)
    return files


# =============================================================================
# The command line
# =============================================================================

CHISEL = ROOT / "hw" / "chisel"
EMITTED_JSON = "emitted.json"


def elaborate(spec_json: Path, out_dir: Path) -> list[Path]:
    """``Emit --spec`` on ``spec_json`` (an ``accelerator.json``) into ``out_dir`` (``sv/`` and
    ``emitted.json``) through sbt in ``hw/chisel/``; the files ``emitted.json`` lists, each
    checked to exist (firtool's failures do not always reach sbt's exit status)."""
    cmd = ["sbt", "-batch",
           f"runMain snax.forge.Emit --spec {spec_json.resolve()} --out {out_dir.resolve()}"]  # fmt: skip
    done = subprocess.run(cmd, cwd=CHISEL, check=False)
    listing = out_dir / EMITTED_JSON
    if done.returncode != 0 or not listing.is_file():
        raise GenError("gen.elaborate", f"Emit --spec {spec_json} failed (exit {done.returncode})")
    files = [out_dir / f for f in json.loads(listing.read_text())["files"]]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise GenError("gen.elaborate", f"{listing} lists files that were not written: {missing}")
    return [*files, listing]


def main(argv: list[str] | None = None) -> int:
    """``python -m snax_forge.gen.rtl DESIGN_POINT [OUT_DIR] [--fixture DIR]`` (pixi
    ``gen-rtl``): ``accelerator.json`` into ``OUT_DIR`` (default ``out/rtl/<design>/``), then the
    accelerator through ``Emit --spec`` into its ``sv/`` with ``emitted.json``; with
    ``--fixture``, a fixture of hw/chisel's specs into ``DIR`` instead (``write_fixture``, no
    elaboration). The flow does not call it (D136). Exit 0 when written, 1 when refused."""
    ap = argparse.ArgumentParser(prog="python -m snax_forge.gen.rtl")
    ap.add_argument("design_point", type=Path, help="a design_point.json (the design step's)")
    ap.add_argument("out_dir", type=Path, nargs="?", help="output directory (out/rtl/<design>)")
    ap.add_argument("--fixture", type=Path, help="write a fixture of the Chisel specs here")
    args = ap.parse_args(argv)

    def write() -> int:
        if args.fixture is not None:
            for f in write_fixture(args.design_point, args.fixture):
                print(f"wrote {cwd_relative(args.fixture / f)}")
            return 0
        out = out_dir(args.out_dir, OUT, args.design_point)
        acc = write_accelerator(args.design_point, out)
        datapaths = ", ".join(u.datapath for g in acc.groups for u in g.units)
        print(f"wrote {cwd_relative(out / ACCELERATOR_JSON)}: {acc.tag}_shell_wrapper, {datapaths}")
        for f in elaborate(out / ACCELERATOR_JSON, out):
            print(f"wrote {cwd_relative(f)}")
        return 0

    return cli(write)


if __name__ == "__main__":
    sys.exit(main())
