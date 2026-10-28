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
``gen.timing`` and the datapath's adapter's), then per unit

    gen.catalogue    the datapath is not in the catalogue, or the unit's timing.latency is not
                     the catalogue's latency
    gen.port_map     the unit's datapath ports are not the catalogue's ports, each once, or a
                     direction differs (inputs and outputs give the unit's)
    gen.params       a param the fill rule gives that is not of its type or not among its values
                     (a dtype, an op, lanes below min)

and, from the command line, ``gen.elaborate``: ``Emit --spec`` failed or did not write a file
its ``emitted.json`` lists.

**The fill rule** (the catalogue's ``fill``): ``inputs.dtype`` the input ports' dtypes in order
(a ``dtypes`` param) or the one input's (a ``dtype``), ``outputs.dtype`` the output's,
``inputs.lanes`` the inputs' lanes, ``unit.opType`` and ``unit.params.op`` the unit's.

**The fixtures** (``write_fixture``): ``hw/chisel/src/test/resources/rtl/<design>/`` holds
``accelerator.json``, the design's streamer hjson (gvsoc.py's) and ``vectors.json``, which the
Chisel specs drive the wrapper with: per BRM port its data.h array (an input, or the output's
golden) and how many beats a task moves (lane i of beat t is element t * lanes + i), and per
task its CSR values (JOBS, JOB_BEATS: lower/c.py's ``job_values``), its firings and the counter
worked from the nest controller's rule with an always-ready streamer (``counter``), the working
in ``working``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from snax_forge.common import cwd_relative, repo_root, to_json
from snax_forge.design.streamers import instance_nodes
from snax_forge.lower.c import job_values
from snax_forge.lower.derive import task_list
from snax_forge.lower.tasks import Configure

from .accelerator import ACCELERATOR_JSON, Accelerator, Unit
from .blocks import GenError
from .data import arrays
from .gvsoc import ROOT, cli, generate, instance, load_point, out_dir

OUT = ROOT / "out" / "rtl"
CATALOGUE = repo_root() / "hw" / "chisel" / "blocks.json"
VECTORS_JSON = "vectors.json"


# =============================================================================
# The checks against the catalogue
# =============================================================================


def catalogue() -> dict[str, Any]:
    """The datapath catalogue, ``hw/chisel/blocks.json``."""
    return json.loads(CATALOGUE.read_text())


def fill(acc: Accelerator, unit: Unit, sig: Mapping[str, Any]) -> dict[str, Any]:
    """The datapath's params of ``unit`` by the catalogue's fill rule (module doc)."""
    ins = [p.split(".", 1)[1] for p in acc.inputs if p.split(".")[0] == unit.name]
    outs = [p.split(".", 1)[1] for p in acc.outputs if p.split(".")[0] == unit.name]
    out = {}
    for name, rule in sig["fill"].items():
        kind = sig["params"][name]["type"]
        if rule == "inputs.dtype":
            dtypes = [unit.ports[p].dtype for p in ins]
            out[name] = dtypes if kind == "dtypes" else dtypes[0]
        elif rule == "outputs.dtype":
            out[name] = unit.ports[outs[0]].dtype
        elif rule == "inputs.lanes":
            out[name] = unit.ports[ins[0]].lanes
        elif rule == "unit.opType":
            out[name] = unit.op_type
        elif rule == "unit.params.op":
            out[name] = unit.params.get("op")
        else:
            raise GenError(
                "gen.catalogue", f"{unit.datapath}.{name}: the fill rule {rule!r} is unknown"
            )
    return out


def check_params(datapath: str, params: Mapping[str, Any], sig: Mapping[str, Any]) -> None:
    """``params`` (``fill``'s) against the datapath's catalogue params ``sig``."""
    for k, t in sig.items():
        v, what = params[k], f"{datapath}.{k}"
        if t["type"] == "int":
            if type(v) is not int or v < t["min"]:
                raise GenError("gen.params", f"{what}: {v!r} is not an int of at least {t['min']}")
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
    if unit.latency != sig["latency"]:
        raise GenError("gen.catalogue", f"unit {unit.name}: timing.latency {unit.latency}, "
                       f"{unit.datapath}'s is {sig['latency']}")  # fmt: skip
    dirs = {**{p.split(".", 1)[1]: "in" for p in acc.inputs if p.split(".")[0] == unit.name},
            **{p.split(".", 1)[1]: "out" for p in acc.outputs if p.split(".")[0] == unit.name}}  # fmt: skip
    used = [p.port for p in unit.ports.values()]
    if sorted(used) != sorted(sig["ports"]):
        raise GenError("gen.port_map", f"unit {unit.name}: ports onto {used}; {unit.datapath}'s "
                       f"are {list(sig['ports'])}, each once")  # fmt: skip
    for brm, p in unit.ports.items():
        if dirs.get(brm) != sig["ports"][p.port]:
            raise GenError("gen.port_map", f"unit {unit.name}: BRM port {brm} ({dirs.get(brm)}) onto "
                           f"{p.port} ({sig['ports'][p.port]})")  # fmt: skip
    check_params(unit.datapath, fill(acc, unit, sig), sig["params"])


def accelerator_spec(design_point: str | Path, root: str | Path | None = None,
                     cat: Mapping[str, Any] | None = None) -> Accelerator:  # fmt: skip
    """The design point's ``accelerator.json`` (accelerator.py), checked against the catalogue."""
    _, _, acc = load_point(design_point, ROOT if root is None else Path(root))
    cat = catalogue() if cat is None else cat
    for g in acc.groups:
        for u in g.units:
            check_unit(acc, u, cat)
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


def vectors(design_point: str | Path, root: str | Path | None = None) -> dict:
    """``vectors.json``: the data and tasks the Chisel specs drive the wrapper with; see the
    module doc."""
    shown, point, acc = load_point(design_point, ROOT if root is None else Path(root))
    name, inst = instance(point)
    unit = acc.groups[0].units[0]
    node = instance_nodes(point.graph)[name]
    data = {a.name: a for a in arrays(point)}
    tasks = [s for s in task_list(point).steps
             if isinstance(s, Configure) and s.type == "accel" and s.component == name]  # fmt: skip
    if len(tasks) != 1:
        raise GenError("gen.instances", f"{len(tasks)} accelerator tasks; a fixture has one")
    (task,) = tasks
    ports = []
    for p in inst.brm.interface.ports:
        c = (node.inputs if p.direction == "in" else node.outputs)[p.name].data
        array = c if p.direction == "in" else f"{c}_golden"
        rate = 1 if p.rate == 1 else task.values[p.rate]
        ports.append({"port": p.name, "array": array, "lanes": inst.lanes(p.name),
                      "beats": task.values["n"] // rate, "values": list(data[array].values)})  # fmt: skip
    n = task.values["n"]
    value, working = counter(unit, n)
    return {
        "design_point": shown,
        "ports": ports,
        "tasks": [{"name": task.task_name, "csrs": job_values(inst, task.values), "firings": n,
                   "counter": value, "working": working}],
    }  # fmt: skip


def write_fixture(design_point: str | Path, out_dir: str | Path,
                  root: str | Path | None = None) -> list[str]:  # fmt: skip
    """A fixture of hw/chisel's specs: ``accelerator.json``, the streamer hjson and
    ``vectors.json`` in ``out_dir``; the files written."""
    out = Path(out_dir)
    acc = write_accelerator(design_point, out, root)
    streamer = f"{acc.tag}_streamer.hjson"
    with tempfile.TemporaryDirectory() as gen:
        generate(design_point, gen, root)
        (out / streamer).write_text((Path(gen) / streamer).read_text())
    (out / VECTORS_JSON).write_text(to_json(vectors(design_point, root)))
    return [ACCELERATOR_JSON, streamer, VECTORS_JSON]


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
