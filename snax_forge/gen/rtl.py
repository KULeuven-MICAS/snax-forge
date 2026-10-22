"""A design point -> its accelerator's RTL spec, ``rtl.json`` (GEN1, D138, C§17).

    write_rtl(design_point, out_dir, root=None) -> RtlSpec
    vectors(design_point, root=None) -> dict
    write_fixture(design_point, out_dir, root=None) -> list[str]
    python -m snax_forge.gen.rtl DESIGN_POINT [OUT_DIR] [--fixture DIR]    (pixi gen-rtl)

``rtl.json`` is the boundary between the Python side and ``hw/chisel/``:
concrete values only, which ``Emit --spec`` reads to build
``<tag>_shell_wrapper``. Python resolves and checks; Scala only constructs.

**What it holds** (C§17): ``design_point`` (its path from the root, as
gvsoc.py names it), ``tag`` (the platform's ``accelerator_slot``) and
``units``, one per accelerated instance (more than one is ``gen.instances``,
lower/c.py's ``accelerator``), each with

    name     the instance
    block    the binding's block
    params   the block's params, from the block's GVSoC unit (blocks.py:
             the same lanes, widths and op) by its resolver (``RTL_PARAMS``):
               ElementwiseTiledSpatial  dataWidth N, lanes, loopCountWidth 32
                                        (a CSR), supportedOps [op], signed
               Accumulator              width N, accWidth N, lanes
    ports    ``{brm, block}`` per BRM port in the BRM's port order (the
             streamer's reader and writer order): the binding's ``ports``,
             by default the same name
    csrs     ``rw``: the catalogue's read-write pins with their widths, then
             ``start``, then ``ro``: busy and the counter (D130); ``tie``: the
             catalogue's tied pins with their values

**The checks**, against the block catalogue ``hw/chisel/blocks.json`` (the
signature of each Chisel block, equal to ``Emit --describe``), each a
``GenError``: first the GVSoC adapter's rules (blocks.py, ``gen.instances``,
``gen.binding``, ``gen.block`` and the adapter's own), then

    gen.catalogue    the block is not in the catalogue, or has no resolver
    gen.params       a param missing, or one the block does not have
    gen.param_type   a param not of its type (int, bool, ops)
    gen.param_limit  an int outside min and max, ops empty, repeated or not
                     among the catalogue's
    gen.port_map     a BRM port onto no block port, a block port left open,
                     two BRM ports on one, or a direction that differs
    gen.csr_pins     the read-write pins are not the binding's csrs fields, or
                     a width is not 1 to 32 bits
    gen.tie          a tie's ops param holds more than one op

and, from the command line, ``gen.elaborate``: ``Emit --spec`` failed or
did not write a file its ``emitted.json`` lists.

**The fixtures** (``write_fixture``): ``hw/chisel/src/test/resources/rtl/
<design>/`` holds ``rtl.json``, the design's streamer hjson (gvsoc.py's) and
``vectors.json``, which the Chisel specs drive the wrapper with: per BRM port
its data.h array (an input, or the output's golden) and how many beats a
task moves (lane i of beat t is element t * lanes + i), and per task the
shell's field values and the counter worked from SnaxShell's rules
(snax-gvsoc ``snax/snax_shell.cpp``) with an always-ready streamer
(``counter``), the working in ``working``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from snax_forge import expr
from snax_forge.brm import Instance
from snax_forge.common import cwd_relative, repo_root, to_json
from snax_forge.design import DesignError
from snax_forge.design.point import DesignPoint
from snax_forge.design.streamers import instance_nodes
from snax_forge.lower import CError
from snax_forge.lower.derive import task_list
from snax_forge.lower.tasks import Configure

from .blocks import ELEMENTWISE_OP, GenError, Unit
from .data import arrays
from .gvsoc import ROOT, accelerator, generate, point_name

OUT = ROOT / "out" / "rtl"
CATALOGUE = repo_root() / "hw" / "chisel" / "blocks.json"
RTL_JSON = "rtl.json"
VECTORS_JSON = "vectors.json"
# A CSR is 32 bits: the widest read-write pin, and an elementwise trip counter's width
CSR_BITS = 32
# The read-only CSRs after start (D130)
RO = ["busy", "counter"]


@dataclass(frozen=True)
class RtlUnit:
    """One unit of ``rtl.json``; see the module doc."""

    name: str
    block: str
    params: dict[str, Any]
    ports: list[tuple[str, str]]
    rw: list[tuple[str, int]]
    start: str
    tie: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "block": self.block,
            "params": dict(self.params),
            "ports": [{"brm": b, "block": k} for b, k in self.ports],
            "csrs": {
                "rw": [{"pin": p, "width": w} for p, w in self.rw],
                "start": self.start,
                "ro": list(RO),
                "tie": dict(self.tie),
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RtlUnit:
        c = d["csrs"]
        return cls(d["name"], d["block"], dict(d["params"]),
                   [(p["brm"], p["block"]) for p in d["ports"]],
                   [(r["pin"], r["width"]) for r in c["rw"]], c["start"], dict(c["tie"]))  # fmt: skip


@dataclass(frozen=True)
class RtlSpec:
    """``rtl.json``; see the module doc."""

    design_point: str
    tag: str
    units: list[RtlUnit]

    def to_dict(self) -> dict[str, Any]:
        return {"design_point": self.design_point, "tag": self.tag,
                "units": [u.to_dict() for u in self.units]}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RtlSpec:
        return cls(d["design_point"], d["tag"], [RtlUnit.from_dict(u) for u in d["units"]])

    def to_json(self) -> str:
        return to_json(self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> RtlSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))


# =============================================================================
# The params of each block
# =============================================================================

Resolver = Callable[[Instance, Unit], dict[str, Any]]
RTL_PARAMS: dict[str, Resolver] = {}
OP_NAME = {v: k for k, v in ELEMENTWISE_OP.items()}


def register_rtl_params(block: str, resolver: Resolver) -> None:
    """Add the resolver of a block's params (a binding's ``block``) from its GVSoC unit."""
    if block in RTL_PARAMS:
        raise ValueError(f"block {block!r} already has an RTL resolver")
    RTL_PARAMS[block] = resolver


def _elementwise_tiled_spatial(inst: Instance, u: Unit) -> dict[str, Any]:
    return {"dataWidth": u.data_width, "lanes": u.lanes, "loopCountWidth": CSR_BITS,
            "supportedOps": [OP_NAME[u.op]], "signed": u.signed}  # fmt: skip


def _accumulator(inst: Instance, u: Unit) -> dict[str, Any]:
    return {"width": u.data_width, "accWidth": u.data_width, "lanes": u.lanes}


register_rtl_params("ElementwiseTiledSpatial", _elementwise_tiled_spatial)
register_rtl_params("Accumulator", _accumulator)


# =============================================================================
# The checks against the catalogue
# =============================================================================


def catalogue(path: str | Path = CATALOGUE) -> dict[str, Any]:
    """The block catalogue, ``hw/chisel/blocks.json``."""
    return json.loads(Path(path).read_text())


def check_params(block: str, params: Mapping[str, Any], sig: Mapping[str, Any]) -> None:
    """``params`` against the block's catalogue params ``sig``; see the module doc."""
    missing = [k for k in sig if k not in params]
    unknown = [k for k in params if k not in sig]
    if missing or unknown:
        raise GenError(
            "gen.params", f"{block}: params missing {missing}, unknown {unknown} (the catalogue "
            f"has {list(sig)})",
        )  # fmt: skip
    for k, t in sig.items():
        v, what = params[k], f"{block}.{k}"
        if t["type"] == "int":
            if type(v) is not int:
                raise GenError("gen.param_type", f"{what}: {v!r} is not an int")
            if v < t["min"] or ("max" in t and v > t["max"]):
                raise GenError(
                    "gen.param_limit", f"{what}: {v} outside [{t['min']}, {t.get('max', '')}]"
                )
        elif t["type"] == "bool":
            if type(v) is not bool:
                raise GenError("gen.param_type", f"{what}: {v!r} is not a bool")
        elif t["type"] == "ops":
            if not isinstance(v, list) or not all(isinstance(o, str) for o in v):
                raise GenError("gen.param_type", f"{what}: {v!r} is not a list of op names")
            if not v or len(set(v)) != len(v) or not set(v) <= set(t["values"]):
                raise GenError(
                    "gen.param_limit", f"{what}: {v} must be distinct ops of {list(t['values'])}"
                )
        else:
            raise GenError(
                "gen.param_type", f"{what}: the catalogue's type {t['type']!r} is unknown"
            )


def check_ports(block: str, ports: list[tuple[str, str]], directions: Mapping[str, str],
                sig: Mapping[str, str]) -> None:  # fmt: skip
    """The BRM -> block port map against the catalogue's ports ``sig``; ``directions`` the BRM
    ports' (``in``, ``out``)."""
    used = [k for _, k in ports]
    for b, k in ports:
        if k not in sig:
            raise GenError(
                "gen.port_map", f"{block}: BRM port {b} onto {k!r}, not one of {list(sig)}"
            )
        if sig[k] != directions[b]:
            raise GenError(
                "gen.port_map", f"{block}: BRM port {b} ({directions[b]}) onto {k} ({sig[k]})"
            )
    if len(set(used)) != len(used) or set(used) != set(sig):
        raise GenError(
            "gen.port_map", f"{block}: BRM ports onto {used}; every block port {list(sig)} once"
        )


def resolve_csrs(
    block: str, params: Mapping[str, Any], fields: list[str], csr: Mapping[str, Any],
    ops: Mapping[str, Mapping[str, int]],
) -> tuple[list[tuple[str, int]], dict[str, int]]:  # fmt: skip
    """The read-write pins with their widths and the ties with their values, from the
    catalogue's ``csr`` section, checked against the binding's ``fields``; ``ops`` the ops
    params' encodings."""
    pins = [r["pin"] for r in csr["rw"]]
    if pins != fields:
        raise GenError(
            "gen.csr_pins", f"{block}: the read-write pins are {pins}, the binding's csrs {fields}"
        )
    rw = []
    for r in csr["rw"]:
        w = r["width"] if isinstance(r["width"], int) else params[r["width"]]
        if not 1 <= w <= CSR_BITS:
            raise GenError("gen.csr_pins", f"{block}.{r['pin']}: width {w} is not 1 to {CSR_BITS}")
        rw.append((r["pin"], w))
    tie = {}
    for t in csr["tie"]:
        held = params[t["op"]]
        if len(held) != 1:
            raise GenError(
                "gen.tie", f"{block}.{t['pin']}: tied to {t['op']}'s op, which holds {held}; one "
                "op is a fixed-function unit",
            )  # fmt: skip
        tie[t["pin"]] = ops[t["op"]][held[0]]
    return rw, tie


def rtl_unit(name: str, inst: Instance, unit: Unit, cat: Mapping[str, Any]) -> RtlUnit:
    """One unit, resolved and checked; see the module doc."""
    block = inst.impl.binding["block"]
    if block not in cat or block not in RTL_PARAMS:
        raise GenError(
            "gen.catalogue", f"{inst.brm.name}.{inst.implementation}: block {block!r} is not in "
            f"the catalogue {sorted(cat)} with an RTL resolver {sorted(RTL_PARAMS)}",
        )  # fmt: skip
    sig = cat[block]
    params = RTL_PARAMS[block](inst, unit)
    check_params(block, params, sig["params"])
    brm_ports = inst.brm.interface.ports
    mapping = inst.impl.binding.get("ports") or {p.name: p.name for p in brm_ports}
    ports = [(p.name, mapping[p.name]) for p in brm_ports]
    check_ports(block, ports, {p.name: p.direction for p in brm_ports}, sig["ports"])
    ops = {k: t["values"] for k, t in sig["params"].items() if t["type"] == "ops"}
    fields = [c["field"] for c in inst.impl.binding["csrs"]]
    rw, tie = resolve_csrs(block, params, fields, sig["csr"], ops)
    return RtlUnit(name, block, params, ports, rw, sig["csr"]["start"], tie)


# =============================================================================
# rtl.json
# =============================================================================


def rtl_spec(design_point: str | Path, root: str | Path | None = None,
             cat: Mapping[str, Any] | None = None) -> RtlSpec:  # fmt: skip
    """The design point's ``rtl.json``, checked; see the module doc."""
    root = ROOT if root is None else Path(root)
    shown = point_name(Path(design_point), root)
    point = DesignPoint.load(design_point)
    name, inst, unit = accelerator(point)
    unit_ = rtl_unit(name, inst, unit, catalogue() if cat is None else cat)
    return RtlSpec(shown, point.platform.accelerator_slot, [unit_])


def write_rtl(design_point: str | Path, out_dir: str | Path,
              root: str | Path | None = None) -> RtlSpec:  # fmt: skip
    """Write ``rtl.json`` to ``out_dir``; see the module doc."""
    spec = rtl_spec(design_point, root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / RTL_JSON).write_text(spec.to_json())
    return spec


# =============================================================================
# The fixtures: vectors.json
# =============================================================================


def counter(unit: Unit, n: int) -> tuple[int, str]:
    """The counter SnaxShell gives a task of ``n`` firings with an always-ready streamer, and
    the working."""
    shell = ("SnaxShell: the counter is reset by the accepted start and counts each busy cycle; "
             "busy rises the cycle after the start")  # fmt: skip
    if unit.wait == "lockstep":
        return n, (
            f"{shell}. Lockstep, inputs always valid and the output always ready: a firing "
            f"every busy cycle, the {n}th the run's last: {n} busy cycles"
        )
    last = (n - 1) * unit.ii + 1
    busy = last + unit.latency
    return busy, (
        f"{shell}. Pipelined, L {unit.latency}, II {unit.ii}, drain 0, inputs always valid and "
        f"the output always ready: firing k (from 0) in busy cycle 1 + {unit.ii} * k, the last ({n - 1}) in "
        f"{last}, its result pushed {unit.latency} later, the run's last cycle (a sum before it "
        f"leaves while the next input is taken): {busy} busy cycles"
    )


def vectors(design_point: str | Path, root: str | Path | None = None, seed: int = 0) -> dict:
    """``vectors.json``: the data and tasks the Chisel specs drive the wrapper with; see the
    module doc."""
    root = ROOT if root is None else Path(root)
    shown = point_name(Path(design_point), root)
    point = DesignPoint.load(design_point)
    name, inst, unit = accelerator(point)
    node = instance_nodes(point.graph)[name]
    data = {a.name: a for a in arrays(point, seed)}
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
    env = {**inst.params, **task.values}
    csrs = {c["field"]: expr.evaluate(c["value"], env) for c in inst.impl.binding["csrs"]}
    n = task.values["n"]
    value, working = counter(unit, n)
    return {
        "design_point": shown,
        "ports": ports,
        "tasks": [{"name": task.task_name, "csrs": csrs, "firings": n, "counter": value,
                   "working": working}],
    }  # fmt: skip


def write_fixture(design_point: str | Path, out_dir: str | Path,
                  root: str | Path | None = None) -> list[str]:  # fmt: skip
    """A fixture of hw/chisel's specs: ``rtl.json``, the streamer hjson and ``vectors.json``
    in ``out_dir``; the files written."""
    out = Path(out_dir)
    spec = write_rtl(design_point, out, root)
    streamer = f"{spec.tag}_streamer.hjson"
    with tempfile.TemporaryDirectory() as gen:
        generate(design_point, gen, root)
        (out / streamer).write_text((Path(gen) / streamer).read_text())
    (out / VECTORS_JSON).write_text(to_json(vectors(design_point, root)))
    return [RTL_JSON, streamer, VECTORS_JSON]


# =============================================================================
# The command line
# =============================================================================

CHISEL = ROOT / "hw" / "chisel"
EMITTED_JSON = "emitted.json"


def elaborate(rtl_json: Path, out_dir: Path) -> list[Path]:
    """``Emit --spec`` on ``rtl_json`` into ``out_dir`` (``sv/`` and ``emitted.json``) through
    sbt in ``hw/chisel/``; the files ``emitted.json`` lists, each checked to exist (firtool's
    failures do not always reach sbt's exit status)."""
    cmd = ["sbt", "-batch",
           f"runMain snax.forge.Emit --spec {rtl_json.resolve()} --out {out_dir.resolve()}"]  # fmt: skip
    done = subprocess.run(cmd, cwd=CHISEL, check=False)
    listing = out_dir / EMITTED_JSON
    if done.returncode != 0 or not listing.is_file():
        raise GenError("gen.elaborate", f"Emit --spec {rtl_json} failed (exit {done.returncode})")
    files = [out_dir / f for f in json.loads(listing.read_text())["files"]]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise GenError("gen.elaborate", f"{listing} lists files that were not written: {missing}")
    return [*files, listing]


def main(argv: list[str] | None = None) -> int:
    """``python -m snax_forge.gen.rtl DESIGN_POINT [OUT_DIR] [--fixture DIR]`` (pixi
    ``gen-rtl``): ``rtl.json`` into ``OUT_DIR`` (default ``out/rtl/<design>/``), then the
    accelerator through ``Emit --spec`` into its ``sv/`` with ``emitted.json``; with
    ``--fixture``, a fixture of hw/chisel's specs into ``DIR`` instead (``write_fixture``, no
    elaboration). The flow does not call it (D136). Exit 0 when written, 1 when refused."""
    ap = argparse.ArgumentParser(prog="python -m snax_forge.gen.rtl")
    ap.add_argument("design_point", type=Path, help="a design_point.json (the design step's)")
    ap.add_argument("out_dir", type=Path, nargs="?", help="output directory (out/rtl/<design>)")
    ap.add_argument("--fixture", type=Path, help="write a fixture of the Chisel specs here")
    args = ap.parse_args(argv)
    try:
        if args.fixture is not None:
            for f in write_fixture(args.design_point, args.fixture):
                print(f"wrote {cwd_relative(args.fixture / f)}")
            return 0
        out = args.out_dir or OUT / DesignPoint.load(args.design_point).name
        spec = write_rtl(args.design_point, out)
        print(
            f"wrote {cwd_relative(out / RTL_JSON)}: {spec.tag}_shell_wrapper, {spec.units[0].block}"
        )
        for f in elaborate(out / RTL_JSON, out):
            print(f"wrote {cwd_relative(f)}")
    except (GenError, CError, DesignError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
