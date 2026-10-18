"""A design point -> a generated accelerator on SNAX-GVSoC (GEN3 step 3, D128, D132).

    generate(design_point, out_dir, install=None) -> Manifest

The design point's one accelerated instance becomes a composite on
snax-gvsoc's hand-written components, as ``snax/snax_tiled_add_acc.py`` is
written by hand: a CSR mux (``interco.router.Router``), a ``SnaxStreamer``,
a ``SnaxShell`` and a ``SnaxUnit``, every number a literal, so a design
point compiles nothing (D128). ``out_dir`` gets:

    forge_<design>_acc.py   the composite (class Forge<Design>Acc)
    forge_<design>.py       the target forge_<design>: snax_alu's board
                            (SnaxAluBoard) with the composite in its
                            accelerator slot
    manifest.json           what was generated (Manifest)

**Only installed components.** The generated files import nothing but
``IMPORTS`` (and each other), never snax_forge: gvrun runs them on the
install's components. Every generated module starts with ``forge_``. With
an ``install`` (snax-gvsoc's install folder), the generator also refuses a
generated module whose name the install's ``targets`` or ``generators``
hold, since those come first on gvrun's path and would shadow it (snax-gvsoc
``GEN3.md``), and checks that every module the files import is there.

**What goes in.** The block's adapter (blocks.py) gives the unit and the
shell's fields. The streamer template comes from the design point's
streamers (D87): readers the input ports in the BRM's port order, writers
the output ports, each with ``spatial_bounds``, ``temporal_dim`` (the
streamer's ``temporal_dims``), ``num_channel`` (``n_ports * N / 64`` for the
port's N-bit dtype) and ``fifo_depth``; ``addr_depth`` and ``prio`` are
SNAX-MODEL's and not in it. The movers, their CSR count and the TCDM port
names follow ``snax/snax_streamer.py`` (``movers_of``, ``nb_csrs``,
``tcdm_port_names``). The TCDM size is the platform's L1. The CSRs: the
streamer's from ``SNAX_CSR_FIRST``, then the shell's fields, start, busy
and the counter (D130).

**Refused**, each a ``GenError`` (``gen.<rule>``): a design name that is
not an identifier (``gen.name``), more or fewer than one accelerated
instance (``gen.instances``: snax_alu's board has one accelerator slot), a
null binding (``gen.binding``), an unregistered block (``gen.block``), what
the block's adapter refuses, a port whose lanes are not whole 64-bit
channels (``gen.dtype``), CSRs past ``SNAX_CSR_LAST`` (``gen.csrs``) and
the install check (``gen.install``).

**The manifest** (paths relative to its folder): ``design``,
``design_point``, ``target``, ``files``, ``target_dirs`` (for gvrun's
``--target-dir``, after the install's own), ``trace`` (the unit's trace
path), ``csrs`` (the layout) and ``tasks``: per accelerator task of the
derived task list (snax_forge/lower/derive.py, D89) its name, its ``beats``
(the unit's trip count, the task's ``n``) and its ``csrs`` values.

Not verified: the order of ``spatial_bounds`` against snax_cluster's
template for more than one spatial dimension (every template so far has one).
"""

from __future__ import annotations

import json
import keyword
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from snax_forge import expr
from snax_forge.brm import Instance
from snax_forge.design.point import DesignPoint
from snax_forge.design.streamers import instances, streamer_name
from snax_forge.lower.derive import task_list
from snax_forge.lower.tasks import Configure
from snax_forge.sdfg.paths import _repo_root
from snax_forge.snax_model.config import to_json

from .blocks import BLOCKS, GenError, Unit, port_dtype

ROOT = _repo_root()
OUT = ROOT / "out" / "gvsoc"
INSTALL = ROOT / "out" / "snax-gvsoc" / "install"

# snax-gvsoc snax/snax_alu_acc.py: the first SNAX CSR (CSR_SNAX_BEGIN) and the
# last one the core sends out (SNAX_CSR_FIRST, SNAX_CSR_LAST)
SNAX_CSR_FIRST = 0x3C0
SNAX_CSR_LAST = 0x5FF
# The unit's trace path: snax_alu.py's SNAX_ACC_CORE 0, named snax_acc<core> by
# pulp/snitch/snitch_cluster/snitch_cluster.py; the composite calls it unit
UNIT_TRACE = "snax_acc0/unit"
PREFIX = "forge_"
# The modules a generated file may import (besides the other generated ones)
IMPORTS = ("gvsoc.systree", "gvsoc.runner", "interco.router", "snax_streamer", "snax_shell",
           "snax_unit", "snax_alu")  # fmt: skip
# Where gvrun finds them in an install: its targets and generators lead its
# path (and shadow a generated module), its python holds the gvsoc package
SHADOWING = ("targets", "generators")
SEARCHED = ("targets", "generators", "python")


@dataclass(frozen=True)
class CsrLayout:
    """The accelerator's CSRs: the streamer's range, then the shell's (D130)."""

    streamer_first: int
    streamer_last: int
    fields: dict[str, int]
    start: int
    busy: int
    counter: int

    @property
    def shell_first(self) -> int:
        return self.streamer_last + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "streamer": {"first": self.streamer_first, "last": self.streamer_last},
            "fields": dict(self.fields),
            "start": self.start,
            "busy": self.busy,
            "counter": self.counter,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CsrLayout:
        return cls(d["streamer"]["first"], d["streamer"]["last"], dict(d["fields"]), d["start"],
                   d["busy"], d["counter"])  # fmt: skip


@dataclass(frozen=True)
class AccTask:
    """One accelerator task: its name, the unit's trip count and the shell's field values."""

    name: str
    beats: int
    csrs: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "beats": self.beats, "csrs": dict(self.csrs)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AccTask:
        return cls(d["name"], d["beats"], dict(d["csrs"]))


@dataclass(frozen=True)
class Manifest:
    """What ``generate`` wrote; see the module doc."""

    design: str
    design_point: str
    target: str
    files: list[str]
    target_dirs: list[str]
    trace: str
    csrs: CsrLayout
    tasks: list[AccTask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "design_point": self.design_point,
            "target": self.target,
            "files": list(self.files),
            "target_dirs": list(self.target_dirs),
            "trace": self.trace,
            "csrs": self.csrs.to_dict(),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        return cls(
            d["design"], d["design_point"], d["target"], list(d["files"]), list(d["target_dirs"]),
            d["trace"], CsrLayout.from_dict(d["csrs"]), [AccTask.from_dict(t) for t in d["tasks"]],
        )  # fmt: skip

    def to_json(self) -> str:
        return to_json(self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        return cls.from_dict(json.loads(Path(path).read_text()))


# =============================================================================
# What the design point gives
# =============================================================================


def accelerator(point: DesignPoint) -> tuple[str, Instance, Unit]:
    """The one accelerated instance, its name and its unit (the block's adapter)."""
    found = instances(point.graph)
    if len(found) != 1:
        raise GenError(
            "gen.instances",
            f"{len(found)} accelerated instances {list(found)}; snax_alu's board has one "
            "accelerator slot",
        )
    (name, inst), = found.items()  # fmt: skip
    b = inst.impl.binding
    what = f"{inst.brm.name}.{inst.implementation}"
    if b is None:
        raise GenError("gen.binding", f"{what}: binding null, nothing to generate")
    if b["block"] not in BLOCKS:
        raise GenError(
            "gen.block", f"{what}: block {b['block']!r} not registered ({sorted(BLOCKS)})"
        )
    return name, inst, BLOCKS[b["block"]](inst)


def streamer_template(point: DesignPoint, name: str, inst: Instance) -> dict[str, Any]:
    """The streamer template, in the hjson's keys; see the module doc."""
    groups: dict[str, list[dict[str, Any]]] = {"data_reader_params": [], "data_writer_params": []}
    for p in inst.brm.interface.ports:
        s = point.streamers[streamer_name(name, p.name)]
        bits, _ = port_dtype(p.dtype, f"port {p.name}")
        if s.n_ports * bits % 64:
            raise GenError(
                "gen.dtype", f"streamer {s.name}: {s.n_ports} x {bits} bits is not whole 64-bit "
                "channels",
            )  # fmt: skip
        mover = {
            "spatial_bounds": list(s.spatial_bounds),
            "temporal_dim": s.options.temporal_dims,
            "num_channel": s.n_ports * bits // 64,
            "fifo_depth": s.options.fifo_depth,
        }
        groups["data_writer_params" if s.write else "data_reader_params"].append(mover)
    return {
        g: {k: [m[k] for m in movers] for k in movers[0]} for g, movers in groups.items() if movers
    }


def movers(template: dict[str, Any]) -> list[dict[str, Any]]:
    """The template's movers, readers first, named as snax_streamer.py's movers_of names them."""
    out = []
    for group, kind in (("data_reader_params", "reader"), ("data_writer_params", "writer")):
        params = template.get(group, {})
        for i in range(len(params.get("num_channel", []))):
            out.append({"name": f"{kind}{i}", **{k: v[i] for k, v in params.items()}})
    return out


def streamer_csrs(template: dict[str, Any]) -> int:
    """snax_streamer.nb_csrs: per mover the pointer (2), the spatial strides, the temporal
    bounds and strides; then start; then busy and the counter."""
    rw = sum(2 + len(m["spatial_bounds"]) + 2 * m["temporal_dim"] for m in movers(template))
    return rw + 1 + 2


def tcdm_port_names(template: dict[str, Any]) -> list[str]:
    """snax_streamer.tcdm_port_names, as the composite names them: streamer.<mover>.lane<l>."""
    return [f"streamer.{m['name']}.lane{lane}" for m in movers(template)
            for lane in range(m["num_channel"])]  # fmt: skip


def csr_layout(template: dict[str, Any], fields: list[str]) -> CsrLayout:
    first = SNAX_CSR_FIRST + streamer_csrs(template)
    nums = {f: first + i for i, f in enumerate(fields)}
    start = first + len(fields)
    layout = CsrLayout(SNAX_CSR_FIRST, first - 1, nums, start, start + 1, start + 2)
    if layout.counter > SNAX_CSR_LAST:
        raise GenError("gen.csrs", f"the shell's last CSR {layout.counter} is past {SNAX_CSR_LAST}")
    return layout


def acc_tasks(point: DesignPoint, name: str, inst: Instance) -> list[AccTask]:
    """The accelerator tasks of the derived task list (D89), with the binding's csrs values."""
    out = []
    for step in task_list(point).steps:
        if isinstance(step, Configure) and step.type == "accel" and step.component == name:
            env = {**inst.params, **step.values}
            csrs = {c["field"]: expr.evaluate(c["value"], env) for c in inst.impl.binding["csrs"]}
            out.append(AccTask(step.task_name, step.values["n"], csrs))
    return out


def check_install(install: Path, modules: list[str], imports: tuple[str, ...] = IMPORTS) -> None:
    """Refuse a generated module the install would shadow, or an import it lacks."""
    for m in modules:
        for d in SHADOWING:
            if (install / d / f"{m}.py").exists() or (install / d / m).is_dir():
                raise GenError(
                    "gen.install", f"{install / d} holds {m!r}, which comes first on gvrun's "
                    "path and would shadow the generated module",
                )  # fmt: skip
    for imp in imports:
        rel = Path(*imp.split("."))
        if not any(
            (install / d / rel).with_suffix(".py").exists()
            or (install / d / rel / "__init__.py").exists()
            for d in SEARCHED
        ):
            raise GenError(
                "gen.install", f"no {imp!r} in {install} ({', '.join(SEARCHED)}): build snax-gvsoc"
            )


# =============================================================================
# The files
# =============================================================================


def _camel(name: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in name.split("_") if p)


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _wrapped(items: list[Any], indent: int, width: int = 96) -> str:
    """``[a, b, ...]`` as a Python list literal, wrapped at ``width``."""
    lines, line = [], ""
    for r in (repr(x) for x in items):
        piece = r + ", "
        if line and indent + len(line) + len(piece) > width:
            lines.append(line.rstrip())
            line = ""
        line += piece
    lines.append(line.rstrip().rstrip(","))
    pad = "\n" + " " * indent
    return "[" + pad.join(lines) + "]" if len(lines) > 1 else "[" + lines[0] + "]"


def _wrapped_args(args: list[str], indent: int, width: int = 96) -> str:
    """``a=1, b=2, ...`` wrapped at ``width``, continuation lines indented by ``indent``."""
    lines, line = [], ""
    for a in args:
        piece = a + ", "
        if line and indent + len(line) + len(piece) > width:
            lines.append(line.rstrip())
            line = ""
        line += piece
    lines.append(line.rstrip().rstrip(","))
    return ("\n" + " " * indent).join(lines)


def _header(point_path: str, what: str) -> str:
    return (
        "#\n"
        f"# Generated by snax-forge's GVSoC generator (snax_forge/gen, GEN3) from the\n"
        f"# design point {point_path}. Do not edit: run\n"
        f"# `pixi run gen-gvsoc {point_path}` again.\n"
        "#\n\n"
        f'"""{what}"""\n'
    )


def composite(design: str, point_path: str, inst_name: str, inst: Instance, unit: Unit,
              template: dict[str, Any], tcdm_size: int, layout: CsrLayout) -> str:  # fmt: skip
    """The text of forge_<design>_acc.py."""
    cls = _camel(f"{PREFIX}{design}_acc")
    b = inst.impl.binding
    groups = []
    for g, params in template.items():
        body = "".join(f"        {k!r}: {v!r},\n" for k, v in params.items())
        groups.append(f"    {g!r}: {{\n{body}    }},\n")
    props = _wrapped_args([f"{k}={v!r}" for k, v in unit.properties().items()], 12)
    ports = []
    for i, p in enumerate(unit.inputs):
        ports.append(f"        unit.o_PORT({p!r}, streamer.i_READER({i}))\n")
    for j, p in enumerate(unit.outputs):
        ports.append(f"        unit.o_PORT({p!r}, streamer.i_WRITER({j}))\n")
    names = tcdm_port_names(template)
    doc = (
        f"The accelerator of design {design}: instance {inst_name}, BRM {inst.brm.name}, "
        f"implementation\n{inst.implementation}, block {b['block']} ({unit.wait}, {unit.control}). "
        "The CSR mux, the\nstreamer, SnaxShell and SnaxUnit, as snax-gvsoc's snax_tiled_add_acc.py.\n\n"
        f"CSRs: the streamer's, {layout.streamer_first} to {layout.streamer_last}; then the "
        f"shell's: {', '.join(f'{f} ({n})' for f, n in layout.fields.items())},\n"
        f"start ({layout.start}), busy ({layout.busy}) and the counter ({layout.counter}), "
        "read-only.\n"
    )
    return (
        _header(point_path, doc)
        + f"""
import gvsoc.systree
import interco.router as router
from snax_streamer import SnaxStreamer
from snax_shell import SnaxShell
from snax_unit import SnaxUnit

# The design point's streamers (D87): readers the input ports in the BRM's
# port order, writers the output ports
STREAMER_TEMPLATE = {{
{"".join(groups)}}}

# The platform's L1 in bytes
TCDM_SIZE = {tcdm_size}

# The first SNAX CSR and the last one the core sends out (snax_alu_acc.py)
SNAX_CSR_FIRST = {layout.streamer_first}
SNAX_CSR_LAST = {SNAX_CSR_LAST}

# Where csr_mux switches from the streamer ({streamer_csrs(template)} CSRs) to the shell
SHELL_CSR_FIRST = {layout.shell_first}

# The binding's configuration fields, in CSR order; start follows them
SHELL_FIELDS = {list(layout.fields)!r}


class {cls}(gvsoc.systree.Component):
    \"\"\"The CSR mux, the streamer, SnaxShell and SnaxUnit.

    Ports
    -----
    input : io slave
        The compute core's CSR accesses, a 32-bit request at 4 * CSR number.
    tcdm_<i> : io master
        The streamer's TCDM ports (nb_tcdm_ports of them).
    tcdm_prio_<i> : wire<bool> master
        The priority bit of each TCDM port's request.
    \"\"\"

    nb_tcdm_ports = {len(names)}
    tcdm_port_names = {_wrapped(names, 23)}

    def __init__(self, parent: gvsoc.systree.Component, name: str):
        super().__init__(parent, name)

        csr_mux = router.Router(self, 'csr_mux')

        streamer = SnaxStreamer(self, 'streamer', template=STREAMER_TEMPLATE,
            csr_base=SNAX_CSR_FIRST, tcdm_size=TCDM_SIZE)
        shell = SnaxShell(self, 'shell', csr_base=SHELL_CSR_FIRST, fields=SHELL_FIELDS)
        unit = SnaxUnit(self, 'unit',
            {props},
            csr_base=SHELL_CSR_FIRST)

        streamer_base = 4 * SNAX_CSR_FIRST
        shell_base = 4 * SHELL_CSR_FIRST
        csr_mux.o_MAP(streamer.i_INPUT(), base=streamer_base,
            size=shell_base - streamer_base)
        csr_mux.o_MAP(shell.i_INPUT(), base=shell_base,
            size=4 * (SNAX_CSR_LAST + 1) - shell_base)

        shell.o_CTRL(unit.i_CTRL())
        unit.o_BUSY(shell.i_BUSY())

{"".join(ports)}
        self.bind(self, 'input', csr_mux, 'input')

        for port in range(0, self.nb_tcdm_ports):
            streamer.o_TCDM(port, self.i_TCDM_INTERNAL(port))
            streamer.o_TCDM_PRIORITY(port, self.i_TCDM_PRIORITY_INTERNAL(port))

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature='io')

    def i_TCDM_INTERNAL(self, port: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'tcdm_{{port}}', signature='io')

    def o_TCDM(self, port: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'tcdm_{{port}}', itf, signature='io')

    def i_TCDM_PRIORITY_INTERNAL(self, port: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'tcdm_prio_{{port}}', signature='wire<bool>')

    def o_TCDM_PRIORITY(self, port: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'tcdm_prio_{{port}}', itf, signature='wire<bool>')
"""
    )


def target(design: str, point_path: str) -> str:
    """The text of forge_<design>.py."""
    acc = f"{PREFIX}{design}_acc"
    board = _camel(f"{PREFIX}{design}_board")
    doc = (
        f"The target {PREFIX}{design}: snax_alu's board (snax_alu.py) with the accelerator of\n"
        f"{acc}.py in the compute core's accelerator slot; everything else as snax_alu.\n"
    )
    return (
        _header(point_path, doc)
        + f"""
import gvsoc.runner
from snax_alu import SnaxAluBoard
from {acc} import {_camel(acc)}


class {board}(SnaxAluBoard):

    accelerator = {_camel(acc)}


class Target(gvsoc.runner.Target):

    gapy_description = "snax_alu's cluster with design {design}'s generated accelerator"
    model = {board}
    name = "{PREFIX}{design}"

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model={board}, name=name)
"""
    )


# =============================================================================
# generate
# =============================================================================


def generate(
    design_point: str | Path, out_dir: str | Path, install: str | Path | None = None
) -> Manifest:
    """Write the composite, the target and the manifest to ``out_dir``; see the module doc."""
    point_path, out = Path(design_point), Path(out_dir)
    point = DesignPoint.load(point_path)
    design = point.name
    if not design.isidentifier() or keyword.iskeyword(design):
        raise GenError(
            "gen.name", f"design {design!r} cannot name a Python module (forge_<design>)"
        )
    name, inst, unit = accelerator(point)
    template = streamer_template(point, name, inst)
    l1 = point.platform.l1
    tcdm_size = l1.n_banks * l1.rows * l1.width_bits // 8
    layout = csr_layout(template, unit.fields)
    tasks = acc_tasks(point, name, inst)
    acc_mod, target_mod = f"{PREFIX}{design}_acc", f"{PREFIX}{design}"
    if install is not None:
        check_install(Path(install), [acc_mod, target_mod])

    shown = _shown(point_path)
    files = {
        f"{acc_mod}.py": composite(design, shown, name, inst, unit, template, tcdm_size, layout),
        f"{target_mod}.py": target(design, shown),
    }
    out.mkdir(parents=True, exist_ok=True)
    for f, text in files.items():
        (out / f).write_text(text)
    manifest = Manifest(
        design=design,
        design_point=_relative(point_path, out),
        target=target_mod,
        files=[*files, "manifest.json"],
        target_dirs=["."],
        trace=UNIT_TRACE,
        csrs=layout,
        tasks=tasks,
    )
    (out / "manifest.json").write_text(manifest.to_json())
    return manifest


def _relative(path: Path, folder: Path) -> str:
    return os.path.relpath(path.resolve(), folder.resolve())
