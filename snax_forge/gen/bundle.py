"""A design point -> its bundle for snax_cluster (D111, D130, C§18).

    write_bundle(design_point, out_dir, root=None) -> (Bundle, written)
    python -m snax_forge.gen.bundle DESIGN_POINT [--out DIR]    (pixi bundle)

A bundle is one folder per design point holding what snax_cluster builds the
design's cluster from. Everything in it is made again by the snax-forge
checked out, never copied from ``out/gvsoc/`` or ``out/rtl/``. ``rtl/`` is
the bundle's own, so it is removed first and holds only this run's files;
the rest of the folder is left as it is. Then, in this order:

    <the root>            gvsoc.py's ``generate``: the three description files
                          (accelerator.json among them), the headers, data.h,
                          main.c and manifest.json; the
                          cluster file's slot names ``rtl/files.f``
                          (``snax_forge_rtl``)
    rtl/sv/, rtl/emitted.json
                          rtl.py's ``elaborate`` (``Emit --spec``, sbt) of the
                          root's accelerator.json, checked against the
                          datapath catalogue first (rtl.py's
                          ``accelerator_spec``)
    rtl/files.f           the RTL's file list: one path per line, relative
                          to ``rtl/``, in ``emitted.json``'s ``files`` order,
                          ``emitted.json`` itself left out
    bundle.json           ``Bundle``: ``design_point`` (as gvsoc.py names
                          it), ``tag`` (accelerator.json's), ``top`` (the
                          wrapper, ``<tag>_shell_wrapper``), ``modules``
                          (``module_tree``), ``snax_forge_commit``
                          (``git rev-parse HEAD`` of the checkout, null
                          outside git) and ``dirty`` (the work tree has
                          uncommitted changes, untracked files included;
                          false outside git)

**The modules** (``module_tree``, C§18), from accelerator.json, not from
Emit's output: the wrapper's three modules, each with the submodules it holds,
by the names hw/chisel's Emit gives them: ``<tag>_csr`` (none), ``<tag>_ctrl``
(``<tag>_<group>_ctrl`` per group) and ``<tag>_datapath`` (``<tag>_<unit>``
per unit, then ``<tag>_<link>`` per link of depth >= 1, a buffer, group by
group).

snax_cluster takes it through the cluster file alone: snaxgen reads the
slot's ``snax_forge_rtl`` and writes ``generated/snax_forge.f``, which its
builds read.

**Refused**, each a ``GenError``: what ``generate``, ``accelerator_spec`` and
``elaborate`` refuse (``gen.*``, the catalogue's checks and a listed file that is not there among them), then
``bundle.empty`` (``emitted.json`` lists no file).
The flow does not call it (D136).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from snax_forge.common import Config, JsonFile, cwd_relative, git

from .accelerator import ACCELERATOR_JSON, Accelerator
from .blocks import GenError
from .gvsoc import ROOT, RTL_FILE_LIST, cli, generate, out_dir
from .rtl import EMITTED_JSON, accelerator_spec, elaborate

OUT = ROOT / "out" / "bundle"
BUNDLE_JSON = "bundle.json"


@dataclass(frozen=True)
class Bundle(Config, JsonFile):
    """``bundle.json``; see the module doc."""

    design_point: str
    tag: str
    top: str
    modules: dict[str, list[str]]
    snax_forge_commit: str | None
    dirty: bool


def module_tree(acc: Accelerator) -> dict[str, list[str]]:
    """The wrapper's modules and the submodules each holds; see the module doc."""
    t = acc.tag
    return {
        f"{t}_csr": [],
        f"{t}_ctrl": [f"{t}_{g.name}_ctrl" for g in acc.groups],
        f"{t}_datapath": [
            *(name for g in acc.groups for name in
              [*(f"{t}_{u.name}" for u in g.units), *(f"{t}_{k.name}" for k in g.links if k.depth)])
        ],
    }  # fmt: skip


def provenance(checkout: Path) -> tuple[str | None, bool]:
    """The checkout's commit (``git rev-parse HEAD``) and whether its work tree has uncommitted
    changes; ``(None, False)`` outside git."""
    head = git(checkout, "rev-parse", "HEAD")
    if head is None:
        return None, False
    return head, bool(git(checkout, "status", "--porcelain"))


def file_list(rtl_dir: Path, emitted: list[Path]) -> list[str]:
    """The files ``elaborate`` wrote and checked (``emitted``), relative to ``rtl_dir``,
    ``emitted.json`` left out."""
    files = [f.relative_to(rtl_dir).as_posix() for f in emitted if f.name != EMITTED_JSON]
    if not files:
        raise GenError(
            "bundle.empty", f"{rtl_dir / EMITTED_JSON} lists no RTL file: files.f would be empty"
        )
    return files


def write_bundle(design_point: str | Path, out_dir: str | Path,
                 root: str | Path | None = None) -> tuple[Bundle, list[Path]]:  # fmt: skip
    """Write the bundle to ``out_dir``; see the module doc. The bundle and the files written,
    in the order written."""
    out = Path(out_dir)
    files_f = out / RTL_FILE_LIST
    rtl_dir = files_f.parent
    shutil.rmtree(rtl_dir, ignore_errors=True)
    manifest = generate(design_point, out, root)
    spec = accelerator_spec(design_point, root)
    rtl_dir.mkdir(parents=True, exist_ok=True)
    files = file_list(rtl_dir, elaborate(out / ACCELERATOR_JSON, rtl_dir))
    files_f.write_text("".join(f"{f}\n" for f in files))
    commit, dirty = provenance(ROOT)
    bundle = Bundle(manifest.design_point, spec.tag, f"{spec.tag}_shell_wrapper",
                    module_tree(spec), commit, dirty)  # fmt: skip
    (out / BUNDLE_JSON).write_text(bundle.to_json())
    written = [
        *(out / f for f in manifest.files), *(rtl_dir / f for f in files),
        rtl_dir / EMITTED_JSON, files_f, out / BUNDLE_JSON,
    ]  # fmt: skip
    return bundle, written


def main(argv: list[str] | None = None) -> int:
    """``python -m snax_forge.gen.bundle DESIGN_POINT [--out DIR]`` (pixi ``bundle``): the
    bundle into ``DIR`` (default ``out/bundle/<design>/``). Exit 0 when written, 1 when refused
    (the rule named), 2 on bad arguments."""
    ap = argparse.ArgumentParser(prog="python -m snax_forge.gen.bundle")
    ap.add_argument("design_point", type=Path, help="a design_point.json (the design step's)")
    ap.add_argument("--out", type=Path, help="output directory (default out/bundle/<design>)")
    args = ap.parse_args(argv)

    def write() -> int:
        out = out_dir(args.out, OUT, args.design_point)
        bundle, written = write_bundle(args.design_point, out)
        commit = bundle.snax_forge_commit or "outside git"
        print(f"bundle {cwd_relative(out)}: {bundle.tag}_shell_wrapper from {bundle.design_point}, "
              f"snax-forge {commit}{', dirty' if bundle.dirty else ''}")  # fmt: skip
        for f in written:
            print(f"wrote {cwd_relative(f)}")
        return 0

    return cli(write)


if __name__ == "__main__":
    sys.exit(main())
