"""A design point -> its bundle for snax_cluster (D111, D139, C§18).

    write_bundle(design_point, out_dir, root=None) -> (Bundle, written)
    python -m snax_forge.gen.bundle DESIGN_POINT [--out DIR]    (pixi bundle)

A bundle is one folder per design point holding what snax_cluster builds the
design's cluster from. Everything in it is made again by the snax-forge
checked out, never copied from ``out/gvsoc/`` or ``out/rtl/``, in this
order:

    <the root>            gvsoc.py's ``generate``: the three hjson files, the
                          headers, data.h, main.c and manifest.json; the
                          cluster file's slot names ``rtl/files.f``
                          (``snax_forge_rtl``)
    rtl/rtl.json          rtl.py's ``write_rtl``
    rtl/sv/, rtl/emitted.json
                          rtl.py's ``elaborate`` (``Emit --spec``, sbt)
    rtl/files.f           the RTL's file list: one path per line, relative
                          to ``rtl/``, in ``emitted.json``'s ``files`` order,
                          ``emitted.json`` itself left out
    bundle.json           ``Bundle``: ``design_point`` (as gvsoc.py names
                          it), ``tag`` (rtl.json's), ``snax_forge_commit``
                          (``git rev-parse HEAD`` of the checkout, null
                          outside git) and ``dirty`` (the work tree has
                          uncommitted changes, untracked files included;
                          false outside git)

snax_cluster takes it through the cluster file alone: snaxgen reads the
slot's ``snax_forge_rtl`` and writes ``generated/snax_forge.f``, which its
builds read (D139). ``files.f`` lists only what this run's ``emitted.json``
lists, so a file left in ``rtl/sv/`` by an earlier run is not built.

**Refused**, each a ``GenError``: what ``generate``, ``write_rtl`` and
``elaborate`` refuse (``gen.*``), then ``bundle.empty`` (``emitted.json``
lists no file) and ``bundle.missing`` (it lists a file that is not there).
The flow does not call it (D136).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snax_forge.common import cwd_relative, to_json
from snax_forge.design import DesignError
from snax_forge.design.point import DesignPoint
from snax_forge.lower import CError

from .blocks import GenError
from .gvsoc import ROOT, RTL_FILE_LIST, generate
from .rtl import EMITTED_JSON, RTL_JSON, elaborate, write_rtl

OUT = ROOT / "out" / "bundle"
BUNDLE_JSON = "bundle.json"


@dataclass(frozen=True)
class Bundle:
    """``bundle.json``; see the module doc."""

    design_point: str
    tag: str
    snax_forge_commit: str | None
    dirty: bool

    def to_dict(self) -> dict[str, Any]:
        return {"design_point": self.design_point, "tag": self.tag,
                "snax_forge_commit": self.snax_forge_commit, "dirty": self.dirty}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Bundle:
        return cls(d["design_point"], d["tag"], d["snax_forge_commit"], d["dirty"])

    def to_json(self) -> str:
        return to_json(self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Bundle:
        return cls.from_dict(json.loads(Path(path).read_text()))


def provenance(checkout: Path) -> tuple[str | None, bool]:
    """The checkout's commit (``git rev-parse HEAD``) and whether its work tree has uncommitted
    changes; ``(None, False)`` outside git."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(checkout), *args], capture_output=True,
                              text=True, check=False)  # fmt: skip

    try:
        head = git("rev-parse", "HEAD")
    except OSError:
        return None, False
    if head.returncode != 0:
        return None, False
    return head.stdout.strip(), bool(git("status", "--porcelain").stdout.strip())


def file_list(rtl_dir: Path) -> list[str]:
    """``emitted.json``'s files, relative to ``rtl_dir``, ``emitted.json`` left out; each must
    be there."""
    listing = rtl_dir / EMITTED_JSON
    files = [f for f in json.loads(listing.read_text())["files"] if f != EMITTED_JSON]
    if not files:
        raise GenError("bundle.empty", f"{listing} lists no RTL file: files.f would be empty")
    missing = [f for f in files if not (rtl_dir / f).is_file()]
    if missing:
        raise GenError("bundle.missing", f"{listing} lists files that are not there: {missing}")
    return files


def write_bundle(design_point: str | Path, out_dir: str | Path,
                 root: str | Path | None = None) -> tuple[Bundle, list[Path]]:  # fmt: skip
    """Write the bundle to ``out_dir``; see the module doc. The bundle and the files written,
    in the order written."""
    out = Path(out_dir)
    files_f = out / RTL_FILE_LIST
    rtl_dir = files_f.parent
    manifest = generate(design_point, out, root)
    spec = write_rtl(design_point, rtl_dir, root)
    elaborate(rtl_dir / RTL_JSON, rtl_dir)
    files = file_list(rtl_dir)
    files_f.write_text("".join(f"{f}\n" for f in files))
    commit, dirty = provenance(ROOT)
    bundle = Bundle(manifest.design_point, spec.tag, commit, dirty)
    (out / BUNDLE_JSON).write_text(bundle.to_json())
    written = [
        *(out / f for f in manifest.files), rtl_dir / RTL_JSON, *(rtl_dir / f for f in files),
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
    try:
        out = args.out or OUT / DesignPoint.load(args.design_point).name
        bundle, written = write_bundle(args.design_point, out)
    except (GenError, CError, DesignError) as e:
        print(e, file=sys.stderr)
        return 1
    commit = bundle.snax_forge_commit or "outside git"
    print(f"bundle {cwd_relative(out)}: {bundle.tag}_shell_wrapper from {bundle.design_point}, "
          f"snax-forge {commit}{', dirty' if bundle.dirty else ''}")  # fmt: skip
    for f in written:
        print(f"wrote {cwd_relative(f)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
