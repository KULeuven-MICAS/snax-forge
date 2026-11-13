"""A design point -> its bundle for snax_cluster.

``write_bundle`` writes one folder per design point holding what snax_cluster builds the
design's cluster from, as CONTRACTS, "Bundle" lists it: gvsoc.py's files at its root, rtl.py's
elaboration in ``rtl/``, the RTL's file list ``rtl/files.f`` and ``bundle.json`` (``Bundle``,
its modules from ``module_tree``). Everything is made again by the snax-forge checked out, never
copied from ``out/gvsoc/`` or ``out/rtl/``. ``rtl/`` is removed first; the rest of the folder is
left as it is, but an earlier ``forge_expected.txt``, which a bundle alone has no run for.
``main`` is ``pixi run bundle``.

``write_expected`` writes ``forge_expected.txt`` after ``pixi run forge``'s SNAX-GVSoC run
passed, its fill lines from ``counter_fills``; the flow does not call it. Refusals are
``GenError``s: what gvsoc.py and rtl.py refuse, ``bundle.empty`` (``emitted.json`` lists no
file) and ``bundle.expected`` (the run printed no ``forge counter`` line or no ``forge data: 0
errors`` line, or a counter's task has no fill).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from snax_forge.common import Config, JsonFile, cwd_relative, git

from .accelerator import ACCELERATOR_JSON, Description
from .blocks import GenError
from .gvsoc import ROOT, RTL_FILE_LIST, cli, generate, out_dir
from .rtl import EMITTED_JSON, accelerator_spec, elaborate

OUT = ROOT / "out" / "bundle"
BUNDLE_JSON = "bundle.json"
EXPECTED = "forge_expected.txt"


@dataclass(frozen=True)
class Bundle(Config, JsonFile):
    """``bundle.json`` (CONTRACTS, "Bundle")."""

    design_point: str
    tag: str
    top: str
    modules: dict[str, list[str]]
    snax_forge_commit: str | None
    dirty: bool
    expected: str | None = None


def module_tree(acc: Description) -> dict[str, list[str]]:
    """The wrapper's modules and the submodules each holds: ``bundle.json``'s ``modules``
    (CONTRACTS, "Bundle")."""
    t = acc.tag
    return {
        f"{t}_csr": [f"{t}_peripherals"],
        f"{t}_ctrl": [f"{t}_{g.name}_ctrl" for g in acc.groups],
        f"{t}_datapath": [
            *(name for g in acc.groups for name in
              [*(f"{t}_{u.name}" for u in g.units), *(f"{t}_{k.name}" for k in g.links if k.depth)])
        ],
        **{f"{t}_{a.name}_port_mux": [] for a in acc.accelerators if a.port_mux is not None},
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
    (out / EXPECTED).unlink(missing_ok=True)
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


def counter_fills(design_point: str | Path, tasks: str | Path) -> dict[str, int]:
    """Each accelerator task's group's streamer fill by task name (CONTRACTS, "Bundle"): the
    estimate's ``fill`` of the group the task (``tasks.json``'s accel configure) runs."""
    from snax_forge.design import DesignPoint
    from snax_forge.estimate import estimate

    point = DesignPoint.load(design_point)
    fill = {g.name: g.fill for g in estimate(point.graph, point.platform, None).groups}
    steps = json.loads(Path(tasks).read_text())["steps"]
    return {s["task_name"]: fill[s["component"]] for s in steps
            if s.get("op") == "configure" and s.get("type") == "accel"}  # fmt: skip


def expected_lines(stdout: str, fills: dict[str, int]) -> list[str]:
    """A SNAX-GVSoC run's ``forge counter`` lines in order, then its ``forge data:`` line, then
    each counter's ``forge fill`` line (the module doc); ``bundle.expected`` without a counter
    line or a clean data line, or for a counter whose task has no fill."""
    lines = stdout.splitlines()
    counters = [x for x in lines if x.startswith("forge counter ")]
    data = [x for x in lines if x.startswith("forge data: ")]
    if not counters or len(data) != 1 or not data[0].startswith("forge data: 0 errors"):
        raise GenError("bundle.expected", f"the run printed counters {counters} and data {data}: "
                       "no counter, or not one line of 0 errors")  # fmt: skip
    tasks = [x.removeprefix("forge counter ").rpartition(":")[0] for x in counters]
    missing = [t for t in tasks if t not in fills]
    if missing:
        raise GenError("bundle.expected", f"no streamer fill for the counters of {missing}: "
                       f"the fills are for {sorted(fills)}")  # fmt: skip
    return [*counters, *data, *(f"forge fill {t}: {fills[t]}" for t in tasks)]


def write_expected(out_dir: str | Path, stdout: str, fills: dict[str, int]) -> Path:
    """``forge_expected.txt`` in the bundle ``out_dir`` from its SNAX-GVSoC run's ``stdout`` and
    each task's fill (``counter_fills``), and ``bundle.json``'s ``expected`` naming it; the
    file's path."""
    out = Path(out_dir)
    path = out / EXPECTED
    path.write_text("".join(f"{x}\n" for x in expected_lines(stdout, fills)))
    made = Bundle.load(out / BUNDLE_JSON)
    (out / BUNDLE_JSON).write_text(replace(made, expected=EXPECTED).to_json())
    return path


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
        print(f"no {EXPECTED}: a bundle alone has no SNAX-GVSoC run (pixi run forge writes it)")
        return 0

    return cli(write)


if __name__ == "__main__":
    sys.exit(main())
