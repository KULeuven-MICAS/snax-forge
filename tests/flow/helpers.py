"""Shared helpers for the flow tests (D136).

The flow runs on SNAX-GVSoC, so CI (which never builds it) runs everything up
to the generator for real and SNAX-GVSoC through ``stub``: ``run_gvsoc``
replaced by a fake make that records each call, makes the install on
``build``, an ELF on ``sw`` and on ``run`` prints what the generated program
would (a window, no data errors, each counter its beats plus 3), writes the
events SNAX-GVSoC would for the level (``tests/run/helpers.synthetic``) and
exits 0; the checkout is a clean one at ``COMMIT``, pinned there, and the GVSoC
build's folder is the test's own. The recipe flows start from the import
fixtures (``PLAIN``), so they need no DaCe. The full flow on SNAX-GVSoC is
marked ``gvsoc`` and skipped where ``out/snax-gvsoc/install`` is missing.

``excerpts_hold`` is the check the walk-throughs in ``examples/`` share: a
table quoted in a README under ``<!-- excerpt: FOLDER/report/FILE -->`` must
be, line for line, part of that report.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from snax_forge.flow import Checkout
from snax_forge.flow import run as flow_run
from snax_forge.lower.tasks import TaskList
from snax_forge.run import load_cluster
from tests.design.helpers import FIXTURES, REPO
from tests.run.helpers import synthetic

PLAIN = FIXTURES / "vecadd.snaxdfg"  # the import of the vecadd kernel
PLAINS = {"vecadd": PLAIN, "vsum": FIXTURES / "vsum.snaxdfg"}

COMMIT = "a" * 40
B_PIN = [("B.l1.base", 576)]
PINS = [*B_PIN, ("C.l1.base", 1152)]
EXCERPT = re.compile(r"<!-- excerpt: (\S+) -->")
INSTALL = REPO / "out" / "snax-gvsoc" / "install"
needs_gvsoc = pytest.mark.skipif(
    not INSTALL.is_dir(), reason="no SNAX-GVSoC install in out/snax-gvsoc (make -C gvsoc build)"
)


@dataclass
class FakeMake:
    """``run_gvsoc``'s stand-in: each call's goal and variables, in order; ``rc`` and
    ``stderr`` per goal for the ones that should fail, ``stdout`` for ``run`` in place of the
    program's lines."""

    workdir: Path
    calls: list[tuple[str, dict[str, str]]] = field(default_factory=list)
    rc: dict[str, int] = field(default_factory=dict)
    stderr: dict[str, str] = field(default_factory=dict)
    stdout: str | None = None

    def __call__(self, goal, variables, log, err=None):
        self.calls.append((goal, dict(variables)))
        log.parent.mkdir(parents=True, exist_ok=True)
        text = f"make {goal}\n"
        if goal == "build":
            gvrun = self.workdir / "install" / "bin" / "gvrun"
            gvrun.parent.mkdir(parents=True, exist_ok=True)
            gvrun.write_text("#!/bin/sh\n")
            (self.workdir / "checkout.txt").write_text(
                f"{flow_run.SNAX_GVSOC.resolve()}\n{COMMIT}\n"
            )
        elif goal == "sw":
            elf = Path(variables["FORGE_BUILD"]) / "forge.elf"
            elf.parent.mkdir(parents=True, exist_ok=True)
            elf.write_bytes(b"\x7fELF fake")
        elif goal == "run":
            text = self.stdout if self.stdout is not None else program_output(variables)
            if variables.get("EVENTS"):
                write_events(variables)
        log.write_text(text + self.stderr.get(goal, "") * (err is None))
        if err is not None:
            err.write_text(self.stderr.get(goal, ""))
        return self.rc.get(goal, 0)


def program_output(variables: dict[str, str]) -> str:
    """What the generated program prints when it passes, from the manifest beside the cluster
    file the run is given: a window, no data errors, each counter its beats plus 3."""
    cluster = Path(variables["GVRUN_ARGS"].partition("hjson=")[2])
    manifest = json.loads((cluster.parent / "manifest.json").read_text())
    lines = ["forge cycles: 226", "forge data: 0 errors (64 words)"]
    lines += [f"forge counter {t['name']}: {t['beats'] + 3}" for t in manifest["tasks"]]
    return "\n".join([*lines, "forge errors: 0", ""])


def write_events(variables: dict[str, str]) -> None:
    """The events a passing run writes to ``EVENTS`` for its ``TRACES``: the synthetic run of
    its cluster (beside the cluster file) and its task list (in the flow's folder)."""
    gen = Path(variables["GVRUN_ARGS"].partition("hjson=")[2]).parent
    cluster = load_cluster(gen / "snax_alu.hjson", gen.parent / "design" / "design_point.json")
    tasks = TaskList.from_dict(json.loads((gen.parent / "tasks.json").read_text()))
    level = "task" if "events" in variables["TRACES"].split() else "off"
    Path(variables["EVENTS"]).write_text("\n".join(synthetic(cluster, tasks, level)) + "\n")


def stub(monkeypatch, tmp_path: Path, pinned: str = COMMIT, dirty: bool = False) -> FakeMake:
    """SNAX-GVSoC stubbed for one test (module doc); the fake make, to read or set up."""
    fake = FakeMake(tmp_path / "snax-gvsoc")
    monkeypatch.setattr(flow_run, "run_gvsoc", fake)
    monkeypatch.setattr(flow_run, "GVSOC_WORKDIR", fake.workdir)
    monkeypatch.setattr(
        flow_run, "checkout", lambda: Checkout(flow_run.SNAX_GVSOC, COMMIT, dirty, pinned)
    )
    return fake


def built(fake: FakeMake, commit: str = COMMIT) -> None:
    """The fake build's folder as ``make build`` leaves it for ``commit``."""
    gvrun = fake.workdir / "install" / "bin" / "gvrun"
    gvrun.parent.mkdir(parents=True, exist_ok=True)
    gvrun.write_text("#!/bin/sh\n")
    (fake.workdir / "checkout.txt").write_text(f"{flow_run.SNAX_GVSOC.resolve()}\n{commit}\n")


def golden(data_h: str, name: str) -> list[int]:
    """The values of array ``name`` in a data.h."""
    m = re.search(rf"\b{re.escape(name)}\[\d+\][^=]*= \{{(.*?)\}};", data_h, re.DOTALL)
    assert m, name
    return [int(x) for x in m[1].replace("\n", " ").split(",") if x.strip()]


def excerpts_hold(readme: str, root: Path) -> int:
    """Every excerpt of ``readme`` (the lines after its marker, up to the first empty one) is
    made of lines of the report it names, a path under ``root``; returns how many there are."""
    lines = readme.splitlines()
    seen = 0
    for i, line in enumerate(lines):
        m = EXCERPT.fullmatch(line.strip())
        if not m:
            continue
        report = set((root / m.group(1)).read_text().splitlines())
        block = []
        for x in lines[i + 1 :]:
            if not x.strip():
                break
            block.append(x)
        assert block, f"excerpt at line {i + 1} is empty"
        missing = [x for x in block if x not in report]
        assert not missing, f"README line {i + 1}: not in {m.group(1)}: {missing}"
        seen += 1
    return seen


def stubbed_flow(base: Path, design: str = "vecadd", level: str = "task", out: Path | None = None,
                 **kw: Any) -> Path:  # fmt: skip
    """A flow folder (``base/<design>_<level>`` unless ``out``) made with SNAX-GVSoC stubbed
    (``stub``, its synthetic events), outside a test's own monkeypatch, so module-scoped
    fixtures can share it; returns the folder."""
    from snax_forge.flow import run_flow

    out = out or base / f"{design}_{level}"
    with pytest.MonkeyPatch.context() as mp:
        stub(mp, out.parent / f".gvsoc_{out.name}")
        run_flow(REPO / "recipes" / f"{design}.json", REPO / "platforms" / "snax_alu.json",
                 graph_path=PLAINS[design], out=out, trace_level=level, **kw)  # fmt: skip
    return out
