"""The composite tests' walk: one of snax-gvsoc's forge/ designs (``FORGE_FIXTURES``) through every
stage once, under a temporary root as tests/gen's fixture tests do.

    walk(tmp, design) -> Walk   sandbox and design (their commands), lower, gen-gvsoc, gen-rtl's
                                description and fixture, the estimate, and the recorded run of
                                tests/run/fixtures/ where the design has one (``RECORDED_LEVELS``)

A composite module walks its design once (a module fixture, its tests in one xdist group) and
asserts one feature per test; ``files`` and ``same_files`` compare a generated folder with its
checked-in fixture byte for byte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from snax_forge.design import DesignPoint
from snax_forge.estimate import Estimate, estimate
from snax_forge.gen import generate
from snax_forge.gen.rtl import write_accelerator, write_fixture
from snax_forge.lower import task_list
from snax_forge.lower.tasks import TaskList
from snax_forge.run import load_cluster
from tests.gen.helpers import designed, files
from tests.helpers import FORGE_FIXTURES, REPO, SUB

FORGE = SUB / "forge"
RTL_FIXTURES = REPO / "hw" / "chisel" / "src" / "test" / "resources" / "rtl"
# The recorded runs of tests/run/fixtures/ and their events' levels
RECORDED_LEVELS = {"vecadd_v1": ("task", "stall"), "mvt_v1_2_2": ("task", "stall"),
                   "mvt_v5_sequential": ("task",), "bicg_v4_1": ("beat",)}  # fmt: skip


@dataclass
class Walk:
    """A design's walk: its root, its design point (its path from the root and loaded), its task
    list, gen-gvsoc's folder and ``accelerator.json``, the cluster run/ reads from them,
    gen-rtl's fixture folder (``vectors.json`` for one accelerator task, else empty), the
    estimate on nangate45 and, where it has one, its recorded runs by level."""

    design: str
    root: Path
    point_path: str
    point: DesignPoint
    tasks: TaskList
    gen: Path
    accelerator: dict[str, Any]
    cluster: Any
    rtl: Path
    estimate: Estimate
    recorded: dict[str, Any]

    @property
    def units(self) -> dict[str, dict[str, Any]]:
        """accelerator.json's units by name, over every accelerator and group."""
        return {u["name"]: u for a in self.accelerator["accelerators"] for g in a["groups"]
                for u in g["units"]}  # fmt: skip

    @property
    def groups(self) -> list[dict[str, Any]]:
        return [g for a in self.accelerator["accelerators"] for g in a["groups"]]


def walk(tmp: Path, design: str) -> Walk:
    """``design``'s walk under ``tmp`` as the root; see the module doc."""
    from tests.run.helpers import load_recorded

    recipe, sets = FORGE_FIXTURES[design]
    rel = designed(tmp, recipe, design, sets)
    gen, rtl = tmp / "gvsoc" / design, tmp / "rtl" / design
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)  # the files name the design point by its path from the root
        point = DesignPoint.load(tmp / rel)
        tasks = task_list(point)
        generate(rel, gen, root=tmp, tasks=tasks)
        write_accelerator(rel, tmp / "spec", tmp)
        write_fixture(rel, rtl, tmp)
    recorded = {lv: load_recorded(design, f"events_{lv}.log")
                for lv in RECORDED_LEVELS.get(design, ())}  # fmt: skip
    return Walk(design, tmp, rel, point, tasks, gen,
                json.loads((gen / "accelerator.json").read_text()),
                load_cluster(gen / "snax_forge.hjson", tmp / rel), rtl,
                estimate(point.graph, point.platform, schedule=point.schedule), recorded)  # fmt: skip


def same_files(out: Path, fixture: Path, skip: tuple[str, ...] = (".md",)) -> None:
    """``out`` holds ``fixture``'s files (those not ending in ``skip``), each byte for byte."""
    want = [f for f in files(fixture) if not f.endswith(skip)]
    assert [f for f in files(out) if not f.endswith(skip)] == want
    for f in want:
        assert (out / f).read_bytes() == (fixture / f).read_bytes(), f
