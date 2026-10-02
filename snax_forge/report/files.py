"""Where the reports read from and write to (REP1, D99).

A folder is one of two things:

* a flow folder (``out/flow/<name>/``): ``run/`` holds the run, ``design/``
  the design point, ``sandbox/recipe.json`` the recipe (a flow from a bound
  graph has none, D108) and ``tasks.json`` the task list. Both reports are complete; they go to ``<folder>/report/``,
  next to ``run/`` and never inside it, since a run directory holds only what
  the model wrote (D44).
* a run directory (``run.json`` in it), e.g. a scenario run: the run report,
  and the design report's cluster part from ``run.json``'s cluster and
  regions. They go to ``<dir>_report/`` beside it. A ``tasks.json`` next to
  the run directory, or the one given (``--tasks``, e.g. a scenario's), names
  its tasks and the containers its streamers read and write.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snax_forge.snax_model.scenario import ClusterConfig

from ..viz import api
from .design import DesignReport, build_design, render_design
from .run import RunReport, build_run, render_run

DESIGN_MD = "design.md"
RUN_MD = "run.md"


class ReportError(ValueError):
    """A folder that is neither a flow folder nor a run directory."""


@dataclass
class Reports:
    design: DesignReport
    run: RunReport
    out: Path


def _kind(folder: Path) -> str:
    if (folder / "run" / "run.json").is_file():
        return "flow"
    if (folder / "run.json").is_file():
        return "run"
    raise ReportError(
        f"{folder}: not a flow folder (no run/run.json) nor a run directory (no run.json)"
    )


def reports_of(
    folder: str | Path, out: str | Path | None = None, tasks_path: str | Path | None = None
) -> Reports:
    """Both reports of ``folder`` (module doc), not yet written; ``tasks_path`` names the
    task list when it is not where the folder keeps one (a scenario run's is in its folder)."""
    folder = Path(folder)
    kind = _kind(folder)
    run_dir = folder / "run" if kind == "flow" else folder
    name = folder.name if kind == "flow" else run_dir.name
    rv = api.load_run(run_dir, name)
    if tasks_path is None:
        tasks_path = folder / "tasks.json" if kind == "flow" else run_dir.parent / "tasks.json"
    tasks_path = Path(tasks_path)
    tasks: Any = None
    if tasks_path.is_file():
        from snax_forge.lower import TaskList

        tasks = TaskList.load(tasks_path)
    point, recipe = None, None
    if kind == "flow" and (folder / "design" / "design_point.json").is_file():
        from snax_forge.design import DesignPoint
        from snax_forge.sandbox import Recipe

        point = DesignPoint.load(folder / "design" / "design_point.json")
        if (folder / "sandbox" / "recipe.json").is_file():
            recipe = Recipe.load(folder / "sandbox" / "recipe.json")
    cluster = ClusterConfig.from_dict(rv.outputs.run["cluster"])
    check = rv.outputs.profile.functional_check or {}
    design = build_design(
        name, cluster, list(rv.outputs.regions), point, recipe, tasks, check.get("kernel", "")
    )
    run = build_run(rv, tasks)
    if out is None:
        out = folder / "report" if kind == "flow" else run_dir.with_name(run_dir.name + "_report")
    return Reports(design, run, Path(out))


def write_reports(
    folder: str | Path, out: str | Path | None = None, tasks_path: str | Path | None = None
) -> Reports:
    """Build and write ``design.md`` and ``run.md``; returns what was written."""
    r = reports_of(folder, out, tasks_path)
    r.out.mkdir(parents=True, exist_ok=True)
    (r.out / DESIGN_MD).write_text(render_design(r.design))
    (r.out / RUN_MD).write_text(render_run(r.run))
    return r


__all__ = ["DESIGN_MD", "RUN_MD", "ReportError", "Reports", "reports_of", "write_reports"]
