"""Where the reports read from and write to.

A folder is a flow folder (``out/flow/<name>/``), or the ``run/`` directory
in one: ``run/`` holds the run (C§7, run/), ``design/`` the design point,
``gen/`` the hjson files, ``sandbox/recipe.json`` the recipe (a flow from a
bound graph has none) and ``tasks.json`` the task list. Both reports go to
``<folder>/report/``, next to ``run/`` and never inside it, since a run
directory holds only what the run wrote.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from snax_forge.run import RunError

from ..viz import api
from .design import DesignReport, build_design, render_design
from .run import RunReport, build_run, render_run

DESIGN_MD = "design.md"
RUN_MD = "run.md"


class ReportError(ValueError):
    """A folder that is not a flow folder."""


@dataclass
class Reports:
    design: DesignReport
    run: RunReport
    out: Path


def reports_of(folder: str | Path, out: str | Path | None = None) -> Reports:
    """Both reports of a flow folder (module doc), not yet written."""
    try:
        rv = api.load_run(folder)
    except RunError as e:
        raise ReportError(f"{folder}: not a flow run ({e})") from None
    from snax_forge.design import DesignPoint
    from snax_forge.sandbox import Recipe

    flow = rv.folder
    point = DesignPoint.load(flow / rv.outputs.run["design_point"])
    recipe = None
    if (flow / "sandbox" / "recipe.json").is_file():
        recipe = Recipe.load(flow / "sandbox" / "recipe.json")
    check = rv.outputs.profile.functional_check or {}
    design = build_design(rv.name, rv.cluster, point, recipe, check.get("kernel", ""),
                          rv.outputs.elf_symbols)  # fmt: skip
    return Reports(design, build_run(rv), Path(out) if out is not None else flow / "report")


def write_reports(folder: str | Path, out: str | Path | None = None) -> Reports:
    """Build and write ``design.md`` and ``run.md``; returns what was written."""
    r = reports_of(folder, out)
    r.out.mkdir(parents=True, exist_ok=True)
    (r.out / DESIGN_MD).write_text(render_design(r.design))
    (r.out / RUN_MD).write_text(render_run(r.run))
    return r


__all__ = ["DESIGN_MD", "RUN_MD", "ReportError", "Reports", "reports_of", "write_reports"]
