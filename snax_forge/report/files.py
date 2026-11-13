"""Where the reports read from and write to.

A folder is a flow folder (``out/flow/<name>/``), or the ``run/`` directory
in one: ``run/`` holds the run (run/), ``design/`` the design point,
``gen/`` the description files, ``sandbox/recipe.json`` the recipe (a flow from a
bound graph has none) and ``tasks.json`` the task list. Both reports go to
``<folder>/report/``, next to ``run/`` and never inside it, since a run
directory holds only what the run wrote.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from snax_forge.char import DEFAULT_TECH
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


def reports_of(folder: str | Path, out: str | Path | None = None,
               tech: str | None = DEFAULT_TECH) -> Reports:  # fmt: skip
    """Both reports of a flow folder (module doc), not yet written; ``design.md``'s cost on
    ``tech`` (None: none)."""
    try:
        rv = api.load_run(folder)
    except RunError as e:
        raise ReportError(f"{folder}: not a flow run ({e})") from None
    from snax_forge.sandbox import Recipe

    flow = rv.folder
    point = rv.point
    recipe = None
    if (flow / "sandbox" / "recipe.json").is_file():
        recipe = Recipe.load(flow / "sandbox" / "recipe.json")
    check = rv.outputs.profile.functional_check or {}
    design = build_design(rv.name, rv.cluster, point, recipe, check.get("kernel", ""),
                          rv.outputs.elf_symbols, tech)  # fmt: skip
    return Reports(design, build_run(rv), Path(out) if out is not None else flow / "report")


def write_reports(folder: str | Path, out: str | Path | None = None,
                  tech: str | None = DEFAULT_TECH) -> Reports:  # fmt: skip
    """Build and write ``design.md`` and ``run.md``; returns what was written."""
    r = reports_of(folder, out, tech)
    r.out.mkdir(parents=True, exist_ok=True)
    (r.out / DESIGN_MD).write_text(render_design(r.design))
    (r.out / RUN_MD).write_text(render_run(r.run))
    return r


__all__ = ["DESIGN_MD", "RUN_MD", "ReportError", "Reports", "reports_of", "write_reports"]
