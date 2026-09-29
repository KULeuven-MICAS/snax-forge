"""Design and run reports in the layout of HLS reports (REP1, D99).

``design.md`` says what was built (design.py), ``run.md`` where the cycles
went (run.py); both are dataclass trees (record.py) rendered as Markdown, so
humans, LLMs and GitHub read the same text. ``write_reports(DIR)`` writes
them for a flow folder or a run directory (files.py); the command line is
``python -m snax_forge.report DIR`` (pixi ``report``).
"""

from .design import DesignReport, build_design, render_design
from .files import ReportError, reports_of, write_reports
from .run import RunReport, achieved_ii, build_run, render_run

__all__ = [
    "DesignReport",
    "ReportError",
    "RunReport",
    "achieved_ii",
    "build_design",
    "build_run",
    "render_design",
    "render_run",
    "reports_of",
    "write_reports",
]
