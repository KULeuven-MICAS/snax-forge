"""The whole path from a kernel to a model run (E2E1, D90).

``run_flow(recipe, platform, ...)`` runs SNAX-SANDBOX, SNAX-DESIGN,
SNAX-LOWER and SNAX-MODEL in turn and checks the run's output against the
kernel's reference and the reference executor (run.py), then writes
``report/`` and ``flow.log`` beside the run. The command line is
``python -m snax_forge.flow`` (pixi ``flow``).
"""

from .run import (
    LOG,
    REPORT,
    Flow,
    FlowError,
    checks_that_ran,
    default_name,
    functional_check,
    read_container,
    regions_of,
    run_flow,
    summary,
)

__all__ = [
    "LOG",
    "REPORT",
    "Flow",
    "FlowError",
    "checks_that_ran",
    "default_name",
    "functional_check",
    "read_container",
    "regions_of",
    "run_flow",
    "summary",
]
