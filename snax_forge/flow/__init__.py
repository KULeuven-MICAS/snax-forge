"""The whole path from a kernel to a model run (E2E1, D90), or from a bound graph (FLOW2, D108).

``run_flow(recipe, platform, ...)`` runs SNAX-SANDBOX, SNAX-DESIGN,
SNAX-LOWER and SNAX-MODEL in turn and checks the run's output against the
kernel's reference and the reference executor (run.py), then writes
``report/`` and ``flow.log`` beside the run; the log ends with each
stage's wall-clock time (D107). ``run_bound(graph, platform, ...)`` is the
same path from a bound ``.snaxdfg`` on, without the sandbox (D108). The
command line is ``python -m snax_forge.flow`` (pixi ``flow``); it takes
either, told apart by the file's suffix.
"""

from .run import (
    LOG,
    REPORT,
    Flow,
    FlowError,
    bound_name,
    checks_that_ran,
    default_name,
    functional_check,
    read_container,
    regions_of,
    run_bound,
    run_flow,
    summary,
)

__all__ = [
    "LOG",
    "REPORT",
    "Flow",
    "FlowError",
    "bound_name",
    "checks_that_ran",
    "default_name",
    "functional_check",
    "read_container",
    "regions_of",
    "run_bound",
    "run_flow",
    "summary",
]
