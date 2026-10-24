"""The whole path from a kernel to a SNAX-GVSoC run (D90, D136), or from a bound graph (D108).

``run_flow(recipe, platform, ...)`` runs SNAX-SANDBOX, SNAX-DESIGN,
SNAX-LOWER and the generator, builds the program and runs it on SNAX-GVSoC,
whose output the program checks itself (run.py), then writes ``flow.log``
beside the run; the log ends with each stage's wall-clock time.
``run_bound(graph, platform, ...)`` is the same path from a bound
``.snaxdfg`` on, without the sandbox. The command line is ``python -m
snax_forge.flow`` (pixi ``flow``); it takes either, told apart by the file's
suffix.
"""

from .run import (
    LOG,
    STAGE_NAMES,
    Checkout,
    Flow,
    FlowError,
    bound_name,
    build_reason,
    check_output,
    checkout,
    default_name,
    run_bound,
    run_exit_code,
    run_flow,
    run_gvsoc,
    summary,
)

__all__ = [
    "LOG",
    "STAGE_NAMES",
    "Checkout",
    "Flow",
    "FlowError",
    "bound_name",
    "build_reason",
    "check_output",
    "checkout",
    "default_name",
    "run_bound",
    "run_exit_code",
    "run_flow",
    "run_gvsoc",
    "summary",
]
