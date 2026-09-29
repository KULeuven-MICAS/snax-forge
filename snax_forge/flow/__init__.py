"""The whole path from a kernel to a model run (E2E1, D90).

``run_flow(recipe, platform, ...)`` runs SNAX-SANDBOX, SNAX-DESIGN,
SNAX-LOWER and SNAX-MODEL in turn and checks the run's output against the
kernel's reference and the reference executor (run.py). The command line is
``python -m snax_forge.flow`` (pixi ``flow``).
"""

from .run import (
    Flow,
    FlowError,
    default_name,
    functional_check,
    read_container,
    regions_of,
    run_flow,
)

__all__ = [
    "Flow",
    "FlowError",
    "default_name",
    "functional_check",
    "read_container",
    "regions_of",
    "run_flow",
]
