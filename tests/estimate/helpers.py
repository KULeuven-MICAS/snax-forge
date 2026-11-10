"""Shared helpers for the estimate tests: the estimate of a recipe's last step on snax_forge
(its graphs from tests/helpers.py's ``steps``)."""

from __future__ import annotations

from snax_forge.design import Platform
from snax_forge.estimate import Estimate, estimate
from tests.helpers import REPO, steps

SNAX_FORGE = REPO / "platforms" / "snax_forge.json"


# The JOB_BEATS CSRs of the recipes SYN2 and the experiments' model synthesized (C§17's table):
# vsum_tree and dot 2-6 one fold's, the elementwise ones none
OPEN_TS = {"vecadd": 0, "vecmul": 0}


def copies(recipe: str, job_width: int = 32) -> int:
    """The bits of CTRL1's copies of JOBS and each JOB_BEATS (job_width each) that SYN2's control
    units still hold and the control unit no longer keeps (D168); CTRL2's R5 refreshes the
    fixtures from the new synthesis."""
    return job_width * (1 + OPEN_TS.get(recipe, 1))


def estimated(name: str, **params: int) -> Estimate:
    """The estimate of recipe ``name``'s last graph on snax_forge."""
    return estimate(steps(name, **params)[-1], Platform.load(SNAX_FORGE))
