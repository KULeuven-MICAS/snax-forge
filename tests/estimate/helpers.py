"""Shared helpers for the estimate tests: the estimate of a recipe's last step on snax_forge
(its graphs from tests/helpers.py's ``steps``)."""

from __future__ import annotations

from snax_forge.design import Platform
from snax_forge.estimate import Estimate, estimate
from tests.helpers import REPO, steps

SNAX_FORGE = REPO / "platforms" / "snax_forge.json"


def estimated(name: str, **params: int) -> Estimate:
    """The estimate of recipe ``name``'s last graph on snax_forge."""
    return estimate(steps(name, **params)[-1], Platform.load(SNAX_FORGE))
