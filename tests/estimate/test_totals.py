"""Every recipe's last graph estimated on the default platform: its cycles, footprint and cost
parts against tests/estimate/totals.json, so a changed coefficient anywhere in the cost model
fails here even where no composite design uses the block."""

import json

import pytest

from tests.numbers import TOTALS, check, recipe_totals
from tests.sandbox.helpers import RECIPES

pytestmark = pytest.mark.slow

SNAPSHOT = json.loads(TOTALS.read_text())


def test_every_recipe_has_a_snapshot():
    assert sorted(SNAPSHOT) == sorted(RECIPES)


@pytest.mark.parametrize("name", RECIPES)
def test_the_totals_match_their_snapshot(name):
    check(recipe_totals(name), SNAPSHOT[name], f"{name}'s estimate")
