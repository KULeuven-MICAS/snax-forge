"""atax_v4 on SNAX-GVSoC at the fixture's size: y = Aᵀ (A x) exactly, every data word, and the
group's BUSY_COUNTER within 10 % of the estimate's beyond the streamers' fill (``sound``)."""

import pytest

from snax_forge.flow import run_flow
from tests.design.helpers import SNAX_FORGE
from tests.flow.helpers import needs_gvsoc, sound, summary
from tests.helpers import FORGE_FIXTURES, recipe_path

pytestmark = [pytest.mark.gvsoc, needs_gvsoc]


def test_atax_v4_runs_with_no_error_and_its_counter_near_the_estimate(tmp_path):
    recipe, sets = FORGE_FIXTURES["atax_v4"]
    values = dict(s.split("=") for s in sets)
    f = run_flow(recipe_path(recipe), SNAX_FORGE, recipe_sets={k: int(v) for k, v in values.items()},
                 out=tmp_path / "atax_v4")  # fmt: skip
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0
    (counter,) = sound(f).values()
    assert counter >= 8  # its beats: M · N / W firings
