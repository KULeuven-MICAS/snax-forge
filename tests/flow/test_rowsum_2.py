"""rowsum 2 on SNAX-GVSoC (B2 R4): the lane accumulator, each lane its own row's running sum, W
rows a job, a column a beat: JOBS = M / W, JOB_BEATS = N, (M / W) * N firings.

The worked example, W = 2, M = 4, N = 3: A's rows (1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)
give y = (6, 15, 24, 33); 6 firings. At the defaults, W = 4, M = 8, N = 64: 128
firings. Contiguous, A's rows are 64 words apart and the four lanes' first reads all fall in
bank 0; the model's lanes are channels of their own, so after that first collision (3 conflicts
on bank 0) they run a cycle apart each and never meet again. Under padded_rows the rows are 65
words apart, the lanes in four banks from the first beat. The counters are checked by D171 (each
within 10 % of the estimate's); as recorded in CORE1 R1 on one core: worked 9, contiguous 129
(the first collision falls in the streamer's head start, its START 10 cycles after the streamer's
start, D155; 134 on two cores), padded 131.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.flow import run_flow
from tests.design.helpers import SNAX_FORGE
from tests.helpers import recipe_path

from .helpers import needs_gvsoc, sound

pytestmark = [pytest.mark.gvsoc, needs_gvsoc]


@pytest.fixture
def worked(monkeypatch):
    """The kernel's inputs as the worked example's."""
    from snax_forge.dfg import reference

    def inputs(spec, graph, seed):
        return {"A": np.arange(1, 13, dtype=np.int64).reshape(4, 3), "y": np.zeros(4, np.int64)}

    monkeypatch.setattr(reference, "kernel_inputs", inputs)


def summary(f) -> str:
    return f"{f.name}: {f.check}"


def test_the_worked_example(tmp_path, worked):
    f = run_flow(recipe_path("rowsum_2"), SNAX_FORGE, recipe_sets={"W": 2, "M": 4, "N": 3},
                 out=tmp_path / "rowsum_2")  # fmt: skip
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and (f.check["data_errors"], f.check["words"]) == (0, 4)
    sound(f)
    assert {t: (x["beats"], x["group"]) for t, x in f.check["tasks"].items()} == {
        "sum_acc": (6, "acc")
    }
    data = (tmp_path / "rowsum_2" / "gen" / "data.h").read_text()
    golden = data.split(" y_golden[", 1)[1].split("{", 1)[1].split("}", 1)[0]
    assert [int(v) for v in golden.replace(",", " ").split()][:4] == [6, 15, 24, 33]


@pytest.mark.parametrize(
    "memory_sets", [(), (("passes.layout", "padded_rows"),)], ids=["contiguous", "padded_rows"]
)  # fmt: skip
def test_rowsum_2_at_its_defaults(tmp_path, memory_sets):
    f = run_flow(recipe_path("rowsum_2"), SNAX_FORGE, memory_sets=memory_sets,
                 out=tmp_path / "rowsum_2")  # fmt: skip
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and (f.check["data_errors"], f.check["words"]) == (0, 8)
    sound(f)
    assert {t: (x["beats"], x["group"]) for t, x in f.check["tasks"].items()} == {
        "sum_acc": (128, "acc")
    }
