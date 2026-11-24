"""vecadd_v1 walked once (one SimdLanes add at int64, the snax_hjson target's default): gen-gvsoc
writes its forge/ fixture and gen-rtl its RTL fixture, and each feature it carries for snax-
gvsoc and the RTL, one a test."""

from tests.composite.helpers import FORGE, RTL_FIXTURES, same_files
from tests.helpers import requires_submodule

pytestmark = requires_submodule


def test_gen_gvsoc_writes_the_forge_fixture(walked):
    same_files(walked.gen, FORGE / walked.design)


def test_gen_rtl_writes_the_rtl_fixture(walked):
    same_files(walked.rtl, RTL_FIXTURES / walked.design)


def test_the_estimate_has_a_group_per_group(walked):
    assert len(walked.estimate.groups) == len(walked.groups)
    assert walked.estimate.total_cycles > 0


def test_one_accelerator_of_one_unit(walked):
    (acc,) = walked.accelerator["accelerators"]
    assert (acc["name"], acc["inputs"], acc["outputs"]) == ("acc", ["acc.a", "acc.b"], ["acc.out"])
    assert [(u["datapath"], u["params"]) for u in walked.units.values()] == [
        ("SimdLanes", {"op": "add"})
    ]


def test_one_task_of_a_job_a_beat(walked):
    (task,) = walked.tasks.accel_steps()
    assert task.values == {"n": 16}
