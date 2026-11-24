"""run/ on a recorded run of one accelerator of two groups sharing one streamer's movers:
mvt_v1_2_2 at W = 8, M = N = 8, trace levels task and stall (fixtures/mvt_v1_2_2/README.md),
converted without GVSoC.

The groups ``mvt`` (``vmul_x1`` linked into ``vsum_x1``, one SNAX-GVSoC component,
``group_mvt``) and ``vadd`` share the set ``mvt_vadd``: mover k of each role carries the k-th
widest port of each group, so reader0 is ``vmul_x1_a`` and ``vadd_a``, reader1 ``vmul_x1_b`` and
``vadd_b``, writer0 ``vsum_x1_out`` (its first channel) and ``vadd_out`` (8 channels). Each of the
four tasks starts every mover of the set; the converter names each group's lines after it, gives
each a profile and matches each group's tasks in order.
"""

import json

import pytest

from snax_forge.run import Cluster, Unit
from snax_forge.run.convert import convert
from snax_forge.run.sources import all_sources
from tests.run.helpers import load_recorded

SET = "mvt_vadd"
TASKS = ["tmp0_mult_mvt", "augassign_18_4_vadd", "tmp2_mult_mvt", "augassign_19_4_vadd"]


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("mvt_v1_2_2")


@pytest.fixture(scope="module")
def converted(recorded):
    cluster, tasks, check, lines, regions, _ = recorded
    return convert(lines, cluster, tasks, check["total_cycles"], "task", check["functional_check"],
                   regions)  # fmt: skip


@pytest.fixture(scope="module")
def stalled():
    cluster, tasks, check, lines, regions, _ = load_recorded("mvt_v1_2_2", "events_stall.log")
    return cluster, regions, convert(lines, cluster, tasks, check["total_cycles"], "stall",
                                     check["functional_check"], regions, all_sources(cluster))  # fmt: skip


def test_the_cluster_has_a_unit_per_group_and_the_shared_movers(recorded):
    cluster, *_ = recorded
    a = cluster.accelerator
    assert (a.name, a.streamer, a.accelerators()) == (
        SET,
        f"{SET}_streamer",
        [(SET, ["mvt", "vadd"])],
    )
    first = a.csrs.sets[0].last + 1
    assert a.units == [Unit("mvt", "mvt", "group_mvt", first, first + 3, first + 4),
                       Unit("vadd", "vadd", "unit_vadd", first + 5, first + 7, first + 8)]  # fmt: skip
    assert [(m.name, m.gvsoc, m.set) for m in a.movers] == [
        (f"{SET}_reader0", "reader0", SET), (f"{SET}_reader1", "reader1", SET),
        (f"{SET}_writer0", "writer0", SET),
    ]  # fmt: skip
    assert [(m.users, m.user_units, m.user_channels) for m in a.movers] == [
        (("vmul_x1_a", "vadd_a"), ("vmul_x1", "vadd"), (8, 8)),
        (("vmul_x1_b", "vadd_b"), ("vmul_x1", "vadd"), (8, 8)),
        (("vsum_x1_out", "vadd_out"), ("vsum_x1", "vadd"), (1, 8)),
    ]
    movers = [m.name for m in a.movers]
    assert [m.name for m in a.movers_of("mvt")] == [m.name for m in a.movers_of("vadd")] == movers
    assert cluster.sources[:5] == ["mvt", "vadd", *movers]
    assert [a.block_of(first + k) for k in (0, 4, 5, 8, 9)] == ["mvt", "mvt", "vadd", "vadd", None]
    assert Cluster.from_dict(json.loads(json.dumps(cluster.to_dict()))) == cluster
