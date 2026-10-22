"""run/sources.py: a trace level and its sources, and what SNAX-GVSoC is told for them."""

import pytest

from snax_forge.run import RunError, cluster_from
from snax_forge.run.sources import gvsoc_options, parse_trace, sources_for

from .helpers import files

LANES = ",".join(f"snax_acc0.streamer.{m}.lane{i}" for m in ("reader0", "reader1", "writer0")
                 for i in range(4))  # fmt: skip
XBAR = "chip/soc/cluster_0/tcdm/snax_xbar/trace_inputs="


@pytest.fixture
def cluster():
    return cluster_from(*files())


def test_a_level_and_its_sources():
    assert parse_trace("task") == ("task", None)
    assert parse_trace("stall:acc,acc_a") == ("stall", ["acc", "acc_a"])
    for text, code in (("cycle", "sources.level"), ("task:acc", "sources.given"),
                       ("beat:", "sources.given")):  # fmt: skip
        with pytest.raises(RunError) as e:
            parse_trace(text)
        assert e.value.code == code


def test_the_default_sources(cluster):
    assert sources_for(cluster, "task", None) == []
    assert sources_for(cluster, "beat", None) == ["acc", "acc_a", "acc_b", "acc_out", "dma"]
    assert sources_for(cluster, "finegrained", None)[-2:] == ["core0", "core1"]
    assert sources_for(cluster, "stall", ["axi", "core1", "acc_b"]) == ["acc_b", "core1", "axi"]
    with pytest.raises(RunError, match=r"sources \['nope'\]: not of this run"):
        sources_for(cluster, "beat", ["nope"])


@pytest.mark.parametrize(
    "level, traces",
    [
        ("off", ["summary"]),
        ("task", ["summary", "events"]),
        ("beat", ["summary", "events", "snax_acc0/unit/beats", "snax_acc0/streamer/beats",
                  "tcdm/snax_xbar/beats",
                  "idma/axi_read/beats", "idma/axi_write/beats"]),
        ("stall", ["summary", "events", "snax_acc0/unit/beats", "snax_acc0/streamer/beats",
                  "tcdm/snax_xbar/beats",
                   "idma/axi_read/beats", "idma/axi_write/beats", "snax_acc0/streamer/stalls",
                   "tcdm/snax_xbar/stalls"]),
    ],
)  # fmt: skip
def test_the_traces_of_each_level(cluster, level, traces):
    o = gvsoc_options(cluster, level, sources_for(cluster, level, None), False)
    assert o.traces == traces
    assert o.args == ([] if level in ("off", "task") else ["--param", f"{XBAR}{LANES},dma"])


def test_finegrained_adds_the_cores_instructions_only(cluster):
    o = gvsoc_options(cluster, "finegrained", sources_for(cluster, "finegrained", None), False)
    assert o.traces[-2:] == ["pe0/insn", "pe1/insn"]
    assert o.args == ["--param", f"{XBAR}{LANES},dma"]
    # named, a core's crossbar lines are written too
    named = gvsoc_options(cluster, "finegrained", ["core1"], False, given=True)
    assert named.traces == ["summary", "events", "tcdm/snax_xbar/beats", "tcdm/snax_xbar/stalls",
                            "pe1/insn"]  # fmt: skip
    assert named.args == ["--param", f"{XBAR}core1"]


def test_the_vcd_names_the_sources_components(cluster):
    o = gvsoc_options(cluster, "beat", ["acc", "acc_out", "dma"], True, given=True)
    assert o.args[2:] == ["--vcd", "--event=.*/snax_acc0/unit/.*",
                          "--event=.*/snax_acc0/streamer/.*", "--event=.*/tcdm/snax_xbar/.*",
                          "--event=.*/idma/fe/dma_busy"]  # fmt: skip
