"""run/sources.py: a trace level and its sources, and what SNAX-GVSoC is told for them."""

import pytest

from snax_forge.run import RunError
from snax_forge.run.sources import parse_trace

LANES = ",".join(f"snax_acc0.streamer_acc.{m}.lane{i}" for m in ("reader0", "reader1", "writer0")
                 for i in range(4))  # fmt: skip
XBAR = "chip/soc/cluster_0/tcdm/snax_xbar/trace_inputs="


def test_a_level_and_its_sources():
    assert parse_trace("task") == ("task", None)
    assert parse_trace("stall:acc,acc_a") == ("stall", ["acc", "acc_a"])
    for text, code in (("cycle", "sources.level"), ("task:acc", "sources.given"),
                       ("beat:", "sources.given")):  # fmt: skip
        with pytest.raises(RunError) as e:
            parse_trace(text)
        assert e.value.code == code
