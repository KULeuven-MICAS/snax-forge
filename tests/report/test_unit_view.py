"""report/design.py's unit_view: a group of chains on a forked input (D189) in one line."""

import json
from pathlib import Path

from snax_forge.report.design import unit_view

FIXTURE = Path(__file__).parents[2] / "gvsoc/snax-gvsoc/forge/bicg_v4_1/accelerator.json"


def test_each_chain_in_order_the_longest_chains_latency():
    file = json.loads(FIXTURE.read_text())
    kind, control, latency, ii = unit_view(file, "bicg_types")
    assert kind == "SimdLanes →[1] TreeAccumulate ‖ Mac"
    assert (control, ii) == ("group control unit", 1)
    units = {u["name"]: u["timing"]["latency"] for g in file["accelerators"][0]["groups"]
             for u in g["units"]}  # fmt: skip
    assert latency == max(units["mul"] + units["sum"], units["mac"])
