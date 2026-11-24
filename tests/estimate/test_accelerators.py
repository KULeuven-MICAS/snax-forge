"""The estimate's accelerators: one streamer set each, its groups, CSRs and streamer,
the same numbers the generator writes (gen/csrs.py's layout), the streamer not costed."""

from __future__ import annotations

import json

from snax_forge.estimate import Estimate, estimate
from tests.helpers import steps

from .helpers import estimated


def test_without_a_platform_no_csrs_and_round_trip():
    e = estimate(steps("dot_v1")[-1], None, None)
    assert [(a.name, a.csrs, a.streamer.csrs, a.streamer.movers) for a in e.accelerators] == [
        ("vmul", None, None, 3), ("vsum", None, None, 2)]  # fmt: skip
    assert e.csrs is None and e.to_dict()["accelerators"][0]["cost"] is None
    for x in (e, estimated("dot_v1")):
        d = x.to_dict()
        assert Estimate.from_dict(json.loads(json.dumps(d))).to_dict() == d
        assert all("cost" not in a["streamer"] for a in d["accelerators"])
