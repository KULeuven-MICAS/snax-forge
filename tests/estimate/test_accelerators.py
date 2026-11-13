"""The estimate's accelerators: one streamer set each, its groups, CSRs and streamer,
the same numbers the generator writes (gen/csrs.py's layout), the streamer not costed."""

from __future__ import annotations

import json

import pytest

from snax_forge.estimate import Estimate, estimate
from snax_forge.estimate.cost import caveat
from tests.helpers import steps

from .helpers import estimated


def test_dot_1_two_streamers_40_csrs_dot_1_enc_one_and_37():
    """dot 1: vmul 960-981, vsum 982-999; dot_1_enc: one set dot, 960-996."""
    one, enc = estimated("dot_1"), estimated("dot_1_enc")
    assert (one.streamers, one.csrs, enc.streamers, enc.csrs) == (2, 40, 1, 37)
    assert [(a.name, a.groups, a.csrs, a.own_csrs, a.streamer.csrs) for a in one.accelerators] == [
        ("vmul", ["vmul"], (960, 981), 4, 18), ("vsum", ["vsum"], (982, 999), 5, 13)]  # fmt: skip
    (dot,) = enc.accelerators
    assert (dot.name, dot.groups, dot.csrs, dot.streamer.movers, dot.streamer.channels,
            dot.streamer.temporal_dims) == ("dot", ["vmul", "vsum"], (960, 996), 5, 17,
                                            [1, 1, 1, 1, 1])  # fmt: skip
    # every port's streamer is costed under its accelerator: vmul a, b, out; vsum in, out
    costed = [[x["name"] for x in a["cost"]["streamers"]] for a in one.to_dict()["accelerators"]]
    assert [len(x) for x in costed] == [3, 2]


@pytest.mark.parametrize(
    "name", ["vecadd_i32", "vsum_tree_i8", "dot_1", "dot_1_enc", "dot_4", "matvec_1", "rowsum"]
)
def test_the_csrs_are_the_generators(tmp_path, name):
    """Each accelerator's CSRs, first to last, as the generator lays the sets out."""
    from snax_forge.gen import generate
    from tests.gen.helpers import point as gen_point

    path = gen_point(tmp_path, name=name, graph=steps(name)[-1])
    generate(path, tmp_path / "gen", tmp_path)
    manifest = json.loads((tmp_path / "gen" / "manifest.json").read_text())
    want = [(s["name"], s["first"]) for s in manifest["csrs"]["streamers"]]
    last = max(manifest["csrs"]["accelerator"].values())
    e = estimated(name)
    assert [(a.name, a.csrs[0]) for a in e.accelerators] == want
    assert e.accelerators[-1].csrs[1] == last
    assert [s["last"] - s["first"] + 1 for s in manifest["csrs"]["streamers"]] == [
        a.streamer.csrs for a in e.accelerators]  # fmt: skip


def test_without_a_platform_no_csrs_and_round_trip():
    e = estimate(steps("dot_1")[-1], None, None)
    assert [(a.name, a.csrs, a.streamer.csrs, a.streamer.movers) for a in e.accelerators] == [
        ("vmul", None, None, 3), ("vsum", None, None, 2)]  # fmt: skip
    assert e.csrs is None and e.to_dict()["accelerators"][0]["cost"] is None
    for x in (e, estimated("dot_1")):
        d = x.to_dict()
        assert Estimate.from_dict(json.loads(json.dumps(d))).to_dict() == d
        assert all("cost" not in a["streamer"] for a in d["accelerators"])


def test_the_caveat_states_the_aim_and_what_is_left_out():
    """About 30 %, the control unit's comb area a rough estimate, the cores, I-cache, DMA
    and csr left out."""
    c = caveat()
    assert "about 30 %" in c and "comb area" in c
    assert all(w in c for w in ("csr", "cores", "I-cache", "DMA"))
