"""share_ports and unshare_ports: two accelerators on one streamer whose movers
they share."""

from __future__ import annotations

import pytest

from snax_forge.design.streamers import SHARE_PORTS, accelerators
from snax_forge.sandbox import TRANSFORMS, SandboxError, encapsulate, share_ports, unshare_ports
from snax_forge.sandbox.share_ports import ShareSiteError
from snax_forge.sandbox.transforms import unencapsulate
from tests.helpers import steps


def dot1():
    return steps("dot_1", W=4)[-1]


def test_share_ports_puts_both_on_one_accelerator_and_back():
    """dot 1: vmul writes tmp0, vsum reads it; the site names the accelerator vmul_vsum, the
    marker records the units it added, and unshare_ports gives the graph back."""
    g = dot1()
    assert TRANSFORMS["share_ports"].site(g, "tmp0") == {
        "container": "tmp0", "accelerator": "vmul_vsum"}  # fmt: skip
    shared = share_ports(g, "tmp0", "vmul_vsum")
    assert accelerators(shared) == {"vmul_vsum": ["vmul", "vsum"]}
    assert shared.containers["tmp0"].attrs[SHARE_PORTS] == {
        "accelerator": "vmul_vsum", "added": ["vmul", "vsum"]}  # fmt: skip
    assert TRANSFORMS["unshare_ports"].site(shared, "tmp0") == {"container": "tmp0"}
    assert unshare_ports(shared, "tmp0").to_dict() == g.to_dict()


def test_an_encapsulated_pair_keeps_its_accelerator():
    """dot_1_enc's dot: share_ports only marks the container, and unshare_ports leaves dot."""
    g = encapsulate(dot1(), "dot", ["vmul", "vsum"])
    assert TRANSFORMS["share_ports"].site(g, "tmp0")["accelerator"] == "dot"
    shared = share_ports(g, "tmp0", "dot")
    assert shared.containers["tmp0"].attrs[SHARE_PORTS]["added"] == []
    assert unshare_ports(shared, "tmp0").to_dict() == g.to_dict()
    with pytest.raises(SandboxError, match=r"share their movers on \['tmp0'\]"):
        unencapsulate(shared, "dot")
    with pytest.raises(ShareSiteError, match="in accelerator 'dot' already, not 'other'"):
        share_ports(g, "tmp0", "other")


@pytest.mark.parametrize(
    "graph, container, text",
    [
        (dot1, "nope", "no container of that name"),
        (dot1, "A", r"written by \[\]; both ends are bound units"),
        (dot1, "out", r"read by \[\]; both ends are bound units"),
        (lambda: steps("dot_2", W=4, N=512)[-1], "tmp0", "a link"),
        (lambda: share_ports(dot1(), "tmp0", "d"), "tmp0", "already shared"),
    ],
)  # fmt: skip
def test_refused_where_it_does_not_apply(graph, container, text):
    with pytest.raises(ShareSiteError, match=text) as e:
        TRANSFORMS["share_ports"].site(graph(), container)
    assert e.value.code == "share.site"


def test_names_another_instance_or_accelerator_are_refused():
    g = dot1()
    for name in ("vmul", "vsum"):
        with pytest.raises(ShareSiteError, match="names another instance"):
            share_ports(g, "tmp0", name)
    with pytest.raises(ShareSiteError, match="must be a name"):
        share_ports(g, "tmp0", "not a name")


def test_the_mvt_recipes_share_their_halves():
    """mvt_1_dot2_shared: the linked group mvt (both halves) and vadd on mvt_vadd;
    mvt_2_shared: each half's dot and vadd on its own shared set."""
    assert accelerators(steps("mvt_1_dot2_shared", W=8)[-1]) == {
        "mvt_vadd": ["vmul_x1", "vsum_x1", "vadd"]}  # fmt: skip
    assert accelerators(steps("mvt_2_shared", W_x1=8, W_x2=8)[-1]) == {
        "mvt_vadd_x1": ["vmul_x1", "vsum_x1", "vadd_x1"],
        "mvt_1_vadd_x2": ["vmul_x2", "vsum_x2", "vadd_x2"]}  # fmt: skip
