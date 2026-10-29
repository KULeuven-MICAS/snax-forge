"""The lane-fold pattern (D149, D150): ``mac`` binds a fused nest of a multiply and the fold of
its products, one partial per lane; a lone tasklet is refused with the fix named."""

from __future__ import annotations

import pytest

from snax_forge.brm import load_brm
from snax_forge.dfg import Memlet, Node, Reduction
from snax_forge.sandbox import SandboxError
from snax_forge.sandbox.patterns import fused_view, match

DTYPES = {"in1": "int64", "in2": "int64", "out": "int64"}


def fused(fold: str = "i_s", code: str = "out = in1 * in2", second: str = "out = in1") -> Node:
    """dot's fused nest at W = 4: mult writes the local tmp0, sum folds it into out_partial at
    ``fold``."""
    i = "4 * i_t + i_s"
    mult = Node("mult", "tasklet", inputs={"in1": Memlet("A", [i]), "in2": Memlet("B", [i])},
                outputs={"out": Memlet("tmp0", [i])}, attrs={"code": code})  # fmt: skip
    total = Node("sum", "tasklet", inputs={"in1": Memlet("tmp0", [i])},
                 outputs={"out": Memlet("out_partial", [fold], Reduction("add", 0))},
                 attrs={"code": second})  # fmt: skip
    return Node("mult_map_s", "map", attrs={"var": "i_s", "range": "0:4", "loop.kind": "spatial"},
                body=[mult, total])  # fmt: skip


def test_a_fused_nest_is_one_tasklet_to_bind():
    first, second, view = fused_view(fused())
    assert (first.id, second.id, view.id) == ("mult", "sum", "mult")
    assert view.attrs["code"] == "out = in1 * in2"
    assert sorted(view.inputs) == ["in1", "in2"] and view.outputs["out"].data == "out_partial"


def test_the_lane_fold_maps_the_nests_connectors():
    assert match(fused(), load_brm("mac"), DTYPES) == {"in1": "a", "in2": "b", "out": "out"}


@pytest.mark.parametrize(
    ("node", "text"),
    [
        (fused().body[0], "fuse_map first"),  # the multiply alone
        (fused().body[1], "split_reduce first"),  # the fold alone
        (fused(fold="0"), "split_reduce first"),  # fused, but folding into one element
        (fused(code="out = in1 + in2"), "is add of 2"),
        (fused(second="out = in1 + 1"), "not a copy"),
    ],
    ids=["multiply", "fold", "one-sum", "op", "copy"],
)
def test_the_lane_fold_refuses(node, text):
    with pytest.raises(SandboxError, match=text):
        match(node, load_brm("mac"), DTYPES)
