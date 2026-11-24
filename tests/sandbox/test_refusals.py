"""The sandbox package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from snax_forge.brm.function import mac_kind
from snax_forge.dfg import DfgError, Graph
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import (
    Recipe,
    SandboxError,
    apply_recipe,
    bind,
    fuse_map,
    split_map,
    split_reduce,
    unencapsulate,
    unfuse_seed,
)
from snax_forge.sandbox.blocks import apply_recipe as apply_block
from snax_forge.sandbox.blocks import lift, matches, unlift
from snax_forge.sandbox.seed import carry_seed, uncarry_seed
from snax_forge.sandbox.transforms import tag_map
from tests.composite.axpy_v1.test_sandbox import BIND, scaled
from tests.composite.axpy_v1.test_sandbox import fused as axpy_fused
from tests.composite.bicg_v4_1.test_carry import _carried
from tests.composite.matvec_acc_v2.test_seed import FOLD
from tests.composite.matvec_acc_v2.test_seed import matvec_acc as seed_matvec_acc
from tests.helpers import graph_dict, imported, recipe_path, steps
from tests.sandbox.helpers import graph, mvt, recipe
from tests.sandbox.test_blocks import _paths
from tests.sandbox.test_compose import fused as compose_fused
from tests.sandbox.test_compose import matvec_acc as compose_matvec_acc
from tests.sandbox.test_encapsulate import _set, dot
from tests.sandbox.test_interchange import run
from tests.sandbox.test_lane_accumulate import LANE, interchanged, tagged


def _a_non_scalar_on_a_slot_is_refused_naming_it():
    d = graph_dict("relu")
    d["symbols"]["N"] = 8
    t = d["body"][0]["body"][0]
    t["inputs"]["in2"] = {"data": "A", "subset": [0]}
    t["attrs"]["code"] = "out = max(in1, in2)"
    g = split_map(Graph.from_dict(d), "numpy_maximum_map", 4)
    bind(g, node="numpy_maximum", **BIND)


def _a_slot_given_as_a_param_and_read_from_a_scalar_is_refused():
    bind(axpy_fused(), node="mult_map_s", **BIND, params={"k1": 3})


def _a_slot_given_that_the_matched_entry_leaves_unset_is_refused():
    bind(axpy_fused(), node="mult_map_s", **BIND, params={"k2": 2})


def _a_scalar_wider_than_the_op_type_is_refused():
    bind(scaled(beta_dtype="int64"), node="mult", **BIND)


def _a_bank_that_does_not_match_the_loops_is_refused_naming_the_split():
    g = steps("matvec_v8")[-2]
    bind(g, "mult_map_inner_s", "mac", "chisel_mac", "mac", {"rows": 4})


def _a_seeded_bank_starts_each_row_from_its_seed_on_the_first_rows_firings():
    f = mac_kind(lanes=2, seed=1, rows=2)
    rate = f.rates({"n": 8, "T": 4})
    assert [f.due(k, rate, "in") for k in range(4)] == [["a", "b", "seed"]] * 2 + [["a", "b"]] * 2
    state, got = ({}, [])
    a, b = (np.array([1, 2]), np.array([3, 3]))
    for k, seed in enumerate(([10, 20], [30, 40], None, None)):
        ins = {"a": a, "b": b} | ({"seed": np.array(seed)} if seed else {})
        got.append(f.fn(k, ins, state, {"T": 4}).get("out"))
    assert got[:2] == [None, None]
    assert [list(x) for x in got[2:]] == [[16, 32], [36, 52]]
    f.fn(0, {"a": a, "b": b, "seed": a}, {}, {"T": 3})


def _a_kernel_not_in_the_graph_is_refused():
    assert matches(mvt(), "vecadd") == []
    lift(mvt(), "vecadd")


def _unlift_gives_the_graph_back():
    g = mvt()
    lifted = lift(lift(g, "matvec_acc"), "dot")
    assert unlift(unlift(lifted, "dot"), "dot_1") == lift(g, "matvec_acc")
    unlift(lifted, "mvt")


def _apply_recipe_refuses_a_param_the_sub_recipe_lacks():
    apply_block(mvt(), "dot/dot_v3", region=["tmp0_mult_map_1", "sum_map_1"], params={"K": 4})


def _lift_offers_the_kernels_matching_at_a_node_and_lifts_that_region_only():
    from snax_forge.sandbox.blocks import lift_choices

    g = mvt()
    assert lift_choices(g, "tmp2_mult") == ["dot", "matvec", "matvec_acc", "vecmul"]
    assert lift_choices(g, "augassign_19_4") == ["matvec_acc"]
    one = lift(g, "dot", at="tmp2_mult")
    assert {b for p in _paths(one).values() for b in p} == {"dot"}
    assert _paths(one)["sum_map_3"] == ["dot"]
    lift(g, "vecadd", at="tmp0_mult")


def _apply_recipe_refuses_a_param_the_region_sets():
    r = Recipe.from_dict(
        {
            "name": "t",
            "kernel": "matvec",
            "params": {"W": 4, "M": 8, "N": 64},
            "symbols": {"M": "M", "N": "N"},
            "steps": [
                {"transform": "tag_map", "params": {"map": "mult_map", "kind": "temporal"}},
                {"transform": "tag_map", "params": {"map": "sum_map", "kind": "temporal"}},
                {
                    "transform": "apply_recipe",
                    "params": {
                        "recipe": "dot/dot_v1",
                        "region": ["mult_map_1", "sum_map_1"],
                        "params": {"W": "W", "N": 16},
                    },
                },
            ],
        }
    )
    apply_recipe(r, imported("matvec"))


def _uncarry_seed_undoes_carry_seed():
    gs, k = _carried()
    before = gs[k - 1]
    assert uncarry_seed(carry_seed(before, "sum_map_1"), "sum_map_1").to_json() == before.to_json()
    carry_seed(gs[k], "sum_map_1")


def _a_local_value_may_not_also_be_a_live_container():
    d = bind(compose_fused(), "mult_map_s", "mac", "chisel_mac", "mac").to_dict()
    d["containers"]["tmp0"] = d["body"][0]["body"][0]["attrs"]["locals"][0]["container"]
    Graph.from_dict(d)


def _temporal_against_spatial_is_refused():
    from snax_forge.sandbox.transforms import tag_map

    g = tag_map(tag_map(compose_matvec_acc(), "mult_map", "temporal"), "sum_map", "spatial")
    fuse_map(g, "mult_map", "sum_map")


def _fuse_map_names_lanes_on_two_dimensions():
    from snax_forge.sandbox import Recipe, apply_recipe
    from tests.helpers import imported, recipe_path

    d = json.loads(recipe_path("matvec_v7").read_text())
    d["steps"] = d["steps"][:3] + [
        {"transform": "tag_map", "params": {"map": "sum_map", "kind": "temporal"}},
        {"transform": "split_map", "params": {"map": "sum_map_1", "factor": "W"}},
        {"transform": "fuse_map", "params": {"first": "mult_map", "second": "sum_map"}},
    ]
    apply_recipe(Recipe.from_dict(d), imported("matvec"))


def _a_narrowing_cast_is_refused_at_bind():
    recipe = Recipe.load(recipe_path("vecmul_v1")).with_params(
        {"dtypes.A": "int32", "dtypes.B": "int32", "dtypes.C": "int32", "dtypes.OP_TYPE": "int8"}
    )
    g = import_kernel(recipe.kernel, recipe.dtypes)
    assert g.node("mult").attrs["code"] == "out = int8(in1) * int8(in2)"
    apply_recipe(recipe, g)


def _unencapsulate_refuses_an_unknown_accelerator():
    unencapsulate(dot(), "dot")


def _the_graph_check_refuses_an_instance_in_two_accelerators():
    d = graph_dict("dot_accelerated")
    _set(d, "mul", "dot")
    second = copy.deepcopy(d["body"][0])
    second["id"] = "mult_map2"
    second["body"][0]["id"] = "mult2"
    second["body"][0]["attrs"]["accelerator"] = "other"
    d["body"].append(second)
    Graph.from_dict(d)


def _the_graph_check_refuses_an_accelerator_named_as_an_instance():
    d = graph_dict("dot_accelerated")
    _set(d, "mul", "sum")
    Graph.from_dict(d)


def _the_graph_check_refuses_an_accelerator_that_is_no_name():
    d = graph_dict("dot_accelerated")
    _set(d, "mul", "a-b")
    Graph.from_dict(d)


def _the_bind_site_names_a_spatial_tag():
    *_, g = run(
        [
            {"transform": "tag_map", "params": {"map": "sum_map", "kind": "spatial"}},
            {"transform": "tag_map", "params": {"map": "sum_map_1", "kind": "spatial"}},
        ],
        M=4,
        N=4,
    )
    from snax_forge.sandbox import bind

    bind(g, "sum", "accumulate", "chisel_adder_tree", "acc", {"W": 4})


def _refused_the_tasklet_instead_of_its_map():
    bind(interchanged(), "sum", **LANE)


def _refused_a_front_op_in_the_tasklet():
    d = copy.deepcopy(interchanged().to_dict())
    tasklet = d["body"][0]["body"][0]["body"][0]["body"][0]
    assert tasklet["id"] == "sum"
    tasklet["attrs"]["code"] = "out = 2 * in1"
    bind(Graph.from_dict(d), "sum_map_s", **LANE)


def _refused_the_reduced_loop_spatial():
    g = tagged({"sum_map": "temporal", "sum_map_1": "spatial"})
    bind(g, "sum_map_1", **LANE | {"params": {"W": 4}})


def _refused_an_output_not_indexed_by_the_lane():
    d = copy.deepcopy(tagged({"sum_map": "temporal", "sum_map_1": "temporal"}).to_dict())
    col = d["body"][0]["body"][0]
    col["body"] = [
        {
            "id": "lane_map",
            "kind": "map",
            "attrs": {"var": "l", "range": "0:4", "loop.kind": "spatial"},
            "body": col["body"],
        }
    ]
    bind(Graph.from_dict(d), "lane_map", **LANE)


def _split_reduce_refuses_a_tagged_map_by_its_own_name():
    g = tag_map(graph("dot"), "sum_map", "temporal")
    split_reduce(g, "sum_map", 4)


def _a_bound_symbol_must_agree():
    d = graph("vecadd").to_dict()
    d["symbols"]["N"] = 32
    apply_recipe(recipe(), Graph.from_dict(d))


def _unfuse_seed_refuses_a_fold_not_seeded():
    unfuse_seed(seed_matvec_acc(), FOLD)


CASES = [
    pytest.param(_a_non_scalar_on_a_slot_is_refused_naming_it, SandboxError, "in2 reads A\\[0\\], the same element in every firing; a slot takes a kernel's scalar argument \\(sdfg.scalar\\), and A is not one", id='a_non_scalar_on_a_slot_is_refused_naming_it'),
    pytest.param(_a_slot_given_as_a_param_and_read_from_a_scalar_is_refused, SandboxError, "k1 = 3 given, but the tasklet computes 'out = b \\+ alpha \\* a', 'simd''s 'out = k1 \\* a \\+ b' with its slot k1 from the scalar alpha", id='a_slot_given_as_a_param_and_read_from_a_scalar_is_refused'),
    pytest.param(_a_slot_given_that_the_matched_entry_leaves_unset_is_refused, SandboxError, "k2 = 2 given, but the tasklet computes 'out = b \\+ alpha \\* a', 'simd''s 'out = k1 \\* a \\+ b', which leaves k2 unset", id='a_slot_given_that_the_matched_entry_leaves_unset_is_refused'),
    pytest.param(_a_scalar_wider_than_the_op_type_is_refused, SandboxError, 'k2 reads the scalar beta, int64, which the op type int32 does not hold', id='a_scalar_wider_than_the_op_type_is_refused'),
    pytest.param(_a_bank_that_does_not_match_the_loops_is_refused_naming_the_split, (SandboxError, DfgError), 'split its loop by 4 times the lanes', id='a_bank_that_does_not_match_the_loops_is_refused_naming_the_split'),
    pytest.param(_a_seeded_bank_starts_each_row_from_its_seed_on_the_first_rows_firings, ValueError, "not a multiple of the bank's 2 rows", id='a_seeded_bank_starts_each_row_from_its_seed_on_the_first_rows_firings'),
    pytest.param(_a_kernel_not_in_the_graph_is_refused, SandboxError, 'no region of the graph matches vecadd', id='a_kernel_not_in_the_graph_is_refused'),
    pytest.param(_unlift_gives_the_graph_back, SandboxError, 'neither a block nor a node in one', id='unlift_gives_the_graph_back'),
    pytest.param(_apply_recipe_refuses_a_param_the_sub_recipe_lacks, SandboxError, "dot_v3 has no param 'K'", id='apply_recipe_refuses_a_param_the_sub_recipe_lacks'),
    pytest.param(_lift_offers_the_kernels_matching_at_a_node_and_lifts_that_region_only, SandboxError, "lift at 'tmp0_mult': vecadd does not match here", id='lift_offers_the_kernels_matching_at_a_node_and_lifts_that_region_only'),
    pytest.param(_apply_recipe_refuses_a_param_the_region_sets, SandboxError, "N is the region's \\(64\\): leave it out of params", id='apply_recipe_refuses_a_param_the_region_sets'),
    pytest.param(_uncarry_seed_undoes_carry_seed, SandboxError, 'carried already', id='uncarry_seed_undoes_carry_seed'),
    pytest.param(_a_local_value_may_not_also_be_a_live_container, DfgError, "'tmp0' is a container of the graph too", id='a_local_value_may_not_also_be_a_live_container'),
    pytest.param(_temporal_against_spatial_is_refused, SandboxError, "fuse_map 'mult_map' 'sum_map': mult_map is temporal, sum_map spatial; fused nests have the same tags", id='temporal_against_spatial_is_refused'),
    pytest.param(_fuse_map_names_lanes_on_two_dimensions, SandboxError, "mult_map_s puts the lanes on tmp0's dimension 0 \\(m\\), sum_map_1_s on its dimension 1 \\(n\\); .* split the map over n in the first nest, or over m in the second", id='fuse_map_names_lanes_on_two_dimensions'),
    pytest.param(_a_narrowing_cast_is_refused_at_bind, SandboxError, 'the cast int8\\(in1\\) of int32 does not hold every value of its input; the hardware only extends', id='a_narrowing_cast_is_refused_at_bind'),
    pytest.param(_unencapsulate_refuses_an_unknown_accelerator, SandboxError, 'no accelerator of that name', id='unencapsulate_refuses_an_unknown_accelerator'),
    pytest.param(_the_graph_check_refuses_an_instance_in_two_accelerators, DfgError, "instance 'mul' is in accelerator 'dot' elsewhere", id='the_graph_check_refuses_an_instance_in_two_accelerators'),
    pytest.param(_the_graph_check_refuses_an_accelerator_named_as_an_instance, DfgError, "accelerator 'sum' is also an instance's name", id='the_graph_check_refuses_an_accelerator_named_as_an_instance'),
    pytest.param(_the_graph_check_refuses_an_accelerator_that_is_no_name, DfgError, 'not a valid name', id='the_graph_check_refuses_an_accelerator_that_is_no_name'),
    pytest.param(_the_bind_site_names_a_spatial_tag, SandboxError, "map 'sum_map' around it is spatial; the maps around the spatial map are its firing loops, temporal: untag_map it, then tag_map it temporal", id='the_bind_site_names_a_spatial_tag'),
    pytest.param(_refused_the_tasklet_instead_of_its_map, SandboxError, 'binds the spatial map around a folding tasklet, not the tasklet: bind its map', id='refused_the_tasklet_instead_of_its_map'),
    pytest.param(_refused_a_front_op_in_the_tasklet, SandboxError, "sum computes '2 \\* in1' before the fold, a front op; a lane_accumulate folds its one input as it is: bind mac", id='refused_a_front_op_in_the_tasklet'),
    pytest.param(_refused_the_reduced_loop_spatial, SandboxError, "sum folds over 'sum_map_1' \\(n_s\\), which is spatial; a lane_accumulate folds each lane over the temporal maps around it, a fold across the lanes binds accumulate", id='refused_the_reduced_loop_spatial'),
    pytest.param(_refused_an_output_not_indexed_by_the_lane, SandboxError, "sum's output y\\['m_t'\\] is not indexed by the lane variable 'l'; each lane folds its own element", id='refused_an_output_not_indexed_by_the_lane'),
    pytest.param(_split_reduce_refuses_a_tagged_map_by_its_own_name, SandboxError, "^split_reduce 'sum_map': already temporal$", id='split_reduce_refuses_a_tagged_map_by_its_own_name'),
    pytest.param(_a_bound_symbol_must_agree, SandboxError, 'N is bound to 32 already, not 64', id='a_bound_symbol_must_agree'),
    pytest.param(_unfuse_seed_refuses_a_fold_not_seeded, SandboxError, "sum's fold is not seeded by fuse_seed", id='unfuse_seed_refuses_a_fold_not_seeded'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
