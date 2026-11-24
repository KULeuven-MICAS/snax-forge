"""The dfg package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

import copy

import pytest
from dace.libraries.standard import Transpose
from dace.sdfg import nodes as dn

from snax_forge.dfg import DfgError, Graph, register_kind
from snax_forge.dfg.execute import ExecutionError, execute
from snax_forge.dfg.import_sdfg import (
    SdfgImportError,
    import_kernel,
    import_sdfg,
)
from snax_forge.dfg.pad import pad_value
from snax_forge.expr import ExprError
from snax_forge.sdfg.expand import ExpandError, expand
from tests.dfg import sdfg_programs as progs
from tests.dfg.helpers import as_dict
from tests.dfg.test_execute import accel, kernel_run
from tests.dfg.test_expand import _nodes
from tests.dfg.test_tail import _vsum


def _the_node_code_must_be_the_brms():
    d = as_dict("vecadd_accelerated")
    accel(d)["attrs"]["code"] = "out = a - b"
    execute(Graph.from_dict(d), kernel_run(64)[0])


def _vec_vec_is_refused():
    expand(progs.vecvec.to_sdfg(simplify=True))


def _shapes_that_do_not_match_are_refused():
    expand(progs.unsupported_matmul_shapes())


def _a_transpose_alone_is_left_for_the_importer():
    sdfg = expand(progs.transpose_only.to_sdfg(simplify=True))
    assert [type(n) for n in _nodes(sdfg) if isinstance(n, dn.LibraryNode)] == [Transpose]
    import_sdfg(sdfg, "t")


def _type_params_are_written_only_when_set_and_checked():
    d = import_kernel("dot", None, "i8").to_dict()
    assert d["type_params"] == {"OP_TYPE": "int32"}
    assert "type_params" not in import_kernel("dot").to_dict()
    d["type_params"] = {"OP_TYPE": "float32"}
    Graph.from_dict(d)


def _a_kind_is_registered_once():
    register_kind("map")


def _readers_that_need_different_pads_are_refused():
    d = _vsum("int8", "max", -128).to_dict()
    second = copy.deepcopy(d["body"][0])
    second["id"], second["body"][0]["id"] = ("min_map", "smallest")
    second["body"][0]["outputs"]["out"].update(data="low", reduce={"op": "min", "identity": 127})
    d["containers"]["low"] = copy.deepcopy(d["containers"]["out"])
    d["body"].append(second)
    pad_value(Graph.from_dict(d), "A")


def _a_container_of_no_dims_is_refused():
    from snax_forge.dfg.graph import Container

    Container.from_dict({"shape": [], "dtype": "int32"}, "c")


def _a_graph_with_an_empty_name_is_refused():
    from snax_forge.dfg import Graph

    Graph.from_dict({"name": ""})


def _a_dim_that_is_not_text_is_refused():
    from snax_forge.dfg.subset import parse_dim

    parse_dim(1.5, "d")


def _a_dim_of_four_parts_is_refused():
    from snax_forge.dfg.subset import parse_dim

    parse_dim("0:1:1:1", "d")


def _a_dim_without_an_end_is_refused():
    from snax_forge.dfg.subset import parse_dim

    parse_dim("0::1", "d")


def _an_empty_seed_is_refused():
    from snax_forge.dfg.reduce import Seed

    Seed.from_dict([], "s")


CASES = [
    pytest.param(_the_node_code_must_be_the_brms, ExecutionError, "'out = a - b', but 'simd' computes", id='the_node_code_must_be_the_brms'),
    pytest.param(_vec_vec_is_refused, ExpandError, 'a \\[N\\] @ b \\[N\\] is not expanded, a has rank 1', id='vec_vec_is_refused'),
    pytest.param(_shapes_that_do_not_match_are_refused, ExpandError, 'a \\[4, 6\\] @ b \\[5\\] do not match', id='shapes_that_do_not_match_are_refused'),
    pytest.param(_a_transpose_alone_is_left_for_the_importer, SdfgImportError, 'library node Transpose is not imported', id='a_transpose_alone_is_left_for_the_importer'),
    pytest.param(_type_params_are_written_only_when_set_and_checked, DfgError, 'type_params.OP_TYPE: must be an integer dtype', id='type_params_are_written_only_when_set_and_checked'),
    pytest.param(_a_kind_is_registered_once, ValueError, 'already registered', id='a_kind_is_registered_once'),
    pytest.param(_readers_that_need_different_pads_are_refused, DfgError, 'container A: its readers need different pads: sum \\(max\\) need -128; smallest \\(min\\) need 127', id='readers_that_need_different_pads_are_refused'),
    pytest.param(_a_container_of_no_dims_is_refused, DfgError, 'c\\.shape: must be a non-empty list', id='a_container_of_no_dims_is_refused'),
    pytest.param(_a_graph_with_an_empty_name_is_refused, DfgError, 'graph name: must be a non-empty string', id='a_graph_with_an_empty_name_is_refused'),
    pytest.param(_a_dim_that_is_not_text_is_refused, ExprError, 'd: must be an int or a string, got 1.5', id='a_dim_that_is_not_text_is_refused'),
    pytest.param(_a_dim_of_four_parts_is_refused, ExprError, 'has more than three parts', id='a_dim_of_four_parts_is_refused'),
    pytest.param(_a_dim_without_an_end_is_refused, ExprError, 'has an empty part', id='a_dim_without_an_end_is_refused'),
    pytest.param(_an_empty_seed_is_refused, DfgError, 's: must be \\{data, subset\\}', id='an_empty_seed_is_refused'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
