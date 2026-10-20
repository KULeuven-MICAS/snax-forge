"""The generator's files against their schemas: snax-forge's own for the streamer and
accelerator files, always; snax_cluster's for the cluster file, where the submodule has it."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import hjson
import pytest
from jsonschema import Draft7Validator, Draft202012Validator, ValidationError

from snax_forge.brm import CONTROLS, WAITS
from snax_forge.gen import generate
from snax_forge.gen.schemas import NAMES, schema, validate

from .helpers import SOURCE, SUB, bound_vsum, point

CLUSTER_SCHEMA = SUB / "sw" / "docs" / "schema" / "snitch_cluster.schema.json"
needs_cluster_schema = pytest.mark.skipif(
    not CLUSTER_SCHEMA.is_file(),
    reason="no snax_cluster schema in the snax-gvsoc submodule (not checked out, or its pin "
    "predates LNK2a round 2)",
)


def parsed(path: Path):
    return json.loads(json.dumps(hjson.loads(path.read_text())))


@pytest.fixture(scope="module", params=["vecadd", "vsum"])
def folder(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(request.param)
    graph = bound_vsum() if request.param == "vsum" else None
    out = tmp / "gvsoc" / request.param
    generate(point(tmp, name=request.param, graph=graph), out, root=tmp)
    return out


@pytest.mark.parametrize("name", NAMES)
def test_the_schemas_are_schemas(name):
    Draft202012Validator.check_schema(schema(name))


def test_the_writers_files_validate(folder):
    validate("streamer", parsed(folder / "snax_alu_streamer.hjson"))
    validate("accelerator", parsed(folder / "snax_alu_acc.hjson"))


@pytest.mark.parametrize(
    ("name", "file", "edit"),
    [
        ("streamer", "snax_alu_streamer.hjson", lambda d: d["data_reader_params"].pop("fifo_depth")),
        ("streamer", "snax_alu_streamer.hjson", lambda d: d.update(snax_library_name="snax-alu")),
        ("streamer", "snax_alu_streamer.hjson",
         lambda d: d["data_writer_params"].update(num_channel=[0])),
        ("streamer", "snax_alu_streamer.hjson", lambda d: d.clear()),
        ("accelerator", "snax_alu_acc.hjson", lambda d: d.pop("fields")),
        ("accelerator", "snax_alu_acc.hjson", lambda d: d.update(unit="SnaxAlu")),
        ("accelerator", "snax_alu_acc.hjson", lambda d: d.update(signed=1)),
        ("accelerator", "snax_alu_acc.hjson", lambda d: d.update(csr_base=978)),
        ("accelerator", "snax_alu_acc.hjson", lambda d: d.update(inputs=[])),
    ],
)  # fmt: skip
def test_a_broken_file_does_not_validate(folder, name, file, edit):
    d = copy.deepcopy(parsed(folder / file))
    edit(d)
    with pytest.raises(ValidationError):
        validate(name, d)


def test_the_kinds_are_the_registered_ones():
    """control and wait name what brm/binding.py registers (D129)."""
    props = schema("accelerator")["properties"]
    assert props["control"]["enum"] == list(CONTROLS)
    assert props["wait"]["enum"] == list(WAITS)


def _resolved(node, top, folder: Path):
    """The cluster file with its refs replaced, as clustergen will once it takes a base_uri:
    ``#/<key>`` from the file itself, ``<file>`` the whole file beside it."""
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            if ref.startswith("#/"):
                return _resolved(top[ref[2:]], top, folder)
            other = parsed(folder / ref)
            return _resolved(other, other, folder)
        return {k: _resolved(v, top, folder) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolved(v, top, folder) for v in node]
    return node


@needs_cluster_schema
def test_the_cluster_file_validates_against_snax_clusters_schema(folder):
    """clustergen's check (util/clustergen/cluster.py: Draft 7 on the cluster), on the file with
    its refs resolved; its snax_acc_cfg entries go unchecked (the schema's "items:" typo, snax-forge
    open item 51)."""
    validator = Draft7Validator(json.loads(CLUSTER_SCHEMA.read_text()))
    cluster = parsed(folder / "snax_alu.hjson")
    validator.validate(_resolved(cluster, cluster, folder)["cluster"])
    source = parsed(SOURCE)
    validator.validate(_resolved(source, source, SOURCE.parent)["cluster"])
    del cluster["cluster"]["addr_width"]  # the schema does check: a required key
    with pytest.raises(ValidationError):
        validator.validate(_resolved(cluster, cluster, folder)["cluster"])


@needs_cluster_schema
def test_the_cluster_file_is_the_source_outside_the_slot(folder):
    cluster, source = parsed(folder / "snax_alu.hjson"), parsed(SOURCE)
    for d in (cluster, source):
        d["snax_alu_core_template"]["snax_acc_cfg"][0].clear()
    del source["snax_alu_streamer_template"]
    assert cluster == source
