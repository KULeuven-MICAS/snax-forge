"""The generator's files against their schemas: snax-forge's own for the streamer and
accelerator files, always; snax_cluster's for the cluster file, where the submodule has it."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator, Draft202012Validator, ValidationError

from snax_forge.brm import CONTROLLERS, INT_DTYPES
from snax_forge.gen import generate
from snax_forge.gen.schemas import NAMES, schema
from tests.helpers import bound_vsum, requires_submodule

from .helpers import SOURCE, SUB, parsed, point, validate

CLUSTER_SCHEMA = SUB / "sw" / "docs" / "schema" / "snitch_cluster.schema.json"


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
    validate("streamer", parsed(folder / "snax_forge_streamer.hjson"))
    validate("accelerator", parsed(folder / "accelerator.json"))


@pytest.mark.parametrize(
    ("name", "file", "edit"),
    [
        ("streamer", "snax_forge_streamer.hjson", lambda d: d["data_reader_params"].pop("fifo_depth")),
        ("streamer", "snax_forge_streamer.hjson", lambda d: d.update(snax_library_name="snax-alu")),
        ("streamer", "snax_forge_streamer.hjson",
         lambda d: d["data_writer_params"].update(num_channel=[0])),
        ("streamer", "snax_forge_streamer.hjson", lambda d: d.clear()),
        ("accelerator", "accelerator.json", lambda d: d.pop("name")),
        ("accelerator", "accelerator.json", lambda d: d.update(fields=["loopCount"])),
        ("accelerator", "accelerator.json", lambda d: d.update(inputs=[])),
        ("accelerator", "accelerator.json", lambda d: d.update(inputs=["a"])),
        ("accelerator", "accelerator.json", lambda d: d["groups"][0].update(links=[{}])),
        ("accelerator", "accelerator.json", lambda d: d["groups"][0]["csrs"][2].update(width=32)),
        ("accelerator", "accelerator.json", lambda d: d["groups"][0]["csrs"][0].pop("width")),
        ("accelerator", "accelerator.json",
         lambda d: d["groups"][0]["units"][0]["ports"]["a"].update(dtype="float64")),
        ("accelerator", "accelerator.json",
         lambda d: d["groups"][0]["units"][0].update(wait="lockstep")),
    ],
)  # fmt: skip
def test_a_broken_file_does_not_validate(folder, name, file, edit):
    d = copy.deepcopy(parsed(folder / file))
    edit(d)
    with pytest.raises(ValidationError):
        validate(name, d)


def test_the_dtypes_are_the_brms():
    """A port's dtype and an opType are the BRM's integer dtypes (D143); counted is the only
    controller, so the file names none (D147)."""
    assert set(schema("accelerator")["$defs"]["dtype"]["enum"]) == set(INT_DTYPES)
    assert list(CONTROLLERS) == ["counted"]


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


@requires_submodule
def test_the_cluster_file_validates_against_snax_clusters_schema(folder):
    """clustergen's check (util/clustergen/cluster.py: Draft 7 on the cluster), on the file with
    its refs resolved, the snax_acc_cfg entries included."""
    validator = Draft7Validator(json.loads(CLUSTER_SCHEMA.read_text()))
    cluster = parsed(folder / "snax_forge.hjson")
    validator.validate(_resolved(cluster, cluster, folder)["cluster"])
    source = parsed(SOURCE)
    validator.validate(_resolved(source, source, SOURCE.parent)["cluster"])
    slot = copy.deepcopy(cluster)
    slot["snax_forge_core_template"]["snax_acc_cfg"][0]["snax_disable_csr_manager"] = "yes"
    with pytest.raises(ValidationError, match="'yes' is not of type 'boolean'"):
        validator.validate(_resolved(slot, slot, folder)["cluster"])
    del cluster["cluster"]["addr_width"]  # the schema does check: a required key
    with pytest.raises(ValidationError):
        validator.validate(_resolved(cluster, cluster, folder)["cluster"])


@requires_submodule
def test_the_cluster_file_is_the_source_outside_the_slot_and_the_clusters_name(folder):
    """The writer changes the slot, cluster.name and its entry in cluster.bender_target, which
    the source already names snax_forge_cluster."""
    cluster, source = parsed(folder / "snax_forge.hjson"), parsed(SOURCE)
    assert cluster["cluster"]["name"] == "snax_forge_cluster"
    assert cluster["cluster"]["bender_target"] == source["cluster"]["bender_target"]
    for d in (cluster, source):
        d["snax_forge_core_template"]["snax_acc_cfg"][0].clear()
        del d["cluster"]["name"], d["cluster"]["bender_target"]
    del source["snax_forge_streamer_template"]
    assert cluster == source
