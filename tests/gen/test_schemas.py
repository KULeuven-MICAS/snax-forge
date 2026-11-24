"""The generator's files against their schemas: snax-forge's own for the streamer and
accelerator files, always; snax_cluster's for the cluster file, where the submodule has it."""

from __future__ import annotations

from pathlib import Path

import pytest

from snax_forge.gen import generate
from tests.helpers import bound_vsum, requires_submodule

from .helpers import SOURCE, SUB, parsed, point

CLUSTER_SCHEMA = SUB / "sw" / "docs" / "schema" / "snitch_cluster.schema.json"


def G(d: dict) -> dict:  # a short name for the parametrize table
    """accelerator.json's first accelerator's first group."""
    return d["accelerators"][0]["groups"][0]


@pytest.fixture(scope="module", params=["vecadd", "vsum"])
def folder(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(request.param)
    graph = bound_vsum() if request.param == "vsum_v1" else None
    out = tmp / "gvsoc" / request.param
    generate(point(tmp, name=request.param, graph=graph), out, root=tmp)
    return out


STREAMER = "streamer_acc.hjson"  # the one set's file, after snax_forge_<design>_


def _streamer(folder: Path) -> Path:
    """The folder's one streamer file, ``snax_forge_<design>_streamer_acc.hjson``."""
    return folder / f"snax_forge_{folder.name}_{STREAMER}"


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
