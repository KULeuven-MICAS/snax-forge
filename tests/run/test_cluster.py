"""run/cluster.py: the cluster as a run is named, from the description files (the slot's sets
form) and the design point."""

import json

import hjson
import pytest

from snax_forge.run import RunError, cluster_from, load_cluster

from .helpers import STREAMER_FILE, files


def test_movers_readers_first_named_by_port(cluster):
    a = cluster.accelerator
    assert [(m.name, m.gvsoc, m.port, m.write, m.lanes, m.fifo_depth, m.set)
            for m in a.movers] == [
        ("acc_a", "reader0", "a", False, 4, 8, "acc"),
        ("acc_b", "reader1", "b", False, 4, 8, "acc"),
        ("acc_out", "writer0", "out", True, 4, 8, "acc"),
    ]  # fmt: skip
    assert a.mover("writer0").ports == [f"acc_out.lane{i}" for i in range(4)]
    assert a.mover("reader1").fifo == "acc_b.fifo"
    with pytest.raises(RunError, match="reader2"):
        a.mover("reader2")


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        (lambda c, s, u, p, m: c["acc_core"].update(xdma=True), "cluster.cores"),  # 2 DMA cores
        (lambda c, s, u, p, m: p["streamers"].pop("acc_b"), "cluster.movers"),  # not the point's
        # the manifest's streamers: another range, no trace path, none at all
        (lambda c, s, u, p, m: m["csrs"]["streamers"][0].update(last=978), "cluster.manifest"),
        (lambda c, s, u, p, m: m["csrs"]["streamers"][0].pop("trace"), "cluster.manifest"),
        (lambda c, s, u, p, m: m.pop("csrs"), "cluster.manifest"),
    ],
)
def test_a_cluster_that_does_not_fit_is_refused(edit, code):
    cluster, streamer, unit, point, manifest = files()
    edit(cluster, streamer, unit, point, manifest)
    with pytest.raises(RunError) as e:
        cluster_from(cluster, streamer, unit, point, manifest)
    assert e.value.code == code


def test_load_from_files(tmp_path):
    cluster, streamer, unit, point, manifest = files()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "snax_forge.hjson").write_text(hjson.dumps(cluster))
    (tmp_path / STREAMER_FILE).write_text(hjson.dumps(streamer["acc"]))
    (tmp_path / "accelerator.json").write_text(json.dumps(unit))
    (tmp_path / "design_point.json").write_text(json.dumps(point))
    c = load_cluster(tmp_path / "snax_forge.hjson", tmp_path / "design_point.json")
    assert c.accelerator == cluster_from(*files()).accelerator
    assert c.meta == {"cluster": "snax_forge.hjson"}
    (tmp_path / "accelerator.json").unlink()
    with pytest.raises(RunError) as e:
        load_cluster(tmp_path / "snax_forge.hjson", tmp_path / "design_point.json")
    assert e.value.code == "cluster.ref"
    (tmp_path / "accelerator.json").write_text(json.dumps(unit))
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(RunError) as e:
        load_cluster(tmp_path / "snax_forge.hjson", tmp_path / "design_point.json")
    assert e.value.code == "cluster.manifest"


def test_the_old_one_streamer_slot_is_refused(tmp_path):
    """The slot in the form before the sets (snax_streamer_cfg, no snax_streamer_sets) does not
    read: cluster.sets."""
    cluster, templates, unit, point, _ = files()
    (slot,) = cluster["acc_core"]["snax_acc_cfg"]
    del slot["snax_streamer_sets"]
    slot["snax_streamer_cfg"] = {"$ref": STREAMER_FILE}
    (tmp_path / "snax_forge.hjson").write_text(hjson.dumps(cluster))
    (tmp_path / STREAMER_FILE).write_text(hjson.dumps(templates["acc"]))
    (tmp_path / "accelerator.json").write_text(json.dumps(unit))
    (tmp_path / "design_point.json").write_text(json.dumps(point))
    with pytest.raises(RunError) as e:
        load_cluster(tmp_path / "snax_forge.hjson", tmp_path / "design_point.json")
    assert e.value.code == "cluster.sets"
