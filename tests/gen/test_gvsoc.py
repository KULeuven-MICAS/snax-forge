"""The GVSoC generator (GEN3 step 3, D132): vecadd on snax_alu, and every refusal."""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import replace
from pathlib import Path

import hjson
import pytest

from snax_forge.brm import Brm, Instance, load_brm
from snax_forge.design import DesignPoint
from snax_forge.dfg import Graph
from snax_forge.gen import (
    BLOCKS,
    ELEMENTWISE_OP,
    IMPORTS,
    GenError,
    Manifest,
    check_install,
    generate,
)
from snax_forge.gen import __main__ as cli
from snax_forge.gen import gvsoc as gen_mod
from snax_forge.gen.blocks import elementwise_tiled_spatial
from tests.design.helpers import REPO, chained, graph_dict

from .helpers import HJSON, SUB, bound_vecmul, fake_install, point, snax_streamer

needs_submodule = pytest.mark.skipif(
    not HJSON.exists(), reason="the snax-gvsoc submodule is not checked out (CI never fetches it)"
)
IMPL = "chisel_tiled_spatial"

# vecadd at W = 4 on snax_alu: the template the design point's streamers give
TEMPLATE_W4 = {
    "data_reader_params": {
        "spatial_bounds": [[4], [4]], "temporal_dim": [1, 1], "num_channel": [4, 4],
        "fifo_depth": [8, 8],
    },
    "data_writer_params": {
        "spatial_bounds": [[4]], "temporal_dim": [1], "num_channel": [4], "fifo_depth": [8],
    },
}  # fmt: skip
# snax-gvsoc snax/snax_tiled_add_acc.py's SnaxUnit and SnaxShell, but signed: the
# hand file has False; int64 gives True (only min and max read it)
TILED_ADD_UNIT = {
    "inputs": ["a", "b"], "outputs": ["out"], "lanes": 4, "data_width": 64, "op": 0,
    "signed": True, "control": "counted", "wait": "lockstep", "latency": 0, "ii": 1,
    "csr_base": 978,
}  # fmt: skip
TILED_ADD_SHELL = {"csr_base": 978, "fields": ["loopCount"]}


@pytest.fixture(scope="module")
def w4(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w4")
    out = tmp / "gvsoc" / "vecadd"
    return out, generate(point(tmp), out, root=tmp)


def _calls(tree: ast.Module, name: str) -> dict:
    """The keyword arguments of the call to ``name`` in a generated file, as literals."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == name:
            return {k.arg: k.value for k in node.keywords}
    raise AssertionError(f"no call to {name}")


def _constants(tree: ast.Module) -> dict:
    return {
        t.id: ast.literal_eval(n.value)
        for n in tree.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)
    }  # fmt: skip


# =============================================================================
# vecadd at W = 4 on snax_alu
# =============================================================================


def test_vecadd_w4_files_and_manifest(w4):
    out, m = w4
    assert m.target == "forge_vecadd" and m.design == "vecadd"
    assert m.files == [
        "forge_vecadd_acc.py", "forge_vecadd.py", "include/streamer_csr_addr_map.h",
        "include/acc_csr.h", "manifest.json",
    ]  # fmt: skip
    assert sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()) == sorted(m.files)
    assert m.target_dirs == ["."] and m.trace == "snax_acc0/unit"
    assert m.design_point == "design/vecadd/design_point.json"
    assert m.symbols == {"N": 64}
    assert list(m.to_dict())[:3] == ["design", "design_point", "symbols"]
    assert (out.parent.parent / m.design_point).is_file()
    assert m.csrs.to_dict() == {
        "streamer": {"first": 960, "last": 977}, "fields": {"loopCount": 978},
        "start": 979, "busy": 980, "counter": 981,
    }  # fmt: skip
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "add_acc", "beats": 16, "csrs": {"loopCount": 16}}
    ]
    assert Manifest.load(out / "manifest.json") == m
    assert Manifest.from_dict(m.to_dict()) == m


def test_the_template_is_the_design_points_streamers(w4):
    out, _ = w4
    consts = _constants(ast.parse((out / "forge_vecadd_acc.py").read_text()))
    assert consts["STREAMER_TEMPLATE"] == TEMPLATE_W4
    assert (consts["TCDM_SIZE"], consts["SNAX_CSR_FIRST"], consts["SNAX_CSR_LAST"]) == (
        131072, 960, 0x5FF
    )  # fmt: skip
    assert consts["SHELL_CSR_FIRST"] == 978 and consts["SHELL_FIELDS"] == ["loopCount"]


@needs_submodule
def test_the_template_is_the_hjsons(w4):
    out, _ = w4
    consts = _constants(ast.parse((out / "forge_vecadd_acc.py").read_text()))
    hand = hjson.loads(HJSON.read_text())["snax_alu_streamer_template"]
    for group in ("data_reader_params", "data_writer_params"):
        assert consts["STREAMER_TEMPLATE"][group] == dict(hand[group])


@needs_submodule
def test_movers_csrs_and_port_names_are_snax_streamers(w4):
    """The generator's copies of snax_streamer.py's helpers give what the real ones give."""
    out, _ = w4
    tree = ast.parse((out / "forge_vecadd_acc.py").read_text())
    template = _constants(tree)["STREAMER_TEMPLATE"]
    ss = snax_streamer()
    assert gen_mod.streamer_csrs(template) == ss.nb_csrs(template) == 18
    assert gen_mod.tcdm_port_names(template) == [
        "streamer." + n for n in ss.tcdm_port_names(template)
    ]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    attrs = {t.id: ast.literal_eval(a.value) for a in cls.body if isinstance(a, ast.Assign)
             for t in a.targets}  # fmt: skip
    assert attrs["nb_tcdm_ports"] == ss.nb_tcdm_ports(template) == 12
    assert attrs["tcdm_port_names"] == gen_mod.tcdm_port_names(template)


def test_unit_and_shell_are_snax_tiled_adds_but_signed(w4):
    out, _ = w4
    tree = ast.parse((out / "forge_vecadd_acc.py").read_text())
    consts = _constants(tree)
    unit = {k: ast.literal_eval(v) if not isinstance(v, ast.Name) else consts[v.id]
            for k, v in _calls(tree, "SnaxUnit").items()}  # fmt: skip
    shell = {k: consts[v.id] for k, v in _calls(tree, "SnaxShell").items()}
    assert unit == TILED_ADD_UNIT and shell == TILED_ADD_SHELL


def test_generated_files_import_only_installed_components(w4):
    out, m = w4
    generated = {f.removesuffix(".py") for f in m.files if f.endswith(".py")}
    for f in generated:
        text = (out / f"{f}.py").read_text()
        tree = ast.parse(text)
        mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert mods <= set(IMPORTS) | generated, f
        assert f.startswith("forge_") and "snax_forge" not in mods
        assert "design point " in text and "snax-forge's GVSoC generator" in text


def test_the_target_puts_the_composite_in_snax_alus_slot(w4):
    out, _ = w4
    tree = ast.parse((out / "forge_vecadd.py").read_text())
    board = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith("Board"))
    assert [b.id for b in board.bases] == ["SnaxAluBoard"]
    assert ast.unparse(board.body[0]) == "accelerator = ForgeVecaddAcc"
    tgt = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Target")
    assert {ast.unparse(a) for a in tgt.body if isinstance(a, ast.Assign)} >= {
        "name = 'forge_vecadd'", "model = ForgeVecaddBoard"
    }  # fmt: skip


def test_vecadd_w8_gives_eight_channels_per_mover(tmp_path):
    m = generate(point(tmp_path, w=8, name="vecadd_w8"), tmp_path / "out", root=tmp_path)
    consts = _constants(ast.parse((tmp_path / "out" / "forge_vecadd_w8_acc.py").read_text()))
    assert consts["STREAMER_TEMPLATE"]["data_reader_params"]["num_channel"] == [8, 8]
    assert gen_mod.tcdm_port_names(consts["STREAMER_TEMPLATE"])[-1] == "streamer.writer0.lane7"
    assert m.csrs.start == 979 and m.tasks[0].beats == 8


def test_vecmul_is_vecadd_with_op_mul(tmp_path, w4):
    """GEN3 step 6a: elementwise_mul's binding gives vecadd's accelerator with op 2."""
    out = tmp_path / "gvsoc" / "vecmul"
    m = generate(point(tmp_path, name="vecmul", graph=bound_vecmul()), out, root=tmp_path)
    tree = ast.parse((out / "forge_vecmul_acc.py").read_text())
    unit = {k: ast.literal_eval(v) if not isinstance(v, ast.Name) else _constants(tree)[v.id]
            for k, v in _calls(tree, "SnaxUnit").items()}  # fmt: skip
    assert unit == {**TILED_ADD_UNIT, "op": 2}
    assert _constants(tree)["STREAMER_TEMPLATE"] == TEMPLATE_W4
    assert m.target == "forge_vecmul" and m.symbols == {"N": 64}
    assert m.design_point == "design/vecmul/design_point.json"
    assert m.files == [
        "forge_vecmul_acc.py", "forge_vecmul.py", "include/streamer_csr_addr_map.h",
        "include/acc_csr.h", "manifest.json",
    ]  # fmt: skip
    assert m.csrs == w4[1].csrs and m.trace == "snax_acc0/unit"
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "mult_acc", "beats": 16, "csrs": {"loopCount": 16}}
    ]
    for h in ("streamer_csr_addr_map.h", "acc_csr.h"):
        mine, add = (out / "include" / h).read_text(), (w4[0] / "include" / h).read_text()
        assert mine.replace("vecmul", "vecadd") == add, h


def test_elementwise_op_is_chisels():
    src = REPO / "hw/chisel/src/main/scala/snax/forge/elementwise/ElementwiseAlu.scala"
    body = src.read_text().split("object ElementwiseOp")[1].split("val width")[0]
    table = {k.lower(): int(v) for k, v in re.findall(r"val (\w+):\s*Int = (\d+)", body)}
    assert table == ELEMENTWISE_OP


def test_two_out_folders_give_byte_identical_files(tmp_path, monkeypatch):
    """Nothing written depends on the out folder or the working directory: the fixture diff."""
    path = point(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b" / "deeper" / "vecadd"
    ma = generate(path, a, root=tmp_path)
    monkeypatch.chdir(path.parent.parent)  # and the design point given relative to it
    mb = generate(Path("vecadd/design_point.json"), b, root=tmp_path)
    assert ma == mb
    for f in ma.files:
        text = (a / f).read_bytes()
        assert text == (b / f).read_bytes(), f
        assert str(tmp_path).encode() not in text, f
    for f in ma.files[:-1]:
        assert b"design point design/vecadd/design_point.json" in (a / f).read_bytes(), f


# =============================================================================
# Refusals
# =============================================================================


def test_a_design_point_outside_the_root(tmp_path):
    with pytest.raises(GenError) as e:
        generate(point(tmp_path / "elsewhere"), tmp_path / "out", root=tmp_path / "root")
    assert e.value.code == "gen.point" and not (tmp_path / "out").exists()


def _instance(edit=None, params=None) -> Instance:
    """elementwise_add's instance, its BRM dict edited (no resolve: the adapter's checks only)."""
    d = copy.deepcopy(load_brm("elementwise_add").to_dict())
    if edit:
        edit(d)
    brm = Brm.from_dict(d)
    return Instance(brm, IMPL, params or {"W": 4, "op": "add"})


def _binding(**kw):
    return lambda d: d["implementations"][IMPL]["binding"].update(kw)


def _ports(**kw):
    def edit(d):
        for p in d["interface"]["ports"]:
            p.update(kw.get(p["name"], {}))

    return edit


ADAPTER_CASES = [
    ("control", _binding(control="stream"), {}, "gen.control"),
    ("wait", _binding(wait="pipelined"), {}, "gen.wait"),
    ("latency", lambda d: d["implementations"][IMPL]["timing"].update(latency=1), {}, "gen.timing"),
    ("ii", lambda d: d["implementations"][IMPL]["timing"].update(initiation_interval=2), {},
     "gen.timing"),
    ("op", None, {"W": 4, "op": "nand"}, "gen.op"),
    ("args", _binding(args={"lanes": "W"}), {}, "gen.args"),
    ("mixed dtype", _ports(b={"dtype": "int32"}), {}, "gen.dtype"),
    ("float", _ports(a={"dtype": "float64"}, b={"dtype": "float64"}, out={"dtype": "float64"}),
     {}, "gen.dtype"),
    ("lanes x N", _ports(a={"dtype": "int8"}, b={"dtype": "int8"}, out={"dtype": "int8"}),
     {}, "gen.lanes"),
    ("lanes vs port", _binding(args={"lanes": "W * 2", "supportedOps": "op"}), {}, "gen.lanes"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("case", "edit", "params", "code"), ADAPTER_CASES, ids=[c[0] for c in ADAPTER_CASES]
)
def test_the_adapter_refuses(case, edit, params, code):
    with pytest.raises(GenError) as e:
        elementwise_tiled_spatial(_instance(edit, params or None))
    assert e.value.code == code


def test_the_adapter_reads_unsigned_and_narrow_dtypes():
    u16 = {"dtype": "uint16"}
    unit = elementwise_tiled_spatial(_instance(_ports(a=u16, b=u16, out=u16)))
    assert (unit.data_width, unit.signed, unit.lanes) == (16, False, 4)


def test_three_ports_are_refused():
    def three_inputs(d):
        d["interface"]["ports"].insert(2, dict(d["interface"]["ports"][1], name="c"))
        d["dataflow"]["ports"]["c"] = d["dataflow"]["ports"]["b"]
        d["function"]["code"] = "out = a + b + c"

    with pytest.raises(GenError) as e:
        elementwise_tiled_spatial(_instance(three_inputs))
    assert e.value.code == "gen.ports"


def test_a_design_name_that_is_no_identifier(tmp_path):
    with pytest.raises(GenError) as e:
        generate(point(tmp_path, name="vecadd-2"), tmp_path / "out", root=tmp_path)
    assert e.value.code == "gen.name" and not (tmp_path / "out").exists()


def test_a_null_symbol_is_refused(tmp_path, monkeypatch):
    # DesignPoint.load refuses one in a file (graph.symbols); gen.symbols holds
    # for a point whose graph lost a value after it was checked
    path = point(tmp_path)
    loaded = DesignPoint.load(path)
    loaded.graph.symbols["N"] = None
    monkeypatch.setattr(gen_mod.DesignPoint, "load", staticmethod(lambda _: loaded))
    with pytest.raises(GenError, match=r"\['N'\]") as e:
        generate(path, tmp_path / "out", root=tmp_path)
    assert e.value.code == "gen.symbols" and not (tmp_path / "out").exists()


def test_two_instances_are_refused(tmp_path):
    gd = graph_dict()
    chained(gd)
    with pytest.raises(GenError) as e:
        generate(point(tmp_path, graph=Graph.from_dict(gd)), tmp_path / "out", root=tmp_path)
    assert e.value.code == "gen.instances" and "['acc', 'acc2']" in str(e.value)


@pytest.mark.parametrize(("binding", "code"), [(None, "gen.binding"), ("Gemm", "gen.block")])
def test_a_null_binding_and_an_unregistered_block(tmp_path, monkeypatch, binding, code):
    real = gen_mod.instances

    def edited(graph):
        (name, inst), = real(graph).items()  # fmt: skip
        b = None if binding is None else {**inst.impl.binding, "block": binding}
        impl = replace(inst.impl, binding=b)
        brm = copy.copy(inst.brm)
        brm.implementations = {IMPL: impl}
        return {name: Instance(brm, IMPL, inst.params)}

    path = point(tmp_path)
    monkeypatch.setattr(gen_mod, "instances", edited)
    with pytest.raises(GenError) as e:
        generate(path, tmp_path / "out", root=tmp_path)
    assert e.value.code == code
    assert "ElementwiseTiledSpatial" in BLOCKS


def test_the_install_check(tmp_path):
    good = fake_install(tmp_path / "a")
    check_install(good, ["forge_vecadd_acc", "forge_vecadd"])
    for rel in ("generators/forge_vecadd.py", "targets/forge_vecadd_acc.py"):
        bad = fake_install(tmp_path / rel.replace("/", "_"), {rel: ""})
        with pytest.raises(GenError, match="would shadow") as e:
            check_install(bad, ["forge_vecadd_acc", "forge_vecadd"])
        assert e.value.code == "gen.install"
    (good / "generators" / "snax_unit.py").unlink()
    with pytest.raises(GenError, match="no 'snax_unit'"):
        check_install(good, ["forge_vecadd"])


def test_generate_checks_the_install_before_writing(tmp_path):
    path = point(tmp_path)
    bad = fake_install(tmp_path, {"generators/forge_vecadd.py": ""})
    with pytest.raises(GenError):
        generate(path, tmp_path / "out", install=bad, root=tmp_path)
    assert not (tmp_path / "out").exists()


def test_the_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(gen_mod, "ROOT", tmp_path)
    path = point(tmp_path)
    out = tmp_path / "out"
    install = fake_install(tmp_path)
    assert cli.main([str(path), "--out", str(out), "--install", str(install)]) == 0
    text = capsys.readouterr().out
    assert "target forge_vecadd" in text and "task add_acc: 16 beats" in text
    assert (out / "manifest.json").is_file()
    (install / "generators" / "forge_vecadd.py").touch()
    assert cli.main([str(path), "--out", str(out), "--install", str(install)]) == 1
    assert "[gen.install]" in capsys.readouterr().err


@pytest.mark.skipif(
    not (SUB / "snax").is_dir(), reason="the snax-gvsoc submodule is not checked out"
)
def test_the_constants_are_snax_alu_accs():
    text = (SUB / "snax" / "snax_alu_acc.py").read_text()
    assert "SNAX_CSR_FIRST = 0x3C0" in text and "SNAX_CSR_LAST = 0x5FF" in text
    assert (gen_mod.SNAX_CSR_FIRST, gen_mod.SNAX_CSR_LAST) == (0x3C0, 0x5FF)
