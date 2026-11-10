"""axpy on SNAX-GVSoC (B2 R3b): y += alpha * x as one simd unit at op axpby, k1 an R/W CSR.

The worked example, W = 2, N = 4, int32: x = (1, -2 | 4, 0), y = (10, 20 | -5, 7). With k1
read from alpha (``recipes/elementwise/axpy.json``), main.c writes K1 = data.h's ``alpha[0]``
before START: alpha = 3 gives y' = (13, 14 | 7, 7) in JOBS = 2 firings. The same design run again
with only data.h's alpha (5) and its golden changed, the program rebuilt and nothing else, gives
y' = (15, 10 | 15, 7): the value is the program's, not the hardware's. Baked
(``axpy_baked.json``, k1 = 3), the program writes no K1 and data.h holds no alpha.
"""

from __future__ import annotations

import json
import re
import shutil

import numpy as np
import pytest

from snax_forge.flow import run as flow_run
from snax_forge.flow import run_flow
from snax_forge.gen import Manifest
from tests.design.helpers import SNAX_FORGE
from tests.helpers import recipe_path

from .helpers import needs_gvsoc

X = [1, -2, 4, 0]
Y = [10, 20, -5, 7]
# The run's BUSY_COUNTER on SNAX-GVSoC: 2 firings and 2 cycles of the shell, as relu's
AXPY_COUNTER = 4
SETS = {"W": 2, "N": 4}

pytestmark = [pytest.mark.gvsoc, needs_gvsoc]


@pytest.fixture
def worked(monkeypatch):
    """The kernel's inputs as the worked example's (alpha 3)."""
    from snax_forge.dfg import reference

    def inputs(spec, graph, seed):
        return {"alpha": np.int32(3), "x": np.array(X, np.int32), "y": np.array(Y, np.int32)}

    monkeypatch.setattr(reference, "kernel_inputs", inputs)


def summary(f) -> str:
    return f"{f.name}: {f.check}"


def array(text: str, name: str) -> list[int]:
    """data.h's array ``name``'s initialiser values."""
    body = text.split(f" {name}[", 1)[1].split("{", 1)[1].split("}", 1)[0]
    return [int(v) for v in body.replace(",", " ").split()]


def test_axpy_reads_k1_from_alpha(tmp_path, worked):
    f = run_flow(recipe_path("axpy"), SNAX_FORGE, recipe_sets=SETS, out=tmp_path / "axpy")
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0
    assert f.check["tasks"] == {"mult_acc": {"counter": AXPY_COUNTER, "beats": 2, "group": "acc"}}
    gen = tmp_path / "axpy" / "gen"
    data = (gen / "data.h").read_text()
    assert array(data, "alpha") == [3] + [0] * 15
    assert array(data, "y_golden") == [13, 14, 7, 7] + [0] * 12
    main = (gen / "main.c").read_text()
    assert main.index("csrw_ss(ACC_RW_K1, alpha[0]);") < main.index("csrw_ss(ACC_RW_START, 1);")


def test_the_same_design_with_alpha_5_in_data_h(tmp_path, worked):
    """The design's files as the first run wrote them, data.h's alpha 5 and its golden
    y + 5x: the program rebuilt alone, run on the same cluster file."""
    f = run_flow(recipe_path("axpy"), SNAX_FORGE, recipe_sets=SETS, out=tmp_path / "axpy")
    assert f.passed, summary(f)
    gen, again = tmp_path / "axpy" / "gen", tmp_path / "alpha5"
    shutil.copytree(gen, again / "gen")
    data = (again / "gen" / "data.h").read_text()
    want = [y + 5 * x for x, y in zip(X, Y, strict=True)]
    assert want == [15, 10, 15, 7]
    data = re.sub(r"(int32_t alpha\[16\][^{]*\{\s*)3,", r"\g<1>5,", data)
    old = ", ".join(str(v) for v in [13, 14, 7, 7])
    assert old in data
    data = data.replace(old, ", ".join(str(v) for v in want), 1)
    (again / "gen" / "data.h").write_text(data)
    for name in ("snax_forge.hjson", "accelerator.json", "main.c"):  # the design unchanged
        assert (again / "gen" / name).read_bytes() == (gen / name).read_bytes()

    manifest = Manifest.load(again / "gen" / "manifest.json")
    sw, run = again / "sw", again / "run"
    flow_run._make("sw", "sw", {"APPS": "forge", "FORGE": str((again / "gen").resolve()),
                                "DESIGN": "alpha5", "FORGE_BUILD": str(sw.resolve())},
                   again / "sw.log")  # fmt: skip
    run.mkdir()
    args = f"--param hjson={(again / 'gen' / manifest.cluster).resolve()}"
    rc = flow_run.run_gvsoc("run", {"TARGET": flow_run.TARGET, "ELF": str((sw / flow_run.ELF)
                                    .resolve()), "GVRUN_ARGS": args,
                                    "WORK_DIR": str((run / "work").resolve())},
                            run / "stdout.txt", run / "stderr.txt")  # fmt: skip
    stdout, stderr = (run / "stdout.txt").read_text(), (run / "stderr.txt").read_text()
    _, result = flow_run.check_output(stdout, flow_run.run_exit_code(rc, stderr),
                                      {t.name: t.beats for t in manifest.tasks})  # fmt: skip
    assert result["exit_code"] == 0 and result["data_errors"] == 0, stdout[-2000:]
    assert result["tasks"]["mult_acc"]["counter"] == AXPY_COUNTER


def test_axpy_baked_writes_no_k1(tmp_path, worked):
    f = run_flow(recipe_path("axpy_baked"), SNAX_FORGE, recipe_sets=SETS,
                 out=tmp_path / "axpy_baked")  # fmt: skip
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0
    assert f.check["tasks"] == {"mult_acc": {"counter": AXPY_COUNTER, "beats": 2, "group": "acc"}}
    gen = tmp_path / "axpy_baked" / "gen"
    assert "alpha" not in (gen / "data.h").read_text()
    assert "K1" not in (gen / "main.c").read_text()
    (unit,) = json.loads((gen / "accelerator.json").read_text())["accelerators"][0]["groups"][0][
        "units"]  # fmt: skip
    assert unit["params"] == {"op": "axpby", "k1": 3} and "scalars" not in unit
    assert array((gen / "data.h").read_text(), "y_golden") == [13, 14, 7, 7] + [0] * 12
