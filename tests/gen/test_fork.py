"""bicg_v3 on the generator: one reader of A for both halves, one group of two chains.

accelerator.json's forked input and JOBS per chain, the program's writes, the Chisel specs'
vectors and the refusals of a fork the description cannot have."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from snax_forge.gen import generate
from snax_forge.gen.accelerator import Description
from snax_forge.gen.blocks import GenError
from snax_forge.lower.derive import task_list
from snax_forge.lower.tasks import check_values

from .helpers import designed


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> tuple[Path, str]:
    tmp = tmp_path_factory.mktemp("bicg_v3")
    path = designed(tmp, "bicg_v3", "bicg_v3")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        generate(path, tmp / "gvsoc", tmp)
    return tmp, path


def test_the_program_writes_each_chains_jobs(generated):
    tmp, _ = generated
    main = (tmp / "gvsoc" / "main.c").read_text()
    for line in ("csrw_ss(BICG_TYPES_RW_JOBS_SUM, 64);", "csrw_ss(BICG_TYPES_RW_JOBS_MAC, 1);",
                 "csrw_ss(BICG_TYPES_RW_MAC_JOB_BEATS, 64);"):  # fmt: skip
        assert line in main
    assert "READER_3" not in main  # A, p and r: three readers, A's for both halves


def test_a_fork_whose_units_fire_unlike_is_refused(generated):
    """LOWER refuses a fork whose units fire unlike: the group would never end."""
    from snax_forge.design.point import DesignPoint

    tmp, path = generated
    p = DesignPoint.load(tmp / path)
    tl = task_list(p)
    (acc,) = tl.accel_steps()
    acc.values["mac"]["n"] = 32
    with pytest.raises(ValueError, match="units that fire mul 64, mac 32 times"):
        check_values(tl, p)


def _description(generated) -> Description:
    tmp, _ = generated
    return Description.from_dict(json.loads((tmp / "gvsoc" / "accelerator.json").read_text()))


@pytest.mark.parametrize(
    "inputs, text",
    [
        ([["mul.a", "mac.b"], "mul.b", "mac.a"], "differ in dtype, lanes, d_in or broadcast"),
        ([["mul.a"], "mul.b", "mac.a", "mac.b"], "two or more outer input ports"),
    ],
)  # fmt: skip
def test_a_fork_the_description_cannot_have(generated, inputs, text):
    d = _description(generated)
    acc = replace(d.accelerators[0], inputs=inputs)
    with pytest.raises(GenError, match=text) as e:
        Description(d.design_point, d.tag, [acc])
    assert e.value.code == "gen.fork"
