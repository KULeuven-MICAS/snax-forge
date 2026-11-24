"""The C backend: vecadd's main.c read as text (no marks, the window, forge_check), vsum's
counter, every refusal, and main.c compiled against snax-gvsoc's sw/ where the image's clang is."""

from __future__ import annotations

import copy
import re
import shutil
import subprocess
from dataclasses import replace

import pytest

from snax_forge.design import DesignPoint
from snax_forge.gen import describe
from snax_forge.gen import gvsoc as gen_mod
from snax_forge.gen.data import arrays as data_arrays
from snax_forge.lower import CError, TaskList, main_c, task_list
from snax_forge.lower import c as c_mod
from snax_forge.lower.c import (
    QUIET_END,
    QUIET_OFF,
    QUIET_ON,
    cores,
    program,
    strip_quiet,
)
from tests.gen.helpers import SUB, bound_vecmul, point
from tests.helpers import SUBMODULE_MISSING, bound_vsum

POINT = "out/design/vecadd/design_point.json"


def parts(tmp_path, graph=None, name="vecadd", acc=None):
    """What main_c reads for vecadd (or ``graph``) on snax_forge: the source, its slot named
    as the generator names it (the tag); the description ``describe``'s unless ``acc`` is
    given."""
    p = DesignPoint.load(point(tmp_path, name=name, graph=graph))
    acc = describe(p, POINT) if acc is None else acc
    cluster = gen_mod.source(p.platform, tmp_path)
    _, slot = gen_mod.find_slot(cluster, p.platform)
    slot["snax_acc_name"] = acc.tag
    return p, task_list(p), cluster, acc, data_arrays(p)


INCLUDES = ["data.h", "acc_csr.h", "streamer_acc_csr_addr_map.h"]


def text(tmp_path, edit_tasks=None, edit_cluster=None, **kw) -> str:
    p, tasks, cluster, acc, arrays = parts(tmp_path, **kw)
    if edit_tasks:
        d = tasks.to_dict()
        edit_tasks(d["steps"])
        tasks = TaskList.from_dict(d)
    if edit_cluster:
        cluster = copy.deepcopy(cluster)
        edit_cluster(cluster)
    return main_c(p, tasks, cluster, acc, arrays, INCLUDES, POINT)


def functions(src: str, quiet: bool = False) -> dict[str, list[str]]:
    """main.c's functions by name (``forge_core_0``, ``forge_core_idle``, ``forge_check``,
    ``main``): their lines, stripped, as the default build compiles them (``strip_quiet``), or
    with ``quiet`` the whole text, the quiet build's directives and lines too."""
    out: dict[str, list[str]] = {}
    name = None
    for line in (src if quiet else strip_quiet(src)).splitlines():
        m = re.fullmatch(r"(?:static __attribute__\(\(noinline\)\) )?int (\w+)\([^)]*\) \{", line)
        if m:
            name = m[1]
            out[name] = []
        elif line == "}":
            name = None
        elif name is not None:
            out[name].append(line.strip())
    return out


BARRIER = "snrt_cluster_hw_barrier();"


def two_cores(c):
    """The two-core cluster: the DMA on a second core, the slot on core 0 alone."""
    c["snax_forge_core_template"]["xdma"] = False
    c["dma_core_template"] = {"isa": "rv32ima", "xdma": True}
    c["cluster"]["hives"][0]["cores"].append({"$ref": "#/dma_core_template"})


def test_a_core_with_no_task_takes_part_in_the_barriers_only(tmp_path):
    def three_cores(c):
        two_cores(c)
        c["cluster"]["hives"][0]["cores"].append({"isa": "rv32ima", "xdma": False})

    src = text(tmp_path, edit_cluster=three_cores)
    f = functions(src)
    assert list(f) == ["forge_core_0", "forge_core_1", "forge_core_idle", "forge_check", "main"]
    assert f["forge_core_idle"] == ["for (int i = 0; i < 3; i++) {", BARRIER, "}", "return 0;"]
    assert "forge_cores[3])(void) = {forge_core_0, forge_core_1, forge_core_idle};" in src


# =============================================================================
# Refusals
# =============================================================================


def test_refused_without_one_dma_core(tmp_path):
    def no_dma(c):
        c["snax_forge_core_template"]["xdma"] = False

    with pytest.raises(CError, match=r"\[c.cores\] .*xdma on cores \[\]"):
        text(tmp_path, edit_cluster=no_dma)
    p, *_ = parts(tmp_path)
    assert cores(gen_mod.source(p.platform, tmp_path), "snax_forge").count == 1  # the source's


def test_refused_a_start_of_some_movers(tmp_path):
    def split(steps):
        i = next(i for i, s in enumerate(steps) if s["op"] == "start" and len(s["tasks"]) == 4)
        steps[i : i + 1] = [{"op": "start", "tasks": steps[i]["tasks"][:1]},
                            {"op": "start", "tasks": steps[i]["tasks"][1:]}]  # fmt: skip

    with pytest.raises(CError, match=r"\[c.start\] .*starts \['acc_a'\] of acc's movers"):
        text(tmp_path, edit_tasks=split)


def test_refused_the_accelerator_before_its_movers(tmp_path):
    def acc_first(steps):
        s = next(s for s in steps if s["op"] == "start" and len(s["tasks"]) == 4)
        s["tasks"] = s["tasks"][-1:] + s["tasks"][:-1]

    with pytest.raises(CError, match=r"\[c.start\] .*acc starts before its movers"):
        text(tmp_path, edit_tasks=acc_first)


def test_refused_a_signal_wait(tmp_path):
    def signal(steps):
        steps[-1]["mode"] = "signal"

    with pytest.raises(CError, match=r"\[c.signal\] a wait on dma in mode 'signal'"):
        text(tmp_path, edit_tasks=signal)


def test_refused_a_dma_pattern_off_the_beat(tmp_path):
    def strided(steps):
        steps[0]["values"]["src"]["strides"] = [128]

    with pytest.raises(CError, match=r"\[c.dma\] .*load_A moves"):
        text(tmp_path, edit_tasks=strided)


def test_refused_a_streamer_task_with_too_many_loops(tmp_path):
    def two_loops(steps):
        s = next(s for s in steps if s.get("task_name") == "add_acc_a")
        s["values"]["temporal_bounds"] += [1]
        s["values"]["temporal_strides"] += [0]

    with pytest.raises(CError, match=r"\[c.tasks\] .*acc_a: 2 loops, registers hold 1"):
        text(tmp_path, edit_tasks=two_loops)


def test_refused_an_accelerator_task_no_wait_covers(tmp_path):
    def no_store(steps):
        steps[:] = [s for s in steps if "store_C" not in (s.get("task_name"), s.get("task"))
                    and s.get("tasks") != ["store_C"]]  # fmt: skip

    with pytest.raises(CError, match=r"\[c.wait\] no wait covers \['add_acc'"):
        text(tmp_path, edit_tasks=no_store)


def test_refused_a_csr_the_description_lacks(tmp_path):
    p, tasks, cluster, acc, arrays = parts(tmp_path)
    group = acc.groups[0]
    no_jobs = replace(group, csrs=[c for c in group.csrs if c.name != "JOBS"])
    no_jobs = replace(no_jobs, csrs=[replace(c, offset=i) for i, c in enumerate(no_jobs.csrs)])
    with pytest.raises(CError, match=r"\[c.csrs\] add_acc: unit acc's group acc has no CSR JOBS"):
        (a,) = acc.accelerators
        main_c(p, tasks, cluster, replace(acc, accelerators=[replace(a, groups=[no_jobs])]), arrays,
               INCLUDES, POINT)  # fmt: skip


STREAMER = "streamer_dot_csr_addr_map.h"


def test_refused_a_start_without_the_idle_movers(tmp_path):
    from tests.gen.helpers import bound_dot1_enc

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_dot1_enc(), name="dot_v1_1")
    d = tasks.to_dict()
    for s in d["steps"]:
        if s["op"] == "start" and "mult_vmul" in s["tasks"]:
            s["tasks"] = [t for t in s["tasks"] if not t.startswith("mult_vsum")]
    d["steps"] = [s for s in d["steps"] if s.get("task_name", "").startswith("mult_vsum") is False]
    with pytest.raises(CError, match=r"\[c.start\] .*the other groups' idle at bound 0"):
        main_c(p, TaskList.from_dict(d), cluster, acc, arrays, [], POINT)


def test_refused_an_idle_mover_at_stride_0(tmp_path):
    """An idle mover starts anyway when its first temporal stride is 0 (Reader.scala)."""
    from tests.gen.helpers import bound_dot1_enc

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_dot1_enc(), name="dot_v1_1")
    d = tasks.to_dict()
    for s in d["steps"]:
        if s.get("task_name") == "mult_vsum_a":
            s["values"]["temporal_strides"] = [0]
    with pytest.raises(CError, match=r"\[c.idle\] mult_vsum_a: an idle mover"):
        main_c(p, TaskList.from_dict(d), cluster, acc, arrays, [], POINT)


def test_an_idle_mover_outside_an_encapsulated_set_is_refused(tmp_path):
    """c.idle's stride check is an encapsulated set's (dot_v1_1, above); in a set of one group an
    idle mover is no other group's, so the start is refused (c.start), whatever its stride."""
    from tests.gen.helpers import bound_recipe

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_recipe("dot_v7", "dot"),
                                           name="dot_v7")  # fmt: skip
    d = tasks.to_dict()
    for s in d["steps"]:
        if s.get("task_name") == "sum_sum_a":
            s["values"]["temporal_bounds"] = [0]
    with pytest.raises(CError, match=r"\[c.start\] sum_sum_a: an idle mover .* of sum, a set of "
                       "one group"):  # fmt: skip
        main_c(p, TaskList.from_dict(d), cluster, acc, arrays, INCLUDES, POINT)


# =============================================================================
# Compiled
# =============================================================================

CLANG = shutil.which("clang")
NM = shutil.which("llvm-nm")
OBJCOPY = shutil.which("llvm-objcopy")
SW = SUB / "sw"
# sw/toolchain.mk's and sw/common.mk's compile flags (the image's clang, snax_cluster's
# toolchain); the generated folder's headers before sw/generated/, as the forge apps put them
FLAGS = [
    "--target=riscv32-unknown-elf", "-mcpu=generic-rv32", "-march=rv32imafdzfh", "-mabi=ilp32d",
    "-mcmodel=medany", "-fno-builtin-memset", "-DPRINTF_DISABLE_SUPPORT_LONG_LONG",
    "-ffast-math", "-fno-builtin-printf", "-fno-common", "-fopenmp", "-ftls-model=local-exec",
    "-O3", "-D__DEFINED_uint64_t",
]  # fmt: skip
SW_INCDIRS = [
    "generated/snax_forge", "runtime/rtl-generic/src", "runtime/common",
    "snRuntime/api", "snRuntime/api/omp", "snRuntime/src", "snRuntime/src/omp",
    "deps/riscv-opcodes", "math/include",
]  # fmt: skip


def _riscv_clang() -> str | None:
    """Why vecadd's main.c cannot be compiled here, or None."""
    if CLANG is None or NM is None:
        return "no clang or llvm-nm on PATH (the image's, snax_cluster's toolchain)"
    if not (SW / "snRuntime").is_dir():
        return SUBMODULE_MISSING
    probe = subprocess.run([CLANG, "-print-targets"], capture_output=True, text=True, check=False)
    if "riscv32" not in probe.stdout:
        return f"{CLANG} has no riscv32 target"
    return None


@pytest.mark.gvsoc  # the image's RISC-V clang: CI's gvsoc job runs it
@pytest.mark.skipif(_riscv_clang() is not None, reason=str(_riscv_clang()))
@pytest.mark.parametrize("design", ["vecadd", "vecmul", "vsum"])
def test_main_c_compiles_against_snax_gvsocs_sw(tmp_path, design):
    """main.c compiles with -Werror; the object defines main and forge_check once each and no
    forge_ symbol but the core functions, their table and the counters (llvm-nm)."""
    gen = tmp_path / "gen"
    graph = {"vecmul": bound_vecmul, "vsum": bound_vsum}.get(design)
    m = gen_mod.generate(point(tmp_path, name=design, graph=graph and graph()), gen,
                         root=tmp_path)  # fmt: skip
    assert {"main.c", "data.h", "include/acc_csr.h"} <= set(m.files) and m.program == "main.c"
    incs = [f"-I{gen}", f"-I{gen / 'include'}", *(f"-I{SW / d}" for d in SW_INCDIRS)]
    run = subprocess.run([CLANG, *FLAGS, *incs, "-Werror", "-c", str(gen / "main.c"), "-o",
                          str(gen / "main.o")], capture_output=True, text=True,
                         check=False)  # fmt: skip
    assert run.returncode == 0, run.stderr
    nm = subprocess.run([NM, "--defined-only", str(gen / "main.o")], capture_output=True,
                        text=True, check=True).stdout  # fmt: skip
    defined = [line.split()[-1] for line in nm.splitlines()]
    assert defined.count("main") == defined.count("forge_check") == 1
    # clang may split the static counters into one symbol per slot (forge_counter.0)
    forge = {x.split(".")[0] for x in defined if x.startswith("forge_")}
    # one core: clang may fold the one-entry table into main
    assert {"forge_check", "forge_counter", "forge_core_0"} <= forge <= {
        "forge_check", "forge_counter", "forge_cores", "forge_core_0"
    }  # fmt: skip


@pytest.mark.gvsoc  # the image's RISC-V clang: CI's gvsoc job runs it
@pytest.mark.skipif(_riscv_clang() is not None or OBJCOPY is None, reason="no RISC-V clang")
@pytest.mark.parametrize("design", ["vecadd", "vecmul", "vsum"])
def test_the_quiet_parts_leave_the_default_build_as_it_was(tmp_path, design):
    """Main.c compiled as it is and with its FORGE_QUIET blocks removed give the same
    object once the debug information is stripped (snRuntime builds with -g, so the lines the
    blocks take move every line number there); with -DFORGE_QUIET it compiles with -Werror."""
    gen = tmp_path / "gen"
    graph = {"vecmul": bound_vecmul, "vsum": bound_vsum}.get(design)
    gen_mod.generate(point(tmp_path, name=design, graph=graph and graph()), gen, root=tmp_path)
    (gen / "stripped.c").write_text(strip_quiet((gen / "main.c").read_text()))
    incs = [f"-I{gen}", f"-I{gen / 'include'}", *(f"-I{SW / d}" for d in SW_INCDIRS)]
    objects = {}
    for name, src, extra in (("main", "main.c", []), ("stripped", "stripped.c", []),
                             ("quiet", "main.c", ["-DFORGE_QUIET"])):  # fmt: skip
        o = gen / f"{name}.o"
        run = subprocess.run([CLANG, *FLAGS, *extra, *incs, "-g", "-Werror", "-c",
                              str(gen / src), "-o", str(o)], capture_output=True, text=True,
                             check=False)  # fmt: skip
        assert run.returncode == 0, (name, run.stderr)
        subprocess.run([OBJCOPY, "--strip-debug", str(o)], check=True)
        objects[name] = o.read_bytes()
    assert objects["main"] == objects["stripped"]
    assert objects["quiet"] != objects["main"]


# =============================================================================
# The quiet build
# =============================================================================


def quiet(tmp_path, **kw):
    p, tasks, cluster, acc, arrays = parts(tmp_path, **kw)
    return program(p, tasks, cluster, acc, arrays, INCLUDES, POINT)


def quiet_build(src: str) -> str:
    """main.c as ``-DFORGE_QUIET`` compiles it: the ``#ifndef FORGE_QUIET`` blocks gone, the
    lines of each ``#ifdef FORGE_QUIET`` block kept without the directives."""
    out, skip = [], False
    for line in src.splitlines():
        s = line.strip()
        if s in (QUIET_ON, QUIET_OFF):
            skip = s == QUIET_OFF
        elif s == QUIET_END:
            skip = False
        elif not skip:
            out.append(line)
    return "\n".join(out)


# vecadd's one wait follows its start; dot_1's vmul is waited on after vsum's configure (D158)
@pytest.mark.parametrize("name", ["vecadd", "dot_1"])
def test_every_start_is_waited_on_at_once_in_the_quiet_build_and_each_build_waits_once(
    tmp_path, name
):
    from tests.gen.helpers import bound_recipe

    graph = bound_recipe("dot_v1", "dot") if name == "dot_v1" else None
    src, _ = quiet(tmp_path, graph=graph, name=name)
    q = functions(quiet_build(src), quiet=True)["forge_core_0"]
    starts = [re.fullmatch(r"csrw_ss\((\w+)_RW_START, 1\);", x) for x in q]
    units = [m[1] for m in starts if m]
    assert units
    for i, x in enumerate(q):
        if x.startswith("snrt_dma_start_"):
            assert q[i + 1] == "snrt_dma_wait_all();", x
        if starts[i]:
            u = starts[i][1]
            assert q[i + 1 : i + 5] == [
                f"while (csrr_ss({u}_RO_STATUS) & 1) {{", "}",
                f"while (csrr_ss({u}_STREAMER_BUSY_CSR)) {{", "}",
            ]  # fmt: skip
    default = functions(src)["forge_core_0"]
    for lines in (q, default):  # one poll per start, in either build
        for u in set(units):
            assert lines.count(f"while (csrr_ss({u}_RO_STATUS) & 1) {{") == units.count(u)
    assert q.count("snrt_dma_wait_all();") == sum(x.startswith("snrt_dma_start_") for x in q)


def test_more_marks_than_obs_o_holds_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(c_mod, "MAX_MARKS", 6)
    with pytest.raises(CError, match=r"\[c.obs\] check: section 7, more than obs_o's 6 marks"):
        quiet(tmp_path)
