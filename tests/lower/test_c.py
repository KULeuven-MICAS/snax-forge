"""The C backend (D135): vecadd's main.c read as text (no marks, the window, forge_check), vsum's
counter, every refusal, and main.c compiled against snax-gvsoc's sw/ where the image's clang is."""

from __future__ import annotations

import copy
import re
import shutil
import subprocess
from dataclasses import replace

import pytest

from snax_forge.design import DesignPoint
from snax_forge.dfg import Graph
from snax_forge.gen import describe
from snax_forge.gen import gvsoc as gen_mod
from snax_forge.gen.data import arrays as data_arrays
from snax_forge.lower import CError, TaskList, main_c, task_list
from snax_forge.lower.c import cores, job_values
from tests.design.helpers import chained
from tests.gen.helpers import SUB, bound_vecmul, point
from tests.helpers import SUBMODULE_MISSING, bound_vsum, graph_dict

POINT = "out/design/vecadd/design_point.json"


def parts(tmp_path, graph=None, name="vecadd", acc=None):
    """What main_c reads for vecadd (or ``graph``) on snax_forge: the source, its slot named
    as the generator names it; the description ``describe``'s unless ``acc`` is given."""
    p = DesignPoint.load(point(tmp_path, name=name, graph=graph))
    acc = describe(p, POINT) if acc is None else acc
    cluster = gen_mod.source(p.platform, tmp_path)
    _, slot = gen_mod.find_slot(cluster, p.platform)
    slot["snax_acc_name"] = p.platform.accelerator_slot
    return p, task_list(p), cluster, acc, data_arrays(p)


INCLUDES = ["data.h", "acc_csr.h", "streamer_csr_addr_map.h"]


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


def functions(src: str) -> dict[str, list[str]]:
    """main.c's functions by name (``forge_core_0``, ``forge_core_idle``, ``forge_check``,
    ``main``): their lines, stripped."""
    out: dict[str, list[str]] = {}
    name = None
    for line in src.splitlines():
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


def test_vecadd_runs_the_dma_on_core_1_and_the_accelerator_on_core_0(tmp_path):
    src = text(tmp_path)
    f = functions(src)
    assert list(f) == ["forge_core_0", "forge_core_1", "forge_check", "main"]
    assert not any("snrt_dma" in x for x in f["forge_core_0"])
    assert not any("csrw_ss" in x or "csrr_ss" in x for x in f["forge_core_1"])
    assert "static int (*const forge_cores[2])(void) = {forge_core_0, forge_core_1};" in src
    assert f["main"] == [
        "uint32_t t0 = snrt_mcycle();", "uint32_t core = snrt_cluster_core_idx();",
        "int err = forge_cores[core]();", "if (core == 0) {", "err += forge_check(t0);", "}",
        "return err;",
    ]  # fmt: skip
    assert "volatile" not in "\n".join(f["main"]) and "snrt_is_compute_core" not in src
    assert f["forge_core_1"][-1] == "return 0;" and f["forge_core_0"][-1] == "return 0;"


def test_vecadd_has_no_marks_and_reads_mcycle_at_the_windows_ends(tmp_path):
    src = text(tmp_path)
    f = functions(src)
    assert "FORGE_MARK" not in src and "asm" not in src and "forge_main" not in src
    assert src.count("snrt_mcycle()") == 2
    assert f["main"][0] == "uint32_t t0 = snrt_mcycle();"
    assert f["forge_check"][:2] == [
        "uint32_t cycles = snrt_mcycle() - t0;", 'printf("forge cycles: %u\\n", cycles);'
    ]  # fmt: skip
    assert "static __attribute__((noinline)) int forge_check(uint32_t t0) {" in src


def test_vecadd_barriers_follow_the_cross_core_waits(tmp_path):
    f = functions(text(tmp_path))
    c0, c1 = f["forge_core_0"], f["forge_core_1"]
    b0 = [i for i, x in enumerate(c0) if x == BARRIER]
    b1 = [i for i, x in enumerate(c1) if x == BARRIER]
    assert len(b0) == len(b1) == 3
    # the loads' wait (core 1), the writer's wait (core 0, its counter stored last), the
    # store's sync (core 1): each the owner's, then the barrier on both
    assert c1[b1[0] - 1] == "snrt_dma_wait_all();"
    assert c0[b0[1] - 1] == "forge_counter[0] = csrr_ss(ACC_RO_COUNTER);"
    assert c1[b1[2] - 1] == "snrt_dma_wait_all();"
    assert c0[b0[0] + 1] == "csrw_ss(STREAMER_START_CSR, 1);"
    assert c1[b1[1] + 1] == "snrt_dma_start_1d(C, (void *)0x10000400, 512);"
    # the function ends after the last barrier; forge_check follows in main
    assert c0[b0[2] + 1] == "return 0;"
    # load_B starts with no wait after load_A: the DMA queues; one wait ends both
    a = c1.index("snrt_dma_start_1d((void *)0x10000200, B, 512);")
    assert "snrt_dma_wait_all();" not in c1[:a] and c1.count("snrt_dma_wait_all();") == 2
    assert c1[a + 1] == "snrt_dma_wait_all();"


def test_a_dependence_on_a_dma_task_brings_its_wait_back(tmp_path):
    def b_after_a(steps):
        next(s for s in steps if s.get("task_name") == "load_B")["after"] = ["load_A"]

    lines = functions(text(tmp_path, edit_tasks=b_after_a))["forge_core_1"]
    i = lines.index("snrt_dma_start_1d((void *)0x10000200, B, 512);")
    assert lines[i - 2 : i] == ["snrt_dma_start_1d((void *)0x10000000, A, 512);",
                                "snrt_dma_wait_all();"]  # fmt: skip
    assert lines.count("snrt_dma_wait_all();") == 3


def test_vecadd_starts_the_movers_with_one_csr_then_the_accelerator(tmp_path):
    src = text(tmp_path)
    assert src.count("csrw_ss(STREAMER_START_CSR, 1);") == 1
    # the two start CSRs back to back
    assert "csrw_ss(STREAMER_START_CSR, 1);\n    csrw_ss(ACC_RW_START, 1);" in src


def test_vecadd_configures_from_the_task_values(tmp_path):
    src = text(tmp_path)
    for line in [
        "csrw_ss(BASE_PTR_READER_1_LOW, 0x10000200);", "csrw_ss(T_BOUND_WRITER_0_0, 16);",
        "csrw_ss(T_STRIDE_READER_0_0, 32);", "csrw_ss(S_STRIDE_READER_0_0, 8);",
        "csrw_ss(ACC_RW_JOBS, 16);", "snrt_dma_start_1d((void *)0x10000000, A, 512);",
        "snrt_dma_start_1d(C, (void *)0x10000400, 512);",
    ]:  # fmt: skip
        assert line in src, line


def test_vecadd_checks_itself(tmp_path):
    src = text(tmp_path)
    assert "if (((volatile int64_t *)C)[i] != C_golden[i]) {" in src
    assert 'printf("forge data: %d errors (%u words)\\n", data_errors, 64u);' in src
    assert 'printf("forge counter add_acc: %u\\n", forge_counter[0]);' in src
    assert "err += forge_counter[0] < 16;" in src
    assert "static uint32_t forge_counter[1];" in src
    f = functions(src)
    assert "forge_counter[0] = csrr_ss(ACC_RO_COUNTER);" in f["forge_core_0"]
    # the wait polls STATUS bit 0 (busy), then the streamer
    assert "while (csrr_ss(ACC_RO_STATUS) & 1) {" in f["forge_core_0"]
    assert f["forge_check"][-2:] == ['printf("forge errors: %d\\n", err);', "return err;"]


def test_vsum_sets_jobs_and_job_beats_and_reads_its_counter(tmp_path):
    """JOBS = n / T, JOB_BEATS = T (D146); the counter checked against the beats n."""
    src = text(tmp_path, graph=bound_vsum(), name="vsum")
    assert "csrw_ss(ACC_RW_JOBS, 1);\n    csrw_ss(ACC_RW_JOB_BEATS, 64);" in src
    assert "err += forge_counter[0] < 64;" in src
    assert 'printf("forge counter sum_acc: %u\\n", forge_counter[0]);' in src


def test_the_same_point_gives_the_same_text(tmp_path):
    assert text(tmp_path) == text(tmp_path)


def test_one_core_has_no_barriers(tmp_path):
    def one_core(c):  # the slot onto the DMA core: one core runs both (open item 80)
        dma = c["dma_core_template"]
        dma["snax_acc_cfg"] = c["snax_forge_core_template"]["snax_acc_cfg"]
        c["cluster"]["hives"][0]["cores"] = [{"$ref": "#/dma_core_template"}]

    src = text(tmp_path, edit_cluster=one_core)
    assert "hw_barrier" not in src
    assert list(functions(src)) == ["forge_core_0", "forge_check", "main"]
    assert "// Core 0: the DMA and the accelerator acc" in src


def test_a_core_with_no_task_takes_part_in_the_barriers_only(tmp_path):
    def three_cores(c):
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
        c["dma_core_template"]["xdma"] = False

    with pytest.raises(CError, match=r"\[c.cores\] .*xdma on cores \[\]"):
        text(tmp_path, edit_cluster=no_dma)
    p, *_ = parts(tmp_path)
    assert cores(gen_mod.source(p.platform, tmp_path), "snax_forge").count == 2


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
        main_c(p, tasks, cluster, replace(acc, groups=[no_jobs]), arrays, INCLUDES, POINT)


def test_job_values():
    """JOBS = n / T and, for a fold, JOB_BEATS = T; T = 0 an empty run (JOBS 0)."""
    from snax_forge.brm import load_brm

    add = load_brm("elementwise_add").resolve("chisel_tiled_spatial", {"W": 4})
    total = load_brm("accumulate").resolve("chisel_adder_tree", {"W": 4})
    assert job_values(add, {"n": 16}) == {"JOBS": 16}
    assert job_values(total, {"n": 16, "T": 16}) == {"JOBS": 1, "JOB_BEATS": 16}
    assert job_values(total, {"n": 12, "T": 4}) == {"JOBS": 3, "JOB_BEATS": 4}
    assert job_values(total, {"n": 0, "T": 0}) == {"JOBS": 0, "JOB_BEATS": 0}


def dot1_text(tmp_path) -> str:
    """dot 1's main.c: groups mul and sum in the accelerator dot (D148)."""
    from tests.gen.helpers import bound_dot1

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_dot1(), name="dot_1")
    return main_c(p, tasks, cluster, acc, arrays, ["data.h", "dot_csr.h", STREAMER], POINT)


STREAMER = "streamer_csr_addr_map.h"


def test_dot_1_runs_each_group_with_its_own_csrs(tmp_path):
    """Per task: every mover (the other group's at bound 0), the group's JOBS (and JOB_BEATS),
    one streamer start, the group's START; then its STATUS, the streamer's busy, its COUNTER."""
    core = functions(dot1_text(tmp_path))["forge_core_0"]
    acc = [x for x in core if "DOT_" in x or "STREAMER_" in x or x.startswith("forge_counter")]
    assert acc == [
        "csrw_ss(DOT_RW_MUL_JOBS, 16);",
        "csrw_ss(STREAMER_START_CSR, 1);", "csrw_ss(DOT_RW_MUL_START, 1);",
        "csrw_ss(DOT_RW_SUM_JOBS, 1);", "csrw_ss(DOT_RW_SUM_JOB_BEATS, 16);",
        "while (csrr_ss(DOT_RO_MUL_STATUS) & 1) {", "while (csrr_ss(STREAMER_BUSY_CSR)) {",
        "forge_counter[0] = csrr_ss(DOT_RO_MUL_COUNTER);",
        "csrw_ss(STREAMER_START_CSR, 1);", "csrw_ss(DOT_RW_SUM_START, 1);",
        "while (csrr_ss(DOT_RO_SUM_STATUS) & 1) {", "while (csrr_ss(STREAMER_BUSY_CSR)) {",
        "forge_counter[1] = csrr_ss(DOT_RO_SUM_COUNTER);",
    ]  # fmt: skip
    bounds = [x for x in core if x.startswith("csrw_ss(T_BOUND_")]
    assert bounds == [
        "csrw_ss(T_BOUND_READER_0_0, 16);", "csrw_ss(T_BOUND_READER_1_0, 16);",
        "csrw_ss(T_BOUND_WRITER_0_0, 16);", "csrw_ss(T_BOUND_READER_2_0, 0);",
        "csrw_ss(T_BOUND_WRITER_1_0, 0);",
        "csrw_ss(T_BOUND_READER_0_0, 0);", "csrw_ss(T_BOUND_READER_1_0, 0);",
        "csrw_ss(T_BOUND_WRITER_0_0, 0);", "csrw_ss(T_BOUND_READER_2_0, 16);",
        "csrw_ss(T_BOUND_WRITER_1_0, 1);",
    ]  # fmt: skip


def test_dot_1_checks_both_counters(tmp_path):
    check = functions(dot1_text(tmp_path))["forge_check"]
    assert 'printf("forge counter mult_mul: %u\\n", forge_counter[0]);' in check
    assert 'printf("forge counter sum_sum: %u\\n", forge_counter[1]);' in check


def test_refused_a_start_without_the_idle_movers(tmp_path):
    from tests.gen.helpers import bound_dot1

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_dot1(), name="dot_1")
    d = tasks.to_dict()
    for s in d["steps"]:
        if s["op"] == "start" and "mult_mul" in s["tasks"]:
            s["tasks"] = [t for t in s["tasks"] if not t.startswith("mult_sum")]
    d["steps"] = [s for s in d["steps"] if s.get("task_name", "").startswith("mult_sum") is False]
    with pytest.raises(CError, match=r"\[c.start\] .*the other groups' idle at bound 0"):
        main_c(p, TaskList.from_dict(d), cluster, acc, arrays, [], POINT)


def test_refused_an_idle_mover_at_stride_0(tmp_path):
    """An idle mover starts anyway when its first temporal stride is 0 (Reader.scala)."""
    from tests.gen.helpers import bound_dot1

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_dot1(), name="dot_1")
    d = tasks.to_dict()
    for s in d["steps"]:
        if s.get("task_name") == "mult_sum_a":
            s["values"]["temporal_strides"] = [0]
    with pytest.raises(CError, match=r"\[c.idle\] mult_sum_a: an idle mover"):
        main_c(p, TaskList.from_dict(d), cluster, acc, arrays, [], POINT)


def test_refused_two_instances(tmp_path):
    _, _, cluster, acc, _ = parts(tmp_path / "one")
    gd = graph_dict()
    chained(gd)
    p = DesignPoint.load(point(tmp_path, graph=Graph.from_dict(gd)))
    with pytest.raises(CError, match=r"\[c.instances\] 2 accelerators \['acc', 'acc2'\]"):
        main_c(p, task_list(p), cluster, acc, [], INCLUDES, POINT)


# =============================================================================
# Compiled
# =============================================================================

CLANG = shutil.which("clang")
NM = shutil.which("llvm-nm")
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
    assert forge == {"forge_check", "forge_counter", "forge_cores", "forge_core_0", "forge_core_1"}
