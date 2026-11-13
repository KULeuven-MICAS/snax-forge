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
    job_values,
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


def test_vecadd_runs_the_dma_and_the_accelerator_on_core_0(tmp_path):
    """The base cluster's one core runs both, with no barrier."""
    src = text(tmp_path)
    f = functions(src)
    assert list(f) == ["forge_core_0", "forge_check", "main"]
    assert "static int (*const forge_cores[1])(void) = {forge_core_0};" in src
    assert BARRIER not in src and "// Core 0: the DMA and the accelerator acc" in src
    assert "on snRuntime: core 0 runs the DMA and the accelerator" in src
    assert f["forge_core_0"].index("snrt_dma_start_1d((void *)0x10000000, A, 512);") < f[
        "forge_core_0"
    ].index("csrw_ss(ACC_RW_START, 1);")


def test_two_cores_run_the_dma_on_core_1_and_the_accelerator_on_core_0(tmp_path):
    src = text(tmp_path, edit_cluster=two_cores)
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


def test_two_cores_barriers_follow_the_cross_core_waits(tmp_path):
    f = functions(text(tmp_path, edit_cluster=two_cores))
    c0, c1 = f["forge_core_0"], f["forge_core_1"]
    b0 = [i for i, x in enumerate(c0) if x == BARRIER]
    b1 = [i for i, x in enumerate(c1) if x == BARRIER]
    assert len(b0) == len(b1) == 3
    # the loads' wait (core 1), the writer's wait (core 0, its counter stored last), the
    # store's sync (core 1): each the owner's, then the barrier on both
    assert c1[b1[0] - 1] == "snrt_dma_wait_all();"
    assert c0[b0[1] - 1] == "forge_counter[0] = csrr_ss(ACC_RO_BUSY_COUNTER);"
    assert c1[b1[2] - 1] == "snrt_dma_wait_all();"
    assert c0[b0[0] + 1] == "csrw_ss(ACC_STREAMER_START_CSR, 1);"
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

    lines = functions(text(tmp_path, edit_tasks=b_after_a, edit_cluster=two_cores))["forge_core_1"]
    i = lines.index("snrt_dma_start_1d((void *)0x10000200, B, 512);")
    assert lines[i - 2 : i] == ["snrt_dma_start_1d((void *)0x10000000, A, 512);",
                                "snrt_dma_wait_all();"]  # fmt: skip
    assert lines.count("snrt_dma_wait_all();") == 3


def test_vecadd_starts_the_movers_with_one_csr_then_the_accelerator(tmp_path):
    """Its set's start (every streamer macro prefixed with the set's name), then START."""
    src = text(tmp_path)
    assert src.count("csrw_ss(ACC_STREAMER_START_CSR, 1);") == 1
    # the two start CSRs back to back
    assert "csrw_ss(ACC_STREAMER_START_CSR, 1);\n    csrw_ss(ACC_RW_START, 1);" in src


def test_vecadd_configures_from_the_task_values(tmp_path):
    src = text(tmp_path)
    for line in [
        "csrw_ss(ACC_BASE_PTR_READER_1_LOW, 0x10000200);",
        "csrw_ss(ACC_T_BOUND_WRITER_0_0, 16);", "csrw_ss(ACC_T_STRIDE_READER_0_0, 32);",
        "csrw_ss(ACC_S_STRIDE_READER_0_0, 8);",
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
    assert "forge_counter[0] = csrr_ss(ACC_RO_BUSY_COUNTER);" in f["forge_core_0"]
    # the wait polls STATUS bit 0 (busy), then its set's streamer
    assert "while (csrr_ss(ACC_RO_STATUS) & 1) {" in f["forge_core_0"]
    assert "while (csrr_ss(ACC_STREAMER_BUSY_CSR)) {" in f["forge_core_0"]
    assert f["forge_check"][-2:] == ['printf("forge errors: %d\\n", err);', "return err;"]


def test_vsum_sets_jobs_and_job_beats_and_reads_its_counter(tmp_path):
    """JOBS = n / T, JOB_BEATS = T; the counter checked against the beats n."""
    src = text(tmp_path, graph=bound_vsum(), name="vsum")
    assert "csrw_ss(ACC_RW_JOBS, 1);\n    csrw_ss(ACC_RW_JOB_BEATS, 64);" in src
    assert "err += forge_counter[0] < 64;" in src
    assert 'printf("forge counter sum_acc: %u\\n", forge_counter[0]);' in src


def test_the_same_point_gives_the_same_text(tmp_path):
    assert text(tmp_path) == text(tmp_path)


def test_one_core_has_no_barriers(tmp_path):
    src = text(tmp_path)  # the base cluster: one core runs both
    assert "hw_barrier" not in src
    assert list(functions(src)) == ["forge_core_0", "forge_check", "main"]
    assert "// Core 0: the DMA and the accelerator acc" in src


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


def test_job_values():
    """JOBS = n / T and, for a fold, JOB_BEATS = T; T = 0 an empty run (JOBS 0)."""
    from snax_forge.brm import load_brm

    add = load_brm("simd").resolve("chisel_tiled_spatial", {"W": 4})
    total = load_brm("accumulate").resolve("chisel_adder_tree", {"W": 4})
    assert job_values(add, {"n": 16}) == {"JOBS": 16}
    assert job_values(total, {"n": 16, "T": 16}) == {"JOBS": 1, "JOB_BEATS": 16}
    assert job_values(total, {"n": 12, "T": 4}) == {"JOBS": 3, "JOB_BEATS": 4}
    assert job_values(total, {"n": 0, "T": 0}) == {"JOBS": 0, "JOB_BEATS": 0}


def dot1_text(tmp_path) -> str:
    """dot_1_enc's main.c: groups vmul and vsum in the accelerator dot."""
    from tests.gen.helpers import bound_dot1_enc

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_dot1_enc(), name="dot_1_enc")
    return main_c(p, tasks, cluster, acc, arrays, ["data.h", "dot_csr.h", STREAMER], POINT)


STREAMER = "streamer_dot_csr_addr_map.h"


def test_dot_1_runs_each_group_with_its_own_csrs(tmp_path):
    """Per task: every mover of the one set dot (the other group's at bound 0), the group's JOBS
    (and JOB_BEATS), the set's streamer start, the group's START; then its STATUS, the set's
    streamer busy, its BUSY_COUNTER."""
    core = functions(dot1_text(tmp_path))["forge_core_0"]
    acc = [x for x in core if "_RW_" in x or "_RO_" in x or "STREAMER_" in x
           or x.startswith("forge_counter")]  # fmt: skip
    assert acc == [
        "csrw_ss(DOT_RW_VMUL_JOBS, 16);",
        "csrw_ss(DOT_STREAMER_START_CSR, 1);", "csrw_ss(DOT_RW_VMUL_START, 1);",
        "csrw_ss(DOT_RW_VSUM_JOBS, 1);", "csrw_ss(DOT_RW_VSUM_JOB_BEATS, 16);",
        "while (csrr_ss(DOT_RO_VMUL_STATUS) & 1) {", "while (csrr_ss(DOT_STREAMER_BUSY_CSR)) {",
        "forge_counter[0] = csrr_ss(DOT_RO_VMUL_BUSY_COUNTER);",
        "csrw_ss(DOT_STREAMER_START_CSR, 1);", "csrw_ss(DOT_RW_VSUM_START, 1);",
        "while (csrr_ss(DOT_RO_VSUM_STATUS) & 1) {", "while (csrr_ss(DOT_STREAMER_BUSY_CSR)) {",
        "forge_counter[1] = csrr_ss(DOT_RO_VSUM_BUSY_COUNTER);",
    ]  # fmt: skip
    bounds = [x for x in core if x.startswith("csrw_ss(DOT_T_BOUND_")]
    assert bounds == [
        "csrw_ss(DOT_T_BOUND_READER_0_0, 16);", "csrw_ss(DOT_T_BOUND_READER_1_0, 16);",
        "csrw_ss(DOT_T_BOUND_WRITER_0_0, 16);", "csrw_ss(DOT_T_BOUND_READER_2_0, 0);",
        "csrw_ss(DOT_T_BOUND_WRITER_1_0, 0);",
        "csrw_ss(DOT_T_BOUND_READER_0_0, 0);", "csrw_ss(DOT_T_BOUND_READER_1_0, 0);",
        "csrw_ss(DOT_T_BOUND_WRITER_0_0, 0);", "csrw_ss(DOT_T_BOUND_READER_2_0, 16);",
        "csrw_ss(DOT_T_BOUND_WRITER_1_0, 1);",
    ]  # fmt: skip


def test_dot_1_checks_both_counters(tmp_path):
    check = functions(dot1_text(tmp_path))["forge_check"]
    assert 'printf("forge counter mult_vmul: %u\\n", forge_counter[0]);' in check
    assert 'printf("forge counter sum_vsum: %u\\n", forge_counter[1]);' in check


def test_refused_a_start_without_the_idle_movers(tmp_path):
    from tests.gen.helpers import bound_dot1_enc

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_dot1_enc(), name="dot_1_enc")
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

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_dot1_enc(), name="dot_1_enc")
    d = tasks.to_dict()
    for s in d["steps"]:
        if s.get("task_name") == "mult_vsum_a":
            s["values"]["temporal_strides"] = [0]
    with pytest.raises(CError, match=r"\[c.idle\] mult_vsum_a: an idle mover"):
        main_c(p, TaskList.from_dict(d), cluster, acc, arrays, [], POINT)


def test_an_idle_mover_outside_an_encapsulated_set_is_refused(tmp_path):
    """c.idle's stride check is an encapsulated set's (dot_1_enc, above); in a set of one group an
    idle mover is no other group's, so the start is refused (c.start), whatever its stride."""
    from tests.gen.helpers import bound_recipe

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_recipe("dot_serial", "dot"),
                                           name="dot_serial")  # fmt: skip
    d = tasks.to_dict()
    for s in d["steps"]:
        if s.get("task_name") == "sum_sum_a":
            s["values"]["temporal_bounds"] = [0]
    with pytest.raises(CError, match=r"\[c.start\] sum_sum_a: an idle mover .* of sum, a set of "
                       "one group"):  # fmt: skip
        main_c(p, TaskList.from_dict(d), cluster, acc, arrays, INCLUDES, POINT)


def test_two_accelerators_each_run_through_their_own_set(tmp_path):
    """Dot_serial's mul and sum are two accelerators, two streamer sets of the one entry;
    each task configures, starts and waits on its own set's macros and its own header's CSRs
    (c.instances retired)."""
    from tests.gen.helpers import bound_recipe

    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=bound_recipe("dot_serial", "dot"),
                                           name="dot_serial")  # fmt: skip
    assert [a.name for a in acc.accelerators] == ["mul", "sum"]
    core = functions(main_c(p, tasks, cluster, acc, arrays, INCLUDES, POINT))["forge_core_0"]
    acc_lines = [x for x in core if "START" in x or "STATUS" in x or "BUSY" in x
                 or x.startswith("forge_counter") or x == BARRIER]  # fmt: skip
    # mul: its set's streamers, its CSRs, its streamer start, START; sum the same once the
    # serial sync, the one wait between them, has waited for mul (its writer too, waits.py)
    # and read its counter
    assert acc_lines == [
        "csrw_ss(MUL_STREAMER_START_CSR, 1);", "csrw_ss(MUL_RW_START, 1);",
        "while (csrr_ss(MUL_RO_STATUS) & 1) {", "while (csrr_ss(MUL_STREAMER_BUSY_CSR)) {",
        "forge_counter[0] = csrr_ss(MUL_RO_BUSY_COUNTER);",
        "csrw_ss(SUM_STREAMER_START_CSR, 1);", "csrw_ss(SUM_RW_START, 1);",
        "while (csrr_ss(SUM_RO_STATUS) & 1) {", "while (csrr_ss(SUM_STREAMER_BUSY_CSR)) {",
        "forge_counter[1] = csrr_ss(SUM_RO_BUSY_COUNTER);",
    ]  # fmt: skip
    i = core.index("csrw_ss(SUM_STREAMER_START_CSR, 1);")
    assert core.index("csrw_ss(SUM_RW_JOB_BEATS, 64);") < core.index("forge_counter[0] = "
                                                                     "csrr_ss(MUL_RO_BUSY_COUNTER);") < i  # fmt: skip
    assert "csrw_ss(SUM_BASE_PTR_READER_0_LOW, 0x10000440);" in core  # mul's output, tmp0
    assert "csrw_ss(SUM_RW_JOBS, 1);" in core and "csrw_ss(SUM_RW_JOB_BEATS, 64);" in core
    assert not any("_T_BOUND_" in x and x.endswith(", 0);") for x in core)  # no idle mover


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


def test_vecadd_marks_each_section_in_order(tmp_path):
    """dot 3's plan on vecadd: 0 from reset, then the two loads, the streamer's configuration,
    the accelerator's, the run, the store and the check, each the next value of CSR 0x7c5."""
    src, marks = quiet(tmp_path)
    assert [(m["mark"], m["kind"], m["label"]) for m in marks] == [
        (1, "dma_in", "dma load_A"), (2, "dma_in", "dma load_B"),
        (3, "streamer_config", "streamer config acc"), (4, "acc_config", "acc config acc"),
        (5, "run", "run acc"), (6, "dma_out", "dma store_C"), (7, "check", "check"),
    ]  # fmt: skip
    lines = [x.strip() for x in src.splitlines()]
    for m in marks:  # each mark once, inside a quiet block, its label as a comment
        line = f"write_csr(0x7c5, {m['mark']});  // {m['label']}"
        assert lines.count(line) == 1 and lines[lines.index(line) - 1] == QUIET_ON


def test_dot_1_repeats_the_configure_and_run_sections_per_accelerator(tmp_path):
    from tests.gen.helpers import bound_recipe

    _, marks = quiet(tmp_path, graph=bound_recipe("dot_1", "dot"), name="dot_1")
    assert [m["kind"] for m in marks] == [
        "dma_in", "dma_in", "streamer_config", "acc_config", "run", "streamer_config",
        "acc_config", "run", "dma_out", "check",
    ]  # fmt: skip
    assert [m["label"] for m in marks if m["kind"] == "run"] == ["run vmul", "run vsum"]


def test_every_start_is_waited_on_at_once_in_the_quiet_build(tmp_path):
    src, _ = quiet(tmp_path)
    f = functions(src, quiet=True)["forge_core_0"]
    for i, x in enumerate(f):
        if x.startswith("snrt_dma_start_"):
            assert f[i + 1 : i + 4] == [QUIET_ON, "snrt_dma_wait_all();", QUIET_END], x
        if x == "csrw_ss(ACC_RW_START, 1);":
            assert f[i + 1 : i + 7] == [
                QUIET_ON, "while (csrr_ss(ACC_RO_STATUS) & 1) {", "}",
                "while (csrr_ss(ACC_STREAMER_BUSY_CSR)) {", "}", QUIET_END,
            ]  # fmt: skip


def test_the_quiet_build_prints_nothing_and_the_default_build_has_no_quiet_part(tmp_path):
    src, _ = quiet(tmp_path)
    lines = src.splitlines()
    prints = [i for i, x in enumerate(lines) if "printf(" in x]
    assert len(prints) == 4 and all(
        lines[i - 1] == QUIET_OFF and lines[i + 1] == QUIET_END for i in prints
    )  # fmt: skip
    default = strip_quiet(src)
    assert QUIET_ON not in default and "write_csr" not in default and "#endif" not in default
    assert default.count("printf(") == 4 and src.count("snrt_mcycle()") == 2


def test_more_marks_than_obs_o_holds_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(c_mod, "MAX_MARKS", 6)
    with pytest.raises(CError, match=r"\[c.obs\] check: section 7, more than obs_o's 6 marks"):
        quiet(tmp_path)
