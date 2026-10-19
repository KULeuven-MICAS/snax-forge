"""The C backend (GEN2a, D135): vecadd's main.c read as text, vsum's counter, every
refusal, and vecadd's main.c compiled against snax-gvsoc's sw/ where the image's clang is."""

from __future__ import annotations

import copy
import re
import shutil
import subprocess

import pytest

from snax_forge.design import DesignPoint
from snax_forge.dfg import Graph
from snax_forge.gen import gvsoc as gen_mod
from snax_forge.gen.data import arrays as data_arrays
from snax_forge.lower import CError, TaskList, main_c, task_list
from snax_forge.lower.c import cores
from tests.design.helpers import chained, graph_dict
from tests.gen.helpers import SUB, bound_vecmul, bound_vsum, point

POINT = "out/design/vecadd/design_point.json"


def parts(tmp_path, graph=None, name="vecadd"):
    """What main_c reads for vecadd (or ``graph``) on snax_alu."""
    p = DesignPoint.load(point(tmp_path, name=name, graph=graph))
    inst_name, inst, unit = gen_mod.accelerator(p)
    layout = gen_mod.csr_layout(gen_mod.streamer_template(p, inst_name, inst), unit.fields)
    cluster = gen_mod.source(p.platform, tmp_path)
    return p, task_list(p), cluster, layout, data_arrays(p)


INCLUDES = ["data.h", "acc_csr.h", "streamer_csr_addr_map.h"]


def text(tmp_path, edit_tasks=None, edit_cluster=None, **kw) -> str:
    p, tasks, cluster, layout, arrays = parts(tmp_path, **kw)
    if edit_tasks:
        d = tasks.to_dict()
        edit_tasks(d["steps"])
        tasks = TaskList.from_dict(d)
    if edit_cluster:
        cluster = copy.deepcopy(cluster)
        edit_cluster(cluster)
    return main_c(p, tasks, cluster, layout, arrays, INCLUDES, POINT)


def functions(src: str) -> dict[str, list[str]]:
    """main.c's functions by name (``forge_core_0``, ``forge_core_idle``, ``main``): their
    lines, stripped."""
    out: dict[str, list[str]] = {}
    name = None
    for line in src.splitlines():
        m = re.fullmatch(
            r"(?:static __attribute__\(\(noinline\)\) )?int (\w+)\((?:void)?\) \{", line
        )
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
    assert list(f) == ["forge_core_0", "forge_core_1", "main"]
    assert not any("snrt_dma" in x for x in f["forge_core_0"])
    assert not any("csrw_ss" in x or "csrr_ss" in x for x in f["forge_core_1"])
    assert "static int (*const forge_cores[2])(void) = {forge_core_0, forge_core_1};" in src
    assert f["main"] == [
        "FORGE_MARK(forge_main_entry);", "int err = forge_cores[snrt_cluster_core_idx()]();",
        "FORGE_MARK(forge_main_exit);", "return err;",
    ]  # fmt: skip
    assert "volatile" not in "\n".join(f["main"]) and "snrt_is_compute_core" not in src
    assert f["forge_core_1"][-1] == "return 0;" and f["forge_core_0"][-1] == "return err;"


def test_vecadd_barriers_follow_the_cross_core_waits(tmp_path):
    f = functions(text(tmp_path))
    c0, c1 = f["forge_core_0"], f["forge_core_1"]
    b0 = [i for i, x in enumerate(c0) if x == BARRIER]
    b1 = [i for i, x in enumerate(c1) if x == BARRIER]
    assert len(b0) == len(b1) == 3
    # the loads' wait (core 1), the writer's wait (core 0, its counter read last), the store's
    # sync (core 1): each the owner's, then the barrier on both
    assert c1[b1[0] - 1] == "FORGE_MARK(forge_load_B_wait);"
    assert c0[b0[1] - 1] == "counter_add_acc = csrr_ss(ACC_RO_PERF_COUNT);"
    assert c1[b1[2] - 1] == "FORGE_MARK(forge_store_C_wait);"
    assert c0[b0[0] + 1] == "FORGE_MARK(forge_add_acc_a_start);"
    assert c1[b1[1] + 1] == "FORGE_MARK(forge_store_C_start);"
    # the check after the last barrier, on the accelerator's core
    assert c0[b0[2] + 1] == "int data_errors = 0;"
    # load_B starts with no wait after load_A: the DMA queues (D135); one wait ends both
    a, b = c1.index("FORGE_MARK(forge_load_B_start);"), c1.index("FORGE_MARK(forge_load_A_wait);")
    assert "snrt_dma_wait_all();" not in c1[:a] and c1.count("snrt_dma_wait_all();") == 2
    assert c1[b - 1 : b + 2] == [
        "snrt_dma_wait_all();", "FORGE_MARK(forge_load_A_wait);", "FORGE_MARK(forge_load_B_wait);"
    ]  # fmt: skip


def test_a_dependence_on_a_dma_task_brings_its_wait_back(tmp_path):
    def b_after_a(steps):
        next(s for s in steps if s.get("task_name") == "load_B")["after"] = ["load_A"]

    lines = functions(text(tmp_path, edit_tasks=b_after_a))["forge_core_1"]
    i = lines.index("FORGE_MARK(forge_load_B_start);")
    assert lines[i - 3 : i] == [
        "FORGE_MARK(forge_load_B_configure);", "snrt_dma_wait_all();",
        "FORGE_MARK(forge_load_A_wait);",
    ]  # fmt: skip
    assert lines.count("snrt_dma_wait_all();") == 3


def test_vecadd_marks_every_task_event_once(tmp_path):
    src = text(tmp_path)
    labels = re.findall(r"FORGE_MARK\((\w+)\);", src)
    tasks = ["load_A", "load_B", "add_acc_a", "add_acc_b", "add_acc_out", "add_acc", "store_C"]
    want = {f"forge_{t}_{e}" for t in tasks for e in ("configure", "start", "wait")}
    assert set(labels) == want | {"forge_main_entry", "forge_main_exit"}
    assert len(labels) == len(set(labels))
    # a wait's marks in start order, then the counter read after the accelerator's
    i = labels.index("forge_add_acc_a_wait")
    assert labels[i : i + 4] == [f"forge_add_acc_{x}_wait" for x in ("a", "b", "out")] + [
        "forge_add_acc_wait"
    ]


def test_vecadd_starts_the_movers_with_one_csr_then_the_accelerator(tmp_path):
    src = text(tmp_path)
    assert src.count("csrw_ss(STREAMER_START_CSR, 1);") == 1
    start = src.index("FORGE_MARK(forge_add_acc_a_start);")
    order = [
        "FORGE_MARK(forge_add_acc_a_start);", "FORGE_MARK(forge_add_acc_b_start);",
        "FORGE_MARK(forge_add_acc_out_start);", "FORGE_MARK(forge_add_acc_start);",
        "csrw_ss(STREAMER_START_CSR, 1);", "csrw_ss(ACC_RW_START, 1);",
    ]  # fmt: skip
    assert [src.index(x, start) for x in order] == sorted(src.index(x, start) for x in order)
    # every mark of the step before its first write: the two start CSRs back to back
    assert "csrw_ss(STREAMER_START_CSR, 1);\n    csrw_ss(ACC_RW_START, 1);" in src


def test_vecadd_configures_from_the_task_values(tmp_path):
    src = text(tmp_path)
    for line in [
        "csrw_ss(BASE_PTR_READER_1_LOW, 0x10000200);", "csrw_ss(T_BOUND_WRITER_0_0, 16);",
        "csrw_ss(T_STRIDE_READER_0_0, 32);", "csrw_ss(S_STRIDE_READER_0_0, 8);",
        "csrw_ss(ACC_RW_LOOP_COUNT, 16);", "snrt_dma_start_1d((void *)0x10000000, A, 512);",
        "snrt_dma_start_1d(C, (void *)0x10000400, 512);",
    ]:  # fmt: skip
        assert line in src, line


def test_vecadd_checks_itself(tmp_path):
    src = text(tmp_path)
    assert "if (((volatile int64_t *)C)[i] != C_golden[i]) {" in src
    assert 'printf("forge data: %d errors (%u words)\\n", data_errors, 64u);' in src
    assert 'printf("forge counter add_acc: %u\\n", counter_add_acc);' in src
    assert "err += counter_add_acc < 16;" in src
    assert src.index("counter_add_acc = csrr_ss(ACC_RO_PERF_COUNT);") < src.index("forge data")


def test_vsum_sets_len_and_reads_its_counter(tmp_path):
    src = text(tmp_path, graph=bound_vsum(), name="vsum")
    assert "csrw_ss(ACC_RW_LOOP_COUNT, 64);\n    csrw_ss(ACC_RW_LEN, 64);" in src
    assert "err += counter_sum_acc < 64;" in src


def test_the_same_point_gives_the_same_text(tmp_path):
    assert text(tmp_path) == text(tmp_path)


def test_one_core_has_no_barriers(tmp_path):
    def one_core(c):  # the slot onto the DMA core: one core runs both (open item 80)
        dma = c["dma_core_template"]
        dma["snax_acc_cfg"] = c["snax_alu_core_template"]["snax_acc_cfg"]
        c["cluster"]["hives"][0]["cores"] = [{"$ref": "#/dma_core_template"}]

    src = text(tmp_path, edit_cluster=one_core)
    assert "hw_barrier" not in src and list(functions(src)) == ["forge_core_0", "main"]
    assert "// Core 0: the DMA and the accelerator acc" in src


def test_a_core_with_no_task_takes_part_in_the_barriers_only(tmp_path):
    def three_cores(c):
        c["cluster"]["hives"][0]["cores"].append({"isa": "rv32ima", "xdma": False})

    src = text(tmp_path, edit_cluster=three_cores)
    f = functions(src)
    assert list(f) == ["forge_core_0", "forge_core_1", "forge_core_idle", "main"]
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
    assert cores(gen_mod.source(p.platform, tmp_path), "snax_alu").count == 2


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


def test_refused_a_label_that_is_not_a_symbol(tmp_path):
    def rename(steps):
        for s in steps:
            if s.get("task_name") == "store_C":
                s["task_name"] = "store-C"
            if s["op"] == "start" and s["tasks"] == ["store_C"]:
                s["tasks"] = ["store-C"]
            if s["op"] == "sync":
                s["task"] = "store-C"

    with pytest.raises(CError, match=r"\[c.label\] 'forge_store-C_configure'"):
        text(tmp_path, edit_tasks=rename)


def test_refused_an_accelerator_task_no_wait_covers(tmp_path):
    def no_store(steps):
        steps[:] = [s for s in steps if "store_C" not in (s.get("task_name"), s.get("task"))
                    and s.get("tasks") != ["store_C"]]  # fmt: skip

    with pytest.raises(CError, match=r"\[c.wait\] no wait covers \['add_acc'"):
        text(tmp_path, edit_tasks=no_store)


def test_refused_a_read_step(tmp_path):
    def read(steps):
        steps.append({"op": "read", "reg": "acc.perf"})

    with pytest.raises(CError, match=r"\[c.read\] vecadd: step 12 \(read\)"):
        text(tmp_path, edit_tasks=read)


def test_refused_a_field_without_a_csr(tmp_path):
    p, tasks, cluster, layout, arrays = parts(tmp_path)
    no_fields = gen_mod.CsrLayout(layout.streamer_first, layout.streamer_last, {}, layout.start,
                                  layout.busy, layout.counter)  # fmt: skip
    with pytest.raises(CError, match=r"\[c.csrs\] add_acc: no CSR for field 'loopCount'"):
        main_c(p, tasks, cluster, no_fields, arrays, INCLUDES, POINT)


def test_refused_two_instances(tmp_path):
    _, _, cluster, layout, _ = parts(tmp_path / "one")
    gd = graph_dict()
    chained(gd)
    p = DesignPoint.load(point(tmp_path, graph=Graph.from_dict(gd)))
    with pytest.raises(CError, match=r"\[c.instances\] 2 accelerated instances"):
        main_c(p, task_list(p), cluster, layout, [], INCLUDES, POINT)


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
    "generated/snax_alu_cluster_old", "runtime/rtl-generic/src", "runtime/common",
    "snRuntime/api", "snRuntime/api/omp", "snRuntime/src", "snRuntime/src/omp",
    "deps/riscv-opcodes", "math/include",
]  # fmt: skip


def _riscv_clang() -> str | None:
    """Why vecadd's main.c cannot be compiled here, or None."""
    if CLANG is None or NM is None:
        return "no clang or llvm-nm on PATH (the image's, snax_cluster's toolchain)"
    if not (SW / "snRuntime").is_dir():
        return "the snax-gvsoc submodule is not checked out (CI never fetches it)"
    probe = subprocess.run([CLANG, "-print-targets"], capture_output=True, text=True, check=False)
    if "riscv32" not in probe.stdout:
        return f"{CLANG} has no riscv32 target"
    return None


@pytest.mark.skipif(_riscv_clang() is not None, reason=str(_riscv_clang()))
@pytest.mark.parametrize("design", ["vecadd", "vecmul", "vsum"])
def test_main_c_compiles_against_snax_gvsocs_sw(tmp_path, design):
    """Each mark's label defined once in the object (llvm-nm): a branch per core in one function
    once let clang copy vsum's exit mark, one label twice."""
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
    labels = re.findall(r"FORGE_MARK\((\w+)\);", (gen / "main.c").read_text())
    nm = subprocess.run([NM, "--defined-only", str(gen / "main.o")], capture_output=True,
                        text=True, check=True).stdout  # fmt: skip
    defined = [line.split()[-1] for line in nm.splitlines()]
    assert sorted(x for x in defined if x.startswith("forge_") and x in labels) == sorted(labels)
    assert all(defined.count(x) == 1 for x in labels)
