"""The C backend: a design point's task list -> ``main.c`` on snRuntime (GEN2a, D135).

    main_c(point, tasks, cluster, layout, arrays, includes, point_path) -> str

SNAX-LOWER's second backend of the task list (derive.py, C§9); SNAX-MODEL's
command list (commands.py) stays as first written until RET1 (D45). It
reads the cluster file the generator writes (gen/gvsoc.py: its cores), the
accelerator's CSR layout (gen's ``CsrLayout``: the field names of
``include/``), data.h's arrays (gen/data.py) and the files to include, in
order; the generator writes ``main.c`` beside them.

**Cores.** Each step runs on the core that owns its component, from the
cluster file: the DMA's core is the one with ``xdma``, the accelerator's
the one whose ``snax_acc_cfg`` holds the slot. ``main`` branches on
``snrt_cluster_core_idx()``, not ``snrt_is_compute_core()``: with one core
snRuntime counts zero compute cores (open item 80). The index is kept in a
``volatile``, so every branch reads it: with a constant, clang threads the
branches into one path per core and copies the code every core passes (the
exit mark) onto each, defining its label twice.

**Configure.** A streamer task is ``csrw_ss`` on streamer_csr_addr_map.h's
names: the base (``BASE_PTR_<M>_LOW`` / ``_HIGH``), the spatial strides,
then the temporal bounds and strides, padded to the streamer's
``temporal_dim`` with bound 1 and stride 0 (C§9); readers are the input
ports in the BRM's port order, writers the outputs, as the streamer file.
An accelerator task is the binding's ``csrs`` fields, from the task's
values through the binding's expressions, on ``<instance>_csr.h``'s names.
A DMA task writes nothing here: snRuntime programs and starts the DMA in one
call, so it is ``snrt_dma_start_1d`` (one loop of whole beats) or
``snrt_dma_start_2d`` (a second loop) at its start.

**Start.** One start step writes ``STREAMER_START_CSR`` once for all of an
accelerator's movers, then the accelerator's start CSR, in the step's
order; a DMA task starts at its place in the step. The step's start marks
all come before its first write, so nothing sits between the streamer's
start and the accelerator's (a mark there gave the streamer a cycle more,
one stall_in cycle fewer on the counter).

**Waits**, as ``lower_program`` places them (D64, ``placed_waits``). On
the owner's core, a wait on the accelerator or one of its streamers polls
the accelerator's busy, then the streamer's; a wait on the DMA is
``snrt_dma_wait_all()``. When the wait is for another core (a task of the
start it comes before runs elsewhere, or it is a sync, which every core
passes), every core then calls ``snrt_cluster_hw_barrier()``: every core
walks the same steps, so the barrier counts match by construction.

**Marks.** ``csrr zero, mcycle`` at a global label ``forge_<task>_<event>``
for each task's configure, start and the end of the wait that covers it,
and ``forge_main_entry`` / ``forge_main_exit``: FLOW3 reads their cycles
from the trace.

**Addresses.** L1 addresses are the memory plan's, absolute. A DMA's L2
side is the container's data.h symbol plus the task's offset into its L2
layout.

**Self-check**, on the accelerator's core after the last step: every output
against its golden (``forge data: <e> errors (<words> words)``), each
accelerator task's counter, read after its wait (``forge counter <task>:
<c>``), and ``forge errors: <err>``, the data errors plus one per counter
below its task's beats, which main returns there (0 on the other cores;
snRuntime sums them at exit).

**Refused**, each a ``CError`` (``c.<rule>``): other than one accelerator
instance, or one without a binding (``c.instances``; the generator refuses
it first, ``gen.instances``), a cluster file without one xdma core or
without the slot (``c.cores``), a start covering only some of an
accelerator's movers or starting it before them (``c.start``), a wait in
``signal`` mode (``c.signal``: nothing to signal with on SNAX), a DMA
pattern that is neither form (``c.dma``), a binding field the CSR layout
lacks (``c.csrs``), a read step (``c.read``), a task list the lowering
refuses (``c.tasks``: values that do not fit their component), a label
that is not a C symbol or not unique (``c.label``), and an accelerator
task no wait covers (``c.wait``: its counter is never read). Same design
point, same text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snax_forge import expr
from snax_forge.brm import Instance
from snax_forge.design.streamers import instances, streamer_name

from .cluster import cluster_file
from .commands import placed_waits
from .tasks import Configure, Start, Sync, TaskList, TaskListError

if TYPE_CHECKING:  # design imports lower, gen imports lower
    from snax_forge.design.point import DesignPoint
    from snax_forge.gen.data import Array

MAIN_C = "main.c"
MARK_ASM = '    asm volatile(".globl " #label "\\n" #label ":\\n\\tcsrr zero, mcycle" ::: "memory")'
SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
INDENT = "    "


class CError(ValueError):
    """A task list or design point the C backend refuses; ``code`` names the rule
    (``c.<rule>``)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def define_name(field: str) -> str:
    """A camelCase field as a define: ``loopCount`` -> ``LOOP_COUNT``, ``opSel`` -> ``OP_SEL``;
    ``<INST>_RW_<FIELD>`` in ``<instance>_csr.h`` (gen/headers.py)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field).upper()


def accelerator(point: DesignPoint) -> tuple[str, Instance]:
    """The one accelerated instance and its name; it must have a binding."""
    found = instances(point.graph)
    if len(found) != 1:
        raise CError("c.instances", f"{len(found)} accelerated instances {list(found)}; the "
                     "program drives one accelerator (open item 79)")  # fmt: skip
    (name, inst), = found.items()  # fmt: skip
    if inst.impl.binding is None:
        raise CError("c.instances", f"{name}: binding null, no CSRs to write")
    return name, inst


@dataclass(frozen=True)
class Cores:
    """The cluster file's cores: how many, the DMA's and the accelerator slot's."""

    count: int
    dma: int
    slot: int


def cores(cluster: Mapping[str, Any], slot: str) -> Cores:
    """The cores of the cluster file (a core that is a ``#/<key>`` ``$ref`` read from that key),
    in hive and core order."""
    flat = []
    try:
        for hive in cluster["cluster"]["hives"]:
            for core in hive["cores"]:
                ref = core.get("$ref")
                flat.append(cluster[ref[2:]] if isinstance(ref, str) and ref[:2] == "#/" else core)
    except (KeyError, TypeError, AttributeError) as e:
        raise CError("c.cores", f"the cluster file's hives and cores do not read: {e!r}") from None
    dma = [i for i, c in enumerate(flat) if c.get("xdma")]
    owner = [i for i, c in enumerate(flat)
             if any(a.get("snax_acc_name") == slot for a in c.get("snax_acc_cfg", []))]  # fmt: skip
    if len(dma) != 1 or len(owner) != 1:
        raise CError("c.cores", f"the cluster file has xdma on cores {dma} and the slot {slot} on cores "
            f"{owner}; the program takes one of each",
        )  # fmt: skip
    return Cores(len(flat), dma[0], owner[0])


@dataclass
class _Block:
    """Lines one core runs (``core`` None: every core)."""

    core: int | None
    lines: list[str] = field(default_factory=list)


class _Program:
    def __init__(self, point: DesignPoint, tasks: TaskList, cluster: Mapping[str, Any],
                 layout: Any, arrays: Sequence[Array]) -> None:  # fmt: skip
        self.point, self.tasks, self.layout = point, tasks, layout
        self.name, self.inst = accelerator(point)
        self.cores = cores(cluster, point.platform.accelerator_slot)
        self.arrays = list(arrays)
        self.conf = tasks.configured()
        self.beat = point.platform.l2.beat_bits // 8
        self.movers = self._movers()
        self.blocks: list[_Block] = []
        self.labels: list[str] = []
        self.started: dict[str, list[str]] = {}  # component -> its tasks, in start order
        self.order: list[str] = []  # every task, in start order
        self.waited: set[str] = set()
        self.counters: list[str] = []  # accelerator tasks, in the order their counters are read

    # -- what the design point gives -------------------------------------------------------

    def _movers(self) -> dict[str, str]:
        """Streamer component -> its mover in the headers (``READER_0``, ``WRITER_0``)."""
        out, n = {}, {"READER": 0, "WRITER": 0}
        for p in self.inst.brm.interface.ports:
            s = self.point.streamers[streamer_name(self.name, p.name)]
            kind = "WRITER" if s.write else "READER"
            out[s.name] = f"{kind}_{n[kind]}"
            n[kind] += 1
        return out

    def core_of(self, component: str) -> int:
        return self.cores.dma if component == "dma" else self.cores.slot

    # -- emitting --------------------------------------------------------------------------

    def emit(self, core: int | None, *lines: str) -> None:
        if not self.blocks or self.blocks[-1].core != core:
            self.blocks.append(_Block(core))
        self.blocks[-1].lines += lines

    def mark(self, core: int | None, task: str, event: str) -> None:
        label = f"forge_{task}_{event}"
        if not SYMBOL.fullmatch(label) or label in self.labels:
            raise CError("c.label", f"{label!r} is not a C symbol or not unique")
        self.labels.append(label)
        self.emit(core, f"FORGE_MARK({label});")

    # -- the steps -------------------------------------------------------------------------

    def configure(self, s: Configure) -> None:
        core = self.core_of(s.component)
        self.mark(core, s.task_name, "configure")
        if s.type == "streamer":
            self.emit(core, *self.streamer(s))
        elif s.type == "accel":
            env = {**self.inst.params, **s.values}
            for c in self.inst.impl.binding["csrs"]:
                if c["field"] not in self.layout.fields:
                    raise CError("c.csrs", f"{s.task_name}: no CSR for field {c['field']!r}")
                define = f"{self.name.upper()}_RW_{define_name(c['field'])}"
                self.emit(core, f"csrw_ss({define}, {expr.evaluate(c['value'], env)});")

    def streamer(self, s: Configure) -> list[str]:
        m, v = self.movers[s.component], s.values
        dims = self.point.streamers[s.component].options.temporal_dims
        bounds, strides = list(v["temporal_bounds"]), list(v["temporal_strides"])
        spatial = list(v["spatial_strides"])  # the lowering checked every count against the dims
        bounds += [1] * (dims - len(bounds))
        strides += [0] * (dims - len(strides))
        base = v["base"]
        lines = [
            f"csrw_ss(BASE_PTR_{m}_LOW, {_hex(base & 0xFFFFFFFF)});",
            f"csrw_ss(BASE_PTR_{m}_HIGH, {base >> 32});",
        ]
        lines += [f"csrw_ss(S_STRIDE_{m}_{i}, {x});" for i, x in enumerate(spatial)]
        lines += [f"csrw_ss(T_BOUND_{m}_{i}, {x});" for i, x in enumerate(bounds)]
        lines += [f"csrw_ss(T_STRIDE_{m}_{i}, {x});" for i, x in enumerate(strides)]
        return lines

    def start(self, s: Start, where: str) -> None:
        types = {t: self.conf[t].type for t in s.tasks}
        movers = [t for t in s.tasks if types[t] == "streamer"]
        if movers:
            have = {self.conf[t].component for t in movers}
            if have != set(self.movers):
                raise CError("c.start", f"{where}: starts {sorted(have)} of {self.name}'s movers "
                    f"{sorted(self.movers)}; the streamer starts all of them with one CSR",
                )  # fmt: skip
            acc = [i for i, t in enumerate(s.tasks) if types[t] == "accel"]
            if acc and acc[0] < s.tasks.index(movers[0]):
                raise CError("c.start", f"{where}: {self.name} starts before its movers")
        for t in s.tasks:  # every mark first, so none sits between the step's writes
            c = self.conf[t]
            self.started.setdefault(c.component, []).append(t)
            self.order.append(t)
            self.mark(self.core_of(c.component), t, "start")
        streamer_started = False
        for t in s.tasks:
            c = self.conf[t]
            core = self.core_of(c.component)
            if c.type == "dma":
                self.emit(core, self.dma(c, where))
            elif c.type == "streamer":
                if not streamer_started:
                    self.emit(core, "csrw_ss(STREAMER_START_CSR, 1);")
                    streamer_started = True
            else:
                self.emit(core, f"csrw_ss({self.name.upper()}_RW_START, 1);")

    def dma(self, c: Configure, where: str) -> str:
        src, dst = c.values["src"], c.values["dst"]
        to_l1 = c.values["direction"] == "l2_to_l1"
        l2, l1 = (src, dst) if to_l1 else (dst, src)
        sym = self.l2_symbol(l2["base"], c.task_name)
        l1p = f"(void *){_hex(l1['base'])}"
        d, s_ = (l1p, sym) if to_l1 else (sym, l1p)
        sb, ss, db, ds = src["bounds"], src["strides"], dst["bounds"], dst["strides"]
        if sb != db or len(sb) not in (1, 2) or ss[0] != self.beat or ds[0] != self.beat:
            raise CError("c.dma", f"{where}: {c.task_name} moves {sb} x {ss} to {db} x {ds}; snRuntime's "
                f"1-D or 2-D copy needs whole {self.beat}-byte beats inside",
            )  # fmt: skip
        size = sb[0] * self.beat
        if len(sb) == 1:
            return f"snrt_dma_start_1d({d}, {s_}, {size});"
        return f"snrt_dma_start_2d({d}, {s_}, {size}, {ds[1]}, {ss[1]}, {sb[1]});"

    def l2_symbol(self, base: int, task: str) -> str:
        """The data.h symbol plus offset of L2 byte ``base``."""
        word = self.point.platform.l2.width_bits // 8
        for a in self.arrays:
            if a.role == "golden":
                continue
            lay = self.point.memory.layouts[a.container]["l2"]
            if lay.base <= base < lay.base + len(a.values) * word:
                off = base - lay.base
                cast = "const uint8_t *" if a.const else "uint8_t *"
                return a.name if off == 0 else f"({cast}){a.name} + {off}"
        raise CError("c.dma", f"{task}: L2 byte {base} is in no array of data.h")

    def wait(self, comp: str, mode: str, covers: Mapping[str, str], consumers: set[int]) -> None:
        if mode != "poll":
            raise CError("c.signal", f"a wait on {comp} in mode {mode!r}: nothing to signal with on "
                "SNAX, only poll",
            )  # fmt: skip
        core, inst = self.core_of(comp), self.name.upper()
        if comp == "dma":
            self.emit(core, "snrt_dma_wait_all();")
        else:
            self.emit(
                core, f"while (csrr_ss({inst}_RO_BUSY)) {{", "}",
                "while (csrr_ss(STREAMER_BUSY_CSR)) {", "}",
            )  # fmt: skip
        ended = [t for c, last in covers.items()
                 for t in self.started[c][: self.started[c].index(last) + 1]
                 if t not in self.waited]  # fmt: skip
        for t in sorted(ended, key=self.order.index):  # in start order
            self.waited.add(t)
            tc = self.core_of(self.conf[t].component)
            self.mark(tc, t, "wait")
            if self.conf[t].type == "accel":
                self.counters.append(t)
                self.emit(tc, f"counter_{t} = csrr_ss({inst}_RO_PERF_COUNT);")
        if consumers - {core}:
            self.emit(None, "snrt_cluster_hw_barrier();")

    def run(self) -> None:
        for i, s in enumerate(self.tasks.steps):
            if not isinstance(s, (Configure, Start, Sync)):
                raise CError("c.read", f"{self.tasks.name}: step {i} ({s.op}) has no C form yet")
        waits: dict[int, list[Any]] = {}
        try:  # the lowering also checks every task's values against its component
            placed = placed_waits(self.tasks, cluster_file(self.point))
        except TaskListError as e:
            raise CError("c.tasks", str(e)) from None
        for w in placed:
            waits.setdefault(w.step, []).append(w)
        everyone = set(range(self.cores.count))
        for i, s in enumerate(self.tasks.steps):
            where = f"{self.tasks.name}: step {i} ({s.op})"
            if isinstance(s, Configure):
                self.configure(s)
            elif isinstance(s, Start):
                needs = {self.core_of(self.conf[t].component) for t in s.tasks}
                for w in waits.get(i, []):
                    self.wait(w.component, w.mode, w.covers, needs)
                self.start(s, where)
            else:
                for w in waits.get(i, []):
                    self.wait(w.component, w.mode, w.covers, everyone)
        unread = [t for t, c in self.conf.items() if c.type == "accel" and t not in self.waited]
        if unread:
            raise CError("c.wait", f"no wait covers {unread}: their counters are never read")

    # -- the file --------------------------------------------------------------------------

    def check(self) -> list[str]:
        outputs = [a for a in self.arrays if a.role == "output"]
        words = sum(a.elements for a in outputs)
        lines = ["int data_errors = 0;"]
        for a in outputs:
            lines += [
                f"for (uint32_t i = 0; i < {a.elements}; i++) {{",
                f"{INDENT}if (((volatile {a.ctype} *){a.name})[i] != {a.name}_golden[i]) {{",
                f"{INDENT * 2}data_errors++;",
                f"{INDENT}}}",
                "}",
            ]
        lines += [
            "err += data_errors;",
            f'printf("forge data: %d errors (%u words)\\n", data_errors, {words}u);',
        ]
        for t in self.counters:
            lines += [
                f'printf("forge counter {t}: %u\\n", counter_{t});',
                f"err += counter_{t} < {self.conf[t].values['n']};",
            ]
        lines.append('printf("forge errors: %d\\n", err);')
        return lines

    def text(self, includes: Sequence[str], point_path: str) -> str:
        self.mark(None, "main", "entry")
        self.run()
        self.emit(self.cores.slot, *self.check())
        self.mark(None, "main", "exit")
        head = [
            "// Generated by snax-forge's generator (snax_forge/gen, GEN2a) from the",
            f"// design point {point_path}. Do not edit: run",
            f"// `pixi run gen-gvsoc {point_path}` again.",
            f"// The task list {self.tasks.name} on snRuntime: core {self.cores.dma} runs the DMA,",
            f"// core {self.cores.slot} the accelerator {self.name}; the waits as the lowering places",
            "// them (D64), a barrier where another core goes on; a mark per task event.",
            "",
            '#include "snrt.h"',
            "",
            *(f'#include "{f}"' for f in includes),
            "",
            "// A mark: mcycle read into x0 at a global label, which names its pc",
            "#define FORGE_MARK(label) \\",
            MARK_ASM,
            "",
            "int main() {",
        ]
        body = [
            "// volatile: read at each branch, so the code every core passes stays one copy",
            "volatile uint32_t core = snrt_cluster_core_idx();",
            "int err = 0;",
        ]
        body += [f"uint32_t counter_{t} = 0;" for t in self.counters]
        out = head + [INDENT + x for x in body]
        for b in self.blocks:
            if b.core is None:
                out += [INDENT + x for x in b.lines]
            else:
                out.append(f"{INDENT}if (core == {b.core}) {{")
                out += [INDENT * 2 + x for x in b.lines]
                out.append(INDENT + "}")
        out += [f"{INDENT}return err;", "}"]
        return "\n".join(out) + "\n"


def _hex(v: int) -> str:
    return f"0x{v:08x}"


def main_c(point: DesignPoint, tasks: TaskList, cluster: Mapping[str, Any], layout: Any,
           arrays: Sequence[Array], includes: Sequence[str], point_path: str) -> str:  # fmt: skip
    """The text of main.c, including ``includes`` in order; see the module doc."""
    return _Program(point, tasks, cluster, layout, arrays).text(includes, point_path)
