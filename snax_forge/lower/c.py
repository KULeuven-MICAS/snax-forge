"""The C backend: a design point's task list -> ``main.c`` on snRuntime (D135).

    main_c(point, tasks, cluster, acc, arrays, includes, point_path) -> str

SNAX-LOWER's backend of the task list (derive.py, C§9). It reads the
cluster file the generator writes (gen/gvsoc.py: its cores), the
accelerator's description (gen/accelerator.py's ``Accelerator``: its CSRs,
named by its ``define`` as ``include/<name>_csr.h`` names them), data.h's
arrays (gen/data.py) and the files to include, in order; the generator
writes ``main.c`` beside them.

**Cores.** Each step runs on the core that owns its component, from the
cluster file: the DMA's core is the one with ``xdma``, the accelerator's
the one whose ``snax_acc_cfg`` holds the slot. Each core with a task gets a
function of its own, ``forge_core_<n>`` (``static``, ``noinline``): its steps
in task-list order, a straight line with the barriers at the same points as
every other core's; a core with none runs ``forge_core_idle``, the barriers
only. ``main`` reads ``mcycle`` first (``t0``), reads
``snrt_cluster_core_idx()`` once (not ``snrt_is_compute_core()``: with one
core snRuntime counts zero compute cores, open item 80) and calls its core's
function through a table; on the accelerator's core it then adds
``forge_check(t0)``, and it returns the sum.

**The window.** The program's cycles are one window, from ``mcycle`` at
main's entry to ``mcycle`` read first in ``forge_check()``, which prints it
as ``forge cycles: <n>``; the flow takes it as ``total_cycles`` (D136, C§7).
No task marks: nothing is read inside the window but the counters, so the
core's timing between the steps is the program's own.

**The accelerator** (D148) is the design point's one: one group per unit
(instance), in the graph's order, each with its own tasks; its movers are
every unit's ports, readers the input ports and writers the outputs, unit by
unit in the BRM's port order, as the streamer file.

**Configure.** A streamer task is ``csrw_ss`` on streamer_csr_addr_map.h's
names: the base (``BASE_PTR_<M>_LOW`` / ``_HIGH``), the spatial strides,
then the temporal bounds and strides, padded to the streamer's
``temporal_dim`` with bound 1 and stride 0 (C§9); an idle mover's bounds are
all 0, so it never starts. An accelerator task writes its group's JOBS = n /
T and, for a fold, its unit's JOB_BEATS = T, from the task's values (n and
the rates, C§9; ``job_values``, D146), on ``<name>_csr.h``'s names.
A DMA task writes nothing here: snRuntime programs and starts the DMA in one
call, so it is ``snrt_dma_start_1d`` (one loop of whole beats) or
``snrt_dma_start_2d`` (a second loop) at its start.

**Start.** One start step writes ``STREAMER_START_CSR`` once for all of the
accelerator's movers (the other groups' idle, at bound 0), then its group's
``START`` (bit 0 set), in the step's order, back to back; a DMA task starts
at its place in the step.

**Waits.** On the SNAX components, as waits.py places them (D64,
``placed_waits``): on the owner's core, a wait on a unit or one of its
streamers polls its group's ``STATUS & 1`` (busy), then the streamer's
busy. On the
DMA, from dependences only: the iDMA queues transfers (the hjson's
``dma_req_fifo_depth``; a full queue stalls the core), so a DMA task starts
without waiting for the one before it, and ``snrt_dma_wait_all()`` comes
only before a start whose task has an ``after`` on a DMA task not yet
waited for, and at a sync on a DMA task. When the wait is for another core (a task of the
start it comes before runs elsewhere, or it is a sync, which every core
passes), every core then calls ``snrt_cluster_hw_barrier()``: every core
walks the same steps, so the barrier counts match by construction.


**Addresses.** L1 addresses are the memory plan's, absolute. A DMA's L2
side is the data.h array its task names (``symbol``).

**Self-check**, ``forge_check(t0)`` (``static``, ``noinline``), on the
accelerator's core after its function: after ``forge cycles:``, every output
against its golden, its valid elements only (a matrix a tail pads, D144,
row by row over its padded rows; ``forge data: <e> errors (<words> words)``,
the words the valid elements), each
accelerator task's counter (``forge counter <task>: <c>``, its group's
``COUNTER``), stored in its
slot of the global ``forge_counter[]`` right after the wait that covers the
task, slots in the order the counters are read, and ``forge errors:
<err>``, the data errors plus one per counter below its task's beats, which
it returns to main (0 on the other cores; snRuntime sums them at exit).

**Refused**, each a ``CError`` (``c.<rule>``): other than one accelerator
(``c.instances``: several groups are one accelerator only through
``encapsulate``, D148; the generator refuses it first, ``gen.instances``), a
unit without a binding (``c.binding``), a cluster file without one xdma core
or without the slot (``c.cores``), a start that does not cover every mover of
the accelerator, the idle ones at bound 0, or that starts a unit before them
(``c.start``), a wait in
``signal`` mode (``c.signal``: nothing to signal with on SNAX), a DMA
pattern that is neither form (``c.dma``), a CSR the description lacks
(``c.csrs``), a read step (``c.read``), a task list whose values do
not fit the design point (``c.tasks``, ``check_values``), and an
accelerator task no wait covers (``c.wait``: its counter is never read).
Same design point, same text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from snax_forge.brm import Instance
from snax_forge.common import CodedError, generated_header
from snax_forge.design.hjson_platform import cores as hjson_cores
from snax_forge.design.problems import DesignError
from snax_forge.design.streamers import accelerators, instances, streamer_name

from .tasks import Configure, Start, Sync, TaskList, TaskListError, check_values
from .waits import PlacedWait, placed_waits

if TYPE_CHECKING:  # design imports lower, gen imports lower
    from snax_forge.design.point import DesignPoint
    from snax_forge.gen.accelerator import Accelerator
    from snax_forge.gen.data import Array

MAIN_C = "main.c"
INDENT = "    "


class CError(CodedError):
    """A task list or design point the C backend refuses; ``code`` names the rule
    (``c.<rule>``)."""


def fold_rate(inst: Instance) -> str | None:
    """The runtime param an output port's rate names, a fold's T; None for a unit whose
    outputs all have rate 1 (D146)."""
    for p in inst.brm.interface.ports:
        if p.direction == "out" and isinstance(p.rate, str):
            return p.rate
    return None


def job_values(inst: Instance, values: Mapping[str, int]) -> dict[str, int]:
    """An accelerator task's CSR values from its values (``n`` and the named rates, C§9):
    ``JOBS`` = n / T and, for a fold, ``JOB_BEATS`` = T (D146); T is 1 for a unit without a
    fold, and T = 0 is an empty run (JOBS 0)."""
    rate = fold_rate(inst)
    t = 1 if rate is None else values[rate]
    out = {"JOBS": values["n"] // t if t else 0}
    if rate is not None:
        out["JOB_BEATS"] = t
    return out


def mover_name(write: bool, index: int) -> str:
    """A streamer mover's name in the headers and the program: ``READER_<i>`` or ``WRITER_<j>``,
    readers the input ports in the BRM's port order, writers the outputs (gen/headers.py)."""
    return f"{'WRITER' if write else 'READER'}_{index}"


def accelerator(
    point: DesignPoint, fail: Callable[[str, str], Exception] | None = None
) -> tuple[str, dict[str, Instance]]:
    """The one accelerator's name and its units, one per group in the graph's order (D148),
    each with a binding. ``fail(rule, message)`` makes the error of the rule broken
    (``instances``, ``binding``): a ``c.<rule>`` CError by default, the generator's own
    (gen/gvsoc.py)."""
    if fail is None:
        fail = lambda rule, message: CError(f"c.{rule}", message)
    found = accelerators(point.graph)
    if len(found) != 1:
        raise fail("instances", f"{len(found)} accelerators {list(found)}; the platform has one "
                   "accelerator slot and the program drives one accelerator: encapsulate puts "
                   "groups into one (D148, open item 79)")  # fmt: skip
    (name, groups), = found.items()  # fmt: skip
    insts = instances(point.graph)
    units = {u: insts[u] for u in groups}
    for inst in units.values():
        if inst.impl.binding is None:
            raise fail("binding", f"{inst.brm.name}.{inst.implementation}: binding null, nothing "
                       "to generate")  # fmt: skip
    return name, units


@dataclass(frozen=True)
class Cores:
    """The cluster file's cores: how many, the DMA's and the accelerator slot's."""

    count: int
    dma: int
    slot: int


def cores(cluster: Mapping[str, Any], slot: str) -> Cores:
    """The cores of the cluster file (design/hjson_platform.py's ``cores``), in hive and core
    order."""
    try:
        flat = [core for _, core in hjson_cores(cluster)]
    except DesignError as e:
        raise CError("c.cores", f"the cluster file's hives and cores do not read: {e}") from None
    dma = [i for i, c in enumerate(flat) if c.get("xdma")]
    owner = [i for i, c in enumerate(flat)
             if any(a.get("snax_acc_name") == slot for a in c.get("snax_acc_cfg", []))]  # fmt: skip
    if len(dma) != 1 or len(owner) != 1:
        raise CError("c.cores", f"the cluster file has xdma on cores {dma} and the slot {slot} on cores "
            f"{owner}; the program takes one of each",
        )  # fmt: skip
    return Cores(len(flat), dma[0], owner[0])


class _Program:
    def __init__(self, point: DesignPoint, tasks: TaskList, cluster: Mapping[str, Any],
                 acc: Accelerator, arrays: Sequence[Array]) -> None:  # fmt: skip
        self.point, self.tasks, self.acc = point, tasks, acc
        self.name, self.units = accelerator(point)
        self.cores = cores(cluster, point.platform.accelerator_slot)
        self.arrays = list(arrays)
        self.conf = tasks.configured()
        self.beat = point.platform.l2.beat_bits // 8
        self.movers = self._movers()
        self.lines: dict[int, list[str]] = {k: [] for k in range(self.cores.count)}
        self.barriers = 0
        self.started: dict[str, list[str]] = {}  # component -> its tasks, in start order
        self.order: list[str] = []  # every task, in start order
        self.waited: set[str] = set()
        self.counters: list[str] = []  # accelerator tasks, in the order their counters are read

    # -- what the design point gives -------------------------------------------------------

    def _movers(self) -> dict[str, str]:
        """Streamer component -> its mover in the headers (``READER_0``, ``WRITER_0``): every
        unit's ports, unit by unit in port order, readers and writers counted apart."""
        out, n = {}, {False: 0, True: 0}
        for unit, inst in self.units.items():
            for p in inst.brm.interface.ports:
                s = self.point.streamers[streamer_name(unit, p.name)]
                out[s.name] = mover_name(s.write, n[s.write])
                n[s.write] += 1
        return out

    def core_of(self, component: str) -> int:
        return self.cores.dma if component == "dma" else self.cores.slot

    def unit_of(self, component: str) -> str:
        """The unit a SNAX component is or serves (a streamer's instance)."""
        return component if component in self.units else self.point.streamers[component].instance

    def csr(self, name: str, component: str, where: str) -> str:
        """The define of CSR ``name`` of ``component``'s unit (its group's or the unit's own)."""
        try:
            return self.acc.find(self.unit_of(component), name)
        except KeyError as e:
            raise CError("c.csrs", f"{where}: {e.args[0]}") from None

    # -- emitting --------------------------------------------------------------------------

    def emit(self, core: int, *lines: str) -> None:
        self.lines[core] += lines

    def barrier(self) -> None:
        """A barrier at this point of every core's line."""
        self.barriers += 1
        for lines in self.lines.values():
            lines.append("snrt_cluster_hw_barrier();")

    # -- the steps -------------------------------------------------------------------------

    def configure(self, s: Configure) -> None:
        core = self.core_of(s.component)
        if s.type == "streamer":
            self.emit(core, *self.streamer(s))
        elif s.type == "accel":
            for k, v in job_values(self.units[s.component], s.values).items():
                self.emit(core, f"csrw_ss({self.csr(k, s.component, s.task_name)}, {v});")

    def streamer(self, s: Configure) -> list[str]:
        m, v = self.movers[s.component], s.values
        dims = self.point.streamers[s.component].temporal_dims
        bounds, strides = list(v["temporal_bounds"]), list(v["temporal_strides"])
        spatial = list(v["spatial_strides"])  # check_values checked every count against the dims
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
                    f"{sorted(self.movers)}; the streamer starts all of them with one CSR, the "
                    "other groups' idle at bound 0",
                )  # fmt: skip
            acc = [i for i, t in enumerate(s.tasks) if types[t] == "accel"]
            if acc and acc[0] < s.tasks.index(movers[0]):
                raise CError("c.start", f"{where}: {self.conf[s.tasks[acc[0]]].component} starts "
                             "before its movers")  # fmt: skip
        for t in s.tasks:
            self.started.setdefault(self.conf[t].component, []).append(t)
            self.order.append(t)
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
                self.emit(core, f"csrw_ss({self.csr('START', c.component, t)}, 1);")

    def dma(self, c: Configure, where: str) -> str:
        src, dst = c.values["src"], c.values["dst"]
        to_l1 = c.values["direction"] == "l2_to_l1"
        l2, l1 = (src, dst) if to_l1 else (dst, src)
        sym = self.l2_symbol(l2["symbol"], c.task_name)
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

    def l2_symbol(self, symbol: str, task: str) -> str:
        """``symbol``, checked to be an array of data.h (not a golden)."""
        if any(a.name == symbol and a.role != "golden" for a in self.arrays):
            return symbol
        raise CError("c.dma", f"{task}: {symbol} is no array of data.h")

    def wait(self, comp: str, mode: str, covers: Mapping[str, str], consumers: set[int]) -> None:
        if mode != "poll":
            raise CError("c.signal", f"a wait on {comp} in mode {mode!r}: nothing to signal with on "
                "SNAX, only poll",
            )  # fmt: skip
        core = self.core_of(comp)
        if comp == "dma":
            self.emit(core, "snrt_dma_wait_all();")
        else:
            self.emit(
                core, f"while (csrr_ss({self.csr('STATUS', comp, comp)}) & 1) {{", "}",
                "while (csrr_ss(STREAMER_BUSY_CSR)) {", "}",
            )  # fmt: skip
        ended = [t for c, last in covers.items()
                 for t in self.started[c][: self.started[c].index(last) + 1]
                 if t not in self.waited]  # fmt: skip
        for t in sorted(ended, key=self.order.index):  # in start order
            self.waited.add(t)
            if self.conf[t].type == "accel":
                slot = len(self.counters)
                self.counters.append(t)
                c = self.conf[t].component
                self.emit(self.core_of(c),
                          f"forge_counter[{slot}] = csrr_ss({self.csr('COUNTER', c, t)});")  # fmt: skip
        if consumers - {core}:
            self.barrier()

    def dma_wait(self, i: int, s: Start) -> list[PlacedWait]:
        """The DMA wait the start ``s`` needs: one when a task of it has an ``after`` on a DMA
        task not waited for yet, ending on the latest DMA task started; else none."""
        pending = [a for t in s.tasks for a in self.conf[t].after
                   if self.conf[a].type == "dma" and a not in self.waited]  # fmt: skip
        if not pending:
            return []
        last = self.started["dma"][-1]
        return [PlacedWait(i, "dma", self.conf[last].wait_mode, {"dma": last})]

    def run(self) -> None:
        for i, s in enumerate(self.tasks.steps):
            if not isinstance(s, (Configure, Start, Sync)):
                raise CError("c.read", f"{self.tasks.name}: step {i} ({s.op}) has no C form yet")
        waits: dict[int, list[Any]] = {}
        try:
            check_values(self.tasks, self.point)
            placed = placed_waits(self.tasks, self.point)
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
                # The DMA queues: none of the lowering's waits on it, one where a dependence needs it
                here = [w for w in waits.get(i, []) if w.component != "dma"]
                here += self.dma_wait(i, s)
                for w in sorted(here, key=lambda w: self.order.index(w.covers[w.component])):
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
        """``forge_check``'s body: the window first, then the self-check."""
        outputs = [a for a in self.arrays if a.role == "output"]
        words = sum(a.elements for a in outputs)
        lines = [
            "uint32_t cycles = snrt_mcycle() - t0;",
            'printf("forge cycles: %u\\n", cycles);',
            "int err = 0;",
            "int data_errors = 0;",
        ]
        for a in outputs:
            lines += _compare(a)
        lines += [
            "err += data_errors;",
            f'printf("forge data: %d errors (%u words)\\n", data_errors, {words}u);',
        ]
        for slot, t in enumerate(self.counters):
            lines += [
                f'printf("forge counter {t}: %u\\n", forge_counter[{slot}]);',
                f"err += forge_counter[{slot}] < {self.conf[t].values['n']};",
            ]
        lines += ['printf("forge errors: %d\\n", err);', "return err;"]
        return lines

    def function(self, name: str, what: str, lines: list[str], args: str = "void") -> list[str]:
        """A ``static noinline`` function returning int: ``lines`` in order."""
        return [f"// {what}", f"static __attribute__((noinline)) int {name}({args}) {{",
                *(INDENT + x for x in lines), "}", ""]  # fmt: skip

    def text(self, includes: Sequence[str], point_path: str) -> str:
        self.run()
        working = {k: v for k, v in self.lines.items()
                   if k == self.cores.slot or any(x != "snrt_cluster_hw_barrier();" for x in v)}  # fmt: skip
        what = (
            f"The task list {self.tasks.name} on snRuntime: core {self.cores.dma} runs the DMA, "
            f"core {self.cores.slot} the accelerator {self.name}; the SNAX waits as the lowering "
            "places them (D64), the DMA's on dependences only (D135), a barrier where another "
            "core goes on. One function per core, a straight line; the window from main's entry "
            "to forge_check()'s."
        )
        head = [
            *generated_header(point_path, what).splitlines(),
            "",
            '#include "snrt.h"',
            "",
            *(f'#include "{f}"' for f in includes),
            "",
            "// Each accelerator task's counter, stored right after the wait that covers it",
            f"static uint32_t forge_counter[{len(self.counters)}];",
            "",
        ]
        out = list(head)
        for k, lines in working.items():
            roles = [r for r, c in (("the DMA", self.cores.dma), (f"the accelerator {self.name}",
                                    self.cores.slot)) if c == k]  # fmt: skip
            out += self.function(f"forge_core_{k}", f"Core {k}: {' and '.join(roles)}",
                                 [*lines, "return 0;"])  # fmt: skip
        table = [f"forge_core_{k}" if k in working else "forge_core_idle"
                 for k in range(self.cores.count)]  # fmt: skip
        if len(working) < self.cores.count:
            idle = [f"for (int i = 0; i < {self.barriers}; i++) {{",
                    f"{INDENT}snrt_cluster_hw_barrier();", "}"] if self.barriers else []  # fmt: skip
            out += self.function("forge_core_idle", "Cores with no task: the barriers only",
                                 [*idle, "return 0;"])  # fmt: skip
        out += self.function("forge_check", "The window's end, then the self-check",
                             self.check(), "uint32_t t0")  # fmt: skip
        out += [
            "// Each core's function, by snrt_cluster_core_idx()",
            f"static int (*const forge_cores[{self.cores.count}])(void) = {{{', '.join(table)}}};",
            "",
            "int main() {",
            f"{INDENT}uint32_t t0 = snrt_mcycle();",
            f"{INDENT}uint32_t core = snrt_cluster_core_idx();",
            f"{INDENT}int err = forge_cores[core]();",
            f"{INDENT}if (core == {self.cores.slot}) {{",
            f"{INDENT * 2}err += forge_check(t0);",
            f"{INDENT}}}",
            f"{INDENT}return err;",
            "}",
        ]
        return "\n".join(out) + "\n"


def _hex(v: int) -> str:
    return f"0x{v:08x}"


def main_c(point: DesignPoint, tasks: TaskList, cluster: Mapping[str, Any], acc: Accelerator,
           arrays: Sequence[Array], includes: Sequence[str], point_path: str) -> str:  # fmt: skip
    """The text of main.c, including ``includes`` in order; see the module doc."""
    return _Program(point, tasks, cluster, acc, arrays).text(includes, point_path)


def _compare(a: Any) -> list[str]:
    """The loop comparing output ``a`` with its golden, its valid elements only: one loop over
    them, or for a matrix a tail pads (D144), one loop per dimension over its own extent,
    each element at its index in the padded rows."""
    test = [
        f"{INDENT}if (((volatile {a.ctype} *){a.name})[i] != {a.name}_golden[i]) {{",
        f"{INDENT * 2}data_errors++;",
        f"{INDENT}}}",
    ]
    if len(a.shape) < 2 or tuple(a.shape) == tuple(a.padded):
        return [f"for (uint32_t i = 0; i < {a.elements}; i++) {{", *test, "}"]
    lines, terms = [], []
    for d, n in enumerate(a.shape):
        stride = 1
        for p in a.padded[d + 1 :]:
            stride *= p
        lines.append(f"{INDENT * d}for (uint32_t i{d} = 0; i{d} < {n}; i{d}++) {{")
        terms.append(f"i{d}" if stride == 1 else f"i{d} * {stride}")
    depth = len(a.shape)
    lines.append(f"{INDENT * depth}uint32_t i = {' + '.join(terms)};")
    lines += [INDENT * (depth - 1) + line for line in test]
    lines += [f"{INDENT * d}}}" for d in reversed(range(depth))]
    return lines
