"""The C backend: a design point's task list -> ``main.c`` on snRuntime.

``main_c`` writes the program of CONTRACTS, "Generated hjson files" (its ``main.c`` row), and
``program`` also returns the quiet build's marks (its ``obs_marks.json`` row); ``strip_quiet``
gives the default build's text. Inputs: the task list (CONTRACTS, "Task list"), the cluster
file's cores (``cores``), the accelerators' ``Description`` and data.h's arrays.

Each core runs ``forge_core_<n>``, its steps in task-list order, or ``forge_core_idle``, the
barriers only, picked by ``snrt_cluster_core_idx()``. A streamer task is ``csrw_ss`` writes padded
to its ``temporal_dim``; an accelerator task writes JOBS, JOB_BEATS and its scalars before START
(``job_values``, ``job_writes``); a DMA task is ``snrt_dma_start_1d`` or ``_2d``, and
``snrt_dma_wait_all()`` comes only where a DMA task is needed. A wait for another core is followed
by ``snrt_cluster_hw_barrier()`` on every core. Each refusal is a ``CError`` (``c.<rule>``).
Same design point, same text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

from snax_forge.brm import Instance
from snax_forge.common import CodedError, generated_header
from snax_forge.design.hjson_platform import CoreCount, dma_and_owner
from snax_forge.design.hjson_platform import cores as hjson_cores
from snax_forge.design.problems import DesignError
from snax_forge.design.streamers import SharedMover, groups, instances, shared_movers, streamer_name
from snax_forge.design.streamers import accelerators as graph_accelerators

from .tasks import Configure, Start, TaskList, TaskListError, check_values
from .waits import PlacedWait, placed_waits

if TYPE_CHECKING:  # design imports lower, gen imports lower
    from snax_forge.design.point import DesignPoint
    from snax_forge.gen.accelerator import Description
    from snax_forge.gen.data import Array

MAIN_C = "main.c"
OBS_MARKS = "obs_marks.json"
INDENT = "    "
QUIET = "FORGE_QUIET"  # the quiet build's define
QUIET_ON, QUIET_OFF, QUIET_END = f"#ifdef {QUIET}", f"#ifndef {QUIET}", f"#endif  // {QUIET}"
OBS_CSR = 0x7C5  # snitch.sv's CsrObsRegister; obs_o its low 8 bits
MAX_MARKS = 255


class CError(CodedError):
    """A task list or design point the C backend refuses; ``code`` names the rule
    (``c.<rule>``)."""


def fold_rate(inst: Instance) -> str | None:
    """The runtime param an output port's rate names, a fold's T; None for a unit whose
    outputs all have rate 1."""
    for p in inst.ports:
        if p.direction == "out" and isinstance(p.rate, str):
            return p.rate
    return None


# NB (D146): a unit fires JOBS · ∏ T over itself and the folds after it; its control unit
# counts exactly these, so JOBS and JOB_BEATS must match the BRM's rates.
def job_values(inst: Instance, values: Mapping[str, int], fixed: bool = False) -> dict[str, int]:
    """An accelerator task's CSR values from its values (``n`` and the named rates):
    ``JOBS`` = n / T and, for a fold, ``JOB_BEATS`` = T, none when its T is ``fixed``
    (the description's ``params.T``); T is 1 for a unit without a fold, and T = 0 is an empty
    run (JOBS 0)."""
    rate = fold_rate(inst)
    t = 1 if rate is None else values[rate]
    out = {"JOBS": values["n"] // t if t else 0}
    if rate is not None and not fixed:
        out["JOB_BEATS"] = t
    return out


def job_writes(units: Mapping[str, Instance], values: Mapping[str, Any],
               fixed: set[str]) -> list[tuple[str, str, int]]:  # fmt: skip
    """An accelerator task's CSR writes, ``(unit, CSR, value)`` in order: a unit's task
    (``units`` one, ``values`` its n and rates) as ``job_values``; a linked group's (``values``
    by unit, the group's units in order) JOBS = n_u / ∏ T_v over its first unit u and the
    group's folds v, then each fold's JOB_BEATS but those in ``fixed`` (CONTRACTS, "Task list")."""
    if "n" in values:
        ((unit, inst),) = units.items()
        return [(unit, k, v) for k, v in job_values(inst, values, unit in fixed).items()]
    first = next(iter(values))
    folds = {u: values[u][r] for u in values if (r := fold_rate(units[u])) is not None}
    t = prod(folds.values())
    out = [(first, "JOBS", values[first]["n"] // t if t else 0)]
    return out + [(u, "JOB_BEATS", v) for u, v in folds.items() if u not in fixed]


def task_beats(values: Mapping[str, Any]) -> int:
    """An accelerator task's beats: its ``n``, a linked group's most firing unit's."""
    return values["n"] if "n" in values else max(v["n"] for v in values.values())


def mover_name(write: bool, index: int) -> str:
    """A streamer mover's name in the headers and the program: ``READER_<i>`` or ``WRITER_<j>``,
    readers the input ports in the BRM's port order, writers the outputs (gen/headers.py)."""
    return f"{'WRITER' if write else 'READER'}_{index}"


def accelerators(
    point: DesignPoint, fail: Callable[[str, str], Exception] | None = None
) -> dict[str, dict[str, Instance]]:
    """The accelerators in the order the graph first uses them (CSR order), each its
    units in the graph's order, every one with a binding; ``fail(rule, message)`` makes the error
    of a unit without one (``binding``): a ``c.binding`` CError by default, the generator's own
    (gen/accelerator.py)."""
    if fail is None:
        fail = lambda rule, message: CError(f"c.{rule}", message)
    insts = instances(point.graph)
    out = {
        name: {u: insts[u] for u in units}
        for name, units in graph_accelerators(point.graph).items()
    }
    for units in out.values():
        for inst in units.values():
            if inst.impl.binding is None:
                raise fail("binding", f"{inst.brm.name}.{inst.implementation}: binding null, "
                           "nothing to generate")  # fmt: skip
    return out


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
    try:
        dma, owner = dma_and_owner(flat, slot)
    except CoreCount as e:
        raise CError("c.cores", f"the cluster file has xdma on cores {e.dma} and the slot {slot} on "
            f"cores {e.owner}; the program takes one of each",
        ) from None  # fmt: skip
    return Cores(len(flat), dma, owner)


class _Program:
    def __init__(self, point: DesignPoint, tasks: TaskList, cluster: Mapping[str, Any],
                 acc: Description, arrays: Sequence[Array]) -> None:  # fmt: skip
        self.point, self.tasks, self.acc = point, tasks, acc
        self.accs = accelerators(point)
        self.units = {u: inst for units in self.accs.values() for u, inst in units.items()}
        self.acc_of = {u: a for a, units in self.accs.items() for u in units}
        # a linked group's task is the group's: its name -> its units
        self.linked = {k: v for k, v in groups(point.graph).items()
                       if len(v) > 1 and v[0] in self.units}  # fmt: skip
        self.fixed = acc.fixed_folds
        self.cores = cores(cluster, acc.tag)
        self.arrays = list(arrays)
        self.conf = tasks.configured()
        self.beat = point.platform.l2.beat_bits // 8
        self.shared = shared_movers(point.graph, point.streamers)
        self.movers = self._movers()
        # each streamer task -> the tasks of the start that launches it
        self.start_of = {t: list(st.tasks) for st in tasks.steps if isinstance(st, Start)
                         for t in st.tasks}  # fmt: skip
        self.sel: dict[str, int] = {}  # a shared set -> its PORT_MUX_SEL as last written
        self.last_task: dict[str, str] = {}  # a set -> its last accelerator task started
        self.lines: dict[int, list[str]] = {k: [] for k in range(self.cores.count)}
        self.barriers = 0
        self.started: dict[str, list[str]] = {}  # component -> its tasks, in start order
        self.order: list[str] = []  # every task, in start order
        self.waited: set[str] = set()
        self.counters: list[str] = []  # accelerator tasks, in the order their counters are read
        self.marks: list[dict[str, Any]] = []  # the quiet build's sections, in order
        self.mark_key: tuple[str, str] | None = None  # the configure section open, if any

    # -- what the design point gives -------------------------------------------------------

    def _movers(self) -> dict[str, tuple[str, str]]:
        """Streamer component -> its accelerator (its streamer set) and its mover in the set's
        header (``READER_0``, ``WRITER_0``): per accelerator every unit's ports, unit by unit in
        port order, readers and writers counted apart."""
        out = {}
        for acc, units in self.accs.items():
            if acc in self.shared:  # every user's port on its shared mover
                n = {False: 0, True: 0}
                for m in self.shared[acc]:
                    for u in m.users:
                        out[u.streamer] = (acc, mover_name(m.write, n[m.write]))
                    n[m.write] += 1
                continue
            n = {False: 0, True: 0}
            for unit, inst in units.items():
                for p in inst.ports:
                    s = self.point.streamers.get(streamer_name(unit, p.name))
                    if s is None:  # a port on a link: no streamer
                        continue
                    out[s.name] = (acc, mover_name(s.write, n[s.write]))
                    n[s.write] += 1
        return out

    def shared_mover(self, component: str) -> SharedMover | None:
        """The shared mover a port's streamer is on, None in a set that shares none."""
        acc = self.movers[component][0]
        return next((m for m in self.shared.get(acc, []) if m.user(component)), None)

    def set_of(self, component: str) -> str:
        """The streamer set (accelerator) a SNAX component is in."""
        return self.acc_of[self.unit_of(component)]

    def prefix(self, acc: str) -> str:
        """The set's macro prefix in its streamer header (``VMUL_``)."""
        return f"{acc.upper()}_"

    @property
    def named(self) -> str:
        """The accelerators as the comments name them: ``the accelerator dot``, ``the
        accelerators mul and sum``."""
        names = list(self.accs)
        if len(names) == 1:
            return f"the accelerator {names[0]}"
        return f"the accelerators {', '.join(names[:-1])} and {names[-1]}"

    def core_of(self, component: str) -> int:
        return self.cores.dma if component == "dma" else self.cores.slot

    def unit_of(self, component: str) -> str:
        """The unit a SNAX component is or serves (a streamer's instance; a linked group's
        first unit)."""
        if component in self.linked:
            return self.linked[component][0]
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

    def quiet(self, core: int, *lines: str) -> None:
        """Lines of the quiet build only (``#ifdef FORGE_QUIET``)."""
        self.emit(core, QUIET_ON, *lines, QUIET_END)

    def next_mark(self, kind: str, label: str) -> str:
        """The next section's mark: its ``csrw`` of CSR 0x7c5, its label as a comment."""
        k = len(self.marks) + 1
        if k > MAX_MARKS:
            raise CError("c.obs", f"{label}: section {k}, more than obs_o's {MAX_MARKS} marks")
        self.marks.append({"mark": k, "kind": kind, "label": label})
        return f"write_csr({_hex(OBS_CSR, 3)}, {k});  // {label}"

    def mark(self, core: int, kind: str, label: str, key: tuple[str, str] | None = None) -> None:
        """A new section in the quiet build, unless ``key`` is the configure section open."""
        if key is not None and key == self.mark_key:
            return
        self.mark_key = key
        self.quiet(core, self.next_mark(kind, label))

    def polls(self, comp: str) -> list[str]:
        """The wait on a SNAX component: its group's STATUS bit 0 (busy), then its set's
        streamer."""
        return [f"while (csrr_ss({self.csr('STATUS', comp, comp)}) & 1) {{", "}",
                f"while (csrr_ss({self.prefix(self.set_of(comp))}STREAMER_BUSY_CSR)) {{", "}"]  # fmt: skip

    def barrier(self) -> None:
        """A barrier at this point of every core's line."""
        self.barriers += 1
        for lines in self.lines.values():
            lines.append("snrt_cluster_hw_barrier();")

    # -- the steps -------------------------------------------------------------------------

    def configure(self, s: Configure) -> None:
        core = self.core_of(s.component)
        if s.type == "streamer":
            lines = self.streamer(s)
            if lines:
                acc = self.movers[s.component][0]
                self.mark(core, "streamer_config", f"streamer config {acc}",
                          ("streamer_config", acc))  # fmt: skip
            self.emit(core, *lines)
        elif s.type == "accel":
            self.mark(core, "acc_config", f"acc config {s.component}", ("acc_config", s.component))
            mine = self.linked.get(s.component, [s.component])
            for unit, k, v in job_writes({u: self.units[u] for u in mine}, s.values, self.fixed):
                self.emit(core, f"csrw_ss({self.csr(k, unit, s.task_name)}, {v});")
            by_unit = s.values if s.component in self.linked else {s.component: s.values}
            for unit in mine:
                for slot in self.units[unit].read:
                    src = by_unit[unit][slot]["from"]
                    self.emit(core, *self.scalar(unit, slot, src, s.task_name))

    def scalar(self, unit: str, slot: str, container: str, task: str) -> list[str]:
        """The writes of a slot read from ``container`` into its CSR (two at 64 bits, low word
        first): data.h's element 0, extended to the scalar's dtype (the unit's opType)."""
        from snax_forge.gen.accelerator import scalar_csrs
        from snax_forge.gen.data import ctype

        (u,) = [x for x in self.acc.units if x.name == unit]
        dtype = u.scalars[slot]
        value = f"{self.l2_symbol(container, task)}[0]"
        names = [self.csr(c, unit, task) for c in scalar_csrs(slot, dtype)]
        if len(names) == 1:
            return [f"csrw_ss({names[0]}, {value});"]
        wide = f"(uint64_t)({ctype(dtype, slot)}){value}"
        return [f"csrw_ss({names[0]}, (uint32_t){wide});",
                f"csrw_ss({names[1]}, (uint32_t)({wide} >> 32));"]  # fmt: skip

    def streamer(self, s: Configure) -> list[str]:
        acc, mover = self.movers[s.component]
        v = s.values

        def m(kind: str) -> str:  # the set's macro: <SET>_<KIND>_<MOVER> (VMUL_BASE_PTR_READER_0)
            return f"{self.prefix(acc)}{kind}_{mover}"

        tb = list(v["temporal_bounds"])
        if tb and not any(tb):  # an idle mover: another group's in an encapsulated set only
            groups = next(len(a.groups) for a in self.acc.accelerators if a.name == acc)
            if groups == 1:
                raise CError("c.start", f"{s.task_name}: an idle mover (every temporal bound 0) of "
                             f"{acc}, a set of one group: only another group's mover in an "
                             "encapsulated set is idle, and this group's run would never "
                             "end")  # fmt: skip
            if not v["temporal_strides"][0]:
                raise CError("c.idle", f"{s.task_name}: an idle mover (every temporal bound 0) "
                             "with first temporal stride 0 starts anyway (Reader.scala: stride 0 "
                             "on loop 0 forces that bound to 1); give it a nonzero stride")  # fmt: skip
        shared = self.shared_mover(s.component)
        if shared is not None and tb and not any(tb):
            # an idle user of a shared mover: written only when no other configure of its start
            # is on that mover (its own group's), and then by the first such idle one
            same = [t for t in self.start_of.get(s.task_name, [])
                    if t in self.conf and self.conf[t].type == "streamer"
                    and self.movers[self.conf[t].component] == (acc, mover)]  # fmt: skip
            if (
                any(any(self.conf[t].values["temporal_bounds"]) for t in same)
                or same[0] != s.task_name
            ):
                return []
        dims = (
            self.point.streamers[s.component].temporal_dims
            if shared is None
            else shared.temporal_dims
        )
        bounds, strides = list(v["temporal_bounds"]), list(v["temporal_strides"])
        spatial = list(v["spatial_strides"])  # check_values checked every count against the dims
        if shared is not None and shared.masked and shared.user(s.component).num_ch == 1:
            # one channel of a masked mover: its spatial stride is no lane's address, so 0; a
            # mask that failed would write the other lanes over this one's word
            spatial = [0] * len(spatial)
        bounds += [1] * (dims - len(bounds))
        strides += [0] * (dims - len(strides))
        base = v["base"]
        lines = [
            f"csrw_ss({m('BASE_PTR')}_LOW, {_hex(base & 0xFFFFFFFF)});",
            f"csrw_ss({m('BASE_PTR')}_HIGH, {base >> 32});",
        ]
        lines += [f"csrw_ss({m('S_STRIDE')}_{i}, {x});" for i, x in enumerate(spatial)]
        lines += [f"csrw_ss({m('T_BOUND')}_{i}, {x});" for i, x in enumerate(bounds)]
        lines += [f"csrw_ss({m('T_STRIDE')}_{i}, {x});" for i, x in enumerate(strides)]
        if shared is not None and shared.masked and any(tb):
            # its user's first channels on (ENABLED_CHANNEL): (1 << num_ch) - 1, by word
            mask = (1 << shared.user(s.component).num_ch) - 1
            for i in range((shared.num_ch + 31) // 32):
                at = m("ENABLED_CHANNEL") + (f" + {i}" if i else "")
                lines.append(f"csrw_ss({at}, {_hex((mask >> (32 * i)) & 0xFFFFFFFF)});")
        return lines

    def start(self, s: Start, where: str) -> None:
        types = {t: self.conf[t].type for t in s.tasks}
        movers = [t for t in s.tasks if types[t] == "streamer"]
        for acc in dict.fromkeys(self.movers[self.conf[t].component][0] for t in movers):
            have = {self.conf[t].component for t in movers
                    if self.movers[self.conf[t].component][0] == acc}  # fmt: skip
            want = {c for c, (a, _) in self.movers.items() if a == acc}
            if have != want:
                raise CError("c.start", f"{where}: starts {sorted(have)} of {acc}'s movers "
                    f"{sorted(want)}; its streamer starts all of them with one CSR, the "
                    "other groups' idle at bound 0",
                )  # fmt: skip
        if movers:
            acc = [i for i, t in enumerate(s.tasks) if types[t] == "accel"]
            if acc and acc[0] < s.tasks.index(movers[0]):
                raise CError("c.start", f"{where}: {self.conf[s.tasks[acc[0]]].component} starts "
                             "before its movers")  # fmt: skip
        for t in s.tasks:
            self.started.setdefault(self.conf[t].component, []).append(t)
            self.order.append(t)
        streamers_started: set[str] = set()
        runs = [self.conf[t].component for t in s.tasks if types[t] == "accel"]
        run_marked = False
        for t in s.tasks:
            c = self.conf[t]
            core = self.core_of(c.component)
            if c.type == "dma":
                kind = "dma_in" if c.values["direction"] == "l2_to_l1" else "dma_out"
                self.mark(core, kind, f"dma {t}")
                self.emit(core, self.dma(c, where))
                self.quiet(core, "snrt_dma_wait_all();")
                continue
            if not run_marked:  # the run: its streamers' start to the end of its tasks
                self.mark(core, "run", f"run {', '.join(runs)}")
                run_marked = True
            if c.type == "streamer":
                acc = self.movers[c.component][0]
                if acc not in streamers_started:
                    self.port_mux(acc, s, where, core)
                    self.emit(core, f"csrw_ss({self.prefix(acc)}STREAMER_START_CSR, 1);")
                    streamers_started.add(acc)
            else:
                self.emit(core, f"csrw_ss({self.csr('START', c.component, t)}, 1);")
                self.quiet(core, *self.polls(c.component))
                self.last_task[self.set_of(c.component)] = t

    # NB (D169): the port mux knows the group only from PORT_MUX_SEL, written before each
    # STREAMER_START of a shared set.
    def port_mux(self, acc: str, s: Start, where: str, core: int) -> None:
        """A shared set's PORT_MUX_SEL, the started group's index in its port mux, written
        when it changes, after the wait on the set's previous task (``c.port_mux``)."""
        sel = self.acc.port_mux_sel(acc)
        if sel is None:
            return
        task = next(t for t in s.tasks if self.conf[t].type == "accel")
        group = self.acc.group(self.unit_of(self.conf[task].component)).name
        (a,) = (a for a in self.acc.accelerators if a.name == acc)
        index = a.port_mux.groups.index(group)
        if self.sel.get(acc) == index:
            return
        prev = self.last_task.get(acc)
        if prev is not None and prev not in self.waited:
            raise CError("c.port_mux", f"{where}: PORT_MUX_SEL of {acc} to {group} ({index}) "
                         f"while its previous task {prev} has no wait: the select changes only on "
                         "an idle streamer")  # fmt: skip
        self.emit(core, f"csrw_ss({sel}, {index});")
        self.sel[acc] = index

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
        """The data.h array (not a golden) of container ``symbol``, the task list's name for
        it: its C name (common.py's ``c_name``: mvt's ``y1`` is ``y1_l2``)."""
        for a in self.arrays:
            if a.container == symbol and a.role != "golden":
                return a.name
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
            self.emit(core, *self.polls(comp))
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
                          f"forge_counter[{slot}] = csrr_ss({self.csr('BUSY_COUNTER', c, t)});")  # fmt: skip
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
                # The DMA queues: none of the lowering's waits on it, one where a dependence needs
                # it
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
            QUIET_ON, self.next_mark("check", "check"), "(void)cycles;", QUIET_END,
            *_loud('printf("forge cycles: %u\\n", cycles);'),
            "int err = 0;",
            "int data_errors = 0;",
        ]  # fmt: skip
        for a in outputs:
            lines += _compare(a)
        lines += [
            "err += data_errors;",
            *_loud(f'printf("forge data: %d errors (%u words)\\n", data_errors, {words}u);'),
        ]
        for slot, t in enumerate(self.counters):
            lines += [
                *_loud(f'printf("forge counter {t}: %u\\n", forge_counter[{slot}]);'),
                f"err += forge_counter[{slot}] < {task_beats(self.conf[t].values)};",
            ]
        lines += [*_loud('printf("forge errors: %d\\n", err);'), "return err;"]
        return lines

    def function(self, name: str, what: str, lines: list[str], args: str = "void") -> list[str]:
        """A ``static noinline`` function returning int: ``lines`` in order."""
        return [f"// {what}", f"static __attribute__((noinline)) int {name}({args}) {{",
                *(x if x.startswith("#") else INDENT + x for x in lines), "}", ""]  # fmt: skip

    def text(self, includes: Sequence[str], point_path: str) -> str:
        self.run()
        working = {k: v for k, v in self.lines.items()
                   if k == self.cores.slot or any(x != "snrt_cluster_hw_barrier();" for x in v)}  # fmt: skip
        what = (
            f"The task list {self.tasks.name} on snRuntime: "
            + (
                f"core {self.cores.dma} runs the DMA and {self.named}; "
                if self.cores.dma == self.cores.slot
                else f"core {self.cores.dma} runs the DMA, core {self.cores.slot} {self.named}; "
            )
            + "the SNAX waits as the lowering "
            "places them, the DMA's on dependences only, a barrier where another "
            "core goes on. One function per core, a straight line; the window from main's entry "
            "to forge_check()'s. Built with -DFORGE_QUIET: every start waited at once, "
            "each section marked in CSR 0x7c5 (obs_marks.json), no prints."
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
            roles = [r for r, c in (("the DMA", self.cores.dma), (self.named,
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


def _hex(v: int, digits: int = 8) -> str:
    return f"0x{v:0{digits}x}"


def _loud(line: str) -> list[str]:
    """A line of the default build only (``#ifndef FORGE_QUIET``): a print."""
    return [QUIET_OFF, line, QUIET_END]


# NB (D116): one program runs on SNAX-GVSoC and on snax_cluster's RTL; it never branches on
# which.
def program(point: DesignPoint, tasks: TaskList, cluster: Mapping[str, Any], acc: Description,
            arrays: Sequence[Array], includes: Sequence[str],
            point_path: str) -> tuple[str, list[dict[str, Any]]]:  # fmt: skip
    """main.c's text, including ``includes`` in order, and the quiet build's marks
    (``obs_marks``); see the module doc."""
    p = _Program(point, tasks, cluster, acc, arrays)
    text = p.text(includes, point_path)
    return text, list(p.marks)


def main_c(point: DesignPoint, tasks: TaskList, cluster: Mapping[str, Any], acc: Description,
           arrays: Sequence[Array], includes: Sequence[str], point_path: str) -> str:  # fmt: skip
    """The text of main.c, including ``includes`` in order; see the module doc."""
    return program(point, tasks, cluster, acc, arrays, includes, point_path)[0]


def strip_quiet(text: str) -> str:
    """main.c as the default build compiles it: the ``#ifdef FORGE_QUIET`` blocks gone, the
    lines of each ``#ifndef FORGE_QUIET`` block kept without the directives."""
    out, skip = [], False
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if s == QUIET_ON:
            skip = True
        elif s == QUIET_END:
            skip = False
        elif s != QUIET_OFF and not skip:
            out.append(line)
    return "".join(out)


def _compare(a: Any) -> list[str]:
    """The loop comparing output ``a`` with its golden, its valid elements only: one loop over
    them, or for a matrix a tail pads, one loop per dimension over its own extent,
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
