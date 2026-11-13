# SNAX-FORGE Contracts

> What the artefacts between the components mean, field by field, so each component can produce its output without reading another's code. Why each is the way it is: `docs/DECISIONS.md` and the module docstrings.
>
> Sections 2–6 and 8 were SNAX-MODEL's, retired in RET1 (D116); their numbers are kept. The artefacts are dataclasses and plain JSON until the M6 freeze (D26); the generated hjson files already have schemas (section 16). Every fenced block under a `<!-- snippet: ... -->` line is copied verbatim out of the file it names, and `tests/test_contracts.py` fails if one drifts (D46).

---

## 1. Conventions

**Byte addresses.** Every memory address (L1 and L2 addresses, streamer `base`, all strides, DMA bases) is a byte address.

**The bank word is the unit of transfer** (D142). A word is the L1's `width_bits` wide and holds num_slc = `width_bits` / b slices (elements) of each container's dtype (b its bits), little-endian as NumPy lays them out. A port of W lanes uses num_ch = W / num_slc streamer channels by default; `connect.pack` and `connect.num_ch` check a port against that (section 14), and lowering lays each container out by its itemsize.

**Integers first** (D28). The default `dtype` is `int64`.

The L1 of `platforms/snax_forge.json` (section 13):

<!-- snippet: platforms/snax_forge.json -->
```json
 "l1": {
  "n_banks": 32,
  "width_bits": 64,
  "rows": 512,
  "base_addr": 268435456,
  "wide_bits": 512,
  "max_xbar_inputs": 128,
  "warn_xbar_inputs": 64
 },
```

## 2. Cluster configuration

Retired in RET1 (SNAX-MODEL).

## 3. Streamer register layout

Retired in RET1 (SNAX-MODEL).

## 4. Accelerator interface

Retired in RET1 (SNAX-MODEL).

## 5. Control program

Retired in RET1 (SNAX-MODEL).

## 6. Scenario and memory

Retired in RET1 (SNAX-MODEL).

## 7. Profile and trace

What the reports and the viewer read of a flow run (D137). The flow folder's default `<name>` (`flow/run.py`'s `default_name`) is the recipe's name with each `--set` appended as `_<path><value>` (`vecmul_dry_W4`, `vecadd_l1.n_banks32`), a `-` between the two where the path ends in a digit (`dot_1_dry_W_acc_1-10_W_acc_2-4`), then `_<level>` above trace `off`. A run directory (`out/flow/<name>/run/`, `run/outputs.py`) holds `run.json` (what ran, D50), `profile.json`, and above trace level `off` `trace.jsonl` and `trace_meta.json`. SNAX-GVSoC writes its events (snax-gvsoc `docs/EVENTS.md`) into `run/events.log`, which `run/convert.py` renames into what follows; the invariants every conversion checks are its own. A passing flow leaves it gzipped, `run/events.log.gz` (no timestamp), which `read_events` reads under either name; a failing one keeps the plain file.

**A failed run** (`flow/run.py`). When SNAX-GVSoC's run exits non-zero or prints no `forge cycles:` line, the failure adds `; SNAX-GVSoC: <line>`, the simulator's first error line from `run/stdout.txt`, then `run/stderr.txt` (GVSoC's fatal line, its component's path and colour codes dropped, or an `ASSERT FAILED` one; `first_error`): the FlowError of a missing line (vecmul at W = 32: `check: no 'forge cycles:' line in the program's output; SNAX-GVSoC: nb_inputs must be in (0, 64], got 99`), or the summary's FAILED check line of a program that exits non-zero. The `sw` step first checks that `clang` and `ld.lld` are on PATH and otherwise stops with `sw: no ld.lld on PATH: the program builds in snax-forge's image, ghcr.io/kuleuven-micas/snax-forge:main (docs/TOOL.md section 1)`, in place of the compiler's `clang: error: invalid linker name in argument '-fuse-ld=lld'`.

**The window.** `total_cycles` is the program's: its `mcycle` read in `forge_check()` minus its read in main, the `forge cycles:` line. The converter finds the same two reads in the accelerator's core's events, checks they are `total_cycles` apart and shifts every cycle so that t = 0 is the read in main (the shift, the boot, in `trace_meta.json`); events outside `[0, total_cycles)` are dropped. The profile, a core's classes and the trace cover this one window.

**Profile** (`run/profile.py`): the totals of one run.

<!-- snippet: snax_forge/run/profile.py -->
```python
@dataclass
class Profile:
    total_cycles: int  # the program's window, its `forge cycles:` line
    accelerators: dict[str, AccelProfile] = field(default_factory=dict)
    streamers: dict[str, StreamerProfile] = field(default_factory=dict)
    cores: dict[str, CoreProfile] = field(default_factory=dict)
    dmas: dict[str, DmaProfile] = field(default_factory=dict)
    ports: dict[str, PortProfile] = field(default_factory=dict)
    banks: BankProfile | None = None
    functional_check: dict[str, Any] | None = None
```

| Part | Holds |
|---|---|
| accelerator | its classes summed over its runs (`busy`: a firing or the II gap after one, `stall_in`, `stall_out`, `flush`), `idle` the rest of the window, `firings`, `utilisation` (`busy` over the window) |
| streamer | per mover `busy`, `stall_xbar`, `stall_fifo`, `idle`, its crossbar ports, its FIFO's name, depth and highest count per lane |
| core | its window as `barrier`, else `poll` (reads of one CSR with one value, first to last), else `sleep` (`wfi`), else `run`; its SNAX CSR writes and reads; each wait (`poll` or `barrier`) with the block, first and last cycle. Classes and waits need the events: `None` and empty at `off` |
| DMA | transfers, busy cycles (what `DMA_BUSY` counts), read and write beats, bytes each way |
| port | per crossbar input, its grants, stalls and the stalls lost to the DMA's superbank; a core's or AXI's over the whole run (`whole_run`) |
| banks | reads, writes and conflicts (cycles with two or more requests on a bank), from the streamers and the DMA only |
| `functional_check` | `kernel`, `seed`, `symbols`; `exit_code`; `data_errors` and `words` (the `forge data:` line); `tasks` (per accelerator task its `counter`, the `forge counter` line, its `beats`, from the manifest, and its `group`, D148); `passed` (exit code 0) |

Names, ports, `write` and the FIFO depths come from the hjson files and the design point.

**Trace** (`run/trace.py`): one event per line of `trace.jsonl`, a flat dict of `t`, `k`, `src` and the kind's fields. The levels are cumulative, each turning on SNAX-GVSoC traces: `off` none, `task` the `events` lines, `beat` the `beats` lines, `stall` the `stalls` lines, `finegrained` the cores' `insn` lines.

<!-- snippet: snax_forge/run/trace.py -->
```
    task         start    task  (t = the cycle the start is taken; busy from t + 1)
    task         done     task, cycles (the run's classes), then firings, first, last for an
                          accelerator  (t = first idle)
    task         csr      pc, csr, block, op (r/w), value, held (a held write's cycles)
    task         poll     pc, csr, block, value, count, last  (t = the first read)
    task         barrier  last  (t = the barrier's read; last = the cycle it is released)
    task         state    mode (run/sleep)  (t = the first cycle in it)
    task         func     name, pc  (t = the function's first instruction)
    task         issue    task, id, dir, from, to, size, from_stride, to_stride, reps  (a copy)
    task         done     task, id  (a DMA copy; t = the cycle it completes)
    beat         grant    task, op, mem, addr, bank, row, value, back, region, elems, vals;
                          b (a SNAX port's), size (under a word)  (src = the port)
    beat         dma      task, id, i, op, mem, addr, size, bank, row, value, l2, region,
                          elems, vals; back (a read's)  (a DMA beat on L1)
    beat         burst    task, id, op, mem, addr, size, back, region, elems  (a request to AXI)
    beat         fire     task, n, b, value, vals  (b, value, vals keyed by reader mover)
    beat         push     task, n, first, last, value, vals
    beat         class    task, mode  (src = acc or the mover; t = the first cycle in it)
    stall        stall    task, addr, bank, row, by, by_addr, region, elems; b (a SNAX
                          port's)  (src = the port; one per cycle refused)
    stall        fifo     task, lane, count  (src = the mover; t = the first cycle it holds)
    finegrained  insn     pc, func, text  (src = the core)
```

`src` is a unit (`acc`; with several groups each, `mul`, `sum`, D148) or a linked group (`dot`: SNAX-GVSoC's one component for its units, `snax_acc<i>/group_<group>`, the manifest's trace path), a mover (`acc_a`), `dma`, `core0`, and for a `grant` or `stall` the crossbar port (`acc_a.lane0`, `core0`, `axi`). A linked group's lines are its boundary's (snax-gvsoc `docs/EVENTS.md`): a run's `count` and `firings` are the head unit's, `rate` the head's firings per result of the last unit, and a `push` line's `first` and `last` the head's firings it folds. `block` is the component whose CSR it is (`acc`, a group's unit or a linked group; `acc_streamer`, `<accelerator>_streamer`, its streamer set's). Each streamer set's streamer is read at its trace path, the manifest's `csrs.streamers[].trace` (`snax_acc<i>/streamer_<set>`, section 16, D157), its `mover` numbered within the set (`reader0` of `streamer_vsum` is `vsum_a`), and the crossbar's inputs are `snax_acc<i>.streamer_<set>.<mover>.lane<l>` (`run/cluster.py`, `run/convert.py`, `run/sources.py`). The beat levels read an accelerator of several groups, each of one unit (dot 1's `mul` and `sum`, D148): each unit's `fire` and `push` lines are its own (`src` the unit, `b`, `value` and `vals` keyed by its own readers, `mul_a` and `mul_b` for the mul), and a streamer line's `task` is its mover's task bracketing it, that task paired with the unit task its start launched it with (`run/beats.py`'s `pairing`, from the task list's `start` steps: a unit's start launches its own set's movers only), so a firing's operands are the grants of the streamer tasks launched with it; a group of several linked units at beat level stays refused (`convert.level`: `trace beat: the beat lines of a group of several linked units are not read yet (open item 96); task and off are`). A DMA copy's `dir` (`l2_to_l1`, `l1_to_l2`) comes from its addresses, and copies are matched to the task list's `load_<C>` and `store_<C>` tasks in issue order.

**The run report and the run viewer** (`report/run.md`, `pixi run view`). `run.md`'s Accelerators section opens with a row per accelerator, its streamer set (D157): its groups, its CSRs (its streamer's, then its own, first to last), its streamer's busy windows (from its start to its busy low: its movers' first start to their last done, one window per start) and each group's BUSY_COUNTER; dot 1 (N = 64) `vmul` 960–981 and `vsum` 982–999, `dot_1_enc` `dot` 960–996 with two windows. Its Units table then has a row per unit (a linked group's under the group's name), its first column `Unit`, and the Tasks table names each task's accelerator. An accelerator's movers are listed in a drop-down per group (`vmul` ▸ `vmul_a`, `vmul_b`, `vmul_out`; `vsum` ▸ `vsum_a`, `vsum_out`): in `run.md` a `<details>` per group, its summary `vmul ▸ vmul_a, vmul_b, vmul_out` and inside the movers' port, direction, num_ch, busy, crossbar and FIFO stalls and FIFO highest against depth; in the viewer's Accelerator section the same, each headed accelerator ▸ group. A streamer task with zero bounds (every temporal bound 0), started only because an encapsulated set's movers share one start (the idle mover of another group in the same encapsulated set, section 9), is marked idle: `mult_vsum_a (idle)` in `dot_1_enc`'s task table, and the schedule's bar `idle: true`, drawn dashed (`viz/index.py`'s `idle_tasks`); dot 1's two sets have none. A task's BUSY_COUNTER is shown beside its expected BUSY_COUNTER, a guide and not a check (`mult_mul 256 (expected 259, a guide)`, D155, section 19). The schedule's lanes go accelerator ▸ group, as the editor's Bindings (D156): per group its movers, then its unit. The cluster view at a cycle (`viz/cluster.py`) gives the unit whose task is open there as `acc`, with `accelerator` the entry's name and `units` every unit's task, class, fire and push, and `streamers`, one per set inside the accelerator, each busy while a task of its movers is open; the schedule's detail gives each unit's fires and pushes under `units`.

**Ids.** Lines are linked by ids, never by order: a streamer beat is its mover and `b` (a reader's numbered in its AGU's order from 0 per run, a writer's in the order the accelerator hands them over; a repeated beat is one `b` for several firings); a firing's `n` is its index in the run, a push's its index among the run's results, equal to the writer's `b`; a DMA beat on L1 is its copy's `id` and `i`. `fire`'s `b`, `value` and `vals` are keyed by reader mover; `push`'s `first` and `last` are the firings it folds.

**Added by the converter**: `task` (the task whose start and done bracket the line on its component, a copy's for the DMA, null for a core's or AXI's access), `mem` (`l1`, `l2`) and the absolute `addr`, `region` (the container the address is in, or null), `elems` (the flat indexes of its elements in the word, a burst's bytes), `vals` (those elements' values by the region's dtype; a fire's and a push's from their beats' grants, null for a mover not traced), an `insn`'s `func`. Every line keeps its raw words: `value` the bank word (`dma`: one per bank it covers), `bank` and `row` the word-interleaved map of `addr`, `back` the cycle the response arrives. The beat and stall lines only feed the trace; the profile stays the summary lines.

**Classes.** An accelerator's or a mover's `done` carries its run's `cycles` by class, from `task` on. From `beat` on, a `class` event gives the first cycle of each run of one class, as SNAX-GVSoC classed it: per component run they cover the cycles its counts add up over, with no gap and no overlap.

**Sources and meta.** `--trace LEVEL[:SOURCES]` names the sources whose beat, stall and instruction lines are written (`acc`, the movers, `dma`, `core<i>`, `axi`); by default the accelerator, its movers and the DMA, at `finegrained` also every core. `summary` and `events` lines are always whole. `trace_meta.json` holds `level`, `total_cycles`, `boot`, `sources` (the cluster's order), `filter` (the sources written) and `start`: per core `{"func": ..., "mode": ...}`, its last `func` at or before t = 0 (or null) and its state there. `--vcd` writes a VCD and `view.gtkw`, never read here. `run.json` records `trace_level`, `trace_sources` and `vcd`.

**Regions** (`run/regions.py`, `run/elf.py`, D86). A container's L1 region is the design point's layout; its L2 region has the plan's shape and dtype at its array's address in the linked program: `run.json`'s `elf_symbols`, container -> `addr` and `size` (its padded shape's bytes, a tail's pads included (D144), in whole DMA beats: `lower.layout.l2_bytes`, the array data.h declares and the DMA's copies touch; a missing symbol or another size stops the flow). Everything takes regions from `regions_of(point, elf_symbols)` only.

## 8. Rules for a new block kind

Retired in RET1 (SNAX-MODEL).

## 9. Task list

SNAX-LOWER's ordered list of tasks (D64): which component runs which task with which values, where each is configured and started, and what it waits for. `lower/c.py` turns it into the program (section 16). It is derived from a design point (`lower/derive.py`, `pixi run lower tasks DESIGN_POINT`) or written by hand. A task list is `name` and `steps`, each with an `op`:

| `op` | Fields | Becomes |
|---|---|---|
| `configure` | `task_name`, `type`, `component`, `after`, `wait_mode`, `values` | the component's configuration, written here |
| `start` | `tasks` | the waits these tasks need, then their `start` writes in list order |
| `sync` | `task`, `mode` | one wait on the task's component, always placed |

`type` says how `values` are read:

```
streamer  base, temporal_bounds, temporal_strides, spatial_strides (bytes),
          in words (D142): a channel moves one bank word of num_slc slices
          of the port's dtype. The fastest spatial loop runs S_0 / num_slc
          channels, its stride num_slc times its element stride times the
          itemsize (8 bytes for a packed port); every other stride is its
          element stride times the itemsize, and base the layout's base
          plus the offset's bytes. Spatial bounds are design-time, in elements, and come from the design
          point's streamers. A shared set's movers (D169) keep a task per port, the
          program writing each mover once per start (its group's port, else the
          first idle one). An idle mover (another group's in the same encapsulated set, D148): every
          temporal bound 0, so it never starts, and every temporal stride
          a bank word, never 0 (a reader whose first loop has stride 0
          runs that loop once whatever its bound, snax_cluster's
          Reader.scala)
dma       direction ("l2_to_l1" or "l1_to_l2"); src and dst, each with
          bounds and strides (bytes, one wide beat per step), the L1 side
          with base, the L2 side with symbol (its container's data.h array)
accel     its start parameters by name: n, then any named rate (e.g. T),
          then each slot read from a scalar as {"from": <container>}
          (axpy's k1: {"from": "alpha"}), which the program writes to
          the slot's CSR from data.h's element before START
```

**Names** (D75): streamer `<instance>_<port>` (`acc_a`), task `<node>_<component>` (`add_acc`), DMA task `load_<container>` or `store_<container>`, with `_<k>` appended for tile `k`.

Only the loops a task uses are given; the backend pads the rest with bound 1 and stride 0. `after` names earlier tasks this one needs finished: data it reads, a buffer it overwrites. `wait_mode` (`poll` or `signal`) is used whenever the lowering waits for this task. Missing `after` and `wait_mode` default to `[]` and `poll`; every field is written back, and unknown keys are errors.

**Checks.** When a list is made: task names are unique; `after`, `start` and `sync` refer to tasks configured earlier; a component has at most one configured task not yet started; a task starts once, after its `after` tasks; every configured task is started. Against the design point (`check_values`, `c.tasks`): the component exists with the task's type, the values have exactly the type's keys, a streamer's loops fit it, a DMA's direction is valid, every bound is at least 1 (or, an idle mover's, every one 0).

**Waits** (`placed_waits`, `lower/waits.py`). Before a `start`, one wait per component, on its latest task, for every `after` task and listed component not yet covered, leaving out a wait another covers. A wait on a writer streamer also covers its unit and that unit's readers started with it; an idle mover's task needs no wait. Where configures, starts and syncs go is the task list's.

**Derived** (`lower/derive.py`): per accelerated node in execution order, `load_<C>` for each input not yet in L1, a task per streamer, the accelerator task (`n` the firings, each named rate from `named_rates`, D102), one `start` of them all, `store_<C>` for each output written for the last time; one `sync` per non-transient output at the end. In an encapsulated accelerator, several groups on one streamer set (D148), each task also configures the other groups' movers idle, in the accelerator's order, and its `start` launches every mover: `dot_1_enc`'s `mult_vmul` starts `vmul_a`, `vmul_b`, `vmul_out` and `vsum_a`, `vsum_out` idle. A linked group (D148) is one such task at its first unit's node, `<node>_<group>`, on the group as its component: its units' streamers on the ports on no link only (no load or store for a link's container), and one accelerator task whose `values` are each unit's (`n` and its named rates) by unit, in the group's order (dot 6's `mult_dot`: `{"mac": {"n": 128, "T": 128}, "sum": {"n": 1, "T": 1}}`); its program writes the group's JOBS = n_u / ∏ T_v over its first unit u and its folds v, then each fold's JOB_BEATS but a fixed one's (section 17). `after` holds data dependences only, so two groups chain through L1 with one wait (D105), on the first's writer. A writer never waits for a reader of its own start, which it starts with: that is an in-place update, allowed only when the reader is a port of its own unit visiting the same addresses in the same order (axpy's `y`, a fold's seed), and refused by name otherwise. Accelerators run in series for now (D158): before each accelerator's `start`, one `sync` on the previous accelerator's task, with its own configures placed before that sync, so it is set up while the previous one runs; none when an `after` of that start already holds the previous accelerator's task, and none between the groups of one encapsulated accelerator, which follow each other through their data (dot_serial: `sum_sum`'s configures, `sync mult_mul`, its start). Each accelerator's tasks hold its own set's movers only; idle movers exist only inside an encapsulated set. vecadd's design point on `platforms/snax_forge.json` gives `tests/run/fixtures/vecadd/tasks.json` exactly; its end:

<!-- snippet: tests/run/fixtures/vecadd/tasks.json -->
```json
  {
   "op": "configure",
   "task_name": "add_acc",
   "type": "accel",
   "component": "acc",
   "after": [],
   "wait_mode": "poll",
   "values": {"n": 16}
  },
  {"op": "start", "tasks": ["add_acc_a", "add_acc_b", "add_acc_out", "add_acc"]},
  {
   "op": "configure",
   "task_name": "store_C",
   "type": "dma",
   "component": "dma",
   "after": ["add_acc_out"],
   "wait_mode": "poll",
   "values": {
    "direction": "l1_to_l2",
    "src": {"base": 268436480, "bounds": [8], "strides": [64]},
    "dst": {"symbol": "C", "bounds": [8], "strides": [64]}
   }
  },
  {"op": "start", "tasks": ["store_C"]},
  {"op": "sync", "task": "store_C", "mode": "poll"}
```

## 10. Block runtime model

One accelerator as plain data (D68, D70): a hand-written JSON file in `snax_forge/brm/library/`, named after it, in the form `Brm.to_json` writes (every field, defaults included). `load_brm(name)`; unknown keys are errors and a missing part is rejected by name.

| Part | Holds | Read by |
|---|---|---|
| `interface` | params (`design` / `runtime`) and ports (name, direction, lanes, rate, dtype) | everything below |
| `function` | an accelerator kind registered in `brm/function.py`, its factory params without timing, and `code`: what one lane computes (D82) | the reference executor; `bind` and the accelerated node (`code`) |
| `dataflow` | a notation (`affine` or `stream`) and one nest per port, in logical indices | the streamer values (`affine`), or nothing beyond the interface (`stream`) |
| `pattern` | `family` (a matcher registered in the sandbox), its `attrs` | SNAX-SANDBOX's `bind` (section 12) |
| `implementations` | per name: `source` (`chisel` only), `supports`, `timing` (`latency`, `initiation_interval`, `drain`, `span`), `binding` (null, or `datapath`, `args`, `controller`, optionally `ports`, D129) | the GVSoC accelerator, the HW generator |

The first four are shared; designs that differ only in timing, supported values or RTL source are implementations of one BRM.

**Ports.** Names are not Python keywords and do not start with `_` (D103); the convention is `a`, `b`, ... for inputs and `out` for the output. A port moves one beat every `rate` firings, an int or a runtime param's name: an input of rate `r` is taken at firings `0, r, 2r, ...`, an output at `r-1, 2r-1, ...`. A SIMD block has rate 1 everywhere, a reduction input rate 1 and output rate `T`. A port may have `when`, `{param: value, ...}` over design params: it exists only when every one holds (`simd`'s `b` only at `streams` 2, `accumulate`'s `seed` only at `seed` 1), and an instance's ports are those present at its params.

**Timing**, each an int or an expression over design params: `latency` (L, a firing to its push), `initiation_interval` (II, the least cycles between firings), `drain` (the cycles, starting with a push, in which no input is taken; 0 by default) and `span` (the cycles a unit reads one input beat in place, section 17; 1 by default). Only L delays a result.

**Params.** A `design` param is fixed per instance (lanes, the op) and reaches the design point and the accelerator hjson; its `default` is used when the design point gives none, `values` limits it, and an implementation's `supports` limits it further. The `runtime` params are exactly `n` (the firings in the task) and every named port rate, in port order: the accelerator task's `values` (section 9); `n` must be a multiple of every resolved rate. A design param with a null `default` is unset unless `bind` gives it or a code entry sets it: resolve leaves it out, and an expression naming it is not evaluated (`simd`'s `k` at two streams). A value field holds an int or an expression over params; a fixed string is a design param with one allowed value.

**The function** names a registered kind (`simd`, `reduce`, `mac`, with an op by name) and its params over the design params; the reference executor runs an accelerated node through it (D79). `code` is one `output = expression` per output port over the input ports, per lane of one firing; a reduction says what one element contributes (`accumulate`'s `out = a`), its fold being its pattern's and kind's (D102). `code` may instead be a list of entries, each `{"params": {...}, "code": ...}` fixing design params (`simd`'s twelve: add, sub, mul, min and max at two streams, the same five at one stream with the slot `k` in place of `b`, then axpby as `out = k1 * a + k2 * b` and `out = k1 * a + b`). A slot is a design param that an entry's code names and that is not a port. `bind` tries the entries in order and every assignment of the tasklet's array inputs to an entry's present ports (given by its params): its code must equal the tasklet's on them with `+`, `*`, `min` and `max` commuting, a slot matching either a part with no names and an int value (`0`, `-3`) that fits the op type (baked, D143) or a scalar operand, an input on an `sdfg.scalar` container whose subset names no variable of the maps around it (read at run time from an R/W CSR, D161). An entry fits only when every slot its code names has a value, baked or read, and every other slot is unset, so axpy (`k2` unset) has one code. The first match's params and baked values become the instance's, and a bind param contradicting them, or giving a read slot a value, is refused (D82). A null `code` cannot be bound.

**The binding** (D129, `brm/binding.py`): `{datapath, args, controller, ports}`. `datapath` a datapath of the catalogue (`hw/chisel/blocks.json`, section 17); `args` its parameters, each an expression over design params; `controller` a registered kind, `counted` the only one (the group's control unit, D147); `ports`, optional and written only when given, maps every BRM port to the datapath's (by default the same name). Which datapaths exist is the generator's (`gen/blocks.py`): `resolve` checks that the binding's takes the instance. The implementation's `drain` must be 0: the control unit has none (`gen.drain`). The CSRs are not the binding's: they follow from the unit's group (section 17), its JOBS J = n / T (T the fold's rate, 1 for a SIMD unit). Expressions are stored canonical. The library's files are the examples: `simd.json`'s `chisel_tiled_spatial` binds `SimdLanes` (`lanes` W, `op`), `accumulate.json`'s `chisel_adder_tree` binds `TreeAccumulate` (its port `a` to the datapath's `in`, `seed` to `seed`).

The BRM `mac` (D149) says what one lane computes, `out = a * b`, its output folded per lane over T beats by its kind `mac` (the product at the inputs' result type, cast to the output's, D143); its implementation `chisel_mac` binds the datapath `Mac` with `accumulate` true and `regOut` false (a bool arg is a design-time variant, kept as it is). Its pattern, `lane_fold`, is a fused nest (section 12): a multiply and a reduce joined by a local value, the fold indexed by the spatial variable:

<!-- snippet: snax_forge/brm/library/mac.json -->
```json
 "function": {"accel": "mac", "params": {"lanes": "W", "op": "op"}, "code": "out = a * b"},
 "dataflow": {"notation": "stream", "ports": {"a": {}, "b": {}, "out": {}}},
 "pattern": {"family": "lane_fold", "attrs": {"op": "mul", "fold": "add"}},
```

`accumulate`'s `chisel_serial` binds `SerialAccumulate`, one lane a cycle, so its span is its lanes. `accumulate`'s design param `seed` (0 or 1, D163) adds the input `seed`, one lane at the output's rate T: the start of each output, read once per job. `bind` sets it to 1 on a seeded fold (section 11) and refuses a seeded fold on a BRM without the port; both implementations take it.

The BRM `lane_accumulate` (rowsum 2) is the lane fold with no front op: ports `a` (W lanes, rate 1) and `out` (W lanes, rate T), code `out = a`, its kind `lane_accumulate` folding each lane over T beats at a's dtype, cast to the output's; pattern `lane_fold` with attrs `{"fold": "add"}` and no `op`. It binds a spatial map whose one tasklet copies its input into an output folded with `add` and indexed by the spatial variable, every map the input names and the output does not being temporal, so each lane folds its own element (one row of A per lane); a front op is refused (bind `mac`), as is an output not indexed by the lane. Its implementation `chisel_lane_accumulate` binds `LaneAccumulate`; it is its own block, not a form of `Mac` (D149).

**Two notations** (D70). An **`affine`** nest of a port: the operand's `shape`, an `offset` and `loops`, outermost first, each with a `bound`, one stride per operand dimension and a `spatial` flag; the element of a step is `offset + sum(i_l * strides_l)`. Temporal loops first (one step is one beat), spatial loops last (the lanes, the last fastest); spatial bounds use design params only and multiply to the port's lanes. A temporal stride of 0 repeats a beat. Section 9 maps the nest through a layout onto a streamer's values, and `bind` checks that the memlets give its order.

A **`stream`** nest, for unit BRMs, is empty: the port's lanes, rate and dtype are the interface's, and the bound memlet gives the element order, so `bind` checks no order, SNAX-LOWER takes the streamer's values from the memlet alone, and the design step takes the spatial bounds from the memlet's ranges. Any key is refused. The library's BRMs are all in it.

## 11. SNAX-DFG

The workload as one `.snaxdfg` JSON file (D71), written by an importer and rewritten by each sandbox transform. `Graph.load(path)`; every field is written, missing keys take defaults, unknown keys are errors, and a graph is validated when it is made.

| Field | Holds |
|---|---|
| `name` | the kernel |
| `symbols` | name → int or `null`: `null` as imported, the value once a recipe binds it; expressions keep the name |
| `containers` | name → `shape` (ints or expressions over symbols), `dtype` (a NumPy dtype name), `transient`, `attrs` (namespaced only; `sdfg.scalar: true` marks a kernel's scalar argument, shape `[1]`, read as element 0, which the reference executor also takes as a 0-d input) |
| `body` | the nodes, in execution order |

**Nodes.** Every node has `id`, `kind`, `inputs`, `outputs`, `attrs` and, for a kind that has one, `body`. `inputs` and `outputs` map a connector to a memlet, `{"data": container, "subset": [dim, ...]}`, one dimension per container dimension: an index (`n`, `4 * n_1_t + n_2_s`, `0`) or a range `begin:end[:step]`, end exclusive. Node ids are unique identifiers, since task names are built from them (D75).

**Folding writes** (D102). A tasklet's output memlet may hold `"reduce": {"op": ..., "identity": ...}`: the elements it writes start at `identity` and every iteration's value is folded in with `op` (`add`, `mul`, `min`, `max`; `register_reduction`). With a seed (D163, `fuse_seed`), `"seed": {"data": ..., "subset": [...]}` takes the identity's place: each element starts from the seed's element, read with the inputs before any write (matvec_acc's sum after `fuse_seed`: `q[m_t]`, written in place, so q is drawn in two rows like every in-place write); its tasklet keeps what `fuse_seed` took in `sandbox.seeded`, for `unfuse_seed`. Written only when set; an error on an input and on an accelerated node. dot as imported (`tests/dfg/fixtures/dot.snaxdfg`) folds its `sum` tasklet's `out[0]` so.

| Kind | Attrs | Body | Connectors |
|---|---|---|---|
| `map` | `var`, `range`; `loop.kind` absent, `tile`, `temporal` or `spatial`, set by `tag_map` (any of the three), `split_map` or `split_reduce`; `loop.dim` and `loop.piece`, the dimension the map runs over and its piece, 1 outer to inner, from which `var` is derived (D162); `loop.split` (the `var` and `range` before `split` or `split_map`, a re-split map's earlier record as `prior`, and `valid`, the iteration count n, when n is not a multiple of the factor: the lanes past it are padding, `dfg/pad.py`, D144) | yes: runs once per value of `var` | none (derived from the body) |
| `tasklet` | `code`: one `output = expression` per output connector | no | one element each: indices only |
| `link` | `container` (the transient it replaced, which the writer and the reader still name, so the reference executor runs the link as it), `depth` (0 a wire, ≥ 1 a buffer), `from` and `to` (the units' instances) | no | none |
| `accelerated` | `instance`, `brm`, `implementation`, `code` (what one lane computes, the BRM's matched code entry, which may name the node's int params: relu's `out = max(a, k)` with `k` 0), `params` (design params; a slot read from a scalar has no value here, its slot connector names the container instead: axpy's input `k1` on `alpha[0]`, which is no BRM port), `replaced` (the subtree `bind` replaced, or null), `locals` (written only when set: a fused nest's local values `bind` took out of the graph, `[{name, index, container}]`, in scope for `replaced` and put back by `unbind`), `accelerator` (written only when set: `encapsulate`'s name for the accelerator its group is in, D148), `group` (written only when set: `link`'s name for the group the unit is in) | yes: empty for a leaf block, a nested block otherwise | the BRM's ports, each port's lanes as a range; one beat |

**Accelerators and groups** (D146, D148). A bound graph nests accelerator ⊃ groups ⊃ units and links: a unit is an accelerated node, a group a connected set of units and links (`link`, section 12), so an unlinked unit is a group of its own, named after its instance; a linked group's units carry its name (`group`). A link node sits before its reader in the body the two share, and is drawn in the accelerated node's style (`out_partial = wire`, `tmp0 = buffer`, its depth, what it replaced). A fused map (`fuse_map`) is one nest of two tasklets, the transient between them a local value, no longer in L1: the container carries `loop.local` (the fused map's id), and the fused map records the two nests it was made of (`loop.fused`); the drawing gives the local value no box: the first tasklet's output connector joins the second's input connector inside the fused map, an edge labelled with the container's name and drawn as a link's wire (`viz/dfg/api.py`); `bind` on the nest removes the local value from the graph, since no live node names it any more, and keeps it in the accelerated node's `locals`, so `unbind` then `unfuse_map` give the graph before `fuse_map` exactly; a map `split_reduce` split records its partials and its final sum (`loop.split_reduce`). By default each group is its own accelerator, named after it; `encapsulate` (section 12) puts several into one, which shares one streamer set (section 16), each group still a task of its own, the groups in the graph's order. The slot, its CSR window and the wrapper (sections 16, 17) are shared by every accelerator of the design. The nodes of one instance carry the same `accelerator` or none, and an accelerator's name is no instance's (`DfgError`). Several nodes may share one instance, with one BRM, implementation and params: one accelerator, run once per node in the body's order (D165; mvt's variant 1, both halves on `vmul`, `vsum` and `vadd`, `recipes/mvt/mvt_1.json`); LOWER gives each node its own task on the same streamers, the estimate counts the instance's area once and each run's cycles. A linked group is shared by `share` (section 12): its units then have a node per nest and it has a link node per nest on the same ports, one piece of hardware (`design/streamers.py`'s `group_links`); LOWER runs the group once per set of nodes its links join (`group_runs`).

**A transposed read** (D164, `sandbox/transpose.py`). `mark_transpose` records on a tasklet `sandbox.transpose`, `{connector: π}`, for an input whose dimensions it reads across its loop order: each dimension's index names one loop dimension (`loop.dim`), and π orders the container's dimensions by the depth of their outermost map (mvt's half 2 reads `A[m, n]` under the loops (n, m): π = (1, 0)). No memlet changes; LOWER takes the strides from the memlets as before. `bind` carries it onto the accelerated node, keyed by the port.

Attrs without a namespace belong to the kind and are all written; namespaced attrs (`loop.*`, `mem.*`, `hw.*`, `user.*`) pass through untouched. More kinds come with `register_kind`.

**Names.** Every name in an expression is a symbol or an enclosing map's `var`; shapes use symbols only; symbols, containers and variables never share a name. An expression is ints, names, `+ - * //`, and `max(a, b)`, `min(a, b)` (two arguments, no other call; `snax_forge/expr.py`). Expressions are stored as `ast.unparse` writes them, one without names as an int.

vecadd after `split_map` (W = 4) and `bind` (`tests/dfg/fixtures/vecadd_accelerated.snaxdfg`): the temporal map keeps what it was split from (`loop.split`), and the accelerated node what it computes and what it replaced (D82), which `unbind` and `join_map` take back. `replaced` is checked in the node's scope, never run, and its ids may repeat live ones.

<!-- snippet: tests/dfg/fixtures/vecadd_accelerated.snaxdfg -->
```json
     "inputs": {
      "a": {"data": "A", "subset": ["4 * n_1_t:4 * n_1_t + 4"]},
      "b": {"data": "B", "subset": ["4 * n_1_t:4 * n_1_t + 4"]}
     },
     "outputs": {"out": {"data": "C", "subset": ["4 * n_1_t:4 * n_1_t + 4"]}},
     "attrs": {
      "instance": "acc",
      "brm": "simd",
      "implementation": "chisel_tiled_spatial",
      "code": "out = a + b",
      "params": {"W": 4, "op": "add", "streams": 2},
      "replaced": {
       "id": "add_map_s",
       "kind": "map",
```

## 12. Recipe

SNAX-SANDBOX's ordered list of transforms (D72), a JSON file in a category folder of `recipes/`: `elementwise/`, `reduce/`, `dot/`, `matvec/`, `mvt/`, and `custom/` for what the editor saves. A recipe is found by its name (`find_recipe`), so names are unique across the folders and no recipe sits in `recipes/` itself. `Recipe.load(path)`; every field is written (`dtypes` only when given), `params`, `symbols` and `steps` default to empty, unknown keys are errors.

| Field | Holds |
|---|---|
| `name` | the recipe; its steps are written to `out/sandbox/<name>/` |
| `kernel` | the kernel whose import the recipe starts from |
| `params` | recipe params, each an int or a string; `--set NAME=VALUE` overrides one |
| `symbols` | a value for graph symbols: an int or an expression over the params |
| `dtypes` | optional: container -> NumPy integer dtype name, in place of the kernel's (D113, D142): the import applies it to the kernel's descriptors before DaCe builds the SDFG, and the kernel's inputs are cast to it (`astype`, wrapping); an unknown dtype or container is an error naming it (`vecadd_i32`: A, B, C `int32`) |
| `steps` | in order: `transform` (a registered name) and its `params` |

A transform's int params take an int or an expression over the recipe params (`"factor": "W"`). Running a recipe binds the symbols (step 0), applies the steps and checks every step against the reference executor on the same seeded inputs; each is written as `<i>_<transform>.snaxdfg`, with the recipe as `recipe.json`.

**The editor's files** (D55, `sandbox/session.py`). `edit-dfg` keeps a session in `out/sessions/<name>/`: `recipe.json`, a recipe holding every step applied so far, and `session.json`, its `cursor` (the step shown, 0 to the steps applied; undo and redo move it) and its `platform`. The graphs are not stored: the session rebuilds them by replaying the recipe. A step applied before the last is refused (`session.drop`) until the steps after the cursor are dropped. Export writes one step's graph to `out/sessions/<name>/export/<k>_<transform>.snaxdfg`. Save writes steps 1 to the cursor as `recipes/custom/<name>.json`, with only the params those steps or the symbols use and the session's own (a recipe opened from a file keeps its params; a link's `D` tried and stepped back from goes, `used_params`), and, as `pixi run sandbox` writes them, to `out/sandbox/<name>/`: a name that is not an identifier or that a recipe in any folder has (it never overwrites) is `save.name`, a `graph`-stage design problem on the cursor's graph and the session's platform `save.graph`, any other `save.design` unless forced.

**The editor's session and forms** (`sandbox/session.py`). A session from a kernel starts from its symbols only, each a param of its name (`edit-dfg --kernel K --set N=…`), with no W unless given; a symbol without a value is refused with one line naming it (`vecmul: give N, e.g. --set N=2048`). Every factor and depth the editor writes names a param (`split`, `split_map` and `split_reduce`'s `factor`, `link`'s `depth`, the transforms' int params), so it can change after the bind (in Bindings, below): the form's default is a param name of its own, `W`, then `W_2`, `W_3`, … for a factor and `D`, `D_2`, … for a depth while the session has it, with its value box (4, or 1 for a depth); a number given instead becomes a new param under the next free name at that value, and the step stores the name (`fresh_param`; vecmul's split given 8 stores `W` with W = 8). When its map is bound, a factor param the editor named is renamed through the session to `W_<instance>` (every step that names it, the session's params, Bindings; dot 1's binds `mul` and `sum` give `W_mul` and `W_sum`), unless several units share it (dot 2–6's `W`, D156). A hand-written recipe's literal factor stays valid (D73), and a recipe opened from a file keeps its numbers. A transform's int param takes an existing param or a new param's name: a new one comes with its value, `POST /api/apply`'s `new_params` (`{"W_mul": 8}`), added to the session's params before the step, so each split can have its own (dot 1's `W_mul` 8 and `W_sum` 4 saves params `{"N": 2048, "W_mul": 8, "W_sum": 4}`); a name the session lacks without one is `session.param` (`split: factor 'W' is not a param; give its value`, with its `param`), which the form avoids by sending a new name's value beside it (below). Undoing the step that added a param keeps it. `bind`'s default instance is `acc_N`, N the bind's order in the session, the next N where the graph has the name as an instance, accelerator or group (D156; vecmul's task `mult_acc_1`, its streamers `acc_1_a`, `acc_1_b`, `acc_1_out`; dot 1's `acc_1` and `acc_2`). Save's result lists each command as `{command, what}`: `pixi run sandbox` writes the steps and checks each against the reference; `pixi run flow` runs the recipe on SNAX-GVSoC and writes the reports; `pixi run forge` the flow plus the bundle for snax_cluster's RTL; its log line names the recipe from the repository (`saved recipes/custom/<name>.json`).

**The recipe's names in the drawing** (D156, `viz/dfg/api.py`'s `unit_params`, `map_ranges`). With a recipe (a session's, or a folder's `recipe.json`), a unit's box gives its lanes under its split's param (`acc_1 · a * b · W_acc_1 8`, a shared param's one name in each box; without a recipe `W 8`), and a split map's Details give its `range` with the param and its iterations (`0:N // W_acc_1 = 256`; a tail `0:(N + W_mul - 1) // W_mul = 205`; the inner map `0:W_acc_1 = 8`).

**The selection and the sites** (D72, `sandbox/session.py`'s `sites`, `transforms.py`'s `site_of`). The editor's selection is one item or several: a node's id, a container's name, a group's or an accelerator's name (its dashed outline). A group stands for its units and an accelerator for its groups. Each transform's site answers for the whole selection: `encapsulate` takes units and groups that are all in no accelerator and ticks exactly their units (the outlines of `acc_1` and `acc_2`: both; one unit's box: that unit); `unencapsulate` takes an accelerator's outline (`dot`) or a unit in one; `fuse_map` and `fuse_seed` take one map, either side of a pair (the form's one `pair` field a choice of every partner), or two as `first` and `second` in the order selected; any other transform refuses several items with `site.one`, `<transform>: takes one selection, not <n>`, listed under "Not here". The route is `GET /api/step/<k>/site?sel=<id>&sel=<id>` (the ids in the order selected); `GET /api/step/<k>/site/<id>` is the same on one item.

**The editor's checks** (D156). Every apply and every param change runs the design checks Save runs, on the cursor's graph and the session's platform, and the page lists their problems under the Estimate, each `[code] where: why` with its fix (the step view's `problems` and `notes`, each `{code, where, message, fix}`); `graph.unbound` while building (`before Save: bind it (…)`) is shown as a note, not a problem, and Save still refuses it (`save.graph`, above); the design notes (`char.missing`, `design.serial_one_lane`) are always listed, also beside problems. Each carries `before_save`; the page lists the problems first, then the notes, a before-Save one tagged so, and the log line's counts (`1 problem`, `2 notes`) link to them. `encapsulate`'s form ticks every bound unit not in an accelerator yet. A param change (`POST /api/params`) replays the steps up to the cursor only, its log line counting the step's problems and notes as an apply's does: the steps after it stay as redo steps, and each is replayed and checked when the cursor reaches it (a refusal then shown on that redo). The log and Compare give one unmapped count, the estimate's `unmapped_cycles` (D155, section 19); the `not_mapped` count is in neither.

**The page** (`viz/static/`, both modes). A map's bar shows its id first (`mult_map · n · 2048 · untagged`). The top bar holds the session's name, the steps, undo and redo, Compare, the technology, Export and Save; no param is there, each is shown with what it configures. The left column scrolls on its own: in edit mode the transforms (with Compare on, a note that a transform applies to one step), then in both modes the Blocks, the Containers and the Bindings (`viz/dfg/api.py`'s `symbols_section`, `containers_section` and `bindings_section` in the step view). **Blocks** (D166) lists the graph's blocks as a tree, each with its recipe and params, a toggle to open or close it and its name to select it; a block not open is drawn as one box inside its firing loops (the maps above it stay, the nests' alike ones as one), each edge labelled with the slice one firing takes (`A[m_t, 0:N]`), an open one as its nodes inside a dotted outline; a double-click on either opens or closes it, with the blocks sharing its units, and the hash's `open` keeps them (`viz/dfg/api.py`'s `fold`). **Containers** start with the Symbols, each kernel symbol and its value (`N 2048`; matvec `M` and `N`), in edit mode an input on the param it is bound to (`POST /api/params`, the session's replay and refusal); then a line per container in the graph's order, `A: N · L1 · 16 KiB` (its shape as written, the dims by ` × `, `M × N`, then its placement and padded bytes; a fused nest's local value `N · local to mult_map`, no bytes, as it has no memory (Details: `local to mult_map, no memory`); a click on the name selects it; its sizes and dtype, read only, D113, are in Details). Focusing or hovering a symbol highlights every container whose shape uses it, in the section and in the graph. **Bindings** show each param with what it configures, though the param stays the split's (D68, D73: `bind` reads W off the spatial map), as drop-downs: one per accelerator, holding one per group, each holding its units' entries with their params editable (D156; dot 1: `dot` ▸ `mul` ▸ `mul · simd · W_mul = 8`, `sum` ▸ …), open by default (`bindings_section`'s `tree`: `{kind: accelerator | group, name, children}` down to `{kind: entry, index}` into `entries`; a linked group not encapsulated is a group at the top, a unit alone and a split not bound yet are entries at the top); one entry per bound unit, `acc · simd · W = 8`, the params of the splits whose inner map it replaced (`<map>_inner`, `<map>_s`, `split_reduce`'s `<map>_final`); a split not bound yet, `mult_map split · W = 8 · not bound`; a link whose depth is a param. In edit mode each param is an input (`POST /api/params`, as a symbol's); one that several entries name (dot 2–6's `W`, shared by the multiply and the sum, which a link requires) says `shared with …` on each, highlights them all on hover, and changes all of them when edited; a session param neither a symbol nor in an entry is listed last as unused. The params come from the session's recipe up to the step shown; in view mode from a sandbox folder's `recipe.json` (`folder_recipe`), read only, and without one the units' design params off the graph. The left and right panels resize by dragging their inner edge, as the log does (one handle code, `dom.js`'s `dragHandle`), from 170 px and 230 px to 45 % of the window, the widths kept in localStorage (a double-click gives back the default), in both modes and Compare. A split's factor (any sweepable int field) that names something the session's params lack shows a value input per such name beside it (`new param W = 4`), which Apply sends as `new_params`. `bind`'s form lists only the (BRM, implementation) pairs that fit, each with its timing, as the sidebar lists only the transforms whose site accepts; the API's bind list keeps the refused ones with their messages, which the form shows under `why` only when none fits. `bind`'s instance field has a choice to share an instance already bound (`share_choices`, D165) for one node as it is bound (a linked group is shared after its links, by `share`): each instance whose BRM and implementation take the node, refused with the params that differ (`vmul_x1 has W = 4, tmp2_mult would have 8`); choosing one fills the instance and picks its binding, and the joining split's editor-named W is renamed to the instance's param, so one param drives every node. A shared instance is one Bindings entry naming its nodes (`vmul · simd · tmp0_mult, tmp2_mult`), and selecting one of its nodes highlights the others (the step view's `shared`). In both modes a selected map shows its partners (the step view's `partners`, per map the maps `fuse_map`, `fuse_seed` and `interchange_map` take with it, found by the transforms' own sites, so the highlight and the editor agree): each in its transform's colour (green, amber, violet) with a tag, a fuse's transient marked with its loops matched (`tmp1[m] · m_t = m (0:8)`), and a near partner, which `fuse_map` refuses only because one side is split, grey and dashed, its tag the refusal and its fix. A node whose output folds (`fold`: a folding tasklet, or a unit whose replaced nest folds) is an upside-down trapezoid. The top-level nodes are drawn in columns, one per independent chain (mvt's two halves side by side), each box and row with its `column`; a container `mark_transpose` marked is drawn once above the columns that read it, its marked edge labelled `Aᵀ` (`viz/dfg/api.py`'s module doc).

**The cost in the viewer and editor** (`view-dfg`, `edit-dfg`; section 20). The technology is the server's `--tech` (nangate45, the only one offered, D167), kept in the URL hash with the other state (`tech=`, D57) and sent to every step and compare request as `?tech=` (`viz/dfg/server.py`; an unknown or retired one is a 400 `char.tech`); the page's technology dropdown hides while there is one technology. A unit's and a buffer link's Details gain a Cost section under Timing (how its table point was found or "not characterized", its area total, comb and flops, its energy per fire, holding per cycle and per task); a group's Details its area, energy and control unit; the Estimate panel each group's area and energy beside its cycles and, where L1's banks allow fewer cycles than the ideal, its bank note under it (`bank bound: mul: the streams move 6144 words over 32 banks, at least 192 cycles of L1 time against the ideal 128`, D155); the design's under the heading "Cost estimate (first-order) on <tech>", its area and energy, the one-line caveat, then the cost by part (`summary.cost.parts`, `cost_parts`, D167): two bars to scale, area and energy, a segment per part in its colour with its share on hover (the colours a meaning of this panel only), and a table of the five parts' area, energy and shares with the total and the total without the banks; then a table per accelerator (`summary.cost.tables`, `cost_tables`): a row per unit, buffer link and control unit (`<group> ctrl`) with its block, datapath (`· not characterized` where the table has no point), area (µm²), energy (nJ) and share of the accelerator's energy, a row per port's streamer (`streamer · FIFO D, L loops, C ch`, `· extrapolated` off the grid), one row "L1 accesses (crossbar and banks)" with its reads, writes and energy (no area: the crossbar's and the banks' are the design's), then the total (vecmul at W = 8: `acc` 55 300 µm², 23.137 nJ, 23.5 %; `acc ctrl` 702 µm², 0.128 nJ; `acc_a`, `acc_b`, `acc_out` 38 500 µm², 10.152 nJ each; L1 accesses 4 096 reads, 2 048 writes, 44.569 nJ, 45.3 %), the blocks not characterized, the parts not costed and the points extrapolated. Compare adds a row for the area without the banks and one per part, and draws the two steps' bars one above the other on one scale, the larger total the whole width (`columns.cost_bars`, `compare_cost_bars`), so the shorter bar is the cheaper design; the parts table sits folded under the bars. Every cost is worked out and rounded in Python (`viz/dfg/api.py`: an area to 3 significant figures, an energy to 3 decimals); the page only draws it.

**Steps from several folders** (`view-dfg`). A step is named after its file's stem; when the files `view-dfg` loads come from more than one folder (two designs' recipe folders, which share the stems `0_input` … `6_link`), after `<folder>/<stem>`, the folder its parent directory's name (`dot4_w8/6_link`), and any name still shared (two folders of one name) gets `-2`, `-3`, … (`viz/dfg/api.py`). `GET /api/steps` gives each step's `name`, `folder` (null from one folder) and `label` (`6 link`, without the folder). The step dropdowns group the steps by folder, an optgroup per folder in the order the paths were given; a panel's head shows its folder, and Compare's column headers read `dot4_w8 · 6 link` and `dot5_w8 · 6 link`. The URL hash holds the step's full name (`step=dot4_w8/6_link`, URL-encoded); a hash with a bare stem that only one step has, or with a step number, still opens that step (`GET /api/find?step=<ref>`). Edit mode is one session, one folder, and its hash keeps step numbers.

**Compare as two columns.** With Compare on, Details and the Estimate panel are each a table: a label column and one value column per step, headed by the two steps' labels, the left graph panel's step on the left (the earlier one unless picked otherwise), with no arrows. A row whose values differ is marked; a field, an item or a group on one side only shows "–" on the other, and an item on one side only is headed `(added)` or `(removed)`. Details shows the selected item's every field on both sides, its cost a row per text; the Estimate the accelerator's cycles and unmapped cycles (D156), per group (every group of either step) its cycles, expected BUSY_COUNTER, channels, area and energy and, where a side has one, its bank note (`bank bound`), L1 and the words read and written, and under the cost heading the accelerator's area, energy, L1 accesses and the blocks not characterized, the caveat below, then the cost tables with each cell `before → after` where the two differ. `GET /api/compare/<j>/<k>` gives each side's texts (`columns`: `labels`, per Details key `rows` and `where`, the Estimate's `rows`, `caveat`, `cost_tables`; a row `{label, under, a, b, differs}`), built in `viz/dfg/api.py`; the page only lays them out. Outside Compare the page is as above.

| Transform | Params | Does |
|---|---|---|
| `split` | `map` (id), `factor` (int) | a map over `b:e` becomes an untagged map over `0:(e - b) // factor` (same id), with `loop.split`, around an untagged map `<id>_inner` over `0:factor`, the two the next pieces of the map's dimension, so `n` becomes `n_1`, `n_2`, and a split of `n_2` gives `n_1`, `n_2`, `n_3`, the pieces below renumbered (D162); when the length n is not a multiple of the factor, over `0:(n + factor - 1) // factor`, with `loop.split.valid` n: the lanes at or past n are padding (D144), which the reference executor neither reads nor writes |
| `split_map` | `map` (id), `factor` (int) | `split`, then `tag_map` `temporal` on the outer map and `spatial` on the inner (D73): one step, as old recipes have it; it keeps the ids `<id>` and `<id>_s` (fresh when taken), which the recipes bind (dot 4–6's `mult_map_s`), the variables the pieces with their tags' suffixes (`n_1_t`, `n_2_s`) |
| `tag_map` | `map` (id), `kind` (`temporal`, `tile` or `spatial`) | an untagged map is tagged as it is, not split, its variable taking the tag's suffix (`_t`, `_tile`, `_s`: rowsum's `n` becomes `n_t`; `untag_map` drops it); on a level of a fused nest whose record holds, both nests are tagged at that level and fused again (`untag_map` likewise); a tagged map is an error |
| `untag_map` | `map` (id) | the map's tag is removed, whatever set it; a map around a bound unit is an error (`unbind` first) |
| `bind` | `node` (a tasklet id, a fused nest's spatial map id, or a spatial map whose one tasklet folds, `lane_accumulate`), `brm`, `implementation`, `instance`, `params` (optional design params) | the tasklet and its spatial map become an accelerated node; every map around the spatial map must be `temporal` (the firing loops: an untagged one is an error naming `tag_map`, a `tile` one is not lowered yet); the lanes param comes from the spatial bound; the dimension over the spatial variable becomes the lanes' range, the others stay indices; the tasklet's code on the port names must be the BRM's `function.code`; a folding output that does not use the spatial variable becomes one lane, `c:c + 1` (D102); a fused nest's two tasklets bind as one, the first's inputs and the second's output, as a fold (`lane_fold`, section 10) or, when the second is a plain tasklet, as one code with the first's expression put in place of the local value (axpy's `y + alpha * x`), and the accelerated node takes the first's id; a tasklet that needs a fused map or a split reduction is refused with the fix named ("fuse_map first", "split_reduce first", D150); a unit on a link is not unbound (unlink first) |
| `unbind` | `node` (accelerated id) | the accelerated node becomes the subtree it `replaced` |
| `join_map` | `map` (id) | a map with `loop.split` and its inner map, whatever their tags, become the map they were split from, the dimension's pieces renumbered (one piece left is the dimension's name again); an inner map split again, or one of another dimension, is refused |
| `encapsulate` | `accelerator` (a name), `units` (instances) | the groups of those units become one accelerator of that name, sharing one streamer set (their nodes' `accelerator`, D148); `units` in the order the graph runs them, which the groups keep; a unit already in an accelerator, a name an instance or an accelerator has, or a unit the graph lacks is an error |
| `unencapsulate` | `accelerator` (a name) | each of its groups becomes its own accelerator again |
| `split_reduce` | `map` (id), `factor` (int) | an untagged map whose one tasklet folds into an element that does not move with it is split as `split_map` splits it, `<map>` (temporal) around a spatial `<map>_s`; the tasklet writes one partial per lane into a new transient `<output>_partial`, indexed by the output's dimensions that move with an enclosing map, then the spatial variable (dot's `out_partial[4]`, matvec's `q_partial[M, 4]`); a final sum, tasklet `<tasklet>_final` in a spatial map `<map>_final` over `0:factor` right after `<map>`, a dimension of its own (`n_f`, so `n_f_s`), folds them into the output; a seeded fold keeps its seed on the final sum, the partials starting from the identity and named after the transient the seed took (matvec_acc's `tmp1_partial`); a length that is not a multiple of the factor is refused (the partials' padding is not placed yet) |
| `join_reduce` | `map` (id) | undoes `split_reduce` |
| `fuse_map` | `first`, `second` (map ids, `second` right after `first`) | two nests with the same ranges and tags level by level down to one tasklet each (an untagged level takes the other's tag, temporal against spatial refused; different ranges where one side is split refused with the split that fixes them, `split_map mult_map_1 4 first`, D150), joined by a transient written and read at the same index, become one nest of two tasklets (two untagged nests before any split too: axpy's recipes fuse the imported maps, then split once) that keeps `first`'s ids and variables, the second's other nodes after the fused one at their level (matvec's final sum inside the row map); the transient becomes a local value (no other node may name it); `bind` on it takes its spatial map's id (dot's `mult_map_s`, matvec's `mult_map_1_s`) and the local value out of the graph (section 11) |
| `unfuse_map` | `map` (id) | undoes `fuse_map`, when nothing changed since (unbind first) |
| `fuse_seed` | `first` (a map of a fold's nest, or a map `split_reduce` split, for its final sum), `second` (a map of the elementwise nest after it) | the fold writes a transient t with op ⊕ whose only reader is that nest, computing `t ⊕ x` with the same op and writing x in place over t's whole subset at t's index: the nest and t go, and the fold writes x at its own subset with `reduce.seed` on x in place of the identity (section 11; D163; matvec_acc's sum and `augassign_19_4`, so q += A p is one fold); exact for any registered op on integers; refused, naming what it found: t with another reader, writer or a link, another op, another subset, a float dtype, a fold seeded already, x named between them or by the fold's nest, a bound fold (unbind first); split, tag, interchange and join carry the seed, so the steps before `bind` go in any order |
| `unfuse_seed` | `map` (a map of the seeded fold's nest) | undoes `fuse_seed`: the nest and t back, names as at import, the fold starting from its identity again |
| `bake_scalar` | `container` (an `sdfg.scalar` container), `value` (an int that fits its dtype, or `csr`) | at an int every tasklet reading the container gets `value` as a literal in place of its connector, and the container goes; the step's check runs the graph before it with the container at `value` and the later steps keep that value (axpy's alpha 3: `mult` reads `out = 3 * in2`, so `bind` bakes `k1 = 3`); at `csr` the connector takes the container's name (`out = alpha * in2`) and the container stays, read at run time (`bind` reads `k1` from it); either way the tasklet has one streamed input, and each rewritten tasklet keeps the bake in its `sandbox.baked` attr (D161); allowed anywhere before `bind`, an accelerated reader is refused (unbind first) |
| `unbake_scalar` | `container` | undoes `bake_scalar`, when nothing changed since (unbind first) |
| `interchange_map` | `map`, `with` (map ids) | two maps of one perfect nest (one inside the other, every map from the outer down to the inner one's parent holding one map and nothing else), adjacent or not, trade places, each keeping its id, variable, range and tag, the maps between them and the inner one's body staying where they are; its own inverse, the same pair in either order; refused when a range would then name the variable of a map inside it, when the swap reorders a fold over two or more maps of the nest whose op depends on the order, and in a fused nest or around a bound unit (unbind first); the editor offers every map above and below the selection, one it cannot trade with marked with its refusal (rowsum 2: `sum_map_s` with `sum_map_1`, so the lanes go across rows) |
| `mark_transpose` | `node` (a map of a nest, or its tasklet), `container` (filled when the tasklet reads only one container across its loop order) | marks the read as transposed (section 11's `sandbox.transpose`), nothing else changing; refused: no input read across its loop order (`tmp0_mult reads A, y1 in loop order`), a container marked already, a bound unit (unbind first) |
| `unmark_transpose` | `node`, `container` | removes the mark |
| `share` | `group` (a group, or a unit: its group), `with` (a group of the same shape) | `group` runs on `with`'s units (D165, `sandbox/share.py`): the same number of units, unit by unit one BRM, implementation and params, and the same links and depths; its units' nodes and its links renamed to `with`'s, what they were kept in `sandbox.shared`; the editor then merges the shared nest's split and depth params into the other's when equal (mvt's `W_vmul_x2` into `W_vmul_x1`); refused naming what differs (`vmul_x2 and vmul_x1 differ: W = 4 against 8`), a group in an accelerator (unencapsulate first), a group that runs several nests already |
| `unshare` | `group` (the shared group's old name, or a node of it) | undoes `share` |
| `share_ports` | `container` (an L1 container), `accelerator` (a name) | the container's writer's and reader's groups go on one accelerator, as `encapsulate` puts them (their nodes' `accelerator`; a group already in one keeps it, and `accelerator` must be its name), and the container is marked `hw.share_ports` (`{accelerator, added}`, the units it gave `accelerator`), so the accelerator's groups share its streamer's movers (D169, `sandbox/share_ports.py`): per role the k-th widest port of each group on mover k; the site fills `accelerator`, else the next free name after `<writer's group>_<reader's group>`. Refused (`share.site`): a container on a link, shared already, not written by one bound group and read by another bound group that runs after it, the two in different accelerators, a name another instance, group or accelerator has; `unencapsulate` refuses an accelerator a container marks |
| `unshare_ports` | `container` (a shared one) | the marker taken off, and `accelerator` off the units `share_ports` gave it |
| `link` | `container`, `depth` (int), `group` (a name) | a transient between two bound units becomes a link node, the units' groups merge under `group` (written on every unit of both groups), and the transient leaves L1; refused: a container that is not a transient with one writer and one reader, or a local value (`link.container`), an end that is not bound (`link.unbound`), the writer's beats not the reader's in the same order, each port's memlet over its firing loops at the recipe's symbols (`link.order`), a name that differs from one already set, another instance's or group's, or units in two accelerators (`link.group`) |
| `unlink` | `container` | undoes `link`, the group taken off every unit left alone; two groups of several units left under one name is `unlink.group` |
| `lift` | `kernel` (a kernel some recipe under `recipes/` is written on), `at` (a node; empty for every region) | the region holding `at`, or every region, matching the kernel's import becomes a block (D166, `sandbox/blocks.py`): `sandbox.block` on each of its nodes, `{name, kernel, heads}`, named after the kernel (`dot`, `dot_1`, ...), nested in a block that holds it or around the blocks it holds; refused: no region of the kernel there, a region holding part of a block |
| `unlift` | `name` (a block with no recipe, or a node in it) | takes the block's record away |
| `apply_recipe` | `recipe` (`category/name`), `region` (the chain heads, in the kernel's nest order) or `block` (a lifted one), `params` (the sub-recipe's, an int or an expression over this recipe's params), `name` | the sub-recipe's steps run on the region, its names renamed onto the region's (a unit's or group's taken as it is, `_1`, `_2`, ... on a clash; a group named after its kernel takes the block's name), the region a block with `recipe` and `params`; one step, one reference check; refused: the region does not match the sub-recipe's kernel (the rule broken named), a sub-step's own refusal |
| `unapply` | `block` (a block a recipe made, or a node in it) | the graph before `apply_recipe` made it (a lifted block stays lifted); refused: a step after it changed what it made, a block inside another a recipe made |
| `name_block` | `nodes` (the region's heads), `name` | the region, each node with everything under it, becomes a block by hand, no kernel and no recipe |

**Recipes of recipes** (D166). A variant built on a smaller kernel's recipe names it in `apply_recipe`: `recipes/matvec/matvec_dot1.json`–`matvec_dot6.json` tag the rows' loops and apply `dot/dot_<k>` on each row (the same graph as `matvec_<k>` for k = 2 to 6, a fuse taking the rows' loop with it), and `recipes/mvt/mvt_1_reuse.json`, `mvt_1_dot2_reuse.json` and `mvt_2_reuse.json` apply `matvec_dot1` or `matvec_dot2` on each half, the `vadd` and the sharing by hand (the hand-written variants' graphs, the units' names aside).

`recipes/elementwise/vecadd_undo.json` takes vecadd's last step back to the first. dot binds the multiply map to `simd` and the sum map to `accumulate`'s `chisel_adder_tree`, whose `out` is one lane every `T = N / W` firings.

`recipes/dot/dot_1_enc.json` then puts the two groups into one accelerator, `dot` (D148), with a task each: the `vmul` group's task sets the `vsum` group's movers to a temporal bound of 0, so they never start, and the other way round (section 9). An idle mover never starts only if its first temporal stride is nonzero: snax_cluster's Reader.scala takes stride 0 on loop 0 as one AGU pass, that bound forced to 1 and the repeater set to the bound, so a reader at stride 0 would start and read (the C backend refuses it, `c.idle`). Without the step, `recipes/dot/dot_1.json`, the two are two accelerators, `vmul` and `vsum`, each its own streamer set, run one after the other (D157, D158).

The other variants are one recipe each (D150): `recipes/dot/dot_2.json`–`dot_6.json` (dot 2 and 3 link `tmp0` at depth 1 and 0; dot 4–6 `split_reduce` the sum, `fuse_map` the two nests and bind `mac`, dot 5 and 6 with `chisel_serial`), and `matvec_1`–`matvec_6` the same on `kernels/blas/matvec.py`, the row maps tagged temporal first. vecadd's, `recipes/elementwise/vecadd.json`:

<!-- snippet: recipes/elementwise/vecadd.json -->
```json
 "params": {"W": 4, "N": 64},
 "symbols": {"N": "N"},
 "steps": [
  {"transform": "split_map", "params": {"map": "add_map", "factor": "W"}},
```

## 13. Platform

The SNAX cluster around the accelerators (D84). A platform file lives in `platforms/` (`small16.json`, without a source hjson; `snax_forge.json`, read from snax-forge's base cluster) and never names a recipe. `Platform.load`; every field is written, missing keys take defaults (`l2` missing: no L2), unknown keys and wrong types are errors, all reported at once (section 14).

| Field | Holds |
|---|---|
| `name` | the platform (an identifier) |
| `base` | the platform this one was made from, or null for an original |
| `changes` | what differs from `base`: `--set` path -> value, in the order set |
| `hjson` | the source cluster file, a path from snax-forge's root, or null (small16): what the generator writes the cluster file from (section 16) |
| `accelerator_slot` | the prefix of the generated entry's name, `<accelerator_slot>_<name>` (`snax_forge_<name>`, `<name>` the design's name, section 16), `snax_forge` from `design from-hjson` (D131), or null; `--set accelerator_slot=NAME` picks another |
| `acc_csr` | `AccCsrConfig` (below): the widths of the generated accelerator's CSRs (D152) |
| `l1`, `l2` | `L1Config`, `L2Config` (below); `l2` null for an L1-only cluster |
| `wait_mode` | `poll` or `signal`: how the program waits for a task (the C backend refuses `signal`, `c.signal`) |
| `streamers` | the streamer shell: `default`, then an entry per streamer it changes |

`L1Config` (`design/platform.py`):

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `n_banks` | single-port banks | 32 | >= 1 |
| `width_bits` | bits per bank word | 64 | multiple of 8 |
| `rows` | words per bank | 512 | >= 1 |
| `base_addr` | byte address of the first L1 word | 0 | — |
| `wide_bits` | widest port (the DMA's) | 512 | power-of-two multiple of `width_bits` |
| `max_xbar_inputs` | the most inputs the TCDM crossbar takes, the limit snax-gvsoc's `snax/snax_tcdm_xbar.cpp` asserts (`nb_inputs` in (0, 128], `kMaxInputs`, PORT1); 128 in `snax_forge.json` and `small16.json` | 128 | >= 1 |
| `warn_xbar_inputs` | above it the note `design.xbar_inputs` (D85): the design passes, its crossbar beyond SNAX's own and its cost extrapolated from the tables (section 20) | 64 | 1 to `max_xbar_inputs` |

`L2Config`:

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `size_bytes` | flat size | 1 MiB | positive multiple of the beat |
| `base_addr` | byte address of the first word | 0 | — |
| `beat_bits` | one DMA beat | 512 | multiple of `width_bits`; equal to the L1's `wide_bits` (`platform.l2`) |
| `width_bits` | as the L1, and must agree with it (`platform.l2`) | 64 | multiple of 8 |

`AccCsrConfig` (D152), written into `accelerator.json` as the CSRs' widths (section 17):

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `job_width` | bits of JOBS and every JOB_BEATS, and of the control units' counters (no copies, D168) | 32 | 1 to 32 |
| `busy_counter_width` | bits of BUSY_COUNTER, which saturates at 2^w − 1; null follows `job_width` | null | 1 to 32, or null |

Neither memory has an element type: each container's dtype decides how many elements a word holds (D142, section 1).

**The streamer shell** (D12). One streamer per port of every accelerated instance, `<instance>_<port>`, but a port on a link (D112): a link's ends have none. The port decides `write`, `n_ports` (its lanes) and, unless an entry gives them, `spatial_bounds` (fastest first: an `affine` nest's spatial loops, a `stream` port's bound memlet ranges). The graph decides `temporal_dims`: the firing loops the port's memlet runs over, at least 1, the most over an instance's nodes (D84); the design point records it per streamer. `default` holds `fifo_depth` (>= 1); an entry named after a streamer overrides it and `spatial_bounds` for it alone, and may give `num_ch` (>= 1, never in `default`): the TCDM channels the streamer uses, without it the wire default W / num_slc (D142, `connect.num_ch`).

**Working copies.** `pixi run design GRAPH --platform P --set platform.PATH=VALUE` sets a field by its path (`l1.n_banks`, `streamers.acc_out.fifo_depth`; the value JSON, else text) and writes `out/design/<name>/platform.json`, `base` the platform it started from and `changes` every path set since; `pixi run design save SRC NAME` keeps it as `platforms/NAME.json`.

**From a cluster file** (D131, `design/hjson_platform.py`). `pixi run design from-hjson HJSON NAME` writes an original `platforms/NAME.json`: the L1 and L2 from the hjson (the reader's docstring lists the keys), `streamers.default` (its `fifo_depth`) from the one accelerator's streamer template, whose movers must agree, the rest small16's, `hjson` (under snax-forge's root, `hjson.path`) and `accelerator_slot` `snax_forge`. `snax_forge.json` is read this way from `platforms/snax_forge_cluster.hjson`: snax-forge's own, `platforms/snax_forge_cluster.hjson`, has one core, which holds the accelerator slot and the DMA (`xdma`, D170).

<!-- snippet: platforms/snax_forge.json -->
```json
 "hjson": "platforms/snax_forge_cluster.hjson",
 "accelerator_slot": "snax_forge",
```

## 14. Design checks

Before the design step writes anything it runs every registered check (`register_check(code, stage, fn)`) on the pairing of a bound graph and a platform and reports every problem, each as `[code] where: why` with a `fix:` line (the `--set` or recipe change that removes it) where one exists (D85). `pixi run design check GRAPH --platform P [--set ...]` runs them only. Stages: `platform` and `graph` independently, `connect` when both passed, `memory` when `connect` passed.

| Code | Fails when |
|---|---|
| `platform.keys` | a platform file that does not parse; an unknown or missing key, a value of the wrong type; a `--set` path that does not exist, is written by the tools (`name`, `base`, `changes`) or is decided by the port (`write`, `n_ports`) |
| `platform.values` | a value its section rejects (`n_banks` 0, a `wait_mode` that is not allowed) |
| `platform.streamer` | a streamer value below 1, spatial bounds that are not a non-empty list of ints >= 1, `num_ch` that is not an int >= 1 |
| `platform.l2` | L2 and L1 disagree on beat width (`beat_bits` vs `wide_bits`) or word width |
| `platform.banks` | with an L2, `n_banks` is not a multiple of the DMA's port group (`wide_bits / width_bits`) |
| `graph.load` | a graph file that does not load (section 11) |
| `graph.symbols` | a symbol still null: bind it in the recipe |
| `graph.unbound` | a tasklet or a map holding nothing to run outside every accelerated node, or no accelerated node: the program drives accelerators only |
| `graph.brm` | an unknown BRM or implementation, design params that do not resolve, connectors that are not the BRM's ports or slot connectors, code that is not the BRM's |
| `graph.instance` | an instance named `dma` or after a streamer, or two streamers with one name |
| `connect.streamer_key` | a streamer entry that matches no port of the graph (the fix names the closest streamer) |
| `connect.derived` | a streamer entry that sets `write` or `n_ports` |
| `connect.lanes` | an entry's `spatial_bounds` that do not give the port's lanes or are not its nest's spatial loops |
| `connect.dtype` | a node whose containers break its BRM's dtype rule (`brm/dtypes.py`, D143: each port's dtype in its `dtypes`, every output at least as wide as every input, no signed input with a `uint64` one; the fix names the output dtypes the inputs accept), or an instance whose nodes put two dtypes on one port |
| `connect.pack` | a port whose elements do not fill whole words (D142), with num_slc slices of its container's dtype per word: the fastest spatial bound not a multiple of num_slc (W < num_slc: the fix makes the container 64-bit or gives the port num_slc lanes), the fastest spatial loop not stepping 1 element (a column), another loop's step or the start not a multiple of num_slc (a row stride that is not: the fix pads the leading dimension). Row strides are over the container's padded shape (a tail's, D144: a 3 x 7 int32 matrix split by 4 has row stride 8). Every port passes at num_slc = 1; a slot connector is no port and is not checked |
| `connect.pad` | a container an accelerator writes and a fold reads through a tail, unless the producer's tasklet evaluated on its inputs' pad values gives the fold's identity (D144; dot 1 at W_mul = W_sum = 10 passes: 0 × 0 = 0, add's identity): otherwise the producer's pad lanes would reach the fold, since the hardware masks nothing, and a producer that never writes the pad words (dot 1 at W_mul = 8, W_sum = 10) is refused too until the program writes the identity there (open item 106); the fix: a size that divides W, or a split factor that divides n; a container whose readers need different pad values (the message names them) |
| `connect.num_ch` | a streamer entry's `num_ch` other than the wire default W / num_slc: any other needs the FIFO slicer, which the flow does not place yet (D142). The message lists the values with rho = num_ch * num_slc / W <= 1, the divisors of W / num_slc; the fix removes it or sets W / num_slc |
| `connect.xbar_inputs` | the TCDM crossbar's inputs as the generator's cluster file gives them to SNAX-GVSoC (`snitch_cluster.py`): every streamer's num_ch (the slot's `snax_tcdm_ports`, D142), then one per core of the platform's `hjson` and one for AXI, a shared set's its movers' channels (D169), above the platform's `l1.max_xbar_inputs` (128), where SNAX-GVSoC's crossbar asserts: `acc: 194 TCDM crossbar inputs, the platform takes at most 128 (192 streamer channels, 2 for the core and AXI; …)` (vecmul at W = 64; W = 32 has 98, which passes with the note `design.xbar_inputs`); the fix: fewer lanes, or `share_ports` on a container between two accelerators (D169) |
| `share.shape` | a shared set's user whose spatial bounds (in words) are not its mover's on every dimension but the slowest, and at most the mover's there: it does not fit in the mover's first channels (D169, `design/streamers.py`'s `shared_movers`) |
| `connect.temporal` | a map around an accelerated node that is not `temporal` (tiles are not lowered yet), or a subset that is not affine in the firing loops |
| `memory.pin` | a memory working copy that does not load; a `--set memory.` path that is not `<container>.<memory>.base` or `passes.<pass>`; a pin on an unknown container (the fix names the closest), on a memory the container does not live in, or on L2 (the link places it, D86); an unknown pass |
| `memory.residency` | a layout in a memory the platform lacks or for a container the graph lacks; a non-transient container that lives nowhere; an accelerator operand without an L1 layout; a slot's container needs none, and stays in the core's memory (L2, or L1 without one) |
| `memory.layout` | a layout whose shape is not its container's |
| `memory.align` | an L1 base that is not a multiple of the word, or a stride that is not one of the element's (its dtype's itemsize, D142); a container the DMA moves whose L1 layout is not contiguous, not whole wide beats (open item 38) or not on a beat |
| `memory.fit` | an L1 layout that leaves L1 (the message names its bytes `[start, end)`; the fix the rows that would hold it, or the pin to L1's first byte), or an L2 layout larger than L2 |
| `memory.runtime` | an L1 layout (with its padding, D105) reaching into the top of L1 that snRuntime keeps at start-up: a 1 KiB stack per core of the platform's `hjson` plus `RUNTIME_MARGIN` (256 B, `design/check.py`); not checked without an `hjson` under snax-forge's root. The fix pins the layout just below |
| `memory.overlap` | two L1 layouts that share bytes, naming the pins (the fix moves the latest pin past everything) |
| `point.keys` | a design point with unknown or missing keys or a malformed streamer entry |
| `point.streamers` | a design point whose `streamers` are not the platform's shell for its graph |

**Notes**, beside the refusals: a design that passes every check but is worth a second look. Each is registered (`register_note(code, fn)`) and run once the checks passed (`run_notes`); the design step prints it as `note [code] where: why` with its `fix:` line and still exits 0, and `design.md` lists it first under Notes. The notes on banks compare only the streamers of one group, since the groups of an accelerator run one after the other (dot 1's `mul`, then `sum`): `bank_bound` (section 19) per group, and `design.md`'s address notes (`report/design.py`), two movers whose containers start in the same banks or lie a whole number of bank rows apart, only for two movers of one group (dot 1 at W = 8 keeps `mul_a and mul_b: A and B lie 16384 B apart, …` and has none pairing a `mul` mover with a `sum` one).

| Code | Noted when |
|---|---|
| `design.serial_one_lane` | a unit bound to `SerialAccumulate` at lanes 1: it reads one lane per beat, the same work as `TreeAccumulate` at lanes 1 (dot 4's shape at W = 1), so valid but with no reason to exist; the fix binds `chisel_adder_tree` |
| `design.xbar_inputs` | more crossbar inputs than the platform's `l1.warn_xbar_inputs` (64, D85): "85 TCDM crossbar inputs, above 64 (l1.warn_xbar_inputs): beyond SNAX's own crossbar, its cost extrapolated from the tables" (mvt_2 at W = 8); the fix: fewer lanes, or `share_ports` (D169) |
| `char.missing` | a bound unit or a buffer link the estimate's technology (`nangate45`) does not characterize, its lookup None (section 20): "not characterized: <datapath> at lanes L (depth D) on <tech>"; the cost counts its comb part as 0 |
| `cost.extrapolated` | a streamer or the crossbar priced outside HWCOST's tables' grid (section 20): "streamer outside the table's grid on channels 32 (grid 1–16): its cost extrapolated on nangate45", the crossbar's "inputs 99 (grid 4–64)"; or an L1 bank the table does not hold, the memory costed 0 (D160, D167) |

## 15. Memory plan and design point

**Memory plan** (D86, `design/memory.py`). A layout per container and memory it lives in: `base` (byte address of index 0), `shape` and one byte stride per dimension (`lower/layout.py`). Banks follow from addresses (word `i` in bank `i % n_banks`, open item 18). Three passes make it, each registered by name (`register_memory_pass(kind, name, fn)`):

| Pass | Decides | Default |
|---|---|---|
| `residency` | the memories of each container | `default`: not transient: L2 and L1 with an L2, else L1; transient: L1, but a link's container and a fused nest's local value (`loop.local`): none (D112, D150) |
| `layout` | shape and strides, base 0 | `contiguous`: the container's shape, its strides row-major over the padded shape (a tail pads the split dimension to a multiple of W, D144), each element its dtype's itemsize (a word holds several narrow ones, D142); the memory checks and the DMA take the padded block. `padded_rows` (`--set memory.passes.layout=padded_rows`, never in a recipe): as `contiguous`, the last dimension's pitch of a container of two or more dimensions rounded up to an odd number of words, in every memory it lives in (L2 too, so the DMA still moves one block), the pad words zero, so lanes reading down a column start in different banks (rowsum 2's A: 8 × 65 words, 4160 B) |
| `placement` | the L1 base per container (an L2 layout keeps base 0: the program's link places its array, section 7's regions) | `contiguous`: graph container order from L1's first byte, no gaps; a layout in both L2 and L1 on a wide beat and padded to whole beats (D105), others on a word; around the pins |

`--set memory.<container>.l1.base=N` pins an L1 base; `--set memory.passes.<pass>=NAME` picks a pass. Passes read a `MemoryContext` (the bound graph, the platform, shapes, memory ranges, and each accelerator port's element-index stream, `accesses`). The design step writes `out/design/<name>/memory.json` (`passes`, `changes`, `layouts`), which `--memory` continues from.

**Design point** (D87, `design/point.py`), written to `out/design/<name>/design_point.json` once every check has passed, never by hand:

| Field | Holds |
|---|---|
| `name` | the design's name: the `name` of the recipe beside the graph in its sandbox folder (the recipe's or Save's, `recipe.json`, never with a `--set` suffix), else the graph's stem; in a flow the recipe's name, its folder keeping every `--set` (D136); the generated entry is named after it (section 16) |
| `graph_from` | the `.snaxdfg` it was made from: a sandbox step, or any bound graph given to the design step or the flow (D108) |
| `graph` | that graph as read: instances, `code`, `replaced`, `loop.split` |
| `platform` | the platform working copy (section 13) |
| `streamers` | the resolved shell: `<instance>_<port>` -> `instance`, `port`, `write`, `n_ports`, `spatial_bounds`, `temporal_dims`, `dtype` (the port's container's), `num_slc` (its slices per word), `num_ch` (the entry's, else W / num_slc; the hjson's `num_channel`), `fifo_depth` |
| `memory` | the memory plan: `passes`, `changes`, `layouts` (container -> memory -> layout) |

It holds no recipe. Loading one runs every check again on what it holds and requires `streamers` to be the platform's shell for the graph.

## 16. Generated hjson files

What `pixi run gen-gvsoc DESIGN_POINT [--out DIR]` writes per design point (D134, `gen/gvsoc.py`), default `out/gvsoc/<design>/`: what SNAX-GVSoC's `snax_hjson` reads (`--param hjson=<dir>/<platform>.hjson`) and the program snax-gvsoc's generic app (`sw/apps/forge/`) builds. For `platforms/snax_forge.json` the description files are `snax_forge.hjson`, one `<tag>_streamer_<set>.hjson` per streamer set (dot 1's `snax_forge_dot_1_streamer_vmul.hjson` and `_vsum`) and `accelerator.json` (D145). Each hjson and C file starts with a `//` comment naming the design point, the rest of an hjson file JSON (a set's streamer header is snax_cluster's text byte for byte, `Streamer.scala`'s comment on top, its timestamp line naming the design point instead, `gen/headers.py`); `accelerator.json` is plain JSON, its `design_point` key naming the source. One design point gives byte-identical files wherever written.

| File | Holds |
|---|---|
| `<platform>.hjson` | the cluster: the platform's `hjson`, its comments dropped and its key order kept, with `cluster.tcdm.banks`, `.size` (KiB), `data_width`, `cluster_base_addr`, `dma_data_width` and `dram` written from the platform, `cluster.name` `snax_forge_cluster` (also in `cluster.bender_target`, in place of the source's name; a source without its own name there is `gen.source`), and the slot set; the source's template, referred to only by the slot, dropped |
| `<tag>_streamer_<set>.hjson` | one per streamer set, its streamer template, the whole file (a shared set's, D169: its movers as section 17's `movers`, each at its widest user's channels and spatial bounds, the most temporal dims and the deepest FIFO, with `configurable_channel` per mover of a role where one is masked): `data_reader_params` (the set's input ports) and `data_writer_params` (its output ports), unit by unit in its groups' order, each in the BRM's port order, per mover `spatial_bounds` in words, fastest first (`[S_0 / k, S_1, ...]`, k the elements of the port's container dtype a word holds, D142), `temporal_dim`, `num_channel` (the streamer's channels, the wire default W / k; the bounds' product) and `fifo_depth` |
| `accelerator.json` | the accelerator's one description, section 17's file |
| `include/` | per set `streamer_<set>_csr_addr_map.h`, its macros prefixed with the set's upper-case name (`VMUL_STREAMER_START_CSR`), and per accelerator `<acc>_csr.h` (`acc` its name in accelerator.json; every CSR by section 17's name, in CSR order: a one-group accelerator's without the group, dot 1's `vmul_csr.h` from `VMUL_RW_START` to `VMUL_RO_BUSY_COUNTER`, an encapsulated one's as `dot_1_enc`'s `dot_csr.h`, from `DOT_RW_VMUL_START` to `DOT_RO_VSUM_BUSY_COUNTER`, a shared one's then `<ACC>_RW_PORT_MUX_SEL`; a masked mover's `<SET>_ENABLED_CHANNEL_<M>` and `_CSR_NUM` after its temporal strides, as Streamer.scala writes them), and the runtime's `snitch_cluster_cfg.h` and `snitch_cluster_addrmap.h`, as snax_cluster's templates render them from the cluster file (`gen/headers.py`, D126) |
| `data.h` | the program's data (`gen/data.py`): per container the DMA moves, an input from the kernel's `make_inputs` (seed 0), or an output zeroed and its `<name>_golden` from the reference executor, each array named after its container, or `<container>_l2` when C, libc or the runtime's headers take the name (`common.py`'s `c_name`: mvt's `y1` is math.h's Bessel function), the name main.c's copies and the run's L2 regions use too; each array the container's dtype, in its padded shape (a tail's pads hold the pad value: 0, or the identity of a max or min fold that reads it, D144), then padded with zeros to whole DMA beats in its itemsize (16 int32 or 64 int8 elements per 64-byte beat) as its L2 layout (D105) and aligned to the beat |
| `main.c` | the program (`lower/c.py`, D135): `main` reads `mcycle` first, then runs its core's function, one per core with each step on its component's core and each accelerator task's counter stored after its wait (`forge_counter[]`); no task marks in the default build. In a shared set (D169) each task writes every mover once, its group's port with a masked mover's `ENABLED_CHANNEL` = (1 << num_ch) − 1 (a one-channel port's spatial stride 0, so a mask that failed would write the other lanes over its word) or idle at bound 0, and after the wait on the set's previous task `PORT_MUX_SEL` (the group's index, only when it changes; `c.port_mux` refuses a change without that wait), then `STREAMER_START`, then the group's `START`. On the accelerator's core `main` then calls `forge_check()`, which reads `mcycle` first and prints `forge cycles: <n>` (the window from main's entry), then the self-check: the outputs against their goldens, the valid elements only, a padded matrix row by row (`forge data:`, its words the valid elements), each accelerator task's counter against its beats (`forge counter <task>:`), `forge errors:`; main returns the error count. Built with `-DFORGE_QUIET` (D135 amended), the quiet build: every start waited on right after it (a DMA transaction by `snrt_dma_wait_all()`, a group's start by its STATUS and streamer-busy polls), each section the next value written to CSR `0x7c5` (`write_csr(0x7c5, k);  // <label>`), no `printf`; its parts in `#ifdef FORGE_QUIET` blocks, each print in `#ifndef FORGE_QUIET`, so the default build compiles the text without them (`strip_quiet`) |
| `obs_marks.json` | the quiet build's sections in order (`lower/c.py`'s `program`), each `{mark, kind, label}`: `mark` the value written to CSR `0x7c5` (the cluster's `obs_o`, core 0's) as the section starts, 1 up, 0 from reset to the first; `kind` `dma_in`, `dma_out` (one DMA transaction), `streamer_config` (a set's streamer CSRs), `acc_config` (a group's JOBS, JOB_BEATS and scalars), `run` (a start to the end of its task) or `check` (`forge_check`, to the end of the program); `label` what it names (`dma load_A`, `run dot`); at most 255 (`c.obs`) |
| `manifest.json` | `design`, `design_point`, `symbols`, `cluster` (the cluster file), `files`, `program` (`main.c`), `trace` (the unit's trace path, `snax_acc0/unit`; a linked group's, `snax_acc0/group_<group>`; with several groups the accelerator's, `snax_acc0`), `csrs` (the CSR layout: `streamers`, one `{name, first, last, trace}` per set, its first and last CSR and its trace `snax_acc0/streamer_<set>`, and `accelerator`, each CSR's header name -> its number), `tasks` (per accelerator task, in order, its `beats` (a linked group's: its most firing unit's n) and its group's `csrs` values: from the task's values n and T (section 9), JOBS = n / T and, for a fold without `params.T`, JOB_BEATS = T, in a linked group named with its unit, `MAC_JOB_BEATS`; with several groups also its group's `trace`, `snax_acc0/unit_<unit>` or a linked group's `snax_acc0/group_<group>`, as snax-gvsoc's composite names them) |

**The slot** is the source's only one, on core0, renamed: `snax_acc_name` the tag, `<accelerator_slot>_<name>`, `<name>` the design's name: the recipe's or Save's `name`, never a flow's `--set` suffix, which only the flow's folder carries (D131; not an identifier, `gen.tag`), its `bender_target` dropped (the RTL comes from the bundle), and its `snax_streamer_cfg`, `snax_num_rw_csr` and `snax_num_ro_csr` (each set has its own). It carries `snax_streamer_sets`, one per accelerator in CSR order, each `{name, snax_streamer_cfg: {$ref: "<tag>_streamer_<set>.hjson"}, snax_num_rw_csr, snax_num_ro_csr}`, the last two its accelerator's rw and ro, counted from its CSR map (section 17); forge always writes the sets form, also for one set (D157). `snax_tcdm_ports` is the sum over the sets of their movers' num_ch; the source's `sparse_interconnect_config` becomes one group per set, of the set's width (its granularity kept), or `snax_narrow_tcdm_ports` that many and `snax_wide_tcdm_ports` 0; then, after the source's other keys, `snax_disable_csr_manager` `true` (D130), `snax_forge_rtl` `"rtl/files.f"` (relative to the cluster file, present only in a bundle; SNAX-GVSoC ignores it), `snax_streamer_sets` and `snax_datapath_cfg` `{$ref: "accelerator.json"}`, which snax_cluster only resolves. snax_cluster's schema lists the keys the slot adds, none with a default (open item 88).

**The CSRs**, per accelerator its streamer's, then its own in section 17's order (`csrs` in the manifest), packed from `0x3C0`; snax_cluster's `snax_csr_range_demux` sends each range to its set (D157). With $s_k$ and $c_k$ accelerator $k$'s streamer and own CSRs, of $K$:

$$b_0 = 960, \qquad b_{k+1} = b_k + s_k + c_k$$

streamer $k$ at $[b_k, b_k + s_k)$, its own CSRs at $[b_k + s_k, b_{k+1})$, and $b_K - 1 \le$ `0x5FD` over the whole design (`gen.csrs`). The shell sees its CSRs contiguous, from 0 (section 17). dot 1:

| Accelerator | Streamer | Own CSRs | In the shell |
|---|---|---|---|
| `vmul` | 960–977 | 978–981 | 0–3 |
| `vsum` | 982–994 | 995–999 | 4–8 |

snax_cluster routes `0x5FE` and `0x5FF` to its task queues on the cluster's CSR port, and SNAX-GVSoC's `SnaxAccelerator` refuses the same.

**Checks.** The streamer file and `accelerator.json` have snax-forge's schemas (`gen/schema/`, JSON Schema 2020-12); the cluster file validates against snax_cluster's (snax-gvsoc's copy, `sw/docs/schema/snitch_cluster.schema.json`). A platform value the cluster file cannot express (an L1 that is not whole KiB) is `gen.platform`, a platform without a source `gen.source`, a source with no slot or several `gen.slot`; what the description refuses is section 17's.

<!-- snippet: snax_forge/gen/schema/accelerator.schema.json -->
```json
 "required": ["design_point", "tag", "accelerators"],
```

## 17. Accelerator RTL spec

`accelerator.json`, the one description of a design point's accelerators (D145, `gen/accelerator.py`): written once by snax-forge, by `pixi run gen-gvsoc` (section 16), so also at the bundle's root (section 18). It is read by SNAX-GVSoC (`snax_cfg.py`) and by `Emit --spec` (`hw/chisel/`), after snax-forge's check against the datapath catalogue (D130); snax_cluster only resolves the slot's `snax_datapath_cfg` `$ref` to it. Plain JSON, concrete values only, no `//` header. The examples are snax-gvsoc's fixtures, `gvsoc/snax-gvsoc/forge/<design>/accelerator.json`.

`pixi run gen-rtl DESIGN_POINT [OUT_DIR]` (`gen/rtl.py`, default `out/rtl/<design>/`) writes the same bytes, checks every unit against the catalogue (`gen.catalogue`, `gen.params`, `gen.port_map`, `gen.timing`: the datapath, the params its fill rule gives, the ports whose `when` holds on them, its latency and span) and every link of depth ≥ 1 against `LinkBuffer`, and runs `Emit --spec accelerator.json --out DIR/`, which lists every file it wrote in `emitted.json`, beside `sv/`. Its java runs on hw/chisel's runtime classpath, which sbt exports into one cached file per environment, `hw/chisel/target/emit-classpath-<hash>.txt` (8 hex digits of the Coursier cache directory and the `java` path, `classpath_file`, so snax-forge's image and the host keep one each), and again when a file under `hw/chisel/src/main/` or `build.sbt` is newer or an entry is gone or cannot be checked. The Chisel specs' fixtures (`hw/chisel/src/test/resources/rtl/<design>/vectors.json`, `gen-rtl --fixture`) read their `accelerator.json` from snax-gvsoc's `forge/`.

One catalogue block alone (CHAR1, for snax-forge-experiments): `Emit --block NAME --params JSON --out DIR`, repeatable like `--spec ... --out ...` and in the same call, takes a datapath of the catalogue and its full params as the catalogue lists them (e.g. `{"inType":"int64","outType":"int64","lanes":8,"op":"add"}`), refusing an unknown name, a missing or extra param and a value outside its type; it builds `make(params)` as the top module, its ports as they are (`clock`, `reset`, `io_*`) and its reset asynchronous and active high, as every datapath's is inside `<tag>_shell_wrapper`, into `DIR/sv/`, one file per module, with `DIR/emitted.json` (`block`, `params`, `top`, `modules`, `files`). The module is the one an accelerator holds, byte for byte, but where the accelerator ties a pin to a constant, which firtool folds into the module there (dot 5's `SerialAccumulate` at T = 1, whose `first` is constant), and for `SimdLanes`' unused `clock` and `reset`, which firtool drops inside an accelerator. `pixi run gen-block LIST.json` (`gen/block.py`) checks each entry of a JSON list of `{"datapath", "params", "out"}` against the catalogue (`gen.block`, `gen.catalogue`, `gen.params`) and runs them all in one java call on gen-rtl's cached classpath.

| Key | Holds |
|---|---|
| `design_point` | the design point by its path from the root, as section 16's files name it |
| `tag` | the generated entry's name, `<accelerator_slot>_<name>` (`snax_forge_<name>`, D131), `<name>` the design's name (section 16); the modules below are named after it |
| `accelerators` | in CSR order, each `{name, inputs, outputs, groups}`, one streamer set (D157): `name` the header's `<ACC>`, `encapsulate`'s `accelerator`, else its one group's name (a one-unit group is named after its instance); `inputs`, `outputs` its outer ports, each `"<unit>.<port>"`, in its streamer's reader and writer order; with `share_ports` (D169) also `movers`, per mover in the streamer's order `{name (reader<k>, writer<k>), role, num_channel, masked, users}` (each user `{group, port, num_channel}`, every outer port on one mover), and `port_mux` `{groups (in PORT_MUX_SEL's order), csr}`, PORT_MUX_SEL the accelerator's last CSR (R/W, its width the bits a group's index takes) |
| an accelerator's `groups` | in CSR order, each `name`, `csrs` (its own), `units` and `links`, each link `{name, from, to, depth}`: `from` an `out` port as `"<unit>.<port>"`, `to` an `in` port; depth 0 is a wire, ≥ 1 a buffer of that many beats |

A unit:

| Key | Holds |
|---|---|
| `name` | the bind's `instance` |
| `brm` | its BRM |
| `datapath` | a catalogue datapath |
| `ports` | per BRM port, in the BRM's order: `port` (the datapath's), `dir` (`in` or `out`), `dtype` (as NumPy names it), `lanes` |
| `opType` | a SIMD or `Mac` unit only: NumPy's result type of its inputs (`brm/dtypes.py`'s `result_type`, D143), the type its op runs at |
| `params` | only what the ports do not give: `op` (a TreeAccumulate's `add`, a SIMD's `axpby`), a baked slot (relu's `k`), a `Mac`'s `accumulate` and `regOut`, and `T`, a rate fixed at design time, so the unit has no JOB_BEATS |
| `scalars` | written only when set: each slot read from a scalar and its type, the op type (axpy's `{"k1": "int32"}`), each an R/W CSR in `csrs` with its datapath `pin` |
| `timing` | `latency`, `ii`, `span`: the cycles a unit reads its input in place, taking it in the last, so its throughput is one beat per max(ii, span) |
| `csrs` | its R/W CSRs |

A CSR: `name`, `access` (`rw` or `ro`), `offset` from the shell's first CSR (contiguous over the accelerators, section 16), `width` (every CSR, 1 to 32: START 1, JOBS and every JOB_BEATS the platform's `acc_csr.job_width`, STATUS 2, BUSY_COUNTER its `busy_counter_width`; D152), `pin` (a scalar's datapath pin). snax-forge computes the offsets and widths once and stores them; a reader checks them (contiguous from 0 in CSR order; JOBS and every JOB_BEATS one width, every BUSY_COUNTER one) and never recomputes them.

dot 4's file, one accelerator `dot` of one group `dot`, the MAC's lane partials wired into the tree:

<!-- snippet: gvsoc/snax-gvsoc/forge/dot_4/accelerator.json -->
```json
{
 "design_point": "out/design/dot_4/design_point.json",
 "tag": "snax_forge_dot_4",
 "accelerators": [
  {
   "name": "dot",
   "inputs": ["mac.a", "mac.b"],
   "outputs": ["sum.out"],
   "groups": [
    {
     "name": "dot",
     "csrs": [
      {"name": "START", "access": "rw", "offset": 0, "width": 1},
      {"name": "JOBS", "access": "rw", "offset": 1, "width": 32},
      {"name": "STATUS", "access": "ro", "offset": 3, "width": 2},
      {"name": "BUSY_COUNTER", "access": "ro", "offset": 4, "width": 32}
     ],
     "units": [
      {
       "name": "mac",
       "brm": "mac",
       "datapath": "Mac",
       "ports": {
        "a": {"port": "a", "dir": "in", "dtype": "int64", "lanes": 4},
        "b": {"port": "b", "dir": "in", "dtype": "int64", "lanes": 4},
        "out": {"port": "out", "dir": "out", "dtype": "int64", "lanes": 4}
       },
       "opType": "int64",
       "params": {"accumulate": true, "regOut": false},
       "timing": {"latency": 1, "ii": 1, "span": 1},
       "csrs": [{"name": "JOB_BEATS", "access": "rw", "offset": 2, "width": 32}]
      },
      {
       "name": "sum",
       "brm": "accumulate",
       "datapath": "TreeAccumulate",
       "ports": {
        "a": {"port": "in", "dir": "in", "dtype": "int64", "lanes": 4},
        "out": {"port": "out", "dir": "out", "dtype": "int64", "lanes": 1}
       },
       "params": {"op": "add", "T": 1},
       "timing": {"latency": 1, "ii": 1, "span": 1},
       "csrs": []
      }
     ],
     "links": [{"name": "out_partial", "from": "mac.out", "to": "sum.a", "depth": 0}]
    }
   ]
  }
 ]
}
```

**Refused**, each a `GenError`, before anything is written: a tag that is not an identifier, or a platform without an `accelerator_slot` (`gen.tag`: a design named `dot-1` gives `snax_forge_dot-1`), a null binding (`gen.binding`), a datapath not registered (`gen.datapath`), a controller other than `counted` (`gen.controller`), a drain other than 0 (`gen.drain`), a latency, ii or span other than the catalogue's (`gen.timing`), and what the datapath refuses: its args (`gen.args`), op (`gen.op`), ports (`gen.ports`), rates (`gen.rate`), lanes not whole 64-bit channels (`gen.lanes`), a dtype that is not intN or uintN (`gen.dtype`). The links and groups: a link not from an `out` to an `in` of the same dtype and lanes (`gen.link`), a port on more than one link (`gen.link_port`), `inputs` or `outputs` not exactly the unlinked ports (`gen.outer`), a group not connected (`gen.group`). Until the phase that lifts each: a group that is not a chain in unit order, only the last unit's output outer (`gen.chain`, until tier A; the sandbox's `link` refuses it first as `link.chain`), and a unit of span > 1 on an outer input (`gen.span`, until B2). A shared accelerator's movers or port mux not as the `accelerators` row says (`gen.share`).

**Derived, never stored**: each port's num_slc = 64 / b and num_ch = W · b / 64 (D142), each link's kind and its `to_input` (the fed port's place among its unit's `in` ports), every unit's firings.

**Types** (D142, D143). A unit's types are its ports' dtypes, fixed at elaboration. Each input is extended to `opType` by its own signedness (a fold's lanes to its output's type before its tree), and the result is extended or wrapped into the output's type; a `Mac` forms the product at `opType`, casts it to `outType` and folds at `outType`. Python decides every type; the datapaths only construct.

**Groups and firings** (D146). A group is a connected set of units and links, an unlinked unit a group of its own, and has one JOBS: the sink gives R = JOBS results, a unit k before its successor s gives R_k = R_s · T_s, one on every T_k-th firing, so it fires n_k = R_k · T_k times, every T counted in beats (the one-lane vsum's `SerialAccumulate` has T = 1 and span W).

**The CSRs** (D146), from offset 0, per accelerator and per group in order: START (R/W), JOBS (R/W), each unit's R/W CSRs (JOB_BEATS for a fold without `params.T`, then its scalars, a 64-bit one as two CSRs, low word first), STATUS (R/O), BUSY_COUNTER (R/O). With f_u = 1 for a fold without `params.T` (else 0) and k_u a unit's scalar CSRs, rw = Σ_g (2 + Σ_u (f_u + k_u)) and ro = 2G. They are the accelerator's interface, all in `<tag>_csr`, BUSY_COUNTER in its submodule `<tag>_peripherals` (D152, D168); each has its width, and a read gives its bits with the upper ones 0. Per group:

- START is a strobe: no register, it reads 0. While the group is idle, a write to another of its R/W CSRs is taken at once and keeps the written value's low bits (JOBS and JOB_BEATS at job_width; the program's values fit, `lower/tasks.py`'s `check_values`), one to START with bit 0 clear is taken and does nothing, and START with bit 0 set is accepted. While it is busy, a write to any of its R/W CSRs (START, JOBS, each JOB_BEATS, each scalar) is held (`csr_req_ready_o` low) until E + 1, E the cycle its run ends, so its settings stay constant through the run (D168); a write to another group's CSRs and every read go through. A program that writes a busy group's settings stalls until that run ends (programming the next task during a run is open item 123).
- The accepted start is a bare strobe to the group's control unit, which reads JOBS and each JOB_BEATS from `<tag>_csr` as wires through the run, as the datapath reads the scalars (D168); a unit without JOB_BEATS has T 1, or its `params.T`, a constant.
- An empty run, JOBS 0 or any JOB_BEATS 0, ends in the cycle a it starts (E = a): busy never rises, BUSY_COUNTER is 0, and it ends in a done like any other.
- STATUS: bit 0 busy, bit 1 done (sticky: set by the done pulse in E, it reads 1 from the next cycle and is cleared by an accepted start; a set wins over a clear in the same cycle). The program polls bit 0. BUSY_COUNTER: the busy cycles after the accepted start, held after the run, saturating at 2^busy_counter_width − 1, so E − a after a run. `<tag>_csr` keeps STATUS from the group's control unit's busy and its done pulse in E, and its `<tag>_peripherals` the BUSY_COUNTER (`<group>_busy_counter`), cleared by the accepted start and + 1 per busy cycle. A run on snax_cluster's RTL may give a BUSY_COUNTER that differs from SNAX-GVSoC's by at most the group's streamer fill (3, or 2 with one outer input; section 19's `fill`, D123): the streamer's head start before START counts, and the core's timing between STREAMER_START and START, which SNAX-GVSoC does not model (open items 65, 67), sets it (`dot_1_enc` at N = 512: `vmul` 128 on the RTL, its START 6 cycles after the streamer's, against 131).
- Pins: `<group>_busy_o` (STATUS bit 0) and `<group>_done_o` (the done pulse registered: high in E + 1). snax_cluster leaves them unconnected (open item 92). No per-unit CSRs: per-unit timing comes from SNAX-GVSoC's events (open item 96).

The header names a CSR `<ACC>_<RW|RO>_[<GROUP>_][<UNIT>_]<NAME>` (one helper, `Accelerator.define`, for the header and the program), the group part dropped when the accelerator has one group and the unit part (on a unit's own CSRs only) when the group has one unit: vecadd's `ACC_RW_START`, dot 4's `DOT_RW_MAC_JOB_BEATS`, dot 1's `VSUM_RW_JOB_BEATS`, `dot_1_enc`'s `DOT_RW_VSUM_JOB_BEATS`.

| | Offsets | rw, ro | Task values |
|---|---|---|---|
| vecadd | START 0, JOBS 1, STATUS 2, BUSY_COUNTER 3 | 2, 2 | JOBS 16 |
| vsum_tree_i32 | START 0, JOBS 1, JOB_BEATS 2, STATUS 3, BUSY_COUNTER 4 | 3, 2 | JOBS 1, JOB_BEATS 16 |
| dot 1 (accelerators vmul, vsum; `dot_1_enc`'s groups vmul, vsum) | VMUL: START 0, JOBS 1, STATUS 2, BUSY_COUNTER 3; VSUM: START 4, JOBS 5, JOB_BEATS 6, STATUS 7, BUSY_COUNTER 8 | 5, 4 | task 1: VMUL JOBS 16; task 2: VSUM JOBS 1, JOB_BEATS 16 |
| dot 2–6 (one group `dot`) | START 0, JOBS 1, JOB_BEATS 2 (unit `sum` for dot 2 and 3, `mac` for dot 4–6), STATUS 3, BUSY_COUNTER 4 | 3, 2 | JOBS 1, JOB_BEATS T = N / W (128 at N = 512, W = 4) |
| matvec 2–6 (one group `matvec`) | as dot 2–6 | 3, 2 | JOBS 16, JOB_BEATS T = N / W (128 at N = 512, W = 4) |
| axpy (`k1` from alpha; at int64 K1_LO 2, K1_HI 3) | START 0, JOBS 1, K1 2, STATUS 3, BUSY_COUNTER 4 | 3, 2 | JOBS 16, K1 alpha[0] |
| matvec_seed (accelerator `matvec`, groups mul, sum; the seed has no CSR) | MUL: START 0, JOBS 1, STATUS 2, BUSY_COUNTER 3; SUM: START 4, JOBS 5, JOB_BEATS 6, STATUS 7, BUSY_COUNTER 8 | 5, 4 | task 1: MUL JOBS 1024; task 2: SUM JOBS 8, JOB_BEATS 128 (M = 8, N = 512, W = 4) |

**The datapath catalogue** (D147), `hw/chisel/blocks.json`, equal to `Emit --describe`: per datapath its params, ports, latency (an int, or an `expr.py` expression over its params, a bool as 0 or 1), `span`, `flops` (its register bits on its params, every flop it declares, with `width(t)` a dtype param's bits: `SimdLanes` 0, `Mac` `lanes * width(outType) * max(accumulate, regOut)`, `TreeAccumulate` and `SerialAccumulate` `width(outType)`, `LinkBuffer` `depth * lanes * width(dtype)`; `BlockFlopsSpec` counts them in the emitted SV, D154), pins and the rule that fills it from the unit's ports. A port is `"in"` or `"out"`, or `{"dir": ..., "when": {param: value, ...}}`; a pin is a name, or `{"name": ..., "when": {...}}`: the port or pin exists only when every listed param has that value. A datapath is data and enables only:

- `SimdLanes(inTypes, opType, outType, lanes, op, streams, k)`: L 0, no pins; `streams` the unit's number of inputs, `b` only at two; at one, `k` (a `const` param, null at two) drives every lane's `b`, and the module name ends `_k<k>` (`_km3` for −3); at op axpby, each lane is `k1 * a + k2 * b` at opType (`+ b` with `k2` unset); a slot read from a scalar is a pin of opType width (a bool param per slot, filled from the unit's `scalars`), its module name part `_k1r`;
- `TreeAccumulate(inType, outType, lanes, op, seed)`: L 1, pins `fire` and `first`; seeded (D163), an input `seed`, one element of outType, and acc ← (first ? seed : acc) + tree(in), the module name ending `_seed` (unseeded, the RTL as before);
- `Mac(inTypes, opType, outType, lanes, accumulate, regOut)`: with `accumulate`, per lane s ← first ? a·b : s + a·b, L 1, pins `fire` and `first`; otherwise a·b + c, L 0, or 1 with `regOut` and its pin `en`;
- `LaneAccumulate(inType, outType, lanes)`: per lane s ← first ? a : s + a at outType, L 1, pins `fire` and `first`, flops lanes · width(outType); its unit's CSRs are a fold's (START, JOBS, JOB_BEATS, STATUS, BUSY_COUNTER: rowsum 2's JOBS M / W, JOB_BEATS N);
- `SerialAccumulate(inType, outType, lanes, seed)`: acc ← (first ∧ k = 0) ? x_k : acc + x_k, lane k in cycle k, span = lanes, L 1, pins `read`, `lane` and `first`; seeded, as `TreeAccumulate`, the seed added on the job's first beat's last lane, the read that fires it;
- `LinkBuffer(depth, lanes, dtype)`, depth ≥ 1: D beats, L 1, pins `load`, `wslot` and `rslot` (the slots 0 bits wide at D = 1).

A wire is no datapath: plain connections with the same dtype and lanes at both ends.

**Modules**, one set per design, shared by its accelerators: `<tag>_shell_wrapper`, `<tag>_csr` ⊃ `<tag>_peripherals`, `<tag>_ctrl` ⊃ `<tag>_<group>_ctrl` (per group of every accelerator), `<tag>_datapath`, and per accelerator whose groups share its movers `<tag>_<accelerator>_port_mux` (D169).

**Control** (D147). One control unit per group, `<tag>_<group>_ctrl`, generated from the description, never written per variant. Its rule is snax-gvsoc's `SnaxGroupRule` (`snax/snax_group.hpp`), with a Python twin, `snax/snax_group_rule.py`, and Chisel's `GroupRule`, one rule on three sides:

- the group is a graph over its links, each `{from, to, to_input, depth}`, the units in topological order; each cycle a forward pass gives what each unit offers whatever its room (avail_k: its outer valids and every in-edge's offer, a wire its producer's offer, a buffer occupancy > 0), and a backward pass in reverse order gives its room (its outer output's ready if it owns it, and every out-edge's take: a wire its consumer's firing, a buffer occupancy < D or that firing), whether it is frozen (its head, or at L 0 a giving firing, has no room), fires and reads;
- a fork, a join, a cycle or two chains are refused, until tier A;
- the outer ports' join is per unit: ready_k = fire_{u(k)}, a unit firing only with every one of its outer valids, so valid never depends on ready; a seeded unit's seed, its last outer input, is needed and taken on a job's first firing only: fire = a.valid ∧ (¬first ∨ seed.valid), seed.ready = fire ∧ first (D153's counts unchanged; the seed is one more input in the fill);
- the class of each busy cycle is judged at the unit owning the outer output: stall_out (it is frozen) > busy (a unit fires, reads or is in its II gap) > stall_in (a unit with outer inputs wants to fire and one has no beat) > flush;
- a unit reads its input in every cycle it wants to, has a beat and is not frozen, at every span: at span S lane l in place, taking the beat (firing) on lane S − 1; at span 1 the read is its firing, so SerialAccumulate at lanes 1 sums one lane per beat; a frozen unit reads nothing, busy counts every read (a span unit's lane reads, at span 1 its firings), a plain Mac's `en` is its firing.

The Chisel control unit keeps a register only for what a firing changes and counts up with no product (D153, D168): it holds no copy of a setting, its tests reading the csr's JOBS and JOB_BEATS live, which stay constant through a run since the csr holds their writes. A unit of T ≠ 1 keeps its beat counter d (`<unit>_job_beats_counter`), 0 at the start; last = (d + 1 = JOB_BEATS), sharing the counter's incrementer, and first = (d = 0); a fixed T keeps the constant test d = T − 1 and a ⌈log₂ T⌉-bit counter. Per segment: a unit at L 0 whose out-link is a wire keeps no count of its results and borrows its consumer s's test (its giving firing is s's firing, so g_k = T_s · g_s + d_s and g_k ≠ R_k exactly when g_s ≠ R_s; a wire chain ends at the segment's last unit); any other unit (the sink, one feeding a buffer, one at L ≥ 1) counts its giving firings up in mixed radix, a digit per unit after it with T ≠ 1 (`<unit>_<downstream>_job_beats_counter`, wrapping at that unit's JOB_BEATS or constant) and then a jobs digit g (`<unit>_jobs_counter`), left = (g ≠ JOBS), and has finished when g equals JOBS. No count of the sink's pushes (done is its JOBS-th push, read off its jobs digit and its pipeline as in CTRL1: at L 0, g + 1 = JOBS on its giving firing; at L ≥ 1, where g has already counted the result being pushed, g = JOBS with its pipeline below the head empty), the II gap only at II > 1; the empty run (JOBS 0 or any JOB_BEATS 0) is tested on the csr's wires in the start's cycle. Every counter is job_width bits (a digit of a fixed T ⌈log₂ T⌉); LOWER refuses a JOBS or JOB_BEATS above 2^job_width − 1 (`lower/tasks.py`'s `_check_widths`), so d and g stay at most 2^job_width − 2 and d + 1 and g + 1 fit. Per cycle the unit decides as CTRL1's did with its copies, so every cycle count is unchanged. The accelerator is three modules (D168): `<tag>_csr` (the CSRs above by offset, ReqRspManager's protocol, D130, STATUS's done bit, the kept read answer and the hold of a busy group's writes: the interface, D152), with its submodule `<tag>_peripherals` (per group the BUSY_COUNTER, from the accepted start and busy; always present, later peripherals go there too), `<tag>_ctrl` (the groups' control units: per group busy, the counters, the done pulse and the pins) and `<tag>_datapath` (one submodule per unit and per buffer, no valid, ready or counter inside). A register goes where its writer and reader are: what the core writes in the csr, what a firing changes in ctrl, data in the datapath.

**Cycles.** The closed cycles of dot's variants, the accelerator alone on ideal streams (latencies vmul 0, Mac 1, tree 1, SerialAccumulate 1, buffer 1), J jobs of T beats at W lanes, which snax-gvsoc's `tests/snax/group/gen.py` checks `snax_group_rule.py` against:

    C2 = JT + 2, C3 = JT + 1, C4 = JT + 2, C5 = J(T + W − 1) + 2, C6 = (J − 1)·max(T, W) + T + W + 2

| Point | T | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| dot, N = 512, W = 4 | 128 | 130 | 129 | 130 | 133 | 134 |
| dot, N = 512, W = 8 | 64 | 66 | 65 | 66 | 73 | 74 |
| row, 16 × 512, W = 4 | 128 | 2050 | 2049 | 2050 | 2098 | 2054 |
| row, 16 × 512, W = 8 | 64 | 1026 | 1025 | 1026 | 1138 | 1034 |

dot 5's MAC stalls (J − 1)(W − 1) cycles; a depth-1 buffer acts as a pipeline register, so dot 2 is dot 3 plus a cycle; dot 6 needs T ≥ W to run without a stall. These cycles plus the streamers' fill, 3 (2 for a one-input unit), are a group's expected BUSY_COUNTER, a guide and not a check: the BUSY_COUNTER counts from the group's START, every cycle until busy drops, so a run's includes bank conflicts and is shortened by the streamers' head start (D155, section 19). A linked group's counter in `vectors.json` comes from the rule's model (`gen/rtl.py`'s `group_counter`, `snax_group_rule.py` stepped with an always-ready streamer), so any shape the rule accepts has one.

**The wrapper**, `<tag>_shell_wrapper`, has the ports `snax_acc_wrapper.sv.tpl` connects with `snax_disable_csr_manager`: `clk_i`, `rst_ni` (active low), `csr_req_addr_i` and `csr_req_data_i` (32 bits), `csr_req_write_i`, `csr_req_valid_i`, `csr_req_ready_o`, `csr_rsp_data_o` (32 bits), `csr_rsp_valid_o`, `csr_rsp_ready_i`; per input port k, numbered across the sets in set order (set 0's readers, then set 1's), `stream2acc_<k>_data_i`, `_valid_i`, `_ready_o`, per output port, the writers numbered the same way, `acc2stream_<k>_data_o`, `_valid_o`, `_ready_i`, the order of the accelerators' `inputs` and `outputs` in `accelerator.json`; data `num_channel × 64` bits with lane i at bits `[w(i+1)-1 : wi]` for the port's w-bit type (D142); per group `<group>_busy_o` and `<group>_done_o`. Each outer port's data goes to `<tag>_datapath`, its valid and ready to `<tag>_ctrl`. An accelerator with `movers` and `port_mux` (D169), which `Emit --spec` reads, has its movers at the wrapper in place of its `inputs` and `outputs`, readers then writers in the file's order, each `num_channel × 64` bits (the template's ports for its streamer hjson, whose movers they are), and `<tag>_<accelerator>_port_mux` between them and its groups' ports, pure wires selected by PORT_MUX_SEL (a register in `<tag>_csr`, reset 0, read back as written, a write never held: it is no group's): a reader's data to each of its users, the low `num_channel × 64` bits, its valid to the selected group's user only and its ready that user's (0 when the selected group has no user on it); a writer the selected user's data zero-padded and its valid (both 0 with none selected), each user's ready the writer's when its group is selected. snax-gvsoc's `SnaxPortMuxRule` without its fatal cases; the groups and their control units as in any other accelerator.

**The shared cases.** snax-gvsoc's `tests/snax/group/cases/` (the group rule), `tests/snax/unit/cases/` (its one-unit case) and `tests/snax/shell/cases/` (`SnaxShellRule`, `snax/snax_shell.hpp`, the CSRs above) are checked against the C++ rules by snax-gvsoc's testbenches and driven through the Chisel by the specs (`SNAX_GVSOC_GROUP_CASES`, `SNAX_GVSOC_UNIT_CASES`, `SNAX_GVSOC_SHELL_CASES`, which CI's chisel job sets to the submodule's folders). Where a case shows the two sides disagree, the Chisel behaviour stands unless it breaks this section or D130, and SNAX-GVSoC follows it.

## 18. Bundle

What `pixi run bundle DESIGN_POINT [--out DIR]` writes per design point (D111, D130, `gen/bundle.py`), default `out/bundle/<design>/`: everything snax_cluster needs to build the design's cluster, made afresh by the snax-forge checked out. `rtl/` is removed before each run; nothing else in `DIR` is.

| File | Holds |
|---|---|
| the root | section 16's files (gen-gvsoc's), the cluster file and `accelerator.json` (section 17) among them |
| `rtl/sv/`, `rtl/emitted.json` | what `Emit --spec` wrote (section 17); firtool also writes `sv/filelist.f`, which `emitted.json` does not list |
| `rtl/files.f` | the RTL's file list: one path per line, relative to `rtl/`, in `emitted.json`'s `files` order, `emitted.json` itself left out |
| `bundle.json` | `design_point` (as section 16 names it), `tag` (section 17), `snax_forge_commit` (`git rev-parse HEAD`, or null outside git), `dirty` (true when the work tree has uncommitted changes, untracked files included; false outside git), `top` (`<tag>_shell_wrapper`, the tag `snax_forge_<design>`) and `modules`, each of the wrapper's modules with the submodules it holds, from `accelerator.json`, not from Emit's output, by the names Emit gives them: `<tag>_csr` (`<tag>_peripherals`), `<tag>_ctrl` (`<tag>_<group>_ctrl` per group of every accelerator), `<tag>_datapath` (`<tag>_<unit>` per unit, then `<tag>_<link>` per buffer, group by group), then per accelerator whose groups share its movers `<tag>_<accelerator>_port_mux` (none; D169); `expected`, `forge_expected.txt` when the bundle has it, else null |
| `forge_expected.txt` | written by `pixi run forge` only, from the bundle's SNAX-GVSoC run: its `forge counter` lines in order, then its `forge data: 0 errors` line, as the program prints them, then per counter in the same order `forge fill <task>: <n>`, the streamer fill of the task's group (section 19's `fill`, `bundle.counter_fills`). A run of the same ELF on snax_cluster's RTL must print the data line as it is, and each counter's gap beyond its fill gives the run's verdict (D123); a file without fill lines takes every fill as 0 (`pixi run bundle` alone writes none) |

**Refused**: what sections 16 and 17 refuse (`gen.elaborate` among them), then `bundle.empty` (`emitted.json` lists no file).

**How snax_cluster takes it** (D130), through the cluster file alone (`CFG_OVERRIDE`):

- clustergen and snaxgen resolve a `$ref` to another file against the cluster file's real folder;
- in RTL mode snaxgen writes `target/snitch_cluster/generated/snax_forge.f` from the slot's `snax_forge_rtl`: the paths it lists, then its own files, each streamer set's streamer files and `<tag>_wrapper.sv`, all absolute; vsim and Verilator read it when it is non-empty;
- each set's streamer header lands in `target/snitch_cluster/sw/snax/<tag>/include/streamer_<set>_csr_addr_map.h`; `snax-forge-sw` compares each with the bundle's `include/` copy and fails when their `#define` lines differ;
- `target/snitch_cluster/snax_forge.mk` runs a program on the RTL: `CFG_OVERRIDE` the bundle's cluster file, `FORGE` the program's folder, `ELF` an ELF to run (default the one built from `FORGE`), `SIM` `vlt` (default) or `vsim`; `snax-forge-sim` builds the simulator, `snax-forge-run` runs the ELF into `logs/snax-forge-$(SIM).txt` and fails on a nonzero exit code.

**The local check** (D123), in place of a CI job in snax_cluster: `pixi run forge-set quick|full` makes a set's bundles with their `forge_expected.txt`, and the line it prints, `make snax-forge-check BUNDLES=... SIM=vsim SNAX_CLUSTER_IMAGE=...` in snax_cluster's `target/snitch_cluster`, runs each bundle's ELF (snax-forge's, `out/forge/<name>/sw/forge.elf`) on its RTL and compares its `forge data` line with `forge_expected.txt`'s and each `forge counter` with its expected one beyond its `forge fill` (D123): `PERFECT`, `NEAR`, `WARNING` or `FAIL` per bundle, the last its only failure, in `logs/snax-forge-check.txt` with each run's `forge cycles`. `docs/TOOL.md` section 6 runs it; CLAUDE.md says when each set is due.

## 19. Estimate

The ideal estimate of a bound graph (D155, `estimate/`), with or without a platform: what a mapping buys before any run, for the viewers and `design.md`; `estimate(graph, platform, tech)` costs it on a technology too (section 20; default `nangate45`, None leaves the cost out). It assumes ideal streams: no bank conflicts, no streamer fill, no DMA and no core control. An `Estimate` is a dataclass with `to_dict` / `from_dict` (`accelerator_cycles`, `unmapped_cycles`, `total_cycles` and `l1_footprint` written, and derived again when read; `unmapped_iterations` written and read):

| Field | Holds |
|---|---|
| `groups` | per group (one unit, or linked units, D146), in the graph's order: `name`, `units`, `links` (their containers), `cycles` (snax-gvsoc's group rule, `snax_group_rule.py`, stepped by `estimate/cycles.py`'s `group_cycles` with every input valid and the output ready, JOBS and each unit's T as the program writes them; `gen/rtl.py`'s `group_counter` calls the same), `bottleneck` (the max over its units of n · max(II, span), n the unit's firings, summed over the group's runs when several nests share it, D165), `ramp` (the cycles to start and finish the pipeline: the sum over its units of L + span − 1, plus 1 per buffer link and 0 per wire, once per run), `stall` (`cycles` − `bottleneck` − `ramp`), `fill` (the streamers' fill a run adds: 3, or 2 for a group whose units have one outer input between them), `counter` (`cycles` + `fill`, the expected BUSY_COUNTER: a guide, not a check, D155), `bank_bound` (⌈(Σ over its outer ports of beats × num_ch) / `l1.n_banks`⌉, the fewest cycles of L1 time the group's streams take, a bound on the streams and not on the BUSY_COUNTER, `cycles` staying ideal (D155), with `n_banks`, the platform's; both null without a platform; a note in the page, `design.md` and `run.md` when it exceeds `cycles`, over the streamers of that group only: `mul: the streams move 6144 words over 32 banks, at least 192 cycles of L1 time against the ideal 128`, dot 1 at W_mul = 16, whose streams took 211 cycles and its BUSY_COUNTER 186), `channels` (the sum of num_ch over its outer ports, D142), `words_per_cycle` (the 64-bit words its outer ports move per cycle), `bits` (per unit and per link, by its container, its datapath's register bits: the catalogue's `flops` on its params, section 17; a wire 0; a unit the generator does not fill is not listed), `ctrl_bits` (the register bits of its control unit from the group description that firtool keeps, every register `GroupRule.scala` and `GroupCtrl.scala` declare but a `v` at L 0 and the slots of a wire or of a depth-1 buffer, which are never written, at the platform's `acc_csr.job_width`, `estimate/bits.py`): CHAR1's cost model's counts, D154; and `cost`, on a technology its units', buffer links' and control unit's area and energy and the group's totals (section 20) |
| `accelerators` | per accelerator, in CSR order (`estimate/accelerators.py`): `name`, `groups` (its groups' names), `own_csrs` (its own: per group START, JOBS, STATUS, BUSY_COUNTER and a fold's JOB_BEATS), `csrs` (its first and last CSR, its streamer's then its own, section 16), `streamer` (its streamer set: `movers`, `channels`, `temporal_dims` per mover, `csrs`) and `cost` (`streamers`, each port's streamer priced, and `l1`, its words through the crossbar and the banks, section 20; null without a technology); without a platform both `csrs` are null, the streamer's spatial bounds being the platform's (dot 1: `vmul` 960–981 and `vsum` 982–999, 2 streamers and 40 CSRs; `dot_1_enc`: `dot` 960–996, 1 and 37) |
| `accelerator_cycles` | the sum of the groups' `cycles`: groups add up through L1 (D105) |
| `not_mapped` | the iterations of each nest no unit runs, counted once per innermost map that holds tasklets (a fused nest of two tasklets counts once), over every map around it except spatial ones (dot at N = 512: 1024 as imported, 640 once `mult_map` is split and tagged, 257 after `split_reduce` on `sum_map`) |
| `unmapped_iterations` | the iterations of each nest no unit runs as `not_mapped` counts them, but over every map around them, spatial ones included: a spatial tag only says which loop will become lanes, and lanes only exist once a unit runs them, so until the bind a tagged loop is still that many iterations of work (vecmul at N = 2048: 2048 as imported, after the split and after both tags, where `not_mapped` is 256; dot at N = 512: 1024, 1024, 1028, the last with the final sum's 4 lanes) |
| `unmapped_cycles` | `unmapped_iterations` × `UNMAPPED_CYCLES_PER_ITERATION` (1, `estimate/__init__.py`): the unmapped work as cycles, an assumption, since a core takes several cycles per iteration; one constant, so a measured value can replace it (D155) |
| `total_cycles` | `accelerator_cycles` + `unmapped_cycles`, so a step before any bind compares with the bound one as cycles (vecmul at N = 2048: 2048 at steps 0 to 3, 256 after the bind at W = 8; dot at N = 512: 1024, 1024, 1028) |
| `containers` | per container: `placement` (`l1`, `local` or `link` with its `depth`), `bytes` (the padded shape, D144, times the itemsize), `reads` and `writes` in 64-bit words: a bound outer port moves beats · num_ch per task, a tasklet no unit runs ceil(iterations · b / 64) per memlet over every map around it; a link's and a local value's traffic is not L1's |
| `l1_footprint`, `l1_size` | the bytes placed in L1, and the platform's L1 size (null without a platform) |
| `ports` | per outer port, `<instance>.<port>`: its `container`, its streamer's `loops` (var, count), `beats` and `num_ch` (ceil(lanes / num_slc), num_slc the elements of its dtype in a 64-bit word) |
| `splits` | per split map with a tail: `n` (`loop.split.valid`), `w` (its inner map's length, or once bound its lanes) and `lane_use` n / (ceil(n / W) · W) |
| `cost` | on a technology (section 20): `tech`, `source` (its table's commits), the design's `area`, `area_design`, `energy`, `kernel_energy`, `parts`, `xbar`, `memory`, `missing`, `not_costed` and `extrapolated`, each accelerator's own figures under `accelerators`; null without one |
| `hw_counts` | HWCOST's counts on a platform (section 20): per port's streamer its FIFO depth, loops, channels, words and beats, the crossbar's inputs and the banks' shape |

**The BUSY_COUNTER and the expected BUSY_COUNTER** (D155). A run's BUSY_COUNTER counts from its group's START, every cycle until busy drops (section 17), so it includes bank conflicts and is shortened by the streamers' head start: the program configures the streamers, then the accelerator, then starts the streamers, then the accelerator (section 9), so a stream may have data waiting when the group starts. Each group's `counter`, its `cycles` plus the streamers' fill, is the expected BUSY_COUNTER, a guide SNAX-DFG shows and `run.md` sets beside the measured one, not a check: a run may be above it or below it (dot 1 at N = 2048, W = 8: `dot_1_enc`, its two groups in one encapsulated set, vmul 256 against 259, its START 10 cycles after the shared streamer's start, and vsum 259; `dot_1`, two accelerators, vmul and vsum 259 against 259, each set's streamer started just before its START; vecmul at W = 1: 2048 against 2051). `tests/estimate/` holds it equal to the `forge counter` of each design of STATUS open item 95 whose SNAX-GVSoC counter it matches (`COUNTERS`), a regression of those designs; it does not match rowsum_2's contiguous layout (131 against 134, the first reads' bank conflicts), the seeded sums (the seed counted as one more input in the fill: matvec_seed's 1028 and matvec_seed_serial's 4100 against 1027 and 4099) and rowsum_serial (67 against 65; the RTL's 67). The unmapped cycles never enter a group's expected BUSY_COUNTER. `bank_bound` is in `to_dict` / `from_dict` with the rest. Against SYN2's 48 points (vecadd, vecmul, vsum_tree, dot 2 to 6 at W = 1 to 32; `tests/estimate/fixtures/syn2_flops.json`, from snax-forge-experiments' DC results), each datapath's `bits` matched the flops DC kept, and `ctrl_bits` matched DC's at all 48 points when it was built (the test holds it within 10 %, so a later control change does not fail on a bit or two): a check of the counts, not a promise of the cost's accuracy, which section 20 states. `design.md` shows the groups, `run.md` each task's expected BUSY_COUNTER beside the measured one.

## 20. Cost model

CHAR1's first-order cost model of a design's groups (D154) and HWCOST's of its streamers, crossbar and L1 banks (D167): what a design costs in area and energy before any synthesis, beside D155's cycles, for `design.md` and the viewers. It aims at the large costs and at showing a designer how a design is costed, for comparing variants against each other, not at accuracy: the intent is about 30 % of synthesis on the large costs, and a small block can be much further off. The evidence: on CHAR1's checked designs (vecmul, vsum, dot 4 and dot 5 at W = 8 and 32) the kernel energy landed at 0.99–1.04 of SYN2's synthesized and simulated accelerators (snax-forge-experiments `results/char/char_table.md`, Model), the Mac at 1.03–1.04, the small sums at 0.06–0.78; the streamer and crossbar tables reproduce their own points within the leave-one-out errors of snax-forge-experiments' `docs/cost_tables.md` (4 % mean on a reader's energy, 2.5 % on the crossbar's); evidence on those points, not a promise. The guide, with where every number comes from and a worked example, is [`snax_forge/char/README.md`](../snax_forge/char/README.md). Only nangate45 is offered (`TECHS`); nangate15's table stays in the repository as data and is refused (`char.tech`), as it has no memories (D159).

**The table** is `snax_forge/char/<tech>.json` (`nangate45`; `nangate15.json` kept as data), a byte copy of snax-forge-experiments' `results/char/<tech>/char.json` (its X37), which names its own source commits. `snax_forge/char/`'s `load(tech)` reads it; `tests/char/` checks every entry against the catalogue: its datapath in `hw/chisel/blocks.json`, its params with each point's lanes (and depth) a valid and complete param set there, its `lanes_rule` known, its points unique.

| Key | Holds |
|---|---|
| `tech`, `clock_ns`, `units`, `source` | the technology, the clock it was measured at, the units (µm², pJ, the flop's fJ per cycle), the snax-forge and snax-forge-experiments commits and the stimulus |
| `flop` | `area_um2` (A_flop), `e_load_fj` (a flop that loads, per cycle) and `e_hold_fj` (one that holds) |
| `ctrl` | `counter_factor` (f_ctrl) |
| `blocks` | per catalogue datapath, entries of `params` (the catalogue's but `lanes` and `depth`), `lanes_rule` (`copy` or `interpolate`) and `points` (`lanes`, LinkBuffer's `depth`, `comb_um2` A_comb and `e_comb_pj` e_comb, a fire's comb energy) |

**The lookup**, `lookup(table, datapath, params, lanes, depth=None)`, takes the entry whose params equal the block's as the generator fills them (lanes and depth aside), then: a point at these lanes (and depth) as is (`measured`); else under `copy` lanes × the lanes-1 point, a LinkBuffer's at the same depth (`copy`); under `interpolate` linear between the two measured lanes around it (`interpolate`); a LinkBuffer depth no point has is linear between the two measured depths around it at lanes 1, then copied (`interpolate`). Anything else gives None: no entry with these params (a dtype other than int64), or lanes or a depth outside the measured range (TreeAccumulate at lanes 64, SerialAccumulate at lanes 1, a LinkBuffer deeper than 16). None never refuses a design: the block's comb part counts 0 and the design note `char.missing` names it (section 14).

**The model**, per unit and per buffer link over one run of the accelerator: n_fire its fires (a unit's firings times its span, as SerialAccumulate reads span beats per firing; a buffer's loads, one per output of its producer, so its producer's firings over its T: a fold gives once per T beats, dot 6's Mac once per job, dot 2's mul at every firing), n_bits its `bits` (section 19) and C_acc the `accelerator_cycles` (with no clock gating every flop holds through every group's cycles); per group n_bits^ctrl its `ctrl_bits`:

$$E_{\mathit{unit}} = n_{\mathit{fire}}\thinspace\big(e_{\mathit{comb}} + n_{\mathit{bits}}\thinspace(e_{\mathit{load}} - e_{\mathit{hold}})\big) + C_{\mathit{acc}}\thinspace n_{\mathit{bits}}\thinspace e_{\mathit{hold}}, \qquad E_{\mathit{ctrl}} = f_{\mathit{ctrl}}\thinspace n_{\mathit{bits}}^{\mathit{ctrl}}\thinspace C_{\mathit{acc}}\thinspace e_{\mathit{hold}}$$

$$A_{\mathit{unit}} = A_{\mathit{comb}} + n_{\mathit{bits}}\thinspace A_{\mathit{flop}}, \qquad A_{\mathit{ctrl}} = 2\thinspace n_{\mathit{bits}}^{\mathit{ctrl}}\thinspace A_{\mathit{flop}}$$

The control unit's area is its flops' area plus an equal comb area (comb : flops = 1 : 1, the 2 above), a rough early estimate (D154); its energy is E_ctrl, its comb energy taken as negligible. The csr is left out (D152), and so are the cores, the I-cache and the DMA, which no table holds: the DMA's traffic is the same for every variant, and the cores' and AXI's ports enter only as crossbar inputs.

**HWCOST's tables** are `snax_forge/char/<tech>/`'s `streamer.csv` (one reader, X52: FIFO depth 2–16 × loops 1–7 × channels 1–16, 168 points, `area_um2` and `e_beat_pj`), `xbar.csv` (the sparse interconnect at 64-bit ports: inputs 4–64 × banks 16, 32, 64, `area_um2` on all 27 points and `e_access_pj` where inputs ≤ banks) `xbar_kinds.json` (the crossbar's read-only and write-only inputs, snax-forge-experiments X71: the factor form's $f_{\mathit{ro}}$ and $f_{\mathit{wo}}$, the crosspoint form's coefficients and the grid they were fitted on) and `ram_table.csv` (CACTI 7, X47, X48: depth 32–4096 in steps of 32 × width 8–256 in steps of 8, `area_um2`, `e_rd_pj`, `e_wr_pj`), copied as they are from snax-forge-experiments by `pixi run import-costs` (`util/import_costs.py`: its main by default, `--experiments DIR` a clean checkout), with `SOURCE` naming its commit and each file's path and sha256; never run in CI. `snax_forge/char/tables.py` reads them (`load_costs`, `streamer_cost`, `xbar_cost`, `bank`; `Costs.kinds`) by snax-forge-experiments' rule (its `report/strm_table.py`, X53, X54): linear along one axis at a time between the two grid values around the query, a grid value exact, each axis in the raw value but the crossbar's energy in 1/inputs; the crossbar's energy where inputs > banks the value at inputs = banks. Outside the grid, where X53 refuses, snax-forge extrapolates along the axis through its two nearest grid values in the same scale, clamped at zero, logs a warning and the design note `cost.extrapolated` names it (section 14). The banks are a lookup, never interpolated (D160): a bank off the grid costs 0 and is noted. A writer is priced as a reader (X52).

**HWCOST's model** (D167), per port's streamer p with D_p its FIFO depth (the platform's `fifo_depth`), L_p its loops (`temporal_dims`), C_p its channels (`num_ch`) and W_p the 64-bit words it moves in L1 in one run over every node of its instance (a shared instance runs once per node, D165; a link's container is not in L1), so W_p / C_p beats; n_in the crossbar's inputs (`design/check.py`'s `crossbar_inputs`: the streamers' channels plus the cores' and AXI's ports), n_ro and n_wo of them a reader's and a writer's channels (`estimate/cost.py`'s `xbar_kinds`, from the same streamers, a shared mover's once; the cores and AXI read and write), f_ro and f_wo the shares of a read-write input's area that a read-only and a write-only input lose (`xbar_kinds.json`, snax-forge-experiments X71: 0.396496 and 0.41077), n_banks the platform's banks, N_rd and N_wr the words read and written:

$$A = \sum_b A_b + \sum_g A_{\mathit{ctrl},g} + \sum_p A_{\mathit{strm}}(D_p, L_p, C_p) + A_{\mathit{xbar}}(n_{\mathit{in}}, n_{\mathit{banks}})\thinspace\Big(1 - f_{\mathit{ro}}\frac{n_{\mathit{ro}}}{n_{\mathit{in}}} - f_{\mathit{wo}}\frac{n_{\mathit{wo}}}{n_{\mathit{in}}}\Big) + n_{\mathit{banks}}\thinspace A_{\mathit{bank}}$$

$$E = \sum_b E_b + \sum_g E_{\mathit{ctrl},g} + \sum_p \frac{W_p}{C_p}\thinspace e_{\mathit{beat}}(D_p, L_p, C_p) + \Big(\sum_p W_p\Big)\thinspace e_{\mathit{acc}}(n_{\mathit{banks}}, n_{\mathit{banks}}) + N_{\mathit{rd}}\thinspace e_{\mathit{rd}} + N_{\mathit{wr}}\thinspace e_{\mathit{wr}}$$

The crossbar's area follows its inputs, but every access has one price for a bank count, a full crossbar's ($e_{\mathit{acc}}$ at inputs = banks, 2.739443 pJ on 32 banks), so its energy follows the accesses alone, whatever the inputs and whether the accelerators run in series: dot 5 linked (20 inputs, 4097 accesses) costs 72.371 nJ against 72.648 nJ unlinked (36 inputs, 4113). A streamer is priced per beat, so its energy follows its words too. The five terms are the parts (`PARTS`): datapath (the units and buffer links, b), control (the control units, g), streamers, crossbar and memory (the banks). Every count is read where it is made, so a later change to it (PORT1's crossbar inputs, PACK1a's channels) changes the cost without touching it. vecmul at W = 8 and N = 2048 (three ports of 8 channels and one loop, 256 beats each; 27 inputs on 32 banks, 16 read-only and 8 write-only; 4096 reads and 2048 writes): datapath 55 300 µm² and 23.137 nJ, control 362 µm² and 0.066 nJ, streamers 116 000 µm² and 30.457 nJ, crossbar 89 600 µm² and 16.831 nJ, memory 433 000 µm² and 27.738 nJ; in all 694 000 µm² (261 000 without the banks) and 98.229 nJ, the memory side 76 % of the energy where the 4 × e_load placeholder it replaced gave 54 %.

**The crossbar's input kinds** (snax-forge-experiments X70, X71). `xbar.csv` prices every input as a read-write port, but a reader's channel only reads (its write bit and data are constants) and a writer's only writes (its response data is never read), and synthesis removes what the constants make dead. The area is therefore the table's times $1 - f_{\mathit{ro}}\thinspace n_{\mathit{ro}} / n_{\mathit{in}} - f_{\mathit{wo}}\thinspace n_{\mathit{wo}} / n_{\mathit{in}}$, the factor form, the default: the table stays the source and a design with no streamer is priced as before. The crosspoint form is an option (`Estimate.xbar_in("crosspoint")`, `tables.xbar_cost(..., form="crosspoint")`): $b\thinspace(c_{\mathit{rw}}\thinspace n_{\mathit{rw}} + c_{\mathit{ro}}\thinspace n_{\mathit{ro}} + c_{\mathit{wo}}\thinspace n_{\mathit{wo}}) + d\thinspace n_{\mathit{in}} + e\thinspace b + g$ with $n_{\mathit{rw}} = n_{\mathit{in}} - n_{\mathit{ro}} - n_{\mathit{wo}}$, fitted on 16–64 inputs and banks and needing no table; `design.md` and the viewers show the factor form. The energy per access is the table's in both. dot 3 at W = 16 on the one-core cluster (D170; 35 inputs, 32 read-only and 1 write-only, 32 banks): 113 776 µm² (crosspoint 110 638), where every input read-write gave 181 823; the synthesized cluster has 102 884 µm² for this interconnect (CORE1 R4; CSYN1's two-core cluster, 36 inputs: 119 308 against 107 685), the rest being constants the factor leaves out (priority, user and strobe bits).

A shared set's movers (D169) are priced once each: the sum over p runs over its users, each with the mover's D and L, its energy per beat at its own C_p, and only the mover's first user carries an area, $A_{\mathit{strm}}$ at the mover's channels (the widest user's); `StreamerCounts`'s `mover`, `mover_channels` and `area` say which. The port mux between the movers and the groups costs nothing (it takes the port's path to the interconnect). dot 1 shared at W = 4: three movers of 4 channels, 15 crossbar inputs against 20.

**The fields** (`estimate/cost.py`; null throughout without a technology):

| Field | Holds |
|---|---|
| a group's `cost.units`, `cost.links` | per unit, and per buffer link by its container: `datapath`, `fires` (n_fire), `found` (`measured`, `copy`, `interpolate`, or null when not characterized), `area` (`comb`, `flops` n_bits A_flop and `total`, µm²) and `energy` (`fire` e_comb + n_bits (e_load − e_hold), pJ per fire; `hold` n_bits e_hold, pJ per cycle; `task` E_unit, nJ) |
| a group's `cost.ctrl` | `bits` (its `ctrl_bits`), `area` (`flops` n_bits^ctrl A_flop, `comb` the same, a rough estimate, and `total`, µm²) and `energy` (E_ctrl, nJ) |
| a group's `cost.area`, `cost.energy` | its units', buffer links' and control unit's added up (µm², nJ) |
| an accelerator's `cost.streamers` | under `accelerators` (section 19), per port's streamer: `name`, `accelerator`, `write`, `depth`, `loops`, `channels`, `words`, `beats` (its counts), `area` (µm²), `e_beat` (pJ), `energy` (nJ) and `extrapolated` (the axes off the grid) |
| an accelerator's `cost.l1` | under `accelerators`, its words through the crossbar and the banks: `words`, `bits` (64 words), `reads`, `writes`, `xbar` and `banks` (their energies, nJ) and `energy` (their sum); the accelerators' rows add up to the crossbar's and memory's energy |
| the estimate's `cost` | `tech`, `source` (the table's), the design's `area` (every part costed, the banks included), `area_design` (without the banks), `energy` (every part costed), `kernel_energy` (the groups', datapath and control), `missing` (the units and links not characterized), `parts` (per part `area` and `energy`, null where not costed), `xbar` (`inputs`, `banks`, `accesses`, `area`, `e_access`, `energy`, `extrapolated`, `read_only`, `write_only`, `form`), `memory` (`banks`, `depth`, `width`, `reads`, `writes`, `area`, `e_rd`, `e_wr`, `energy`, `found`), `not_costed` (the parts not costed and why: no platform, no tables) and `extrapolated` (the streamers and the crossbar priced off the grid) |
| the estimate's `hw_counts` | what HWCOST prices, so `from_dict` rebuilds the cost: `streamers` (each its counts), `xbar_inputs` (channels, others) and `bank_shape` (banks, depth, width); null pieces without a platform |

**What is shown.** The fields keep full precision; what a reader sees is rounded for display only: an area to 3 significant figures (`estimate/cost.py`'s `sig3`: 63 200 µm²), an energy to 3 decimal places in its unit (`dec3`: per run and per task in nJ, 26.285 nJ; per fire and holding per cycle in pJ, 97.259 pJ and 3.660 pJ), so two close designs can be told apart (dot 4 and dot 5 at W = 8: kernels of 26.285 and 26.339 nJ). The decimals are display precision, not accuracy, which stays as stated above; every cost is shown under the heading "Cost estimate (first-order) on <tech>" with one line (`CAVEAT`): first-order, aimed at about 30 % on the large costs, the control unit's comb area a rough estimate, the streamers priced per beat and the crossbar per access, the csr, the cores, the I-cache and the DMA left out. `design.md` (`pixi run report DIR --tech T`, default nangate45) shows each group's area and energy beside its cycles, then per unit, buffer link and control unit how it is costed, per port its streamer, per accelerator its words through the crossbar and the banks, the cost by part with each part's share of the area and of the energy, the total and the total without the banks, what is not characterized, not costed or extrapolated; the viewer and the editor show the same, the parts as two bars to scale (section 12).

**The checks.** `tests/estimate/test_cost.py` is a consistency check, not one of accuracy: it asks that the code computes snax-forge-experiments' formula on the same inputs. On vecmul, vsum_tree, dot 4 and dot 5 at W = 8 and 32 with N = 2048 on `platforms/snax_forge.json`, on both technologies, the estimate's kernel energy (`kernel_energy`: units, links and ctrl, no L1 accesses) is within 0.5 % (the rounding of the experiments' table) of that model of the same designs (`results/char/char_table.md`, Model, copied into `tests/estimate/fixtures/char_model.json` by `util/char_model.py`), with the same C_acc (SYN2's window) and ctrl bits (DC's). The accuracy evidence is the experiments' own comparison with SYN2, above; `tests/char/` checks the tables against the catalogue and section 19 the bit counts. `tests/char/test_tables.py` checks the tables against `SOURCE` and the rule against `cost_tables.md`'s four worked examples to their printed digits; `test_cost.py` prices vecmul at W = 8, dot 4 and mvt_1_dot2 by hand from the tables' rows, and vecmul at W = 32 off the grid. The CHAR1 check runs on nangate45's eight rows only (its nangate15 rows skipped since nangate15 is not offered).

**Known limits.** The control unit's comb area is only its 1 : 1 estimate. A streamer is priced per beat and a crossbar access at one price: a streamer's clock while it waits for its accelerator's turn, the crossbar's per-cycle part and its conflicts are not counted (left for energy from what ran, after the paper); a writer is priced as a reader. An accumulator's isolated fire costs 25–30 % more than a back-to-back one (its adders settle twice), which the busy and idle windows do not model. Deep LinkBuffers come out about 1.4–1.8× high on nangate45 (PT's stream slope there is 0.56–0.71 of one held slot per depth, snax-forge-experiments X35). Comb energy is counted per fire only, so a unit fed by another unit is undercounted when its inputs change more often than it fires: dot 4's TreeAccumulate sum fires once but sees 256 new Mac outputs and comes out at 0.06–0.19 of SYN2, the kernel staying within 4 % because such sums are small.
