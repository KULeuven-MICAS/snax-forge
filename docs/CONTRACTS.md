# SNAX-FORGE Contracts

> What the artefacts between the components mean, field by field, so each
> component can produce its output without reading another's code. Why each
> is the way it is: `docs/DECISIONS.md` and the module docstrings.
>
> Sections 2–6 and 8 were SNAX-MODEL's, retired in RET1 (D116); their numbers
> are kept. The artefacts are dataclasses and plain JSON until the M6 freeze
> (D26); the generated hjson files already have schemas (section 16). Every
> fenced block under a `<!-- snippet: ... -->` line is copied verbatim out of
> the file it names, and `tests/test_contracts.py` fails if one drifts (D46).

---

## 1. Conventions

**Byte addresses.** Every memory address (L1 and L2 addresses, streamer
`base`, all strides, DMA bases) is a byte address.

**The bank word is the unit of transfer** (D142). A word is the L1's
`width_bits` wide and holds num_slc = `width_bits` / b slices (elements) of
each container's dtype (b its bits), little-endian as NumPy lays them out. A
port of W lanes uses num_ch = W / num_slc streamer channels by default;
`connect.pack` and `connect.num_ch` check a port against that (section 14),
and lowering lays each container out by its itemsize.

**Integers first** (D28). The default `dtype` is `int64`.

The L1 of `platforms/snax_forge.json` (section 13):

<!-- snippet: platforms/snax_forge.json -->
```json
 "l1": {"n_banks": 32, "width_bits": 64, "rows": 512, "base_addr": 268435456, "wide_bits": 512},
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

What the reports and the viewer read of a flow run (D137). A run directory
(`out/flow/<name>/run/`, `run/outputs.py`) holds `run.json` (what ran, D50),
`profile.json`, and above trace level `off` `trace.jsonl` and
`trace_meta.json`. SNAX-GVSoC writes its events (snax-gvsoc `docs/EVENTS.md`)
into `run/events.log`, which `run/convert.py` renames into what follows; the
invariants every conversion checks are its own.

**The window.** `total_cycles` is the program's: its `mcycle` read in
`forge_check()` minus its read in main, the `forge cycles:` line. The
converter finds the same two reads in the accelerator's core's events,
checks they are `total_cycles` apart and shifts every cycle so that t = 0 is
the read in main (the shift, the boot, in `trace_meta.json`); events outside
`[0, total_cycles)` are dropped. The profile, a core's classes and the trace
cover this one window.

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

Names, ports, `write` and the FIFO depths come from the hjson files and the
design point.

**Trace** (`run/trace.py`): one event per line of `trace.jsonl`, a flat dict
of `t`, `k`, `src` and the kind's fields. The levels are cumulative, each
turning on SNAX-GVSoC traces: `off` none, `task` the `events` lines, `beat`
the `beats` lines, `stall` the `stalls` lines, `finegrained` the cores'
`insn` lines.

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

`src` is a unit (`acc`; with several groups each, `mul`, `sum`, D148), a
mover (`acc_a`), `dma`, `core0`, and for a `grant` or `stall` the crossbar
port (`acc_a.lane0`, `core0`, `axi`). `block` is the component whose CSR it
is (`acc`, a group's unit; `acc_streamer`, `<accelerator>_streamer`). Beat
levels of an accelerator of several groups are refused (`convert.level`;
open item 96). A DMA copy's `dir`
(`l2_to_l1`, `l1_to_l2`) comes from its addresses, and copies are matched to
the task list's `load_<C>` and `store_<C>` tasks in issue order.

**Ids.** Lines are linked by ids, never by order: a streamer beat is its
mover and `b` (a reader's numbered in its AGU's order from 0 per run, a
writer's in the order the accelerator hands them over; a repeated beat is
one `b` for several firings); a firing's `n` is its index in the run, a
push's its index among the run's results, equal to the writer's `b`; a DMA
beat on L1 is its copy's `id` and `i`. `fire`'s `b`, `value` and `vals` are
keyed by reader mover; `push`'s `first` and `last` are the firings it folds.

**Added by the converter**: `task` (the task whose start and done bracket
the line on its component, a copy's for the DMA, null for a core's or AXI's
access), `mem` (`l1`, `l2`) and the absolute `addr`, `region` (the container
the address is in, or null), `elems` (the flat indexes of its elements in the
word, a burst's bytes), `vals` (those elements' values by the region's dtype;
a fire's and a push's from their beats' grants, null for a mover not traced),
an `insn`'s `func`. Every line keeps its raw words: `value` the bank word
(`dma`: one per bank it covers), `bank` and `row` the word-interleaved map of
`addr`, `back` the cycle the response arrives. The beat and stall lines only
feed the trace; the profile stays the summary lines.

**Classes.** An accelerator's or a mover's `done` carries its run's `cycles`
by class, from `task` on. From `beat` on, a `class` event gives the first
cycle of each run of one class, as SNAX-GVSoC classed it: per component run
they cover the cycles its counts add up over, with no gap and no overlap.

**Sources and meta.** `--trace LEVEL[:SOURCES]` names the sources whose
beat, stall and instruction lines are written (`acc`, the movers, `dma`,
`core<i>`, `axi`); by default the accelerator, its movers and the DMA, at
`finegrained` also every core. `summary` and `events` lines are always
whole. `trace_meta.json` holds `level`, `total_cycles`, `boot`, `sources`
(the cluster's order), `filter` (the sources written) and `start`: per core
`{"func": ..., "mode": ...}`, its last `func` at or before t = 0 (or null)
and its state there. `--vcd` writes a VCD and `view.gtkw`, never read here.
`run.json` records `trace_level`, `trace_sources` and `vcd`.

**Regions** (`run/regions.py`, `run/elf.py`, D86). A container's L1 region
is the design point's layout; its L2 region has the plan's shape and dtype
at its array's address in the linked program: `run.json`'s `elf_symbols`,
container -> `addr` and `size` (its bytes padded to whole DMA beats; a
missing symbol or another size stops the flow). Everything takes regions
from `regions_of(point, elf_symbols)` only.

## 8. Rules for a new block kind

Retired in RET1 (SNAX-MODEL).

## 9. Task list

SNAX-LOWER's ordered list of tasks (D64): which component runs which task
with which values, where each is configured and started, and what it waits
for. `lower/c.py` turns it into the program (section 16). It is derived from
a design point (`lower/derive.py`, `pixi run lower tasks DESIGN_POINT`) or
written by hand. A task list is `name` and `steps`, each with an `op`:

| `op` | Fields | Becomes |
|---|---|---|
| `configure` | `task_name`, `type`, `component`, `after`, `wait_mode`, `values` | the component's configuration, written here |
| `start` | `tasks` | the waits these tasks need, then their `start` writes in list order |
| `sync` | `task`, `mode` | one wait on the task's component, always placed |
| `read` | `reg` | a read of `block.register`; no C form yet (`c.read`) |

`type` says how `values` are read:

```
streamer  base, temporal_bounds, temporal_strides, spatial_strides (bytes),
          in words (D142): a channel moves one bank word of num_slc slices
          of the port's dtype. The fastest spatial loop runs S_0 / num_slc
          channels, its stride num_slc times its element stride times the
          itemsize (8 bytes for a packed port); every other stride is its
          element stride times the itemsize, and base the layout's base
          plus the offset's bytes. At num_slc = 1 these are the element
          strides as before. Spatial
          bounds are design-time, in elements, and come from the design
          point's streamers. An idle mover (another group's, D148): every
          temporal bound 0, so it never starts, and every temporal stride
          a bank word, never 0 (a reader whose first loop has stride 0
          runs that loop once whatever its bound, snax_cluster's
          Reader.scala)
dma       direction ("l2_to_l1" or "l1_to_l2"); src and dst, each with
          bounds and strides (bytes, one wide beat per step), the L1 side
          with base, the L2 side with symbol (its container's data.h array)
accel     its start parameters by name: n, then any named rate (e.g. T)
```

**Names** (D75): streamer `<instance>_<port>` (`acc_a`), task
`<node>_<component>` (`add_acc`), DMA task `load_<container>` or
`store_<container>`, with `_<k>` appended for tile `k`.

Only the loops a task uses are given; the backend pads the rest with bound
1 and stride 0. `after` names earlier tasks this one needs finished: data
it reads, a buffer it overwrites. `wait_mode` (`poll` or `signal`) is used
whenever the lowering waits for this task. Missing `after` and `wait_mode`
default to `[]` and `poll`; every field is written back, and unknown keys
are errors.

**Checks.** When a list is made: task names are unique; `after`, `start` and
`sync` refer to tasks configured earlier; a component has at most one
configured task not yet started; a task starts once, after its `after`
tasks; every configured task is started. Against the design point
(`check_values`, `c.tasks`): the component exists with the task's type, the
values have exactly the type's keys, a streamer's loops fit it, a DMA's
direction is valid, every bound is at least 1 (or, an idle mover's, every
one 0).

**Waits** (`placed_waits`, `lower/waits.py`). Before a `start`, one wait per
component, on its latest task, for every `after` task and listed component
not yet covered, leaving out a wait another covers. A wait on a writer
streamer also covers its unit and that unit's readers started with it; an
idle mover's task needs no wait. Where configures, starts and syncs go is
the task list's.

**Derived** (`lower/derive.py`): per accelerated node in execution order,
`load_<C>` for each input not yet in L1, a task per streamer, the
accelerator task (`n` the firings, each named rate from `named_rates`,
D102), one `start` of them all, `store_<C>` for each output written for the
last time; one `sync` per non-transient output at the end. In an
accelerator of several groups (D148) each task also configures the other
groups' movers idle, in the accelerator's order, and its `start` launches every
mover: dot 1's `mult_mul` starts `mul_a`, `mul_b`, `mul_out` and `sum_a`,
`sum_out` idle. A linked group (D148) is one such task at its first unit's
node, `<node>_<group>`, on the group as its component: its units' streamers
on the ports on no link only (no load or store for a link's container), and
one accelerator task whose `values` are each unit's (`n` and its named
rates) by unit, in the group's order (dot 6's `mult_dot`: `{"mac": {"n":
128, "T": 128}, "sum": {"n": 1, "T": 1}}`); its program writes the group's
JOBS = n_u / ∏ T_v over its first unit u and its folds v, then each fold's
JOB_BEATS but a fixed one's (section 17). `after` holds data dependences only, so two groups chain
through L1 with one wait (D105), on the first's writer. vecadd's design point on `platforms/snax_forge.json` gives
`tests/run/fixtures/vecadd/tasks.json` exactly; its end:

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

One accelerator as plain data (D68, D70): a hand-written JSON file in
`snax_forge/brm/library/`, named after it, in the form `Brm.to_json` writes
(every field, defaults included). `load_brm(name)`; unknown keys are errors
and a missing part is rejected by name.

| Part | Holds | Read by |
|---|---|---|
| `interface` | params (`design` / `runtime`) and ports (name, direction, lanes, rate, dtype) | everything below |
| `function` | an accelerator kind registered in `brm/function.py`, its factory params without timing, and `code`: what one lane computes (D82) | the reference executor; `bind` and the accelerated node (`code`) |
| `dataflow` | a notation (`affine` or `stream`) and one nest per port, in logical indices | the streamer values (`affine`), or nothing beyond the interface (`stream`) |
| `pattern` | `family` (a matcher registered in the sandbox), its `attrs` | SNAX-SANDBOX's `bind` (section 12) |
| `implementations` | per name: `source` (`chisel` only), `supports`, `timing` (`latency`, `initiation_interval`, `drain`, `span`), `binding` (null, or `datapath`, `args`, `controller`, optionally `ports`, D129) | the GVSoC accelerator, the HW generator |

The first four are shared; designs that differ only in timing, supported
values or RTL source are implementations of one BRM.

**Ports.** Names are not Python keywords and do not start with `_` (D103);
the convention is `a`, `b`, ... for inputs and `out` for the output. A port
moves one beat every `rate` firings, an int or a runtime param's name: an
input of rate `r` is taken at firings `0, r, 2r, ...`, an output at `r-1,
2r-1, ...`. An elementwise block has rate 1 everywhere, a reduction input
rate 1 and output rate `T`.

**Timing**, each an int or an expression over design params: `latency` (L,
a firing to its push), `initiation_interval` (II, the least cycles between
firings), `drain` (the cycles, starting with a push, in which no input is
taken; 0 by default) and `span` (the cycles a unit reads one input beat in
place, section 17; 1 by default). Only L delays a result.

**Params.** A `design` param is fixed per instance (lanes, the op) and
reaches the design point and the accelerator hjson; its `default` is used
when the design point gives none, `values` limits it, and an
implementation's `supports` limits it further. The `runtime` params are
exactly `n` (the firings in the task) and every named port rate, in port
order: the accelerator task's `values` (section 9); `n` must be a multiple
of every resolved rate. A value field holds an int or an expression over
params; a fixed string is a design param with one allowed value.

**The function** names a registered kind (`elementwise`, `reduce`, `mac`,
with an op by name) and its params over the design params; the reference executor
runs an accelerated node through it (D79). `code` is one `output =
expression` per output port over the input ports, per lane of one firing; a
reduction says what one element contributes (`accumulate`'s `out = a`), its
fold being its pattern's and kind's (D102). A null `code` cannot be bound.

<!-- snippet: snax_forge/brm/library/elementwise_add.json -->
```json
 "function": {
  "accel": "elementwise",
  "params": {"lanes": "W", "n_inputs": 2, "op": "op"},
  "code": "out = a + b"
 },
```

**The binding** (D129, `brm/binding.py`): `{datapath, args, controller,
ports}`. `datapath` a datapath of the catalogue (`hw/chisel/blocks.json`,
section 17); `args` its parameters, each an expression over design params;
`controller` a registered kind, `counted` the only one (the group's control
unit, D147); `ports`, optional and written only when given, maps every BRM port to
the datapath's (by default the same name). Which datapaths exist is the
generator's (`gen/blocks.py`): `resolve` checks that the binding's takes the
instance. The implementation's `drain` must be 0: the control unit has
none (`gen.drain`). The CSRs are not the binding's:
they follow from the unit's group (section 17), its JOBS J = n / T (T the
fold's rate, 1 for an elementwise unit). Expressions are stored canonical.

<!-- snippet: snax_forge/brm/library/accumulate.json -->
```json
  "chisel_accumulator": {
   "source": "chisel",
   "supports": {"W": [1]},
   "timing": {"latency": 1, "initiation_interval": 1, "drain": 0, "span": 1},
   "binding": {
    "datapath": "TreeAccumulate",
    "args": {"lanes": "W"},
    "controller": "counted",
    "ports": {"a": "in", "out": "out"}
   }
  },
```

<!-- snippet: snax_forge/brm/library/elementwise_add.json -->
```json
  "chisel_tiled_spatial": {
   "source": "chisel",
   "supports": {},
   "timing": {"latency": 0, "initiation_interval": 1, "drain": 0, "span": 1},
   "binding": {
    "datapath": "ElementwiseLanes",
    "args": {"lanes": "W", "op": "op"},
    "controller": "counted"
   }
  }
```

The BRM `mac` (D149) says what one lane computes, `out = a * b`, its output
folded per lane over T beats by its kind `mac` (the product at the inputs'
result type, cast to the output's, D143); its implementation `chisel_mac`
binds the datapath `Mac` with `accumulate` true and `regOut` false (a bool
arg is a design-time variant, kept as it is). Its pattern, `lane_fold`, is a
fused nest (section 12): a multiply and a reduce joined by a local value, the
fold indexed by the spatial variable:

<!-- snippet: snax_forge/brm/library/mac.json -->
```json
 "function": {"accel": "mac", "params": {"lanes": "W", "op": "op"}, "code": "out = a * b"},
 "dataflow": {"notation": "stream", "ports": {"a": {}, "b": {}, "out": {}}},
 "pattern": {"family": "lane_fold", "attrs": {"op": "mul", "fold": "add"}},
```

<!-- snippet: snax_forge/brm/library/mac.json -->
```json
   "binding": {
    "datapath": "Mac",
    "args": {"lanes": "W", "accumulate": true, "regOut": false},
    "controller": "counted"
   }
```

`accumulate`'s `chisel_serial` binds `SerialAccumulate`, one lane a cycle, so
its span is its lanes:

<!-- snippet: snax_forge/brm/library/accumulate.json -->
```json
  "chisel_serial": {
   "source": "chisel",
   "supports": {},
   "timing": {"latency": 1, "initiation_interval": 1, "drain": 0, "span": "W"},
   "binding": {
    "datapath": "SerialAccumulate",
    "args": {"lanes": "W"},
    "controller": "counted",
    "ports": {"a": "in", "out": "out"}
   }
  }
```

**Two notations** (D70). An **`affine`** nest of a port: the operand's
`shape`, an `offset` and `loops`, outermost first, each with a `bound`, one
stride per operand dimension and a `spatial` flag; the element of a step is
`offset + sum(i_l * strides_l)`. Temporal loops first (one step is one beat),
spatial loops last (the lanes, the last fastest); spatial bounds use design
params only and multiply to the port's lanes. A temporal stride of 0 repeats a
beat. Section 9 maps the nest through a layout onto a streamer's values, and
`bind` checks that the memlets give its order.

A **`stream`** nest, for unit BRMs, is empty: the port's lanes, rate and dtype
are the interface's, and the bound memlet gives the element order, so `bind`
checks no order, SNAX-LOWER takes the streamer's values from the memlet alone,
and the design step takes the spatial bounds from the memlet's ranges. Any key
is refused. The library's BRMs are all in it:

<!-- snippet: snax_forge/brm/library/elementwise_add.json -->
```json
 "dataflow": {"notation": "stream", "ports": {"a": {}, "b": {}, "out": {}}},
```

## 11. SNAX-DFG

The workload as one `.snaxdfg` JSON file (D71), written by an importer and
rewritten by each sandbox transform. `Graph.load(path)`; every field is
written, missing keys take defaults, unknown keys are errors, and a graph is
validated when it is made.

| Field | Holds |
|---|---|
| `name` | the kernel |
| `symbols` | name → int or `null`: `null` as imported, the value once a recipe binds it; expressions keep the name |
| `containers` | name → `shape` (ints or expressions over symbols), `dtype` (a NumPy dtype name), `transient`, `attrs` (namespaced only; `sdfg.scalar: true` marks a kernel's scalar argument, shape `[1]`, read as element 0, which the reference executor also takes as a 0-d input) |
| `body` | the nodes, in execution order |

**Nodes.** Every node has `id`, `kind`, `inputs`, `outputs`, `attrs` and,
for a kind that has one, `body`. `inputs` and `outputs` map a connector to a
memlet, `{"data": container, "subset": [dim, ...]}`, one dimension per
container dimension: an index (`i`, `4 * i_t + i_s`, `0`) or a range
`begin:end[:step]`, end exclusive. Node ids are unique identifiers, since
task names are built from them (D75).

**Folding writes** (D102). A tasklet's output memlet may hold `"reduce":
{"op": ..., "identity": ...}`: the elements it writes start at `identity`
and every iteration's value is folded in with `op` (`add`, `mul`, `min`,
`max`; `register_reduction`). Written only when set; an error on an input
and on an accelerated node. dot as imported:

<!-- snippet: tests/dfg/fixtures/dot.snaxdfg -->
```json
     "id": "sum",
     "kind": "tasklet",
     "inputs": {"in1": {"data": "tmp0", "subset": ["i"]}},
     "outputs": {"out": {"data": "out", "subset": [0], "reduce": {"op": "add", "identity": 0}}},
     "attrs": {"code": "out = in1"}
```

| Kind | Attrs | Body | Connectors |
|---|---|---|---|
| `map` | `var`, `range`; `loop.kind` absent, `tile`, `temporal` or `spatial`; `loop.split` (the `var` and `range` before `split_map`, and `valid`, the iteration count n, when n is not a multiple of the factor: the lanes past it are padding, `dfg/pad.py`, D144); `loop.tagged` (true when `tag_map` set the kind) | yes: runs once per value of `var` | none (derived from the body) |
| `tasklet` | `code`: one `output = expression` per output connector | no | one element each: indices only |
| `link` | `container` (the transient it replaced, which the writer and the reader still name, so the reference executor runs the link as it), `depth` (0 a wire, ≥ 1 a buffer), `from` and `to` (the units' instances) | no | none |
| `accelerated` | `instance`, `brm`, `implementation`, `code` (what one lane computes, the BRM's `function.code`), `params` (design params), `replaced` (the subtree `bind` replaced, or null), `accelerator` (written only when set: `encapsulate`'s name for the accelerator its group is in, D148), `group` (written only when set: `link`'s name for the group the unit is in) | yes: empty for a leaf block, a nested block otherwise | the BRM's ports, each port's lanes as a range; one beat |

**Accelerators and groups** (D146, D148). A bound graph nests accelerator ⊃
groups ⊃ units and links: a unit is an accelerated node, a group a connected
set of units and links (`link`, section 12), so an unlinked unit is a group
of its own, named after its instance; a linked group's units carry its
name (`group`). A link node sits before its reader in the body the two
share, and is drawn in the accelerated node's style (`out_partial = wire`,
`tmp0 = buffer`, its depth, what it replaced). A fused map (`fuse_map`) is
one nest of two tasklets, the transient between them a local value, no
longer in L1: the container carries `loop.local` (the fused map's id), and
the fused map records the two nests it was made of (`loop.fused`); a map
`split_reduce` split records its partials and its final sum
(`loop.split_reduce`). By default each group is its own
accelerator, named after it; `encapsulate` (section 12) puts several into
one, which shares one slot, one CSR window and one wrapper (section 17),
each group still a task of its own, the groups in the graph's order. The
nodes of one instance carry the same `accelerator` or none, and an
accelerator's name is no instance's (`DfgError`).

Attrs without a namespace belong to the kind and are all written;
namespaced attrs (`loop.*`, `mem.*`, `hw.*`, `user.*`) pass through
untouched. More kinds come with `register_kind`.

**Names.** Every name in an expression is a symbol or an enclosing map's
`var`; shapes use symbols only; symbols, containers and variables never
share a name. An expression is ints, names, `+ - * //`, and `max(a, b)`,
`min(a, b)` (two arguments, no other call; `snax_forge/expr.py`).
Expressions are stored as `ast.unparse` writes them, one without names as
an int.

vecadd after `split_map` (W = 4) and `bind`: the temporal map keeps what it
was split from, and the accelerated node what it computes and what it
replaced (D82), which `unbind` and `join_map` take back. `replaced` is
checked in the node's scope, never run, and its ids may repeat live ones.

<!-- snippet: tests/dfg/fixtures/vecadd_accelerated.snaxdfg -->
```json
   "attrs": {
    "var": "i_t",
    "range": "0:N // 4",
    "loop.kind": "temporal",
    "loop.split": {"var": "i", "range": "0:N"}
   },
```

<!-- snippet: tests/dfg/fixtures/vecadd_accelerated.snaxdfg -->
```json
     "inputs": {
      "a": {"data": "A", "subset": ["4 * i_t:4 * i_t + 4"]},
      "b": {"data": "B", "subset": ["4 * i_t:4 * i_t + 4"]}
     },
     "outputs": {"out": {"data": "C", "subset": ["4 * i_t:4 * i_t + 4"]}},
     "attrs": {
      "instance": "acc",
      "brm": "elementwise_add",
      "implementation": "chisel_tiled_spatial",
      "code": "out = a + b",
      "params": {"W": 4, "op": "add"},
      "replaced": {
       "id": "add_map_s",
       "kind": "map",
```

## 12. Recipe

SNAX-SANDBOX's ordered list of transforms (D72), a hand-written JSON file in
`recipes/`. `Recipe.load(path)`; every field is written (`dtypes` only when
given), `params`, `symbols` and `steps` default to empty, unknown keys are
errors.

| Field | Holds |
|---|---|
| `name` | the recipe; its steps are written to `out/sandbox/<name>/` |
| `kernel` | the kernel whose import the recipe starts from |
| `params` | recipe params, each an int or a string; `--set NAME=VALUE` overrides one |
| `symbols` | a value for graph symbols: an int or an expression over the params |
| `dtypes` | optional: container -> NumPy integer dtype name, in place of the kernel's (D113, D142): the import applies it to the kernel's descriptors before DaCe builds the SDFG, and the kernel's inputs are cast to it (`astype`, wrapping); an unknown dtype or container is an error naming it (`vecadd_i32`: A, B, C `int32`) |
| `steps` | in order: `transform` (a registered name) and its `params` |

A transform's int params take an int or an expression over the recipe params
(`"factor": "W"`). Running a recipe binds the symbols (step 0), applies the
steps and checks every step against the reference executor on the same
seeded inputs; each is written as `<i>_<transform>.snaxdfg`, with the recipe
as `recipe.json`.

| Transform | Params | Does |
|---|---|---|
| `split_map` | `map` (id), `factor` (int) | a map over `b:e` becomes a `temporal` map over `0:(e - b) // factor` (same id) around a `spatial` map `<id>_s` over `0:factor`; when the length n is not a multiple of the factor, over `0:(n + factor - 1) // factor`, with `loop.split.valid` n: the lanes at or past n are padding (D144), which the reference executor neither reads nor writes |
| `tag_map` | `map` (id), `kind` (`temporal` or `tile`) | an untagged map is tagged as it is, not split, and records it (`loop.tagged`); a tagged map is an error |
| `untag_map` | `map` (id) | the tag `tag_map` set is removed; a map `split_map` tagged is an error (`join_map` undoes it) |
| `bind` | `node` (a tasklet id, or a fused nest's spatial map id), `brm`, `implementation`, `instance`, `params` (optional design params) | the tasklet and its spatial map become an accelerated node; every map around the spatial map must be `temporal` (the firing loops: an untagged one is an error naming `tag_map`, a `tile` one is not lowered yet); the lanes param comes from the spatial bound; the dimension over the spatial variable becomes the lanes' range, the others stay indices; the tasklet's code on the port names must be the BRM's `function.code`; a folding output that does not use the spatial variable becomes one lane, `c:c + 1` (D102); a fused nest's two tasklets bind as one, the first's inputs and the second's output (`lane_fold`, section 10), and the accelerated node takes the first's id; a tasklet that needs a fused map or a split reduction is refused with the fix named ("fuse_map first", "split_reduce first", D150); a unit on a link is not unbound (unlink first) |
| `unbind` | `node` (accelerated id) | the accelerated node becomes the subtree it `replaced` |
| `join_map` | `map` (id) | a temporal map with `loop.split` and its spatial map become the map they were split from |
| `encapsulate` | `accelerator` (a name), `units` (instances) | the groups of those units become one accelerator of that name (their nodes' `accelerator`, D148); `units` in the order the graph runs them, which the groups keep; a unit already in an accelerator, a name an instance or an accelerator has, or a unit the graph lacks is an error |
| `unencapsulate` | `accelerator` (a name) | each of its groups becomes its own accelerator again |
| `split_reduce` | `map` (id), `factor` (int) | an untagged map whose one tasklet folds into an element that does not move with it is split as `split_map` splits it, `<map>` (temporal) around a spatial `<map>_s`; the tasklet writes one partial per lane into a new transient `<output>_partial`, indexed by the output's dimensions that move with an enclosing map, then the spatial variable (dot's `out_partial[4]`, matvec's `q_partial[M, 4]`); a final sum, tasklet `<tasklet>_final` in a spatial map `<map>_final` over `0:factor` right after `<map>`, folds them into the output; a length that is not a multiple of the factor is refused (the partials' padding is not placed yet) |
| `join_reduce` | `map` (id) | undoes `split_reduce` |
| `fuse_map` | `first`, `second` (map ids, `second` right after `first`) | two nests with the same ranges and tags level by level down to one tasklet each, joined by a transient written and read at the same index, become one nest of two tasklets that keeps `first`'s ids and variables, the second's other nodes after the fused one at their level (matvec's final sum inside the row map); the transient becomes a local value (no other node may name it); `bind` on it takes its spatial map's id (dot's `mult_map_s`, matvec's `mult_map_1_s`) |
| `unfuse_map` | `map` (id) | undoes `fuse_map`, when nothing changed since (unbind first) |
| `link` | `container`, `depth` (int), `group` (a name) | a transient between two bound units becomes a link node, the units' groups merge under `group` (written on every unit of both groups), and the transient leaves L1; refused: a container that is not a transient with one writer and one reader, or a local value (`link.container`), an end that is not bound (`link.unbound`), the writer's beats not the reader's in the same order, each port's memlet over its firing loops at the recipe's symbols (`link.order`), a name that differs from one already set, another instance's or group's, or units in two accelerators (`link.group`) |
| `unlink` | `container` | undoes `link`, the group taken off every unit left alone; two groups of several units left under one name is `unlink.group` |

`recipes/vecadd_undo.json` takes vecadd's last step back to the first. dot
binds the multiply map to `elementwise_mul` and the sum map to
`accumulate`'s `chisel_adder_tree`, whose `out` is one lane every `T = N /
W` firings:

<!-- snippet: recipes/dot_1.json -->
```json
  {"transform": "split_map", "params": {"map": "sum_map", "factor": "W"}},
  {
   "transform": "bind",
   "params": {
    "node": "sum",
    "brm": "accumulate",
    "implementation": "chisel_adder_tree",
    "instance": "sum"
   }
  },
```

`recipes/dot_1.json` then puts the two groups into one accelerator (D148),
with a task each: the `mul` group's task sets the `sum` group's movers to a
temporal bound of 0, so they never start, and the other way round (section
9). An idle mover never starts only if its first temporal stride is nonzero:
snax_cluster's Reader.scala takes stride 0 on loop 0 as one AGU pass, that
bound forced to 1 and the repeater set to the bound, so a reader at stride 0
would start and read (the C backend refuses it, `c.idle`). Without the step the two are two accelerators, which the platform's one
slot refuses (`gen.instances`, `recipes/dot_serial.json`):

<!-- snippet: recipes/dot_1.json -->
```json
  {"transform": "encapsulate", "params": {"accelerator": "dot", "units": ["mul", "sum"]}}
```

The other variants, one recipe each (D150, F1), after the import:

| Recipe | After the import |
|---|---|
| `dot_2`, `dot_3` | split_map mult_map W; bind mult → elementwise_mul; split_map sum_map W; bind sum → accumulate chisel_adder_tree; link tmp0, depth 1 or 0, group dot |
| `dot_4` | split_map mult_map W; split_reduce sum_map W; fuse_map mult_map sum_map; bind mult_map_s → mac; bind sum_final → accumulate chisel_adder_tree; link out_partial, depth 0, group dot |
| `dot_5`, `dot_6` | as dot_4, sum_final → chisel_serial; link out_partial, depth 0 or 1 |
| `matvec_1`–`matvec_6` | the same on matvec (`kernels/blas/matvec.py`), the row maps tagged temporal first (`tag_map mult_map`, and `sum_map` before its split), the j maps `mult_map_1` and `sum_map_1` split or `split_reduce`d, `fuse_map mult_map sum_map`, the MAC on `mult_map_1_s`; group `matvec`, the partial `q_partial`; `matvec_1` is dot_1's two groups, `encapsulate`d as `matvec` |

<!-- snippet: recipes/vecadd.json -->
```json
 "params": {"W": 4, "N": 64},
 "symbols": {"N": "N"},
 "steps": [
  {"transform": "split_map", "params": {"map": "add_map", "factor": "W"}},
```

## 13. Platform

The SNAX cluster around the accelerators (D84). A platform file lives in
`platforms/` (`small16.json`, without a source hjson; `snax_forge.json`, read
from snax-forge's base cluster) and never names a recipe. `Platform.load`;
every field is written, missing keys take defaults (`l2` missing: no L2),
unknown keys and wrong types are errors, all reported at once (section 14).

| Field | Holds |
|---|---|
| `name` | the platform (an identifier) |
| `base` | the platform this one was made from, or null for an original |
| `changes` | what differs from `base`: `--set` path -> value, in the order set |
| `hjson` | the source cluster file, a path from snax-forge's root, or null (small16): what the generator writes the cluster file from (section 16) |
| `accelerator_slot` | the generated accelerator's name, `snax_forge` from `design from-hjson` (D131), or null; `--set accelerator_slot=NAME` picks another |
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

`L2Config`:

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `size_bytes` | flat size | 1 MiB | positive multiple of the beat |
| `base_addr` | byte address of the first word | 0 | — |
| `beat_bits` | one DMA beat | 512 | multiple of `width_bits`; equal to the L1's `wide_bits` (`platform.l2`) |
| `width_bits` | as the L1, and must agree with it (`platform.l2`) | 64 | multiple of 8 |

Neither memory has an element type: each container's dtype decides how many
elements a word holds (D142, section 1). A platform that still sets `dtype`
or `elems_per_word` in `l1` or `l2` is refused (`platform.keys`).

**The streamer shell** (D12). One streamer per port of every accelerated
instance, `<instance>_<port>`, but a port on a link (D112): a link's ends
have none. The port decides `write`, `n_ports` (its
lanes) and, unless an entry gives them, `spatial_bounds` (fastest first: an
`affine` nest's spatial loops, a `stream` port's bound memlet ranges). The
graph decides `temporal_dims`: the firing loops the port's memlet runs over,
at least 1, the most over an instance's nodes (D84); the design point records
it per streamer. `default` holds `fifo_depth` (>= 1); an entry named after a
streamer overrides it and `spatial_bounds` for it alone, and may give
`num_ch` (>= 1, never in `default`): the TCDM channels the streamer uses,
without it the wire default W / num_slc (D142, `connect.num_ch`). A platform that sets
`temporal_dims` is refused (`platform.keys`), as is `--set` of it.

<!-- snippet: platforms/small16.json -->
```json
 "wait_mode": "poll",
 "streamers": {"default": {"fifo_depth": 2}}
```

**Working copies.** `pixi run design GRAPH --platform P --set
platform.PATH=VALUE` sets a field by its path (`l1.n_banks`,
`streamers.acc_out.fifo_depth`; the value JSON, else text) and writes
`out/design/<name>/platform.json`, `base` the platform it started from and
`changes` every path set since; `pixi run design save SRC NAME` keeps it as
`platforms/NAME.json`.

**From a cluster file** (D131, `design/hjson_platform.py`). `pixi run design
from-hjson HJSON NAME` writes an original `platforms/NAME.json`: the L1 and
L2 from the hjson (the reader's docstring lists the keys), `streamers.default`
(its `fifo_depth`) from the one accelerator's streamer template, whose movers
must agree, the rest small16's, `hjson` (under snax-forge's root, `hjson.path`) and
`accelerator_slot` `snax_forge`. `snax_forge.json` is read this way from
`platforms/snax_forge_cluster.hjson`:

<!-- snippet: platforms/snax_forge.json -->
```json
 "hjson": "platforms/snax_forge_cluster.hjson",
 "accelerator_slot": "snax_forge",
```

## 14. Design checks

Before the design step writes anything it runs every registered check
(`register_check(code, stage, fn)`) on the pairing of a bound graph and a
platform and reports every problem, each as `[code] where: why` with a
`fix:` line (the `--set` or recipe change that removes it) where one exists
(D85). `pixi run design check GRAPH --platform P [--set ...]` runs them only.
Stages: `platform` and `graph` independently, `connect` when both passed,
`memory` when `connect` passed.

| Code | Fails when |
|---|---|
| `platform.keys` | a platform file that does not parse; an unknown or missing key, a value of the wrong type; `dtype` or `elems_per_word` in `l1` or `l2`, no longer a platform field (the fix removes it: each container's dtype decides the packing, D142); a `--set` path that does not exist, is written by the tools (`name`, `base`, `changes`) or is decided by the port (`write`, `n_ports`) |
| `platform.values` | a value its section rejects (`n_banks` 0, a `wait_mode` that is not allowed) |
| `platform.streamer` | a streamer value below 1, spatial bounds that are not a non-empty list of ints >= 1, `num_ch` that is not an int >= 1 |
| `platform.l2` | L2 and L1 disagree on beat width (`beat_bits` vs `wide_bits`) or word width |
| `platform.banks` | with an L2, `n_banks` is not a multiple of the DMA's port group (`wide_bits / width_bits`) |
| `graph.load` | a graph file that does not load (section 11) |
| `graph.symbols` | a symbol still null: bind it in the recipe |
| `graph.unbound` | a tasklet or a map holding nothing to run outside every accelerated node, or no accelerated node: the program drives accelerators only |
| `graph.brm` | an unknown BRM or implementation, design params that do not resolve, connectors that are not the BRM's ports, code that is not the BRM's |
| `graph.instance` | an instance named `dma` or after a streamer, or two streamers with one name |
| `connect.streamer_key` | a streamer entry that matches no port of the graph (the fix names the closest streamer) |
| `connect.derived` | a streamer entry that sets `write` or `n_ports` |
| `connect.lanes` | an entry's `spatial_bounds` that do not give the port's lanes or are not its nest's spatial loops |
| `connect.dtype` | a node whose containers break its BRM's dtype rule (`brm/dtypes.py`, D143: each port's dtype in its `dtypes`, every output at least as wide as every input, no signed input with a `uint64` one; the fix names the output dtypes the inputs accept), or an instance whose nodes put two dtypes on one port |
| `connect.pack` | a port whose elements do not fill whole words (D142), with num_slc slices of its container's dtype per word: the fastest spatial bound not a multiple of num_slc (W < num_slc: the fix makes the container 64-bit or gives the port num_slc lanes), the fastest spatial loop not stepping 1 element (a column), another loop's step or the start not a multiple of num_slc (a row stride that is not: the fix pads the leading dimension). Row strides are over the container's padded shape (a tail's, D144: a 3 x 7 int32 matrix split by 4 has row stride 8). Every port passes at num_slc = 1 |
| `connect.pad` | a container an accelerator writes and a fold reads through a tail: the producer's pad lanes would reach the fold, since the hardware masks nothing (D144; the fix: a size that divides W, or a split factor that divides n); a container whose readers need different pad values (the message names them) |
| `connect.num_ch` | a streamer entry's `num_ch` other than the wire default W / num_slc: any other needs the FIFO slicer, which the flow does not place yet (D142). The message lists the values with rho = num_ch * num_slc / W <= 1, the divisors of W / num_slc; the fix removes it or sets W / num_slc |
| `connect.temporal` | a map around an accelerated node that is not `temporal` (tiles are not lowered yet), or a subset that is not affine in the firing loops |
| `memory.pin` | a memory working copy that does not load; a `--set memory.` path that is not `<container>.<memory>.base` or `passes.<pass>`; a pin on an unknown container (the fix names the closest), on a memory the container does not live in, or on L2 (the link places it, D86); an unknown pass |
| `memory.residency` | a layout in a memory the platform lacks or for a container the graph lacks; a non-transient container that lives nowhere; an accelerator operand without an L1 layout |
| `memory.layout` | a layout whose shape is not its container's |
| `memory.align` | an L1 base that is not a multiple of the word, or a stride that is not one of the element's (its dtype's itemsize, D142); a container the DMA moves whose L1 layout is not contiguous, not whole wide beats (open item 38) or not on a beat |
| `memory.fit` | an L1 layout that leaves L1 (the message names its bytes `[start, end)`; the fix the rows that would hold it, or the pin to L1's first byte), or an L2 layout larger than L2 |
| `memory.runtime` | an L1 layout (with its padding, D105) reaching into the top of L1 that snRuntime keeps at start-up: a 1 KiB stack per core of the platform's `hjson` plus `RUNTIME_MARGIN` (256 B, `design/check.py`); not checked without an `hjson` under snax-forge's root. The fix pins the layout just below |
| `memory.overlap` | two L1 layouts that share bytes, naming the pins (the fix moves the latest pin past everything) |
| `point.keys` | a design point with unknown or missing keys or a malformed streamer entry |
| `point.streamers` | a design point whose `streamers` are not the platform's shell for its graph |

## 15. Memory plan and design point

**Memory plan** (D86, `design/memory.py`). A layout per container and memory
it lives in: `base` (byte address of index 0), `shape` and one byte stride
per dimension (`lower/layout.py`). Banks follow from addresses (word `i` in
bank `i % n_banks`, open item 18). Three passes make it, each registered by
name (`register_memory_pass(kind, name, fn)`):

| Pass | Decides | Default |
|---|---|---|
| `residency` | the memories of each container | `default`: not transient: L2 and L1 with an L2, else L1; transient: L1, but a link's container and a fused nest's local value (`loop.local`): none (D112, D150) |
| `layout` | shape and strides, base 0 | `contiguous`: the container's shape, its strides row-major over the padded shape (a tail pads the split dimension to a multiple of W, D144), each element its dtype's itemsize (a word holds several narrow ones, D142); the memory checks and the DMA take the padded block |
| `placement` | the L1 base per container (an L2 layout keeps base 0: the program's link places its array, section 7's regions) | `contiguous`: graph container order from L1's first byte, no gaps; a layout in both L2 and L1 on a wide beat and padded to whole beats (D105), others on a word; around the pins |

`--set memory.<container>.l1.base=N` pins an L1 base; `--set
memory.passes.<pass>=NAME` picks a pass. Passes read a `MemoryContext` (the
bound graph, the platform, shapes, memory ranges, and each accelerator
port's element-index stream, `accesses`). The design step writes
`out/design/<name>/memory.json` (`passes`, `changes`, `layouts`), which
`--memory` continues from.

**Design point** (D87, `design/point.py`), written to
`out/design/<name>/design_point.json` once every check has passed, never by
hand:

| Field | Holds |
|---|---|
| `name` | the design (the graph's sandbox folder, else its stem; in a flow, the recipe's name and params, D136) |
| `graph_from` | the `.snaxdfg` it was made from: a sandbox step, or any bound graph given to the design step or the flow (D108) |
| `graph` | that graph as read: instances, `code`, `replaced`, `loop.split` |
| `platform` | the platform working copy (section 13) |
| `streamers` | the resolved shell: `<instance>_<port>` -> `instance`, `port`, `write`, `n_ports`, `spatial_bounds`, `temporal_dims`, `dtype` (the port's container's), `num_slc` (its slices per word), `num_ch` (the entry's, else W / num_slc; the hjson's `num_channel`), `fifo_depth` |
| `memory` | the memory plan: `passes`, `changes`, `layouts` (container -> memory -> layout) |

It holds no recipe. Loading one runs every check again on what it holds and
requires `streamers` to be the platform's shell for the graph.

## 16. Generated hjson files

What `pixi run gen-gvsoc DESIGN_POINT [--out DIR]` writes per design point
(D134, `gen/gvsoc.py`), default `out/gvsoc/<design>/`: what SNAX-GVSoC's
`snax_hjson` reads (`--param hjson=<dir>/<platform>.hjson`) and the program
snax-gvsoc's generic app (`sw/apps/forge/`) builds. For
`platforms/snax_forge.json` the three description files are `snax_forge.hjson`,
`snax_forge_streamer.hjson` and `accelerator.json` (D145). Each hjson and C
file starts with a `//` comment naming the design point, the rest of an hjson
file JSON; `accelerator.json` is plain JSON, its `design_point` key naming
the source. One design point gives byte-identical files wherever written.

| File | Holds |
|---|---|
| `<platform>.hjson` | the cluster: the platform's `hjson`, its comments dropped and its key order kept, with `cluster.tcdm.banks`, `.size` (KiB), `data_width`, `cluster_base_addr`, `dma_data_width` and `dram` written from the platform, `cluster.name` `snax_forge_cluster` (also in `cluster.bender_target`, in place of the source's name; a source without its own name there is `gen.source`), and the slot set; the source's template, referred to only by the slot, dropped |
| `<slot>_streamer.hjson` | the streamer template, the whole file: `data_reader_params` (the input ports) and `data_writer_params` (the output ports), unit by unit in the groups' order, each in the BRM's port order, per mover `spatial_bounds` in words, fastest first (`[S_0 / k, S_1, ...]`, k the elements of the port's container dtype a word holds, D142), `temporal_dim`, `num_channel` (the streamer's channels, the wire default W / k; the bounds' product) and `fifo_depth` |
| `accelerator.json` | the accelerator's one description, section 17's file |
| `include/` | `streamer_csr_addr_map.h` and `<name>_csr.h` (`name` accelerator.json's; every CSR by section 17's name, in CSR order: dot 1's `dot_csr.h` from `DOT_RW_MUL_START` to `DOT_RO_SUM_COUNTER`), and the runtime's `snitch_cluster_cfg.h` and `snitch_cluster_addrmap.h`, as snax_cluster's templates render them from the cluster file (`gen/headers.py`, D126) |
| `data.h` | the program's data (`gen/data.py`): per container the DMA moves, an input from the kernel's `make_inputs` (seed 0), or an output zeroed and its `<name>_golden` from the reference executor; each array the container's dtype, in its padded shape (a tail's pads hold the pad value: 0, or the identity of a max or min fold that reads it, D144), then padded with zeros to whole DMA beats in its itemsize (16 int32 or 64 int8 elements per 64-byte beat) as its L2 layout (D105) and aligned to the beat |
| `main.c` | the program (`lower/c.py`, D135): `main` reads `mcycle` first, then runs its core's function, one per core with each step on its component's core and each accelerator task's counter stored after its wait (`forge_counter[]`); no task marks. On the accelerator's core `main` then calls `forge_check()`, which reads `mcycle` first and prints `forge cycles: <n>` (the window from main's entry), then the self-check: the outputs against their goldens, the valid elements only, a padded matrix row by row (`forge data:`, its words the valid elements), each accelerator task's counter against its beats (`forge counter <task>:`), `forge errors:`; main returns the error count |
| `manifest.json` | `design`, `design_point`, `symbols`, `cluster` (the cluster file), `files`, `program` (`main.c`), `trace` (the unit's trace path, `snax_acc0/unit`; with several groups the accelerator's, `snax_acc0`), `csrs` (the CSR layout: `streamer`, its `first` and `last` CSR, and `accelerator`, each CSR's header name -> its number), `tasks` (per accelerator task, in order, its `beats` (a linked group's: its most firing unit's n) and its group's `csrs` values: from the task's values n and T (section 9), JOBS = n / T and, for a fold without `params.T`, JOB_BEATS = T, in a linked group named with its unit, `MAC_JOB_BEATS`; with several groups also a one-unit group's `trace`, `snax_acc0/unit_<unit>`, as snax-gvsoc's composite names it; a linked group's trace is G1's) |

**The slot** is the source's only one, renamed: `snax_acc_name` the
platform's `accelerator_slot`, its `bender_target` dropped (the RTL comes
from the bundle). `snax_tcdm_ports` is the sum of its movers' num_ch; the
source's `sparse_interconnect_config` becomes one group of that width (its
granularity kept), or `snax_narrow_tcdm_ports` that many and
`snax_wide_tcdm_ports` 0; `snax_num_rw_csr` and `snax_num_ro_csr` the
accelerator's rw and ro, counted from its CSR map (section 17); then, after
the source's other keys, `snax_disable_csr_manager` `true` (D130), `snax_forge_rtl`
`"rtl/files.f"` (relative to the cluster file, present only in a bundle;
SNAX-GVSoC ignores it), `snax_streamer_cfg` `{$ref:
"<slot>_streamer.hjson"}` and `snax_datapath_cfg` `{$ref:
"accelerator.json"}`, which snax_cluster only resolves. snax_cluster's
schema lists the keys the slot adds, none with a default (open item 88).

**The CSRs.** The streamer's from `0x3C0`, then the accelerator's in section
17's order (`csrs` in the manifest), ending at `0x5FD` at most
(`gen.csrs`): snax_cluster routes `0x5FE` and `0x5FF` to its task queues on
the cluster's CSR port, and SNAX-GVSoC's `SnaxUnitAcc` refuses the same.

**Checks.** The streamer file and `accelerator.json` have snax-forge's schemas
(`gen/schema/`, JSON Schema 2020-12); the cluster file validates against
snax_cluster's (snax-gvsoc's copy, `sw/docs/schema/snitch_cluster.schema.json`).
A platform value the cluster file cannot express (an L1 that is not whole
KiB) is `gen.platform`, a platform without a source `gen.source`, a source
with no slot or several `gen.slot`; what the description refuses is section
17's.

<!-- snippet: snax_forge/gen/schema/accelerator.schema.json -->
```json
 "required": ["name", "design_point", "tag", "inputs", "outputs", "groups"],
```

## 17. Accelerator RTL spec

`accelerator.json`, the one description of a design point's accelerator
(D145, `gen/accelerator.py`): written once by snax-forge, by `pixi run
gen-gvsoc` (section 16), so also at the bundle's root (section 18). It is
read by SNAX-GVSoC (`snax_cfg.py`) and by `Emit --spec` (`hw/chisel/`), after
snax-forge's check against the datapath catalogue (D130); snax_cluster only
resolves the slot's `snax_datapath_cfg` `$ref` to it. Plain JSON, concrete
values only, no `//` header.

`pixi run gen-rtl DESIGN_POINT [OUT_DIR]` (`gen/rtl.py`, default
`out/rtl/<design>/`) writes the same bytes, checks every unit against the
catalogue below (`gen.catalogue`, `gen.params`, `gen.port_map`,
`gen.timing`: the datapath, the params its fill rule gives, the ports whose
`when` holds on them, its latency and span) and every link of depth ≥ 1
against `LinkBuffer`, and runs
`Emit --spec accelerator.json --out DIR/`. The Chisel specs' fixtures
(`hw/chisel/src/test/resources/rtl/<design>/`) are this file, the streamer
hjson and `vectors.json` (`gen-rtl --fixture`).

| Key | Holds |
|---|---|
| `name` | the accelerator's name, the header's `<ACC>`: `encapsulate`'s `accelerator`, else its one group's name (a one-unit group is named after its instance); dot 2–6 are `dot`, matvec 2–6 `matvec` |
| `design_point` | the design point by its path from the root, as section 16's files name it |
| `tag` | the platform's `accelerator_slot`, `snax_forge` by default; the wrapper is `<tag>_shell_wrapper` |
| `inputs`, `outputs` | the outer ports, each `"<unit>.<port>"`, in the streamer's reader and writer order |
| `groups` | in CSR order, each `name`, `csrs` (its own), `units` and `links`, each link `{name, from, to, depth}`: `from` an `out` port as `"<unit>.<port>"`, `to` an `in` port; depth 0 is a wire, ≥ 1 a buffer of that many beats |

A unit:

| Key | Holds |
|---|---|
| `name` | the bind's `instance` |
| `brm` | its BRM |
| `datapath` | a catalogue datapath |
| `ports` | per BRM port, in the BRM's order: `port` (the datapath's), `dir` (`in` or `out`), `dtype` (as NumPy names it), `lanes` |
| `opType` | an elementwise or `Mac` unit only: NumPy's result type of its inputs (`brm/dtypes.py`'s `result_type`, D143), the type its op runs at |
| `params` | only what the ports do not give: `op` (a TreeAccumulate's `add`, the only op it runs), a `Mac`'s `accumulate` and `regOut`, and `T`, a rate fixed at design time, so the unit has no JOB_BEATS (a rate that depends on a kernel symbol stays a JOB_BEATS R/W CSR) |
| `timing` | `latency`, `ii`, `span`: the cycles a unit reads its input in place, taking it in the last, so its throughput is one beat per max(ii, span); `ii` stays for units with their own state (a divider, a CORDIC) |
| `csrs` | its R/W CSRs |

A CSR: `name`, `access` (`rw` or `ro`), `offset` from the accelerator's
first CSR, `width` (R/W only), `pin` (a scalar's datapath pin). snax-forge computes the
offsets once and stores them; a reader checks them (contiguous from 0 in CSR
order: per group START, JOBS, the units' R/W, STATUS, COUNTER, not the file's
nesting) and never recomputes them.

**Refused**, each a `GenError`, before anything is written: other than one
accelerator (`gen.instances`: several groups are one accelerator only
through `encapsulate`, D148), a null binding (`gen.binding`), a datapath not registered
(`gen.datapath`), a controller other than `counted` (`gen.controller`), a
drain other than 0 (`gen.drain`), a latency other than the catalogue's (its
expression evaluated on the unit's params), an ii other than 1 or a span other
than the catalogue's (`gen.timing`), and what the datapath refuses: its args
(`gen.args`), op (`gen.op`), ports (`gen.ports`), rates (`gen.rate`), lanes,
a port not whole 64-bit channels among them (`gen.lanes`) and a dtype that is
not intN or uintN (`gen.dtype`). The links and groups (F1 implements these): a
link not from an `out` to an `in` of the same dtype and lanes (`gen.link`), a
port on more than one link (`gen.link_port`), `inputs` or `outputs` not exactly
the unlinked ports (`gen.outer`), and a group not connected (`gen.group`).

**Derived, never stored**: each port's num_slc = 64 / b and num_ch = W · b /
64 (D142), each link's kind and what the control unit needs, every unit's
firings. Nothing of the old files' `unit`, `control`, `config`, `fields`, `wait`, `in_widths`, `in_signed`,
`out_widths`, `out_signed`, `supportedOps` or `tie` remains.

**Types** (D142, D143). A unit's types are its ports' dtypes, fixed at
elaboration: `Emit --spec` builds the datapath for that one choice. Each
input is extended to `opType` by its own signedness (a fold's lanes to its
output's type before its tree), and the result is extended or wrapped into
the output's type; a `Mac` forms the product at `opType`, casts it to
`outType` and folds at `outType`. Python decides every type; the datapaths
only construct.

**Groups and firings** (D146). A group is a connected set of units and
links, an unlinked unit a group of its own, and has one JOBS: a unit u fires
n_u = JOBS · ∏ T_v, over u and every fold v downstream of it (synchronous
dataflow's balance equations: JOBS the scale, the folds' T the rates), every
T counted in beats: the one-lane vsum (`SerialAccumulate`) has T = 1 and
span W.

**The CSRs** (D146), from offset 0, per group in order: START (R/W), JOBS
(R/W), each unit's R/W CSRs (JOB_BEATS for a fold without `params.T`, then
its scalars, a 64-bit one as two CSRs, low word first), STATUS (R/O), COUNTER
(R/O). With f_u = 1 for a fold without `params.T` (else 0) and k_u a unit's
scalar CSRs, rw = Σ_g (2 + Σ_u (f_u + k_u)) and ro = 2G. Per group:

- A write to an R/W CSR other than START, or to START with bit 0 clear, is
  taken at once. START with bit 0 set is accepted when the group is idle,
  else held until E+1 (E the cycle its run ends).
- STATUS: bit 0 busy, bit 1 done (sticky: set by the done pulse, it reads 1
  from the next cycle and is cleared by an accepted start; a set wins over a
  clear in the same cycle), bit 2 kept free. The program polls bit 0.
- COUNTER: the busy cycles after the accepted start, held after the run.
- An empty run (JOBS = 0 or any T = 0) ends in the cycle a it starts, E = a:
  busy never rises, done pulses in a, STATUS reads done from a+1, COUNTER
  is 0. Every accepted start ends in a done; there is no ERROR CSR.
- Pins: `<group>_busy_o` (STATUS bit 0) and `<group>_done_o` (the done pulse
  registered: high in E+1, a+1 for an empty run). snax_cluster leaves them
  unconnected (open item 92).
- No per-unit CSRs: per-unit timing comes from SNAX-GVSoC's events (open
  item 96).

The header names a CSR `<ACC>_<RW|RO>_[<GROUP>_][<UNIT>_]<NAME>` (one
helper, `Accelerator.define`, for the header and the program), `<ACC>` the
file's `name` (D148), the group part dropped when the accelerator has one
group and the unit part (on a unit's own CSRs only) when the group has one
unit: vecadd's
`ACC_RW_START`, `ACC_RW_JOBS`, `ACC_RO_STATUS`, `ACC_RO_COUNTER`; vsum adds
`ACC_RW_JOB_BEATS`; dot 2–6's are `DOT_RW_START`, `DOT_RW_JOBS`,
`DOT_RW_MAC_JOB_BEATS` (`DOT_RW_SUM_JOB_BEATS` for dot 2 and 3),
`DOT_RO_STATUS`, `DOT_RO_COUNTER`; matvec's group and accelerator are
`matvec` (`MATVEC_RW_JOBS`, `MATVEC_RW_MAC_JOB_BEATS`,
`MATVEC_RW_SUM_JOB_BEATS`).

| | Offsets | rw, ro | Task values |
|---|---|---|---|
| vecadd | START 0, JOBS 1, STATUS 2, COUNTER 3 | 2, 2 | JOBS 16 |
| vsum_tree_i32 | START 0, JOBS 1, JOB_BEATS 2, STATUS 3, COUNTER 4 | 3, 2 | JOBS 1, JOB_BEATS 16 |
| dot 1 (groups mul, sum) | MUL: START 0, JOBS 1, STATUS 2, COUNTER 3; SUM: START 4, JOBS 5, JOB_BEATS 6, STATUS 7, COUNTER 8 | 5, 4 | task 1: MUL_JOBS 16; task 2: SUM_JOBS 1, SUM_JOB_BEATS 16 |
| dot 2–6 (one group `dot`) | START 0, JOBS 1, JOB_BEATS 2 (unit `sum` for dot 2 and 3, `mac` for dot 4–6), STATUS 3, COUNTER 4 | 3, 2 | JOBS 1, JOB_BEATS T = N / W (128 at N = 512, W = 4) |
| matvec 2–6 (one group `matvec`) | as dot 2–6 | 3, 2 | JOBS 16, JOB_BEATS T = N / W (128 at N = 512, W = 4) |

dot 1's file, snax-gvsoc's fixture (`forge/dot_1/`): the groups `mul` and
`sum` in `encapsulate`'s order, each a unit of its own:

<!-- snippet: gvsoc/snax-gvsoc/forge/dot_1/accelerator.json -->
```json
{
 "name": "dot",
 "design_point": "out/design/dot_1/design_point.json",
 "tag": "snax_forge",
 "inputs": ["mul.a", "mul.b", "sum.a"],
 "outputs": ["mul.out", "sum.out"],
 "groups": [
  {
   "name": "mul",
   "csrs": [
    {"name": "START", "access": "rw", "offset": 0, "width": 32},
    {"name": "JOBS", "access": "rw", "offset": 1, "width": 32},
    {"name": "STATUS", "access": "ro", "offset": 2},
    {"name": "COUNTER", "access": "ro", "offset": 3}
   ],
   "units": [
    {
     "name": "mul",
     "brm": "elementwise_mul",
     "datapath": "ElementwiseLanes",
     "ports": {
      "a": {"port": "a", "dtype": "int64", "lanes": 4},
      "b": {"port": "b", "dtype": "int64", "lanes": 4},
      "out": {"port": "out", "dtype": "int64", "lanes": 4}
     },
     "opType": "int64",
     "params": {"op": "mul"},
     "timing": {"latency": 0, "ii": 1},
     "csrs": []
    }
   ],
   "links": []
  },
  {
   "name": "sum",
   "csrs": [
    {"name": "START", "access": "rw", "offset": 4, "width": 32},
    {"name": "JOBS", "access": "rw", "offset": 5, "width": 32},
    {"name": "STATUS", "access": "ro", "offset": 7},
    {"name": "COUNTER", "access": "ro", "offset": 8}
   ],
   "units": [
    {
     "name": "sum",
     "brm": "accumulate",
     "datapath": "TreeAccumulate",
     "ports": {
      "a": {"port": "in", "dtype": "int64", "lanes": 4},
      "out": {"port": "out", "dtype": "int64", "lanes": 1}
     },
     "params": {"op": "add"},
     "timing": {"latency": 1, "ii": 1},
     "csrs": [{"name": "JOB_BEATS", "access": "rw", "offset": 6, "width": 32}]
    }
   ],
   "links": []
  }
 ]
}
```

dot 4's file: one group `dot`, the MAC's lane partials wired into the tree.

(F2 makes this a snippet)
```json
{
 "name": "dot",
 "design_point": "out/design/dot_4/design_point.json",
 "tag": "snax_forge",
 "inputs": ["mac.a", "mac.b"],
 "outputs": ["sum.out"],
 "groups": [
  {
   "name": "dot",
   "csrs": [
    {"name": "START", "access": "rw", "offset": 0, "width": 32},
    {"name": "JOBS", "access": "rw", "offset": 1, "width": 32},
    {"name": "STATUS", "access": "ro", "offset": 3},
    {"name": "COUNTER", "access": "ro", "offset": 4}
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
```

**The datapath catalogue** (D147), `hw/chisel/blocks.json`: per datapath
its params, ports, latency (an int, or an expression over its params whose
syntax is F1's), `span`, pins and the rule that fills it from the unit's
ports; no `csr`, no `tie`. A port is `"in"` or `"out"`, or `{"dir": ...,
"when": {param: value, ...}}`; a pin is a name, or `{"name": ..., "when":
{...}}`. `when` is a match: the port or pin exists only when every listed
param has that value, decided at elaboration; one without `when` is
unconditional. A datapath is data and enables only: `ElementwiseLanes` (L
0) and `TreeAccumulate` (L 1, pins `fire` and `first`); F2 adds:

- `Mac(inTypes, opType, outType, lanes, accumulate, regOut)`: with
  `accumulate`, per lane s ← first ? a·b : s + a·b, L 1, pins `fire` and
  `first`; otherwise a·b + c, `c`'s dtype and lanes those of `out`, L 0, or
  1 with `regOut` and its pin `en` (`regIn` is not built);
- `SerialAccumulate(inType, outType, lanes)`: acc ← (first ∧ k = 0) ? x_k :
  acc + x_k, lane k in cycle k, span = lanes, L 1, pins `read`, `lane` and
  `first`;
- `LinkBuffer(depth, lanes, dtype)`, depth ≥ 1: D beats, L 1; `load` and the
  ring's write and read slots (`wslot`, `rslot`, 0 bits wide at D = 1) come
  from the control unit. A link of depth ≥ 1 is checked against it (its ends'
  lanes and dtype, `link.lanes` and `link.dtype` in its fill rule).

A wire is no datapath: plain connections with the same dtype and lanes at
both ends. The catalogue's entry for `TreeAccumulate`:

<!-- snippet: hw/chisel/blocks.json -->
```json
 "TreeAccumulate": {
  "params": {
   "inType": {
    "type": "dtype",
    "values": ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
   },
   "outType": {
    "type": "dtype",
    "values": ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
   },
   "lanes": {"type": "int", "min": 1},
   "op": {"type": "op", "values": ["add"]}
  },
  "ports": {"in": "in", "out": "out"},
  "latency": 1,
  "span": 1,
  "pins": ["fire", "first"],
  "fill": {
   "inType": "inputs.dtype",
   "outType": "outputs.dtype",
   "lanes": "inputs.lanes",
   "op": "unit.params.op"
  }
 },
```

and `Mac`'s, its latency an `expr.py` expression over its params (a bool
as 0 or 1):

<!-- snippet: hw/chisel/blocks.json -->
```json
 "Mac": {
  "params": {
   "inTypes": {
    "type": "dtypes",
    "values": ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
   },
   "opType": {
    "type": "dtype",
    "values": ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
   },
   "outType": {
    "type": "dtype",
    "values": ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
   },
   "lanes": {"type": "int", "min": 1},
   "accumulate": {"type": "bool"},
   "regOut": {"type": "bool"}
  },
  "ports": {"a": "in", "b": "in", "c": {"dir": "in", "when": {"accumulate": false}}, "out": "out"},
  "latency": "max(accumulate, regOut)",
  "span": 1,
  "pins": [
   {"name": "fire", "when": {"accumulate": true}},
   {"name": "first", "when": {"accumulate": true}},
   {"name": "en", "when": {"accumulate": false, "regOut": true}}
  ],
  "fill": {
   "inTypes": "inputs.dtype",
   "opType": "unit.opType",
   "outType": "outputs.dtype",
   "lanes": "inputs.lanes",
   "accumulate": "unit.params.accumulate",
   "regOut": "unit.params.regOut"
  }
 },
```

**Control** (D147). One control unit per group, `<tag>_<group>_ctrl`,
generated from the description (F2), never written per variant:

- once per group: JOBS, the job counters (done is the last unit's JOBS-th
  result), busy, STATUS, COUNTER;
- per unit: its beat digit d_k (mod T_k), its latency pipeline, and a lane
  counter when span > 1;
- per link of depth D ≥ 1: an occupancy count (a full bit at D = 1);
- the outer ports' join, one rule per port: ready_k = armed ∧ ⋀_{j≠k}
  valid_j;
- between units, unit k gives into a link when it has room or the next unit
  takes in that cycle. At depth 0 with span W, the MAC holds its result until
  the reader's last lane.

A wire chain is B1's nest with L = Σ L_k and T_job = ∏ T_k, plus one enable
per stage:

    fire_{k+1}(c) = fire_k(c − L_k) ∧ gives_k(c − L_k),  first_k = [d_k = 0],  gives_k = [d_k = T_k − 1]

A one-unit group is B1's nest controller (`NestCtrl`) unchanged, so every
existing fixture's counter stays: SnaxUnit's rule in Chisel, a Decoupled
config (JOBS, JOB_BEATS; ready when idle), the join of the input valids, an
L-deep valid register that stalls as a whole (it advances when its last
stage is empty or `out.ready`), the II gap, busy, and a done pulse at the
run's last push (E). To the datapath it gives `fire`, `first` (the job's
first beat) and the register's advance. At L 0 `out.valid` is a firing
that gives an output, without `out.ready` (valid never depends on ready),
and that firing happens when `out.ready` is high too. Lockstep is L = 0,
II = 1. Every group's control unit: START is taken when the group is
idle and held (`csr_req_ready_o` low) until E + 1 otherwise; an empty run
(JOBS 0, or JOB_BEATS 0) ends in its start's cycle without starting the
nest; STATUS is bit 0 busy, bit 1 done (sticky: set by the done pulse,
cleared by a taken start, set wins); COUNTER counts the busy cycles from the
taken start to E. The pins are registered: `<group>_busy_o`, and
`<group>_done_o` high in E + 1 (an empty run's start cycle + 1). The
accelerator is three modules, `<tag>_csr` (the CSRs above by offset,
ReqRspManager's protocol, D130: a START write with bit 0 set is that
group's config, JOBS and JOB_BEATS as written, 1 for JOB_BEATS without
one), `<tag>_ctrl` (the groups' control units) and `<tag>_datapath` (one
submodule per unit and per buffer, no valid, ready or counter inside); Emit
lists every file it wrote in `emitted.json`, beside `sv/`.

**Cycles.** The exact cycles of dot's variants, the accelerator alone on
ideal streams (latencies vmul 0, Mac 1, tree 1, SerialAccumulate 1, buffer
1), J jobs of T beats at W lanes:

    C2 = JT + 2, C3 = JT + 1, C4 = JT + 2, C5 = J(T + W − 1) + 2, C6 = (J − 1)·max(T, W) + T + W + 2

| Point | T | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| dot, N = 512, W = 4 | 128 | 130 | 129 | 130 | 133 | 134 |
| dot, N = 512, W = 8 | 64 | 66 | 65 | 66 | 73 | 74 |
| row, 16 × 512, W = 4 | 128 | 2050 | 2049 | 2050 | 2098 | 2054 |
| row, 16 × 512, W = 8 | 64 | 1026 | 1025 | 1026 | 1138 | 1034 |

dot 5's MAC stalls (J − 1)(W − 1) cycles; a depth-1 buffer acts as a
pipeline register, so dot 2 is dot 3 plus a cycle; dot 6 needs T ≥ W to run
without a stall.

**The wrapper**, `<tag>_shell_wrapper`, has the ports
`snax_acc_wrapper.sv.tpl` connects with `snax_disable_csr_manager`: `clk_i`,
`rst_ni` (active low), `csr_req_addr_i` and `csr_req_data_i` (32 bits),
`csr_req_write_i`, `csr_req_valid_i`, `csr_req_ready_o`, `csr_rsp_data_o`
(32 bits), `csr_rsp_valid_o`, `csr_rsp_ready_i`; per input port k, in order,
`stream2acc_<k>_data_i`, `_valid_i`, `_ready_o`, per output port
`acc2stream_<k>_data_o`, `_valid_o`, `_ready_i`, data `num_channel × 64`
bits with lane i at bits `[w(i+1)-1 : wi]` for the port's w-bit type
(num_slc = 64 / w lanes a word, D142); per group `<group>_busy_o` and
`<group>_done_o`. Inside, the three modules: each outer port's data goes to
`<tag>_datapath`, its valid and ready to `<tag>_ctrl`.

**One rule on both sides.** A group's control model is snax-gvsoc's
`SnaxShellRule` (`snax/snax_shell.hpp`) today, which `SnaxShell` runs; G1
generalizes it into one control model per group, with the datapath classes
in `snax/snax_datapath.hpp`. Its cases
(`tests/snax/shell/cases/`) are checked against the rule by snax-gvsoc's
`tb.cpp` and driven through vecadd's and vsum's wrappers by `ShellCasesSpec`
(`SNAX_GVSOC_SHELL_CASES`). The nest controller's rule is `snax/snax_unit.hpp`
with `tests/snax/unit/cases/`, which the specs drive the controller and its
datapaths through (`SNAX_GVSOC_UNIT_CASES`); CI's chisel job sets both to the
submodule's folders. Where a case shows the two sides disagree, the Chisel behaviour
stands unless it breaks this section or D130, and SNAX-GVSoC follows it.

## 18. Bundle

What `pixi run bundle DESIGN_POINT [--out DIR]` writes per design point
(D111, D130, `gen/bundle.py`), default `out/bundle/<design>/`: everything
snax_cluster needs to build the design's cluster, made afresh by the
snax-forge checked out. `rtl/` is removed before each run; nothing else in
`DIR` is.

| File | Holds |
|---|---|
| the root | section 16's files (gen-gvsoc's), the cluster file and `accelerator.json` (section 17) among them |
| `rtl/sv/`, `rtl/emitted.json` | what `Emit --spec` wrote (section 17); firtool also writes `sv/filelist.f`, which `emitted.json` does not list |
| `rtl/files.f` | the RTL's file list: one path per line, relative to `rtl/`, in `emitted.json`'s `files` order, `emitted.json` itself left out |
| `bundle.json` | `design_point` (as section 16 names it), `tag` (section 17), `snax_forge_commit` (`git rev-parse HEAD`, or null outside git), `dirty` (true when the work tree has uncommitted changes, untracked files included; false outside git), `top` (`<tag>_shell_wrapper`) and `modules`, each of the wrapper's modules with the submodules it holds, from `accelerator.json`, not from Emit's output, by the names F2's Emit gives them: `<tag>_csr` (none), `<tag>_ctrl` (`<tag>_<group>_ctrl` per group), `<tag>_datapath` (`<tag>_<unit>` per unit, then `<tag>_<link>` per buffer, a link of depth ≥ 1, group by group); vecadd's `{"snax_forge_csr": [], "snax_forge_ctrl": ["snax_forge_acc_ctrl"], "snax_forge_datapath": ["snax_forge_acc"]}` |

**Refused**: what sections 16 and 17 refuse (`gen.elaborate` among them),
then `bundle.empty` (`emitted.json` lists no file).

**How snax_cluster takes it** (D130), through the cluster file alone
(`CFG_OVERRIDE`):

- clustergen and snaxgen resolve a `$ref` to another file against the
  cluster file's real folder;
- in RTL mode snaxgen writes `target/snitch_cluster/generated/snax_forge.f`
  from the slot's `snax_forge_rtl`: the paths it lists, then its own
  `<tag>_Streamer.sv`, `<tag>_streamer_wrapper.sv` and `<tag>_wrapper.sv`,
  all absolute; vsim and Verilator read it when it is non-empty;
- the streamer header lands in
  `target/snitch_cluster/sw/snax/<tag>/include/streamer_csr_addr_map.h`;
  `snax-forge-sw` fails when its `#define` lines and the bundle's
  `include/` copy differ;
- `target/snitch_cluster/snax_forge.mk` runs a program on the RTL:
  `CFG_OVERRIDE` the bundle's cluster file, `FORGE` the program's folder,
  `ELF` an ELF to run (default the one built from `FORGE`), `SIM` `vlt`
  (default) or `vsim`; `snax-forge-sim` builds the simulator,
  `snax-forge-run` runs the ELF into `logs/snax-forge-$(SIM).txt` and fails
  on a nonzero exit code;
- snax_cluster keeps the vecadd and vsum bundles as fixtures
  (`target/snitch_cluster/cfg/snax_forge/<design>/`, no `manifest.json`),
  which its `snax-forge-ci.yml` builds with Verilator and runs on every PR,
  checking each README's `forge counter` line.
