# SNAX-FORGE Contracts

> What the artefacts between the components mean, field by field, so each
> component can produce its output without reading another's code. Why each
> is the way it is: the module docstrings and `docs/DECISIONS.md`.
>
> Sections 1, 7 and 9–18 are the flow: conventions, a run's profile and
> trace (D137), the task list (D64), the BRM (D68, D70), the SNAX-DFG (D77),
> the recipe (D72), the platform (D84), the design checks (D85), the memory
> plan and design point (D86, D87), the generated hjson files (D134), the
> accelerator RTL's spec (D138) and the bundle for snax_cluster (D139).
> Sections 2–6 and 8 were SNAX-MODEL's and were retired with it in RET1
> (D116); their numbers are kept, their text is in git history (commit
> `cad6619`, before RET1).
>
> These are dataclasses and plain JSON until the M6 freeze (D26), which brings
> versioned schemas for them; the generated hjson files already have schemas
> (section 16).
>
> Every fenced block below is copied verbatim out of a checked-in file, and
> `tests/test_contracts.py` fails if one of them drifts. The
> `<!-- snippet: ... -->` line above each block names its source.

---

## 1. Conventions

**Byte addresses.** Every memory address (L1 and L2 addresses, streamer
`base`, all strides, DMA bases) is a byte address.

**The bank word is the unit of transfer** (D13). A word is the L1's
`width_bits` wide and holds `elems_per_word` elements of `dtype`, one in v1:
nothing has moved packed words yet, and the design check `connect.dtype`
refuses them (open item 21).

**Integers first** (D28). The default `dtype` is `int64`. A float
reduction's result depends on accumulation order, which the reference
executor matches only once a BRM defines that order.

The L1 of `platforms/snax_forge.json` (section 13):

<!-- snippet: platforms/snax_forge.json -->
```json
 "l1": {
  "n_banks": 32,
  "width_bits": 64,
  "rows": 512,
  "dtype": "int64",
  "elems_per_word": 1,
  "base_addr": 268435456,
  "wide_bits": 512
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

What the reports (D99) and the viewer read of a flow run (D136, D137). A
run directory (`out/flow/<name>/run/`, `run/outputs.py`) holds `run.json`
(what ran), `profile.json`, and at a trace level above `off`
`trace.jsonl` and `trace_meta.json`. SNAX-GVSoC writes its events in its
own grammar (snax-gvsoc `docs/EVENTS.md`) into `run/events.log`; the
converter (`run/convert.py`) renames them into what follows and never
recounts (D137).

**The window.** `total_cycles` is the program's: its `mcycle` read in
`forge_check()` minus its `mcycle` read in main, the `forge cycles:` line.
With events, the converter takes the same two reads from the accelerator's
core's `mcycle` lines (each picked by the function span it falls in),
checks that they are exactly `total_cycles` apart, and shifts every cycle
so that t = 0 is the read in main; the shift (the boot's length) is in
`trace_meta.json`, and events outside `[0, total_cycles)` are dropped. The
profile, a core's classes and the trace cover this one window.

**Profile** (`run/profile.py`): the totals of one run. The fields:

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

Per part: an accelerator gives its classes summed over its runs (`busy` a
firing or the II gap after one, `stall_in`, `stall_out`, `flush`), `idle`
the rest of the window, its `firings` and `utilisation` (`busy` over the
window); a streamer, one entry per mover, gives `busy`, `stall_xbar`,
`stall_fifo` and `idle` likewise, its crossbar ports, and its FIFO's name,
depth and highest count per lane; a core gives its cycles in the window as
`barrier`, else `poll` (a run of reads of one CSR with one value, first to
last), else `sleep` (`wfi`), else `run`, its SNAX CSR writes and reads, and
each wait (`poll` or `barrier`) with the block, its first and last cycle;
the classes and waits need the events (`None` and empty at trace `off`); a
DMA gives its transfers, the cycles its busy signal is high (what
`DMA_BUSY` counts), its read and write beats and the bytes each way; a port,
per crossbar input, its grants, stalls and the stalls lost to the DMA's
superbank, a core's or AXI's over the whole run (`whole_run`); a bank its
reads, writes and conflicts (cycles with two requests or more on it), from
the streamers and the DMA only. Names, ports, `write` and the FIFO depths
come from the hjson files and the design point.

`functional_check` holds the `kernel`, `seed` and `symbols` of the run,
then what the program printed and returned: `exit_code` (the run's),
`data_errors` and `words` (the `forge data:` line, every output word against
its golden), `tasks` (per accelerator task its `counter`, the `forge
counter` line, and its `beats`, from the manifest), and `passed`, true when
the exit code is 0.

**Trace** (`run/trace.py`): one event per line of `trace.jsonl`. The
levels are cumulative, each the SNAX-GVSoC traces it turns on (snax-gvsoc
`docs/EVENTS.md`): `off` none (the profile only), `task` the `events` lines,
`beat` the `beats` lines, `stall` the `stalls` lines, `finegrained` the
cores' `insn` lines. Every event is a flat dict of `t`, `k`, `src` and the
kind's own fields:

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

`src` is a C§7 name: `acc`, a mover (`acc_a`), `dma`, `core0`, and for a
`grant` or `stall` the crossbar port (`acc_a.lane0`, `core0`, `axi`). A core's
`poll` folds the reads of one CSR that return one value; a single read is a
`csr`. `block` is the component whose CSR it is (`acc`, `acc_streamer`). A
DMA copy's `dir` (`l2_to_l1`, `l1_to_l2`) comes from its addresses and the
cluster's L1 range, and copies are matched to the task list's `load_<C>` and
`store_<C>` tasks in issue order.

**Beats** (FLOW3 round 2a). Lines are linked by ids, never by order: a
streamer beat is its mover and `b` (a reader's beats numbered in its AGU's
order from 0 per run, a writer's in the order the accelerator hands them
over; a reader with a repeat hands one beat out several times, so several
firings name one `b`); a firing's `n` is its index in the run, a push's its
index among the run's results, equal to the writer's `b`; a DMA beat on L1
is its copy's `id` and `i`, its index within the copy. `fire`'s `b`,
`value` and `vals` are dicts keyed by reader mover; `push`'s `first` and
`last` are the firings it folds (equal at rate 1). The converter renames
(crossbar input index to port, `reader0` to `acc_a`, `by` -1 to `dma`),
shifts every cycle and `back` by the boot, drops lines outside the window,
and adds what no reader should look up: `task` (the task whose start and
done bracket the line on its component, a copy's task for the DMA, null for
a core's or AXI's access and outside its tasks), `mem` (`l1`, `l2`) and the
absolute `addr`, `region` (the container the address is in, or null: in L1
by the design point's layouts, in L2 by the ELF's symbols, below), `elems`
(the flat indexes of its elements in the word, a burst's bytes) and `vals` (those elements' values by the region's dtype; a
fire's and a push's from their beats' grants, null for a mover not
traced); an `insn`'s `func` is the function its core is in. Every line keeps its raw words
too: `value` is the bank word (`dma`: one per bank it covers), `bank` and
`row` the word-interleaved map of `addr`. `back` is the cycle the response
arrives, known at the grant; there is no request or response event. The
beat and stall lines only feed the trace: the profile stays the summary
lines (D137).

**Classes** (FLOW3 round 3). An accelerator's or a mover's `done` carries
its run's `cycles`, the classes of its summary line by name (`busy`,
`stall_in`, `stall_out`, `flush`; a mover's `busy`, `stall_xbar`,
`stall_fifo`, `idle`), from `task` on. From `beat` on, a `class` event
gives the first cycle of each run of one class, as SNAX-GVSoC classed the
cycle for those counts (never derived here from the other lines): per
component run they cover the cycles its counts add up over (a mover's past
its `done`, up to the streamer's end), with no gap and no overlap, and the
lengths of each class's runs add up to its count.

**Sources.** `--trace LEVEL[:SOURCES]` names the C§7 sources whose beat,
stall and instruction lines are written (`acc`, the movers, `dma`,
`core<i>`, `axi`, joined by `,`); by default at `beat` and `stall` the
accelerator, its movers and the DMA, at `finegrained` also every core.
`summary` and `events` lines are always whole. `trace_meta.json` holds the
`level`, `total_cycles`, the `boot` shift, the `sources` in the cluster's
order, the `filter` (the sources the beat lines were written for) and
`start`: per core in the cluster's order, `{"func": ..., "mode": ...}`, the
name of its last `func` line at or before the window's first cycle (null
without one) and its state there (`run`/`sleep`, the one its classes start
from). No `func` line is written at t = 0, whose t would not be the
function's first instruction.
`--vcd` writes a VCD and `view.gtkw` beside the run for gtkwave, never read
here. `run.json` records the `trace_level`, the `trace_sources` and `vcd`.

**Regions** (`run/regions.py`, `run/elf.py`, D86). Both memories are known
before the run: a container's L1 region is the design point's layout; its L2
region has the plan's shape and dtype at its array's address in the linked
program. The flow reads the ELF's symbol table once the program is built
(`sw`) into `run.json`'s `elf_symbols`, container -> `addr` and `size` (its
bytes padded to whole DMA beats, as `data.h` declares it; a container in L2
with no symbol or another size stops the flow). The converter, the reports
and the viewer take regions from `regions_of(point, elf_symbols)` only. From
`task` on, the converter checks that each container's L2 bytes its copies
touch are its array's, padded to whole beats, and that no copy leaves it.

The invariants every conversion checks are `run/convert.py`'s.

## 8. Rules for a new block kind

Retired in RET1 (SNAX-MODEL).

## 9. Task list

SNAX-LOWER's ordered list of tasks (D64): which component runs which task
with which values, where each task is configured and started, and what it
waits for. It is the hand-off to the backend: `main_c` (`lower/c.py`, D135)
turns it into C on snRuntime, the program SNAX-GVSoC runs (section 16). It
is derived from a design point (D89, `snax_forge/lower/derive.py`, `pixi run
lower tasks DESIGN_POINT`; the rules below) or written by hand.

A task list is `name` and `steps`, each step with an `op`:

| `op` | Fields | Becomes |
|---|---|---|
| `configure` | `task_name`, `type`, `component`, `after`, `wait_mode`, `values` | the component's configuration, written here |
| `start` | `tasks` | the waits these tasks need, then their `start` writes in list order |
| `sync` | `task`, `mode` | one wait on the task's component, always placed |
| `read` | `reg` | a read of `block.register`; the C backend has no form for it yet (`c.read`) |

`type` is the component's type and says how `values` are read:

```
streamer  base, temporal_bounds, temporal_strides, spatial_strides (bytes);
          spatial bounds are design-time and come from the design point's
          streamers
dma       direction ("l2_to_l1" or "l1_to_l2"); src and dst, each with base,
          bounds and strides (bytes, one wide beat per step)
accel     its start parameters by name: n, then any named rate (e.g. T)
```

**Names** (D75). A streamer is named after the accelerator port it
serves, `<instance>_<port>` (`acc_a`, `acc_out`). A task is
`<node>_<component>` (`add_acc_a`, `add_acc`), a DMA task
`load_<container>` or `store_<container>` (`load_A`, `store_C`), with node
and container names as the imported graph has them, and `_<k>` appended for
tile `k` (`add_acc_a_2`).

Only the loops a task uses are given; the backend pads the rest with bound
1 and stride 0. `after` names earlier tasks this one needs finished before
it starts: data it reads, and a buffer it overwrites. `wait_mode` (`poll`
or `signal`) is used whenever the lowering waits for this task; a `sync`
gives its own `mode`. Missing `after` and `wait_mode` default to `[]` and
`poll`; every field is written back, and unknown keys are errors.

**Structural rules**, checked when a task list is made: task names are
unique; `after`, `start` and `sync` refer to tasks configured earlier; a
component has at most one configured task that has not started, because its
registers are buffered; a task starts once, after its `after` tasks have
started; every configured task is started.

**Values** are checked against the design point before the backend runs
(`check_values`, `lower/tasks.py`; `c.tasks`): the component exists and has
the task's type, the values have exactly the type's keys, a streamer's loops
fit it (no more temporal loops than its `temporal_dims`, one spatial stride
per spatial bound), a DMA's direction is `l2_to_l1` or `l1_to_l2`, and every
bound is at least 1.

**Waits** (`placed_waits`, `lower/waits.py`). Before a `start`, the lowering
places one wait per component, for every `after` task and every listed
component whose latest task is not yet covered, leaves out a wait that
another of these waits covers (D64), and orders the rest by when the task
each one ends on was started. A wait is on a component, so it ends on that
component's latest task and covers every earlier one. A wait on a writer
streamer also covers the accelerator attached to it and that accelerator's reader streamers when
the same start launched them: their data flows into the writer, so it
finishes last. Nothing else is added; where configures, starts and syncs
go, and so how programming overlaps running blocks, is the task list's.

**Derived from a design point** (D89). The accelerated nodes run in
execution order, one group each; every map around a node is one of its
temporal firing loops (a single tile). A group is: `load_<C>` (configure,
start) for each input, in BRM port order, whose container has an L2 and an
L1 layout and is not in L1 yet; `<node>_<instance>_<port>` for every port,
the streamer's values from the memlet through the L1 layout (a port with a
named rate over the firing loops its memlet uses only, one beat per `T`
firings); `<node>_<instance>` with `n` = the firing count and each named rate
from `named_rates` (D104, D105: dot's `sum` gets `{"n": 16, "T": 16}`); one
`start` of the streamer tasks and then the
accelerator task; `store_<C>` after the writer's task for each output
written for the last time. At the end one `sync` per non-transient
container the graph writes, on its store (or its writer without an L2).
`after` holds data dependences only: a reader waits for the task that last
put its container in L1, a writer for the readers and writer since. So a
node that reads what an earlier node wrote waits for that node's writer, and
the lowering turns that into the one wait between the two accelerators
(dot's `sum_sum_a` after `mult_mul_out`: chaining through L1, D105). Every
configure and sync uses the platform's `wait_mode`. Streamer values, with
firing loops `v_k` (begin `b_k`, `count_k`, step `s_k`) and a subset
dimension `d` whose begin is `c_d + sum_k a_dk v_k`: `base` = layout base +
sum_d (c_d + sum_k a_dk b_k) * stride_d; temporal loop k (innermost first):
bound `count_k`, stride sum_d a_dk s_k stride_d; one spatial loop per range
dimension (the last first): its length and step * stride_d. The spatial
bounds must be the streamer's and the addresses must be those the BRM's
nest gives through the same layout (`lower/streams.py`). vecadd's design
point on `platforms/snax_forge.json` gives `tests/run/fixtures/vecadd/tasks.json`
exactly.

The end of `vecadd`'s task list: the adder's four tasks are configured one
by one (the last one is shown), started together, and the store waits for
the writer.

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
    "dst": {"base": 2147484672, "bounds": [8], "strides": [64]}
   }
  },
  {"op": "start", "tasks": ["store_C"]},
  {"op": "sync", "task": "store_C", "mode": "poll"}
```

## 10. Block runtime model

What SNAX-BRM hands the rest of the flow (D68, D70): one accelerator as
plain data. A BRM is a hand-written JSON file in
`snax_forge/brm/library/`, one per accelerator, named after it and kept in
the form `Brm.to_json` writes (every field, defaults included). Loaded with
`load_brm(name)`; unknown keys are errors and a missing part is rejected by
name.

| Part | Holds | Read by |
|---|---|---|
| `interface` | params (`design` / `runtime`) and ports (name, direction, lanes, rate, dtype) | everything below |
| `function` | an accelerator kind registered in `brm/function.py`, its factory params without timing, and `code`: what one lane computes (D82) | the reference executor; `bind` and the accelerated node (`code`) |
| `dataflow` | a notation and one nest per port, in logical indices | the streamer values |
| `pattern` | `family` (a matcher registered in the sandbox), its `attrs`; `predicate` null | SNAX-SANDBOX's `bind` (section 12) |
| `implementations` | per name: `source` (`chisel` only), `supports`, `timing` (`latency`, `initiation_interval`, `drain`), `binding` (null, or `block`, `args`, `control`, `csrs`, `wait`, optionally `ports`, D129) | the GVSoC accelerator (GEN3), the HW generator (GEN1) |

The shared part (the first four) is what every implementation has in
common; designs that differ only in timing, supported values or RTL source
are implementations of one BRM.

**Port names** are used in `code` and in the streamer names
`<instance>_<port>` (D75), so a Python keyword (`in`) and a leading
underscore (`_in`) are rejected. The convention is `a`, `b`, ... for inputs
and `out` for the output (D103).

**Rates** (D68). A port moves one beat every `rate` firings, an int or the
name of a runtime param: an input of rate `r` is taken at firings `0, r, 2r,
...`, an output of rate `r` is given at firings `r-1, 2r-1, ...`. An
elementwise block has rate 1 everywhere, a reduction input rate 1 and output
rate `T`.

**Timing** is three numbers, each an int or an expression over design
params: `latency` (L, a firing to its push), `initiation_interval` (II, the
least cycles between firings) and `drain` (the cycles, starting with a push,
in which no input is taken; optional, 0 by default). Only L delays a result;
II and the drain limit how often inputs are taken.

**Params.** A `design` param is fixed per instance (lanes, the op) and
reaches the design point and the accelerator hjson (section 16); its
`default` is used when the design point gives none, and `values` limits what
any implementation may take. An implementation's `supports` limits it
further for that RTL. A `runtime` param is a start parameter of the
accelerator, given per task: the runtime params are exactly `n` (the
firings in the task) and every named port rate, in port order without
duplicates, and they are the accelerator task's `values` in section 9
(`accumulate`'s `n` and `T`); `n` must be a multiple of every resolved rate.
A value field
holds an int or an expression over params (`+ - * //`); a fixed string is
a design param with one allowed value.

**The function** names an accelerator kind registered in `brm/function.py`
(`elementwise`, `reduce`, with an op by name) and the kind's params over the
design params; the reference executor runs an accelerated node through it,
firing by firing (D79):

<!-- snippet: snax_forge/brm/library/elementwise_add.json -->
```json
 "function": {
  "accel": "elementwise",
  "params": {"lanes": "W", "n_inputs": 2, "op": "op"},
  "code": "out = a + b"
 },
```

`code` is one `output = expression` per output port over the input ports,
per lane of one firing, in the grammar of the value fields. It says what
the accelerator computes without running it: `bind` compares a tasklet
against it, and the accelerated node carries it. A reduction says what one
element contributes (`accumulate`'s `out = a`); that it is folded, and with
which op, is its pattern's (`family: reduce`, `attrs.op`) and its kind's
(`reduce`, `op`), so no IR's word for a reduction enters a BRM (D103). A
BRM with a null `code` cannot be bound. The library tests check `code`
against the registered kind.

<!-- snippet: snax_forge/brm/library/accumulate.json -->
```json
  "chisel_accumulator": {
   "source": "chisel",
   "supports": {"W": [1]},
   "timing": {"latency": 1, "initiation_interval": 1, "drain": 0},
   "binding": {
    "block": "Accumulator",
    "args": {"lanes": "W"},
    "control": "counted",
    "csrs": [{"field": "loopCount", "value": "n"}, {"field": "len", "value": "T"}],
    "wait": "pipelined",
    "ports": {"a": "in", "out": "out"}
   }
  },
```

<!-- snippet: snax_forge/brm/library/elementwise_add.json -->
```json
  "chisel_tiled_spatial": {
   "source": "chisel",
   "supports": {},
   "timing": {"latency": 0, "initiation_interval": 1, "drain": 0},
   "binding": {
    "block": "ElementwiseTiledSpatial",
    "args": {"lanes": "W", "supportedOps": "op"},
    "control": "counted",
    "csrs": [{"field": "loopCount", "value": "n"}],
    "wait": "lockstep"
   }
  }
```

**The binding** (D129, `brm/binding.py`) names the Chisel block and how
the BRM fills it; null until the implementation is generated. `block` is
the block's name (the generators register what they can build); `args` its
parameters, each an expression over design params (`ElementwiseTiledSpatial`'s
`lanes` = W and `supportedOps` = op; the Accumulator's `lanes` = W, which
`chisel_accumulator` and `chisel_adder_tree` both bind, D133); `control` (`counted`:
the block counts its firings, the only kind) and `wait` (`lockstep`,
`pipelined`) are registered kinds (D19); `csrs` the configuration fields in
CSR order, each `{field, value}` with the value an expression over params,
start following them: one for `ElementwiseTiledSpatial` (`loopCount`), two
for the Accumulator (`loopCount` = `n`, then `len` = `T`). A fixed op gives no `opSel`
field. `ports`, optional, maps every BRM port to the block's port of that role, by default
the same name (`accumulate`'s `a` is the Accumulator's `in`); it is written only when given.
Expressions are stored canonical.

`resolve(implementation, params)` makes an instance and checks a bound
implementation against the generator's blocks (`gen.BLOCKS`): its `block`
must be registered and the block's adapter must take the instance.

**The affine nest** of a port lists the operand's `shape`, an `offset` and
`loops`, outermost first; each loop has a `bound`, one stride per operand
dimension and a `spatial` flag, and the element of a step is
`offset + sum(i_l * strides_l)`. Temporal loops come first (one step of
them is one beat, the last one innermost), spatial loops last (the lanes,
the last one fastest). Spatial bounds use design params only and multiply
to the port's lanes; per task the nest gives n / rate beats inside the
shape. A temporal stride of 0 repeats a beat, as SNAX's reader does (D69).
The nest says nothing about addresses: section 9 gives how SNAX-LOWER maps
it through a layout onto a streamer's values.

<!-- snippet: snax_forge/brm/library/elementwise_add.json -->
```json
   "a": {
    "shape": ["n * W"],
    "offset": [0],
    "loops": [
     {"bound": "n", "strides": ["W"], "spatial": false},
     {"bound": "W", "strides": [1], "spatial": true}
    ]
   },
```

Element `t * W + j` in lane `j` of beat `t`, on `b` and `out` alike: the
ports agree on element positions, which is the BRM author's responsibility
(D70).

## 11. SNAX-DFG

The workload as SNAX-FORGE's own dataflow graph (D71, D77): one `.snaxdfg`
JSON file, written by an importer and rewritten by each sandbox transform. Loaded with `Graph.load(path)`; every field is
written, missing keys take defaults, unknown keys are errors, and a graph is
validated when it is made. Generated files live under `out/`; the fixtures
in `tests/dfg/fixtures/` are the exception.

| Field | Holds |
|---|---|
| `name` | the kernel |
| `symbols` | name → int or `null`: `null` as imported, the value once a recipe binds it; expressions keep the name |
| `containers` | name → `shape` (ints or expressions over symbols), `dtype` (a NumPy dtype name), `transient`, `attrs` (namespaced only) |
| `body` | the nodes, in execution order |

**Nodes.** Every node has `id`, `kind`, `inputs`, `outputs`, `attrs` and,
for a kind that has one, `body`. `inputs` and `outputs` map a connector
name to a memlet, `{"data": container, "subset": [dim, ...]}`, one
dimension per container dimension. A dimension is an index (`i`,
`4 * i_t + i_s`, `0`) or a range `begin:end` or `begin:end:step` with the
end exclusive. Node ids are identifiers and unique in the graph, because
derived task names are built from them (D75).

**Folding writes** (D102). A tasklet's output memlet may hold a third key,
`"reduce": {"op": ..., "identity": ...}`: the elements it writes are set to
`identity` when the node runs, and every iteration's value is folded in with
`op` instead of overwriting (a reduction). `op` is a registered name (`add`,
`mul`, `min`, `max`; `register_reduction`), each associative and commutative on
integers; `identity` is an int. `reduce` is written only when it is set, the one
field that is not always written, so graphs without a reduction read as
before; it is an error on an input and on an accelerated node, whose BRM
folds on its own. dot as imported: the sum map folds `tmp0` into `out[0]`:

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
| `map` | `var`, `range`; `loop.kind` absent, `tile`, `temporal` or `spatial`; `loop.split` (the `var` and `range` before `split_map`) | yes: runs once per value of `var` | none (derived from the body) |
| `tasklet` | `code`: one `output = expression` per output connector | no | one element each: indices only |
| `accelerated` | `instance`, `brm`, `implementation`, `code` (what one lane computes, the BRM's `function.code`), `params` (design params), `replaced` (the subtree `bind` replaced, or null) | yes: empty for a leaf block, a nested block otherwise | the BRM's ports, each port's lanes as a range; one beat |

Attrs without a namespace belong to the kind and are all written;
namespaced attrs (`loop.*`, `mem.*`, `hw.*`, `user.*`) are passed through
untouched. More kinds are added with `register_kind` (principle 6).

**Names.** Every name in an expression is a symbol or the `var` of an
enclosing map, and nothing else; shapes use symbols only. Symbols,
containers and variables never share a name. Expressions use the grammar
of section 10's value fields (ints, names, `+ - * //`) and are stored as
`ast.unparse` writes them, an expression without names as an int.

vecadd after `split_map` (W = 4) and `bind` to `elementwise_add`: the
accelerated node does one beat of four lanes, inside the temporal map of
N // 4 beats that becomes its streamers' temporal loop and its `n`. The
temporal map keeps what it was split from:

<!-- snippet: tests/dfg/fixtures/vecadd_accelerated.snaxdfg -->
```json
   "attrs": {
    "var": "i_t",
    "range": "0:N // 4",
    "loop.kind": "temporal",
    "loop.split": {"var": "i", "range": "0:N"}
   },
```

and the accelerated node what it computes and what it replaced (the
spatial map and the tasklet, as `split_map` left them, D82):

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

So a bound graph says on its own what each accelerator does, and
`unbind` and `join_map` (section 12) take it back to the imported graph.
`replaced` is history: checked in the node's scope, never run, and its ids
may repeat live ones. Whether the connectors, params and `code` agree with
the BRM is checked where the BRM is loaded: by `bind` and the reference
executor.

## 12. Recipe

SNAX-SANDBOX's ordered list of transforms (D72): what a thinker edits to
change the mapping of a kernel. A recipe is a hand-written JSON
file in `recipes/`, loaded with `Recipe.load(path)`; every field is written,
`params`, `symbols` and `steps` default to empty, unknown keys are errors.

| Field | Holds |
|---|---|
| `name` | the recipe; its steps are written to `out/sandbox/<name>/` |
| `kernel` | the kernel whose import the recipe starts from |
| `params` | recipe params, each an int or a string; `--set NAME=VALUE` overrides one |
| `symbols` | a value for graph symbols: an int or an expression over the params |
| `steps` | in order: `transform` (a registered name) and its `params` |

A transform's int params take an int or an expression over the recipe
params (`"factor": "W"`); its other params are taken as written. Running a
recipe binds the symbols (step 0), applies the steps, and checks every step
against the reference executor: on the same seeded inputs, every
non-transient container equals the input graph's. Each step is written as
`<i>_<transform>.snaxdfg`, with the recipe it ran as `recipe.json`.

| Transform | Params | Does |
|---|---|---|
| `split_map` | `map` (id), `factor` (int) | a map over `b:e` becomes a `temporal` map over `0:(e - b) // factor` (same id) around a `spatial` map `<id>_s` over `0:factor` |
| `bind` | `node` (tasklet id), `brm`, `implementation`, `instance`, `params` (optional design params) | the tasklet and its spatial map become an accelerated node; the lanes param comes from the spatial bound; the tasklet's code on the port names must be the BRM's `function.code`; a folding output that does not use the spatial variable becomes one lane, `c:c + 1` (D104) |
| `unbind` | `node` (accelerated id) | the accelerated node becomes the subtree it `replaced` |
| `join_map` | `map` (id) | a temporal map with `loop.split` and its spatial map become the map they were split from |

The vecadd recipe: W lanes, one temporal map of N / W beats, bound to
`elementwise_add`. `recipes/vecadd_undo.json` (`unbind`, then `join_map`)
takes its last step back to the first. The dot recipe binds two accelerators
with one `W` for both: the multiply map to `elementwise_mul` (instance
`mul`), the sum map to `accumulate`'s `chisel_adder_tree` (instance `sum`),
whose `out` is one lane every `T = N / W` firings (D104):

<!-- snippet: recipes/dot.json -->
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
  }
```

<!-- snippet: recipes/vecadd.json -->
```json
 "params": {"W": 4, "N": 64},
 "symbols": {"N": "N"},
 "steps": [
  {"transform": "split_map", "params": {"map": "add_map", "factor": "W"}},
```

## 13. Platform

The SNAX cluster around the accelerators (D84): everything of the cluster
that is not the accelerator. A platform file lives in `platforms/`
(`small16.json`, a small one without a source hjson; `snax_forge.json`, read
from snax_cluster's hjson); it never names a recipe,
and a recipe never names it. `Platform.load(path)`; every field is written,
missing keys take their defaults (`l2` missing: no L2), unknown keys and
values of the wrong type are errors, and loading reports every problem at
once (section 14).

| Field | Holds |
|---|---|
| `name` | the platform (an identifier) |
| `base` | the platform this one was made from, or null for an original |
| `changes` | what differs from `base`: `--set` path -> value, in the order set |
| `hjson` | the source cluster file, a path from snax-forge's root, or null (small16): what the generator writes the cluster file from (section 16) |
| `accelerator_slot` | the generated accelerator's name, `snax_forge` from `design from-hjson` (D131), or null; the generator gives it to the source's only slot (section 16); `--set accelerator_slot=NAME` picks another |
| `l1`, `l2` | `L1Config`, `L2Config` (below); `l2` null for an L1-only cluster |
| `wait_mode` | `poll` or `signal`: how the program waits for a task (the C backend refuses `signal`, `c.signal`) |
| `streamers` | the streamer shell: `default`, then an entry per streamer it changes |

`L1Config` (`design/platform.py`):

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `n_banks` | single-port banks | 32 | >= 1 |
| `width_bits` | bits per bank word | 64 | multiple of 8 |
| `rows` | words per bank | 512 | >= 1 |
| `dtype` | element type, any NumPy dtype name | `int64` | fits the word |
| `elems_per_word` | elements packed in one word (1 only, section 1) | 1 | >= 1, fits the word |
| `base_addr` | byte address of the first L1 word | 0 | — |
| `wide_bits` | widest port (the DMA's) | 512 | power-of-two multiple of `width_bits` |

`L2Config`:

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `size_bytes` | flat size | 1 MiB | positive multiple of the beat |
| `base_addr` | byte address of the first word | 0 | — |
| `beat_bits` | one DMA beat | 512 | multiple of `width_bits`; equal to the L1's `wide_bits` (`platform.l2`) |
| `width_bits`, `dtype`, `elems_per_word` | as the L1, and must agree with it (`platform.l2`) | 64, `int64`, 1 | fits the word |

**The streamer shell.** One streamer per port of every accelerated node's
instance, named `<instance>_<port>` (D75). The port decides `write` (its
direction is `out`), `n_ports` (its lanes) and, unless an entry gives them,
`spatial_bounds` (the port nest's spatial loops, fastest first); an entry
may not set `write` or `n_ports`. `default` holds `temporal_dims` and
`fifo_depth` (each >= 1); an entry named after a streamer
overrides any of them, and `spatial_bounds`, for that streamer only. These
are knobs, never a template: the generator lowers the streamer per design
point from them (section 16); a streamer input of its own is open item 78.

<!-- snippet: platforms/small16.json -->
```json
 "wait_mode": "poll",
 "streamers": {"default": {"temporal_dims": 1, "fifo_depth": 2}}
```

**Changes and working copies.** `pixi run design GRAPH --platform P --set
platform.PATH=VALUE` sets any field by its path (`l1.n_banks`,
`streamers.default.temporal_dims`, `streamers.acc_out.fifo_depth`; the
value is JSON, else text, and must have the field's type) and writes the
working copy `out/design/<name>/platform.json`: complete values, `base`
the platform it all started from, `changes` every path set since. Passing
the working copy as `--platform` continues from it, adding to `changes`.
`pixi run design save SRC NAME` keeps it as `platforms/NAME.json`, with
`base` and `changes` as its record; continuing from a saved platform keeps
the root `base`, so `changes` is always the difference from it.

**From snax_cluster's hjson** (LNK1, D131, `design/hjson_platform.py`).
`pixi run design from-hjson HJSON NAME` writes `platforms/NAME.json`, an
original (`base` null), from a cluster file (`$ref`s resolved): the L1 from
`cluster.tcdm`, `data_width`, `cluster_base_addr` and `dma_data_width`; the
L2 from the `dram` range; `streamers.default`'s `temporal_dims` and
`fifo_depth` from the one accelerator's streamer template, whose movers must
agree. What the hjson does not hold (dtype, elements per word,
`wait_mode`) is small16's. The platform records the
file (`hjson`, which must be under snax-forge's root, `hjson.path`) and
`accelerator_slot`, `snax_forge`, not the source's `snax_acc_name`; the
source must still have exactly one accelerator, its name an identifier.
`snax_forge.json` comes from `platforms/snax_alu_cluster.hjson`, a verbatim
copy of snax-gvsoc's `sw/cfg/snax_alu_cluster.hjson` refreshed by hand (its
diff the check, as D126); snax-gvsoc's `_old` copy gives the same platform.

<!-- snippet: platforms/snax_forge.json -->
```json
 "hjson": "platforms/snax_alu_cluster.hjson",
 "accelerator_slot": "snax_forge",
```

## 14. Design checks

Before the design step writes anything it runs every registered check on
the pairing of a bound graph and a platform (D85) and reports every
problem found, each as `[code] where: why` with a `fix:` line where one
exists: the `--set` or recipe change that removes it. `pixi run design
check GRAPH --platform P [--set ...]` runs them only. Stages: `platform`
and `graph` run independently; `connect` runs when both passed, `memory`
when `connect` passed. A check is
registered with `register_check(code, stage, fn)`, `fn(design)` yielding
problems.

| Code | Fails when |
|---|---|
| `platform.keys` | a platform file that does not parse; an unknown or missing key, a value of the wrong type; a `--set` path that does not exist, is written by the tools (`name`, `base`, `changes`) or is decided by the port (`write`, `n_ports`) |
| `platform.values` | a value its section rejects (`n_banks` 0, a `wait_mode` that is not allowed) |
| `platform.streamer` | a streamer value below 1, spatial bounds that are not a non-empty list of ints >= 1 |
| `platform.l2` | L2 and L1 disagree on beat width (`beat_bits` vs `wide_bits`), word width, dtype or elements per word |
| `platform.banks` | with an L2, `n_banks` is not a multiple of the DMA's port group (`wide_bits / width_bits`) |
| `graph.load` | a graph file that does not load (section 11) |
| `graph.symbols` | a symbol still null: bind it in the recipe |
| `graph.unbound` | a tasklet or a map holding nothing to run outside every accelerated node, or no accelerated node: the program drives accelerators only |
| `graph.brm` | an unknown BRM or implementation, design params that do not resolve, connectors that are not the BRM's ports, code that is not the BRM's |
| `graph.instance` | an instance named `dma` or after a streamer, or two streamers with one name |
| `connect.streamer_key` | a streamer entry that matches no port of the graph (the fix names the closest streamer) |
| `connect.derived` | a streamer entry that sets `write` or `n_ports` |
| `connect.lanes` | an entry's `spatial_bounds` that do not give the port's lanes or are not its nest's spatial loops |
| `connect.dtype` | a container whose dtype is not the L1's or not its port's, or an L1 with more than one element per word (open item 21) |
| `connect.temporal` | a map around an accelerated node that is not `temporal` (tiles are not lowered yet), a subset that is not affine in the firing loops, or a streamer with fewer `temporal_dims` than the node's firing loops (the fix names the count) |
| `memory.pin` | a memory working copy that does not load; a `--set memory.` path that is not `<container>.<memory>.base` or `passes.<pass>`; a pin on an unknown container (the fix names the closest) or on a memory the container does not live in; an unknown pass. The plan is made after these pass |
| `memory.residency` | a layout in a memory the platform lacks or for a container the graph lacks; a non-transient container that lives nowhere; an accelerator operand without an L1 layout |
| `memory.layout` | a layout whose shape is not its container's |
| `memory.align` | a base or stride that is not a multiple of the word; a container the DMA moves (in L2 and L1) that is not contiguous, not whole wide beats (open item 38) or not on a beat |
| `memory.fit` | a layout that leaves its memory, the message naming the memory's bytes `[start, end)`. Past the end the fix names the rows or L2 size that would hold it, after "pin it lower" for a pinned layout; below the start (a nonzero base) it names the pin that puts the layout at the memory's first byte |
| `memory.runtime` | an L1 layout (with its padding, D105) reaching into the top of L1 that snRuntime keeps at start-up (`start.S` in snax-gvsoc's `sw/snRuntime/src/`): a 1 KiB stack (`1 << SNRT_LOG2_STACK_SIZE`) for each core of the platform's `hjson`, plus 256 B for CLS (`.cdata`, `.cbss`), main's return word and the stacks' misalignment. The 256 B is a declared default (`RUNTIME_MARGIN`, `design/check.py`); a platform without an `hjson` runs no program and is not checked, nor is one whose `hjson` is not under snax-forge's root (the generator reads it from its own root, `gen.source`). The fix pins the layout just below |
| `memory.overlap` | two layouts in one memory that share bytes, naming the pins (the fix moves the latest pin past everything) |
| `point.keys` | a design point with unknown or missing keys or a malformed streamer entry |
| `point.streamers` | a design point whose `streamers` are not the platform's shell for its graph |

## 15. Memory plan and design point

**Memory plan** (D86, `snax_forge/design/memory.py`). A layout per
container and memory it lives in: `base` (byte address of index 0),
`shape` and one byte stride per dimension, the form of
`snax_forge/lower/layout.py`. Banks follow from addresses (word `i` in bank
`i % n_banks`, open item 18) and are not stated. Three passes make it, each
registered by name (`register_memory_pass(kind, name, fn)`):

| Pass | Decides | Default |
|---|---|---|
| `residency` | the memories of each container | `default`: not transient: L2 and L1 with an L2, else L1; transient: L1 |
| `layout` | shape and strides, base 0 | `contiguous`: row-major, one element per word |
| `placement` | the base per container and memory (in L2 the program's link replaces it: C§7's regions, D86) | `contiguous`: graph container order from the memory's first byte, no gaps; a layout in both L2 and L1 on a wide beat and to the end of its last beat (the DMA moves whole beats, so a size that is not whole beats is padded, D105), others on a word; around the pins |

`--set memory.<container>.<memory>.base=N` pins a base; `--set
memory.passes.<pass>=NAME` picks a pass. Passes read a `MemoryContext`:
the graph with its symbols bound, the platform, shapes, words, memory
ranges and, for policies that avoid bank conflicts, each accelerator
port's element-index stream (`accesses`, beats x lanes x dimensions, from
its memlet over the maps around it) grouped by the node that runs them
together. The design step writes `out/design/<name>/memory.json`:
`passes`, `changes` (every memory path set, in order) and `layouts`;
`--memory` continues from it.

**Design point** (D87, `snax_forge/design/point.py`). Written by the
design step to `out/design/<name>/design_point.json` once every check has
passed; not edited by hand. It is what SNAX-LOWER reads.

| Field | Holds |
|---|---|
| `name` | the design (the graph's sandbox folder, else its stem; in a flow, the recipe's name and params, D90) |
| `graph_from` | the `.snaxdfg` it was made from: a sandbox step, or any bound graph given to the design step or the flow (D108) |
| `graph` | that graph as read: instances, `code`, `replaced`, `loop.split` |
| `platform` | the platform working copy (section 13) |
| `streamers` | the resolved shell: `<instance>_<port>` -> `instance`, `port`, `write`, `n_ports`, `spatial_bounds`, `temporal_dims`, `fifo_depth` |
| `memory` | the memory plan: `passes`, `changes`, `layouts` (container -> memory -> layout) |

There is no recipe in it: the graph holds every instance, and the sandbox
folder keeps the recipe, when there was one (a flow from a bound graph has
none, D108). Loading one runs every check again on what it
holds (the stored layouts as they are) and requires `streamers` to be the
platform's shell for the graph.

## 16. Generated hjson files

What `pixi run gen-gvsoc DESIGN_POINT [--out DIR]` writes per design point
(D134, `snax_forge/gen/gvsoc.py`), default `out/gvsoc/<design>/`, and what
SNAX-GVSoC's `snax_hjson` reads (`--param hjson=<dir>/<platform>.hjson`,
snax-gvsoc `snax/snax_cfg.py`), with the program snax-gvsoc's generic app
builds from it (`sw/apps/forge/` there). For `platforms/snax_forge.json`
the three hjson files are `snax_forge.hjson`, `snax_forge_streamer.hjson`
and `snax_forge_acc.hjson`. Each hjson and C file starts with a
`//` comment naming the design point by its path from the root; the rest of
an hjson file is JSON, which hjson reads. One design point gives
byte-identical files wherever they are written.

| File | Holds |
|---|---|
| `<platform>.hjson` | the cluster: the platform's `hjson`, its comments dropped and its key order kept, with `cluster.tcdm.banks`, `.size` (KiB), `data_width`, `cluster_base_addr`, `dma_data_width` and `dram` written from the platform, `cluster.name` `snax_forge_cluster` (also in `cluster.bender_target`, in place of the source's name), and the slot set; the source's template, referred to only by the slot, dropped |
| `<slot>_streamer.hjson` | the streamer template, the whole file: `data_reader_params` (the input ports in the BRM's port order) and `data_writer_params` (the output ports), per mover `spatial_bounds`, `temporal_dim`, `num_channel` (64-bit channels) and `fifo_depth` |
| `<slot>_acc.hjson` | the datapath block, the whole file: `unit` (`SnaxUnit`), `inputs` and `outputs` (one per reader and writer, in order), `lanes`, `data_width`, `op`, `signed`, `control`, `config` (left out when `["count"]`), `wait`, `latency`, `ii`, `fields` (the shell's configuration CSRs; start follows) |
| `include/` | `streamer_csr_addr_map.h` and `<instance>_csr.h` (`gen/headers.py`) |
| `data.h` | the program's data (`gen/data.py`): per container the DMA moves, an input from the kernel's `make_inputs` (seed 0), or an output zeroed and its `<name>_golden` from the reference executor; each array the container's dtype, padded to whole DMA beats as its L2 layout (D105) and aligned to the beat |
| `main.c` | the program (`lower/c.py`, D135): `main` reads `mcycle` first, then runs its core's function, one per core with each step on its component's core and each accelerator task's counter stored after its wait (`forge_counter[]`); no task marks. On the accelerator's core `main` then calls `forge_check()`, which reads `mcycle` first and prints `forge cycles: <n>` (the window from main's entry), then the self-check: the outputs against their goldens (`forge data:`), each accelerator task's counter against its beats (`forge counter <task>:`), `forge errors:`; main returns the error count |
| `manifest.json` | `design`, `design_point`, `symbols`, `cluster` (the cluster file), `files`, `program` (`main.c`), `trace`, `csrs` (the CSR layout), `tasks` (per accelerator task its `beats` and `csrs` values) |

**The slot** is the source's only one, renamed: `snax_acc_name` is the
platform's `accelerator_slot`, and its `bender_target` goes (the RTL comes
from the bundle, section 18). `snax_tcdm_ports` is the streamer's channels; the source's
`sparse_interconnect_config` becomes one group of that width (its
granularity kept), or `snax_narrow_tcdm_ports` that many and
`snax_wide_tcdm_ports` 0; `snax_num_rw_csr` is the fields and start,
`snax_num_ro_csr` 2 (busy and the counter, D130); then, after the source's
other keys in their order, `snax_disable_csr_manager` `true` (the wrapper
carries its own manager, D130), `snax_forge_rtl` `"rtl/files.f"` (the RTL's
file list, relative to the cluster file; it exists only in a bundle,
section 18, and SNAX-GVSoC ignores the key), `snax_streamer_cfg`
`{$ref: "<slot>_streamer.hjson"}` and `snax_datapath_cfg`
`{$ref: "<slot>_acc.hjson"}`. SNAX-GVSoC's reader checks the counts against
the two files; snax_cluster's clustergen and snaxgen resolve the cross-file
`$ref`s against the cluster file's real folder (section 18). snax_cluster's
schema lists the slot's keys `snax_acc_name`, `snax_tcdm_ports`,
`snax_narrow_tcdm_ports`, `snax_wide_tcdm_ports`, `snax_num_rw_csr`,
`snax_num_ro_csr`, `snax_disable_csr_manager` (boolean), `snax_datapath_cfg`
(object) and `snax_forge_rtl` (string), none with a default (open item
88); the others the slot keeps (`sparse_interconnect_config`,
`snax_streamer_cfg`) it does not list, and it allows them.

**The CSRs.** The streamer's from `0x3C0`, then the shell's fields, start,
busy and the counter (`csrs` in the manifest). An accelerator's range is
`0x3C0` to `0x5FD` (`SNAX_ACC_CSR_LAST`): the core sends out up to `0x5FF`,
but snax_cluster reserves `0x5FE` and `0x5FF` for its task queues
(`CSR_SNAX_READ_TASK_READY_QUEUE`, `CSR_SNAX_WRITE_TASK_DONE_QUEUE`) and
routes them to the cluster's top-level CSR port, not to the accelerator;
SNAX-GVSoC's `SnaxUnitAcc` refuses the same. A counter past `0x5FD` is
`gen.csrs`.

**The cluster.** `cluster.name` becomes `snax_forge_cluster` and, in
`cluster.bender_target`, the entry equal to the source's `cluster.name`
becomes it too (D139); a source whose `bender_target` does not hold its own
name is `gen.source`.

**Checks.** The streamer and accelerator files have snax-forge's schemas
(`snax_forge/gen/schema/`, JSON Schema 2020-12); the cluster file, its
`snax_acc_cfg` entries included, validates against snax_cluster's
(snax-gvsoc's copy, `sw/docs/schema/snitch_cluster.schema.json`, where the
submodule is checked out). A platform value the cluster file cannot express (an L1 that is not
whole KiB, an L1 or L2 field it does not give back) is `gen.platform`, a
platform without a source `gen.source`, a source with no slot or several
`gen.slot`.

<!-- snippet: snax_forge/gen/schema/accelerator.schema.json -->
```json
 "required": ["unit", "inputs", "outputs", "lanes", "data_width", "op", "signed", "control",
              "wait", "latency", "ii", "fields"],
```

## 17. Accelerator RTL spec

What `pixi run gen-rtl DESIGN_POINT [OUT_DIR]` writes first (D138,
`snax_forge/gen/rtl.py`), default `out/rtl/<design>/rtl.json`, and what
`Emit --spec` reads to build the accelerator (`hw/chisel/`): concrete values
only, no expressions, checked against the block catalogue
`hw/chisel/blocks.json` and the schema `snax_forge/gen/schema/rtl.schema.json`.

| Key | Holds |
|---|---|
| `design_point` | the design point by its path from the root, as C§16's files name it |
| `tag` | the platform's `accelerator_slot`, `snax_forge` by default; the wrapper is `<tag>_shell_wrapper` |
| `units` | one unit (more is `gen.instances`; links come with M13): `name` (the instance), `block` (a catalogue block), `params` (every block param, typed and in its limits), `ports` (`{brm, block}` per BRM port, in the BRM's port order, which is the streamer's reader and writer order) and `csrs` |

A unit's `csrs` are in CSR order from offset 0 (`snax_csr_mux_demux`
removes the streamer's range): `rw`, the configuration fields (`pin`, the
binding's `csrs` field, and its `width`), then `start`, the start CSR's pin
(bit 0 written 1 starts), then `ro`, busy and the counter. `tie` gives the
block's pins tied to a constant (`opSel`, the op's encoding). No module
names: `Emit` picks them and lists what it wrote in `emitted.json`, beside
`sv/`. vecadd's, a fixture of the Chisel specs:

<!-- snippet: hw/chisel/src/test/resources/rtl/vecadd/rtl.json -->
```json
{
 "design_point": "out/design/vecadd/design_point.json",
 "tag": "snax_forge",
 "units": [
  {
   "name": "acc",
   "block": "ElementwiseTiledSpatial",
   "params": {
    "dataWidth": 64,
    "lanes": 4,
    "loopCountWidth": 32,
    "supportedOps": ["add"],
    "signed": true
   },
   "ports": [{"brm": "a", "block": "a"}, {"brm": "b", "block": "b"}, {"brm": "out", "block": "out"}],
   "csrs": {
    "rw": [{"pin": "loopCount", "width": 32}],
    "start": "start",
    "ro": ["busy", "counter"],
    "tie": {"opSel": 0}
   }
  }
 ]
}
```

**The wrapper**, `<tag>_shell_wrapper`, has the ports
`snax_acc_wrapper.sv.tpl` connects with `snax_disable_csr_manager`: `clk_i`,
`rst_ni` (active low), `csr_req_addr_i` and `csr_req_data_i` (32 bits),
`csr_req_write_i`, `csr_req_valid_i`, `csr_req_ready_o`, `csr_rsp_data_o`
(32 bits), `csr_rsp_valid_o`, `csr_rsp_ready_i`; per input port k, in order,
`stream2acc_<k>_data_i`, `_valid_i`, `_ready_o`, per output port
`acc2stream_<k>_data_o`, `_valid_o`, `_ready_i`, data `num_channel × 64`
bits with lane i at bits `[64i+63 : 64i]`. Inside: `CsrManager`
(ReqRspManager's protocol, D130), `CtrlAdapter` (an accepted start is the
block's start with the fields on its pins; ready is not busy; the counter
is reset by an accepted start, counts busy cycles and holds after) and the
unit.

**One rule on both sides.** The shell's rule (a start, a start held while
busy and answered in E + 1, busy, the counter) is snax-gvsoc's
`SnaxShellRule` (`snax/snax_shell.hpp`), which `SnaxShell` runs; its
hand-worked cases are `tests/snax/shell/cases/` (the format in that folder's
README), which snax-gvsoc's `tb.cpp` checks the rule against, stepped with
the unit's, and `ShellCasesSpec` drives vecadd's wrapper through
(`SNAX_GVSOC_SHELL_CASES`). The unit's rule is `snax/snax_unit.hpp` with
`tests/snax/unit/cases/`, which the block specs drive the blocks through
(`SNAX_GVSOC_UNIT_CASES`). CI's chisel job sets both to the submodule's
folders. Where a case shows the two sides disagree, the Chisel behaviour
stands unless it breaks this section or D130, and SNAX-GVSoC follows it.

## 18. Bundle

What `pixi run bundle DESIGN_POINT [--out DIR]` writes per design point
(D111, D139, `snax_forge/gen/bundle.py`), default `out/bundle/<design>/`:
everything snax_cluster needs to build the design's cluster, made afresh by
the snax-forge checked out, never copied from `out/gvsoc/` or `out/rtl/`.
`rtl/` is the bundle's own and is removed before each run, so it never
keeps a stale file; nothing else in `DIR` is removed.

| File | Holds |
|---|---|
| the root | section 16's files (gen-gvsoc's), the cluster file among them |
| `rtl/rtl.json` | section 17's spec |
| `rtl/sv/`, `rtl/emitted.json` | what `Emit --spec` wrote (section 17); firtool also writes `sv/filelist.f`, which `emitted.json` does not list |
| `rtl/files.f` | the RTL's file list: one path per line, relative to `rtl/`, in `emitted.json`'s `files` order, `emitted.json` itself left out |
| `bundle.json` | `design_point` (as section 16 names it), `tag` (section 17), `snax_forge_commit` (`git rev-parse HEAD`, or null outside git) and `dirty` (true when the work tree has uncommitted changes, untracked files included; false outside git) |

**Refused**: what sections 16 and 17 refuse, then `bundle.empty`
(`emitted.json` lists no file) and `bundle.missing` (it lists a file that
is not there).

**How snax_cluster takes it** (D139, snax_cluster #659), through the
cluster file alone (`CFG_OVERRIDE`), so snax_cluster holds no dependency on
snax-forge:

- clustergen and snaxgen read the cluster file with one hjson loader that
  resolves a `$ref` to another file against the cluster file's real folder;
- the slot names `rtl/files.f` (`snax_forge_rtl`, relative to the cluster
  file); in RTL mode snaxgen writes `target/snitch_cluster/generated/snax_forge.f`:
  the paths that list names, then its own `<tag>/<tag>_Streamer.sv`,
  `<tag>_streamer_wrapper.sv` and `<tag>_wrapper.sv` (no CSR manager), all
  absolute; it is empty for a cluster file without the key;
- the builds read it only when it is non-empty: the vsim script gets a
  `vlog -sv ... -f snax_forge.f` line after Bender's, Verilator a
  `-f snax_forge.f`; VCS and synthesis do not read it;
- the streamer template has no `snax_library_name`, which snax_cluster's
  streamer generator then takes as the tag: the streamer header lands in
  `target/snitch_cluster/sw/snax/<tag>/include/streamer_csr_addr_map.h`
  (gitignored), the file GEN2b compares with the bundle's `include/`;
- the cluster is always `snax_forge_cluster` with one Bender target of that
  name: the testharness imports `<cluster>_pkg` from the cluster wrapper,
  which must therefore come through Bender's ordered file list, not through
  `snax_forge.f`;
- snax_cluster keeps a vecadd bundle as a fixture,
  `target/snitch_cluster/cfg/snax_forge/vecadd/` (refreshed by its README):
  the hjson files, `bundle.json`, `rtl/files.f` and the files it lists, and
  the program, `main.c`, `data.h` and `include/` (not `manifest.json`), all
  untouched;
- `target/snitch_cluster/snax_forge.mk` (GEN2b, D140) runs a program on the
  RTL with snax_cluster's own targets: `CFG_OVERRIDE` the bundle's cluster
  file, `FORGE` the program's folder (default `CFG_OVERRIDE`'s), `ELF` an
  ELF to run (default the one built from `FORGE`), `SIM` `vlt` (default) or
  `vsim`. `snax-forge-sim` builds `bin/snitch_cluster.$(SIM)` (for `vsim`
  after `vsim_preparation`); `snax-forge-sw` builds `sw/apps/snax-forge`
  for `FORGE`; `snax-forge-run` runs the ELF, writes its output to
  `logs/snax-forge-$(SIM).txt` and fails on the program's nonzero exit code;
- the header check: `snax-forge-sw` fails when the `#define` lines of
  `$(FORGE)/include/streamer_csr_addr_map.h` and of the header snaxgen
  wrote (above) differ; comments are ignored;
- its `.github/workflows/snax-forge-ci.yml` builds the fixture with
  Verilator on every PR, failing on an empty or broken `snax_forge.f` or a
  duplicate module name, then builds the fixture's program, runs it with
  `snax-forge-run SIM=vlt` (pass on exit code 0) and checks the README's
  expected `forge counter` line.
