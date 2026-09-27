# SNAX-FORGE Contracts

> What the artefacts between the components *mean*, field by field, so each
> component can produce its output without reading another's code.
> Sections 1–8 are the model side (D26, D46): cluster configuration,
> streamer registers, accelerator interface, control program, scenario,
> profile and trace, and the rules a new block kind must follow. Sections
> 9–15 are the rest of the flow: the task list (D64), the BRM (D68, D70),
> the SNAX-DFG (D77), the recipe (D80), the platform (D84), the design
> checks (D85), and the memory plan and design point (D86, D87).
>
> Until the M6 freeze these are plain dataclasses and plain JSON; versioned
> schemas are F2's job (D26). Every value marked **default** is a declared
> platform default, not a measurement (D51); nothing else about the format
> depends on it.
>
> Every fenced block below is copied verbatim out of a checked-in file, a
> file `scenarios/make.py` generates (D67), or the output of
> `scenarios/reduce`, and `tests/snax_model/test_contracts.py` fails if one
> of them drifts. The `<!-- snippet: ... -->` line above each block names its
> source; `run:` means "produced by running that scenario". Why each
> contract is the way it is: the module docstrings and `docs/DECISIONS.md`.

---

## 1. Conventions

**Byte addresses vs register indices.** Memory addresses — L1 and L2
addresses, streamer `base`, all strides, DMA bases — are byte addresses.
Control-program addresses are *register indices*, not bytes: register `n` of
a block at base `b` is address `b + n` (section 5).

**The bank word is the unit of transfer** (D13). A word is
`L1Config.width_bits` wide and holds `elems_per_word` elements of `dtype`.
In v1 `elems_per_word` is 1: one element per word. The data model is ready
for packing (storage is `[banks, rows, elems_per_word]`, writes take a
per-element strobe), but nothing above the L1 has been exercised with
`elems_per_word > 1`; treat anything else as unsupported until a kernel
needs it.

**Integers first** (D28). The default `dtype` is `int64`. Floating point
works through the L1 and the streamers, but a reduction's result then
depends on accumulation order, which the reference executor cannot match
until a BRM defines that order (D28, M5).

**One cycle** is one sweep of the phases `CONTROL, COMPUTE, REQUEST,
ARBITRATE, MEMORY, RESPONSE` (D29). A cycle number in any artefact — a
trace timestamp, a wait's `done`, a class interval — counts these sweeps
from 0.

<!-- snippet: scenarios/clusters/alu4.json -->
```json
 "l1": {
  "n_banks": 16,
  "width_bits": 64,
  "rows": 64,
  "read_latency": 1,
  "dtype": "int64",
  "elems_per_word": 1,
  "base_addr": 0,
  "wide_bits": 512
 },
```

## 2. Cluster configuration

The cluster file (`ClusterConfig`) holds `l1`, optional `l2`, `components`
and `register_map` (D41). Every config class writes all of its fields
(`to_dict`), missing keys take the class default, and unknown keys are an
error.

**Who fills what** (D51, D53). The accelerator entries are the user's: an
accelerator's `lanes`, rates, `latency`, `ii` and `op` (later its BRM).
Everything else describes the SNAX platform, with the defaults below; they
are design knobs for SNAX-DSE, not measurements. SNAX-LOWER derives the whole
file (LOW1c, D88, `snax_forge/lower/cluster.py`): from a design point with
`pixi run lower cluster out/design/<name>/design_point.json`, and for the
checked-in clusters through the same builder, which `scenarios/make.py`
writes. Components: `xbar`, `dma` (with an L2), then per accelerator in the
order the graph first uses it its streamers in port order
(`<instance>_<port>`, from the platform's shell, section 13) and the
accelerator itself, then `ctl`; the register map lists all but the xbar and
the controller, with spatial bounds only where they are not `[n_ports]`.

**`components` is one ordered list** of every ticked component, the xbar and
the controller included. The builder adds them in exactly that order, and
that order fixes three things: the scheduler's tick order, the xbar's port
order (which decides round-robin tie-breaks, D31) and the trace's source
order (D39). The L1 and the L2 are shared elements, not components (D30,
D34), so they have their own keys. There is one L1, one xbar and at most one
L2; streamers and the DMA find them without naming them. The register map is
built when the builder reaches the controller, so every block it lists must
come before it.

### L1Config (mem.py)

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `n_banks` | single-port banks, one access per bank per cycle | 32 | >= 1 |
| `width_bits` | bits per bank word | 64 | multiple of 8 |
| `rows` | words per bank | 512 | >= 1 |
| `read_latency` | cycles from a read to its data | 1 | >= 0 |
| `dtype` | element type, any NumPy dtype name | `int64` | fits the word |
| `elems_per_word` | elements packed in one word | 1 | >= 1, fits the word |
| `base_addr` | byte address of the first L1 word | 0 | — |
| `wide_bits` | widest port allowed (the DMA's) | 512 | power-of-two multiple of `width_bits` |

A port of `w` bits covers `w / width_bits` consecutive banks in an aligned
group; `n_banks` must be a multiple of that group, checked when the port is
added (D33). The address-to-bank map is word-interleaved (word `i` in bank
`i % n_banks`, row `i // n_banks`); a different map can be passed in Python
but cannot yet be named in a scenario (open item 18).

### L2Config (l2.py)

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `size_bytes` | flat size | 1 MiB | positive multiple of `beat_bytes` |
| `base_addr` | byte address of the first word | 0 | — |
| `read_latency` | cycles from a read to the DMA's buffer | 1 **default** | >= 0 |
| `beat_bits` | one access | 512 | multiple of `width_bits`; must equal `L1Config.wide_bits` |
| `width_bits`, `dtype`, `elems_per_word` | as the L1, and must agree with it | 64, `int64`, 1 | fits the word |

One read and one write per cycle (independent AXI channels); a second of
either in one cycle is a bug in the DMA and raises.

### StreamerConfig (streamer.py)

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `write` | false: reader (L1 -> FIFO); true: writer | false | — |
| `n_ports` | interconnect ports = lanes per beat (D12) | 1 | >= 1 |
| `temporal_dims` | temporal counters in hardware | 1 | >= 1 |
| `fifo_depth` | data buffer per lane | 8 | >= 1 |
| `addr_depth` | address buffer per port | 8 | >= 1 |
| `prio` | static TCDM priority of every port | 0 | — |

### DmaConfig (dma.py) — every value a declared **default** (D51)

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `startup` | start in `s` -> first source request in `s + startup` | 2 | >= 1 |
| `beat_interval` | minimum cycles between beats on each side | 1 | >= 1 |
| `l1_read_extra` | L1 source: extra cycles from read data to the buffer | 0 | >= 0 |
| `done_latency` | last write -> its response | 0 | >= 0 |
| `dims` | loops per side held in registers | 2 | >= 1 |

### ControllerConfig (ctrl.py) — every cost a declared **default** (D51)

| Field | Meaning | Default | Checked |
|---|---|---|---|
| `write_cost`, `read_cost` | cycles per `csr_write` / `csr_read` | 1, 1 | >= 1 |
| `kind_write_cost`, `kind_read_cost` | the same per block kind, e.g. `{"dma": 2}` | `{}` | >= 1 |
| `poll_interval` | `P`: cycles from one poll sample to the next | 1 | >= 1, >= the block's read cost |
| `signal_latency` | `S`: done -> the controller continues | 1 | >= 1 |

### Components

A component entry is `name`, `kind` and `config`; an accelerator uses
`accel`, `params` and `attach` instead. Kinds are registered
(`register_component`, D43): `xbar`, `streamer`, `dma`, `accel`,
`controller`. Accelerator kinds are registered separately
(`register_accel`): `elementwise` and `reduce` are the two stubs, and
`params` are the stub's own keyword arguments with `op` given by a
registered name (`add`, `sub`, `mul`, `min`, `max`, `and`, `or`, `xor`).

<!-- snippet: scenarios/clusters/alu4.json -->
```json
  {
   "name": "dma",
   "kind": "dma",
   "config": {"startup": 2, "beat_interval": 1, "l1_read_extra": 0, "done_latency": 0, "dims": 2}
  },
```

<!-- snippet: scenarios/clusters/red4.json -->
```json
  {
   "name": "acc",
   "kind": "accel",
   "accel": "reduce",
   "params": {"lanes": 4, "lanes_out": 1, "op": "add", "latency": 1, "ii": 1},
   "attach": {"in": "acc_in", "out": "acc_out"}
  },
```

`attach` names, per accelerator port, the streamer whose FIFO it uses: a
reader's FIFO for an input, a writer's for an output. Every port must be
attached exactly once, to a streamer listed before the accelerator, and the
streamer's `n_ports` must equal the port's `lanes`.

`register_map` gives `window` (registers per block, a power of two, default
32), `blocks` (the blocks in address order; left out, every component except
the xbar and the controller, in component order), `bases` (a block's window
base, a multiple of `window`) and `spatial_bounds` (per streamer, design-time
spatial bounds whose product is `n_ports`; default `(n_ports,)`).

## 3. Streamer register layout

This is what BRM2 is tested against.

A streamer's configuration registers, in offset order after the three status
registers (section 5):

```
base            byte address of element (0, 0)
tbound[0..D-1]  temporal loop bounds,  D = StreamerConfig.temporal_dims
tstride[0..D-1] temporal loop strides in bytes, signed
sstride[0..S-1] spatial loop strides in bytes, signed, S = len(spatial_bounds)
```

They enumerate, with loop 0 innermost and spatial dimension 0 fastest:

```
for i[D-1] in range(tbound[D-1]):          # outermost
  ...
    for i[0] in range(tbound[0]):          # one beat per step
      parfor j[0..S-1]:                    # one interconnect port per lane
        lane  = j[0] + sbound[0] * (j[1] + sbound[1] * ...)
        addr  = base + sum(i[t] * tstride[t]) + sum(j[k] * sstride[k])
```

One temporal step is one beat of `n_ports` elements, issued as one request
per port. `n_beats` is the product of the temporal bounds; if any of them is
0 the streamer never becomes busy (the AGU ignores the start), and a bound
of 1 runs once. **Spatial bounds are design time**: they are part of
`register_map.spatial_bounds`, not registers, as in the RTL, where only the
spatial strides are CSRs. Unused temporal loops are padded with bound 1 and
stride 0, which the model treats as fewer loops.

A reader with `tstride[0] == 0` repeats instead of re-reading, as the RTL
does (D69): it reads each group of `tbound[0]` beats once and hands that beat
to the accelerator `tbound[0]` times, so the accelerator sees every beat of
the enumeration above while L1 sees one read per group. A stride of 0 on any
other loop, or on a writer, is an ordinary stride.

**Where the values come from** (D73, D89). SNAX-LOWER derives a
streamer's values from the memlet on the accelerated node's connector,
through the container's L1 layout in the memory plan (base, shape and one
byte stride per dimension, section 15): one temporal loop per firing loop
around the node, innermost first, and one spatial loop per range dimension
of the memlet, fastest first, whose bounds must be the streamer's
design-time spatial bounds (so their product is `n_ports`); a loop's byte
stride is its index strides dotted with the layout's, and `base` is the
layout's address of the memlet's first element (`lower/derive.py`). The
BRM's per-port nest (D70) then checks the order: mapped through the same
layout (`lower/streams.py`), it must visit the same addresses beat by beat.
The first nest notation is `affine`; later ones are open item 1, and this
register side does not change with them.

<!-- snippet: scenarios/vecadd/scenario.json -->
```json
  {"op": "csr_write", "reg": "acc_a.base", "value": 0},
  {"op": "csr_write", "reg": "acc_a.tbound[0]", "value": 16},
  {"op": "csr_write", "reg": "acc_a.tstride[0]", "value": 32},
  {"op": "csr_write", "reg": "acc_a.sstride[0]", "value": 8},
```

16 beats of 4 words: 4 lanes 8 bytes apart (`sstride[0]`), one beat every 32
bytes (`tstride[0]`), from L1 byte 0.

## 4. Accelerator interface

This is what BRM3 and BRM4 are accepted against: plugged into the model, a
BRM must give the same cycles and the same data as the matching stub.

An accelerator is a set of ports, two timing numbers and a Python function:

| Part | Meaning |
|---|---|
| port `name` | matches an `attach` key and a function argument |
| port `direction` | `in` (popped from a reader's FIFO) or `out` (pushed into a writer's) |
| port `lanes` | elements per beat; equals the attached streamer's `n_ports` |
| port `rate` | the port moves one beat every `rate` firings (D25); an int, or the name of a start parameter |
| `latency` (`L`) | pipeline stages between a firing and its push; 0 = same cycle |
| `ii` | minimum cycles between firings |
| `fn` | the Python implementation, called once per firing |

A *firing* is one step of the datapath. Input port with rate `r` is popped at
firings `0, r, 2r, ...`; output port with rate `r` is pushed at firings
`r-1, 2r-1, ...`. So an elementwise block has rate 1 everywhere, and a
reduction has input rate 1 and output rate `T`.

```python
fn(k, ins, state, params) -> outs
```

`k` is the firing index in the task; `ins` maps every input port due at `k`
to its beat (a NumPy array, first axis = lanes); `state` is a dict that lives
for one task (the reduction's partial sum) and changes only when a firing
happens; `params` are the start parameters. `outs` must hold exactly the
output ports due at `k`, each with `lanes` elements.

Registers: `n` (firings in the task), then every port rate that is named
rather than an int, in port order without duplicates. A reduce stub
therefore has `n` and `T`; `n` must be a multiple of every resolved rate.

<!-- snippet: scenarios/reduce/scenario.json -->
```json
  {"op": "csr_write", "reg": "acc.n", "value": 16},
  {"op": "csr_write", "reg": "acc.T", "value": 4},
```

Timing, as the model runs it (D35): a start in cycle `s` makes the block busy
from `s+1`; a firing in cycle `t` is pushed in `t + L`; the output FIFO is
checked at push time, and if the head cannot be pushed the whole pipeline
freezes for that cycle (no firing, no pop); `II` counts wall-clock cycles, so
a freeze longer than `II` does not delay the next firing; `done_cycle` is the
cycle after the last push. An `AccelConfig` is not serialisable — it holds
`fn` — which is why a cluster file names a registered kind and its params
(D43).

## 5. Control program

This is LOW1b's target. The program is a plain list of commands, one command
per entry, as the controller runs them; nothing expands inside the model
(principle 4, D42).

| Command | Fields | Effect |
|---|---|---|
| `csr_write` | `addr` or `reg`, `value` | writes a configuration register, or launches the block when it is `start` |
| `csr_read` | `addr` or `reg` | samples committed state into the run's read log |
| `wait` | `block`, `mode` (`poll` or `signal`) | blocks until the block is done |

A register is named either by raw index (`"addr": 4`) or by name
(`"reg": "dma.src_base"`), exactly one of the two, and a command keeps the
form it was written in through a round trip.

**Every block has the same window** of `window` registers (default 32, one
word each, D36):

```
offset 0   start        write-only; writing 1 launches the block
offset 1   busy         read-only; the component's committed busy
offset 2   busy_cycles  read-only; busy cycles of the current or last task
offset 3.. configuration registers, read/write, listed by the block's adapter
```

Configuration registers are **buffered**: they are shadow values, and a start
copies them into the block's start argument, so the next task can be
programmed while the block runs. They reset to 0. A start while the block is
busy is an error — the program must wait first. `busy_cycles` read in cycle
`r` after a start that landed in `s` is `min(r, D) - s - 1`, with `D` the
block's `done_cycle`; a task with no work reads 0.

**Timing** (D37, every cost a declared **default**, D51). The program starts
in cycle 0 and runs one command at a time. A command beginning in `t` with
cost `c` covers `[t, t + c - 1]` and takes effect in its last cycle; the next
begins in `t + c`. A start landing in `w` makes the block busy from `w + 1`.
With `c_r` the block's read cost, `P` the poll interval, `S` the signal
latency and `D` the block's `done_cycle`:

```
poll:    sample i is taken in cycle t + i*P + c_r - 1, and the wait ends on
         the first sample that reads busy = 0, so its length is
             W_poll   = i* * P + c_r ,   i* = max(0, ceil((D - t - c_r + 1) / P))
signal:  the wait covers [t, max(t, D) + S - 1], so its length is
             W_signal = max(t, D) - t + S
```

Both modes give the same data; only the cycle count differs. A wait may only
follow a start of its block in program order (checked when the controller is
built), so `done_cycle` always belongs to the task the wait is for.

<!-- snippet: scenarios/reduce/scenario.json -->
```json
  {"op": "csr_write", "reg": "acc_in.start", "value": 1},
  {"op": "csr_write", "reg": "acc_out.start", "value": 1},
  {"op": "csr_write", "reg": "acc.start", "value": 1},
  {"op": "wait", "block": "acc_out", "mode": "signal"},
  {"op": "wait", "block": "acc", "mode": "poll"},
  {"op": "csr_read", "reg": "acc.busy_cycles"}
```

Mapping these register blocks onto the real SNAX interfaces (ReqRspManager
CSRs, iDMA instructions) is SNAX-LOWER's C backend, by register *name*
(D36, open item 9). `RegisterMap.to_dict` lists every register with its
address, and every run writes that listing into `run.json`.

## 6. Scenario and memory

A scenario is the input of one run and nothing else: expected results live in
the tests (D41). Two files, because one cluster serves many programs:

* the **cluster file** (section 2), shared;
* the **scenario file**: `name`, optional `max_cycles`, `cluster` (a path
  relative to the scenario file, or the cluster object inline), `memory` and
  `program`.

<!-- snippet: scenarios/vecadd/scenario.json -->
```json
{
 "name": "vecadd",
 "max_cycles": 5000,
 "cluster": "../clusters/alu4.json",
 "memory": [{"mem": "l2", "addr": 0, "npy": "a.npy"}, {"mem": "l2", "addr": 512, "npy": "b.npy"}],
```

`memory` is a list of fills applied in order before the run. Each has `mem`
(`"l1"` or `"l2"`), a byte `addr` (the L1 one includes `base_addr`) and
exactly one source:

| Source | Meaning |
|---|---|
| `data` | inline words, `[n]` or `[n, elems_per_word]` |
| `npy` | a `.npy` file, path relative to the scenario file |
| `random` | `seed`, `n`, `low`, `high`: `default_rng(seed).integers(low, high, n)`, integers only (D28) |

The run writes into a directory (D44, D50): `run.json` (scenario name,
cluster file, trace level, total cycles, the cluster configuration, the
register map, the `csr_read` values), `profile.json`, `trace.jsonl` and
`trace_meta.json` when traced, `l1.npy` and `l2.npy` (flat words
`[n_words, elems_per_word]` in address order). Files a run does not produce
are removed, so the directory always matches its `run.json`. The bytes are
identical on every run and with skipping on and off.

## 7. Profile and trace

This is what the views (M4a, M4b) and VIS7 read.

**Profile** (`profile.py`): the totals of one run, read off the counters the
components already keep. It never recounts, and it changes nothing (D38).
`skip_idle` is deliberately not in it. The fields:

<!-- snippet: snax_forge/snax_model/profile.py -->
```python
@dataclass
class Profile:
    total_cycles: int
    controller: ControllerProfile | None = None
    accelerators: dict[str, AccelProfile] = field(default_factory=dict)
    streamers: dict[str, StreamerProfile] = field(default_factory=dict)
    dmas: dict[str, DmaProfile] = field(default_factory=dict)
    banks: BankProfile | None = None
    ports: dict[str, PortProfile] = field(default_factory=dict)
    l2: dict[str, L2Profile] = field(default_factory=dict)
    functional_check: dict[str, Any] | None = None  # MOD9 / E2E1
```

Per part: the controller gives cycles per class (`command` = control
overhead, kept apart from `wait`), the command and read counts, the poll
count and every wait with its first cycle, the block's `done_cycle` and its
last cycle; an accelerator gives cycles per class, utilisation
(`busy / total`, the share of cycles its datapath is occupied: a firing and
the II gap after it, D59), beats per port and `firings` (equal to `busy`
only when `ii` = 1); a streamer gives
cycles per class, its xbar ports and its FIFO occupancy (max, time-weighted
mean and a histogram per lane, D40); a DMA gives cycles per class, beats and
bytes each way and the peak buffer; `banks` gives per-bank reads, writes,
grants, conflicts, stalls and cycles blocked by a wider grant; `ports` gives
per xbar port its owner, width, grants, stalls and the stalls caused by a
wider grant. `functional_check` is null for a scenario run on its own; a run
of the flow (E2E1, D90, `pixi run flow`) fills it: `kernel`, `seed`,
`symbols`, `passed`, and per `inout` container of the kernel the `memory` it
was read back from (L2 when it has an L2 layout), its `elements`, whether it
equals the kernel's `reference` and the reference executor (`ref1`) on the
same inputs, and the number of `mismatches` against the reference.

Cycle classes, one per cycle per component, in this order:

| Component | Classes |
|---|---|
| accelerator | `busy` (a firing) > `stall_out` (frozen on a full output) > `busy` (II gap of a running task, D59) > `stall_in` (a due input empty) > `idle` |
| streamer | `busy` (a port granted) > `stall_xbar` (requested, none granted) > `stall_fifo` (an address but no credit or no data) > `idle` |
| DMA | `busy` (a beat moved) > `stall_l1` (an L1 request refused) > `idle` (bandwidth gap) > `stall_mem` (waiting for memory) > `idle` |
| controller | `command` (control overhead) / `wait` / `idle` |

**Trace** (`trace.py`): an independent event log, written from each cycle's
final wires in commit. Levels: `off`; `task` = controller commands, block
starts and dones, and the class intervals; `beat` = adds xbar grants and
stalls (these *are* the L1 accesses: there is no separate access event),
read responses, accelerator firings, DMA beats, controller polls and FIFO
count changes.
Every event is a flat dict of `t`, `k`, `src` and the kind's own fields:

<!-- snippet: snax_forge/snax_model/trace.py -->
```
    task  cmd       pc, op, reg | block, value, mode, done, last  (t = first cycle)
    task  start     (t = cycle the start landed; busy from t + 1)
    task  done      (t = done_cycle: first cycle busy reads 0)
    beat  grant     port, mem, w, addr, banks, row
    beat  stall     port, mem, w, addr, banks, row, wider
    beat  resp      mem, addr, then port, banks, row (L1) | i (L2)  (t = data returns)
    beat  fire      n (firing index)
    beat  dma_beat  side (src/dst), i (beat index), mem (l1/l2), addr
    beat  poll      block, value (busy as sampled)
    beat  fifo      lane, count (t = first cycle the count holds)
```

An action event carries the cycle it happens in; a state change (`done`,
`fifo`) the first cycle its new state is visible. A read shows up twice (D62):
its `grant` in the cycle the request is accepted, and its `resp` in the
cycle the data returns (`grant` + `L1Config.read_latency`, from the xbar
with the same `port`, `addr`, `banks` and `row`). An L2 read is a DMA
`dma_beat` with `side: src`, `mem: l2`, and a `resp` from the DMA with the
same `i` and `addr`, `L2Config.read_latency` later. Writes have no `resp`.
A reader's FIFO count rises the cycle after the `resp` (`Queue`, D32). Inside a cycle the order is
state changes, then phase, then source in registration order, then emission
order — identical with skipping on and off.

<!-- snippet: run:reduce/trace.jsonl -->
```json
{"t": 0, "k": "cmd", "src": "ctl", "pc": 0, "op": "csr_write", "last": 0, "reg": "acc_in.base", "value": 0}
{"t": 14, "k": "fifo", "src": "acc_in.fifo", "lane": 0, "count": 1}
```

**Class intervals** are not events: `on_gap` reports a skipped range only
when the component wakes, after other components' later events. They are
kept per component and written to `trace_meta.json` as half-open runs
`[class, start, stop)` covering `[0, total)`:

<!-- snippet: run:reduce/trace_meta.json -->
```json
  "acc_in": [["idle", 0, 12], ["busy", 12, 28], ["idle", 28, 35]],
```

Two things need the intervals rather than the totals, so they need at least a
task-level trace: the anchor report's overlap of accelerator-active phases
with control overhead (section 7 of ARCHITECTURE.md, deferred) and FIFO occupancy
over the owner's busy window (D56, shown by the viewer since VIS1). The
first is open item 11; the second closed it for the views.

**Filter** (D49). A beat-level run writes roughly a kilobyte per cycle, so a
long run needs `--trace-source NAME` (repeatable) or `--trace-window A:B`.
These drop beat events only; the task-level skeleton and every profile number
stay complete, and `trace_meta.json` records what was filtered.

## 8. Rules for a new block kind

A component that sleeps wrongly does not crash — it reports wrong
statistics. These are the rules that keep skipping honest (D29, D40, D47);
`tests/snax_model/test_gaps.py` checks them.

**R1. `next_wake` is a pure answer about committed state.** It is asked only
after a commit, it may be asked several times in the same cycle, and other
components ask it too (the xbar asks its port owners, a reader asks its
FIFO's consumer). Every call in a cycle must give the same answer. The only
state it may change is a memo keyed by that cycle — today the accelerator's
recorded gap class.

**R2. A shared element changes only in a cycle in which the component
changing it is ticked.** That is what makes it safe for a component to
answer `None` while waiting for someone else: the scheduler asks *every*
component again after every simulated cycle, so a change that happened is
seen before the next one can. A new shared element (a FIFO, a memory, a
queue) must keep this property, or a sleeping reader of it will miss a
change.

**R3. Wake at every cycle in which your own cycle class can change, and
classify a whole gap with the class of its first cycle.** Two ways are in
use: recompute the class from committed state in `on_gap` (streamer, DMA),
or record it when the gap begins and end the gap with a tick as soon as it
would change (accelerator). Changes driven by time alone — `II`,
`beat_interval`, a startup delay — are wakes too.

**R4. The class runs cover `[0, total)` exactly.** `build_profile` checks
this for every classified component on every run and raises otherwise;
`ClassLog` additionally checks that the runs are contiguous while a trace
records them.

**R5. A start before the run (`start(arg, cycle=None)`) is setup only.** It
behaves as a start in cycle -1 and is not traced (open item 14). Do not use
it once a trace is bound.

A block kind also needs a register adapter (`register_adapter`, D36): it
lists the configuration registers in offset order from the component's own
config, decodes their values into the component's start argument, and
encodes an argument back. The register *names* are the contract (section 5),
so pick them as SNAX-LOWER's C backend will want to read them. To appear in
task lists (section 9) it also needs its `values` form, registered under the
adapter's kind with `register_values` in `snax_forge/lower/values.py`. What
must hold for it to fit a platform and a graph is a design check,
registered with `register_check` (section 14).

## 9. Task list

SNAX-LOWER's ordered list of tasks (LOW1b, D45, D64): which component runs
which task with which values, where each task is configured and started,
and what it waits for. `lower_program(tasks, cluster)` turns it into the
program of section 5; the model never reads a task list. LOW1a produces it
from a design point (D89, `snax_forge/lower/derive.py`, `pixi run lower tasks
DESIGN_POINT`; the rules below); it can still be written by hand (D63). Every
scenario has one as `tasks.json`, the hand-written source its `scenario.py`
lowers into the program of its generated `scenario.json` (D65, D66, D67).

A task list is `name` and `steps`, each step with an `op`:

| `op` | Fields | Becomes |
|---|---|---|
| `configure` | `task_name`, `type`, `component`, `after`, `wait_mode`, `values` | every configuration register of the component, written here |
| `start` | `tasks` | the waits these tasks need, then their `start` writes in list order |
| `sync` | `task`, `mode` | one `wait` on the task's component, always emitted |
| `read` | `reg` | one `csr_read` of `block.register` |

`type` is the component's adapter kind and says how `values` are read:

```
streamer  base, temporal_bounds, temporal_strides, spatial_strides (bytes);
          spatial bounds are design-time and come from the cluster file
dma       direction ("l2_to_l1" or "l1_to_l2"); src and dst, each with base,
          bounds and strides (bytes, one wide beat per step)
accel     its start parameters by name: n, then any named rate (e.g. T)
```

**Names** (D75, D83). A streamer is named after the accelerator port it
serves, `<instance>_<port>` (`acc_a`, `acc_out`). A task is
`<node>_<component>` (`add_acc_a`, `add_acc`), a DMA task
`load_<container>` or `store_<container>` (`load_A`, `store_C`), with node
and container names as the imported graph has them, and `_<k>` appended for
tile `k` (`add_acc_a_2`). Scenarios without a kernel (fmul, reduce, dma)
name their containers `A`, `B`, `C` as vecadd does.

Only the loops a task uses are given; the adapter pads the rest with bound
1 and stride 0. `after` names earlier tasks this one needs finished before
it starts: data it reads, and a buffer it overwrites. `wait_mode` (`poll`
or `signal`) is used whenever the lowering waits for this task; a `sync`
gives its own `mode`. Missing `after` and `wait_mode` default to `[]` and
`poll`; every field is written back, and unknown keys are errors.

**Structural rules**, checked when a task list is made: task names are
unique; `after`, `start` and `sync` refer to tasks configured earlier; a
component has at most one configured task that has not started, because its
registers are buffered (section 5); a task starts once, after its `after`
tasks have started; every configured task is started.

**Waits.** Before a `start`, the lowering adds one wait per component, for
every `after` task and every listed component whose latest task is not yet
covered, leaves out a wait that another of these waits covers (D66), and
orders the rest by when the task each one ends on was started. A wait is on a component, so it ends on that component's latest
task and covers every earlier one. A wait on a writer streamer also covers
the accelerator attached to it and that accelerator's reader streamers when
the same start launched them: their data flows into the writer, so it
finishes last. Nothing else is added; where configures, starts and syncs
go, and so how programming overlaps running blocks, is the task list's.

**Derived from a design point** (LOW1a, D89). The accelerated nodes run in
execution order, one group each; every map around a node is one of its
temporal firing loops (a single tile). A group is: `load_<C>` (configure,
start) for each input, in BRM port order, whose container has an L2 and an
L1 layout and is not in L1 yet; `<node>_<instance>_<port>` for every port,
the streamer's values from the memlet through the L1 layout; `<node>_<instance>`
with `n` = the firing count; one `start` of the streamer tasks and then the
accelerator task; `store_<C>` after the writer's task for each output
written for the last time. At the end one `sync` per non-transient
container the graph writes, on its store (or its writer without an L2).
`after` holds data dependences only: a reader waits for the task that last
put its container in L1, a writer for the readers and writer since. Every
configure and sync uses the platform's `wait_mode`. Streamer values, with
firing loops `v_k` (begin `b_k`, `count_k`, step `s_k`) and a subset
dimension `d` whose begin is `c_d + sum_k a_dk v_k`: `base` = layout base +
sum_d (c_d + sum_k a_dk b_k) * stride_d; temporal loop k (innermost first):
bound `count_k`, stride sum_d a_dk s_k stride_d; one spatial loop per range
dimension (the last first): its length and step * stride_d. The spatial
bounds must be the streamer's and the addresses must be those the BRM's
nest gives through the same layout. vecadd's design point with B and C
pinned to 576 and 1152 gives `scenarios/vecadd/tasks.json` exactly.

The end of `vecadd`'s task list: the adder's four tasks are configured one
by one (the last one is shown), started together, and the store waits for
the writer.

<!-- snippet: scenarios/vecadd/tasks.json -->
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
    "src": {"base": 1152, "bounds": [8], "strides": [64]},
    "dst": {"base": 1024, "bounds": [8], "strides": [64]}
   }
  },
  {"op": "start", "tasks": ["store_C"]},
  {"op": "sync", "task": "store_C", "mode": "poll"}
```

From the first of the adder's configures on, this lowers to the 13
configuration writes of `acc_a`, `acc_b`, `acc_out` and `acc`, `wait dma` (for
the loads in `add_acc_a`'s and `add_acc_b`'s `after`), the four starts, the
DMA's 11 configuration writes, `wait acc_out`, the DMA's start and a final `wait dma`: the
last 32 of the 57 commands of `scenarios/vecadd/scenario.json`.

## 10. Block runtime model

What SNAX-BRM hands the rest of the flow (BRM1–BRM3, D68, D70): one
accelerator as plain data. A BRM is a hand-written JSON file in
`snax_forge/brm/library/`, one per accelerator, named after it and kept in
the form `Brm.to_json` writes (every field, defaults included). Loaded with
`load_brm(name)`; unknown keys are errors and a missing part is rejected by
name.

| Part | Holds | Read by |
|---|---|---|
| `interface` | params (`design` / `runtime`) and ports (direction, lanes, rate, dtype) | everything below |
| `function` | a registered accelerator kind (D43), its factory params without timing, and `code`: what one lane computes (D82) | the accelerator entry; `bind` and the accelerated node (`code`) |
| `dataflow` | a notation and one nest per port, in logical indices | the streamer values |
| `pattern` | `family` (a matcher registered in the sandbox), its `attrs`; `predicate` null | SNAX-SANDBOX's `bind` (section 12) |
| `implementations` | per name: `source` (`chisel` only), `supports`, `timing`, `binding` (null until M10) | the accelerator entry, the HW generator (later) |

The shared part (the first four) is what every implementation has in
common; designs that differ only in timing, supported values or RTL source
are implementations of one BRM.

**Params.** A `design` param is fixed per instance and ends up in the
cluster file (lanes, the op); its `default` is used when the design point
gives none, and `values` limits what any implementation may take. An
implementation's `supports` limits it further for that RTL. A `runtime`
param is a start parameter of the accelerator: the runtime params are
exactly `n` and every named port rate, and they are the registers of
section 4 and the accelerator task's `values` in section 9. A value field
holds an int or an expression over params (`+ - * //`); a fixed string is
a design param with one allowed value.

**The accelerator entry** of an instance (`brm.resolve(implementation,
params)`, then `accel_entry()`) is the function's params resolved, in their
order, then the implementation's `latency` and `ii` (its
`initiation_interval`). The first library BRM resolves to exactly the
accelerator of `scenarios/clusters/alu4.json`:

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
against it, and the accelerated node carries it. It is null for a BRM
that cannot say it per lane yet (a reduction, DFG3 / BRM4), which cannot
be bound. The library tests check it against the registered kind.

<!-- snippet: snax_forge/brm/library/elementwise_add.json -->
```json
  "chisel_tiled_spatial": {
   "source": "chisel",
   "supports": {},
   "timing": {"latency": 0, "initiation_interval": 1},
   "binding": null
  }
```

`resolve` builds the `AccelConfig` through the kind and rejects a BRM whose
declared ports (name, direction, lanes, rate, in order) or timing differ
from it.

**The affine nest** of a port lists the operand's `shape`, an `offset` and
`loops`, outermost first; each loop has a `bound`, one stride per operand
dimension and a `spatial` flag, and the element of a step is
`offset + sum(i_l * strides_l)`. Temporal loops come first (one step of
them is one beat, the last one innermost), spatial loops last (the lanes,
the last one fastest). Spatial bounds use design params only and multiply
to the port's lanes; per task the nest gives n / rate beats inside the
shape. The nest says nothing about addresses: section 3 gives how
SNAX-LOWER maps it through a layout onto streamer registers.

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

The workload as SNAX-FORGE's own dataflow graph (DFG1, D71, D77): one
`.snaxdfg` JSON file, written by an importer (IMP1) and rewritten by each
sandbox transform (SBX1). Loaded with `Graph.load(path)`; every field is
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
the BRM is checked where the BRM is loaded: by `bind` (SBX1) and the
reference executor (REF1).

## 12. Recipe

SNAX-SANDBOX's ordered list of transforms (SBX1, D72, D80): what a thinker
edits to change the mapping of a kernel. A recipe is a hand-written JSON
file in `recipes/`, loaded with `Recipe.load(path)`; every field is written,
`params`, `symbols` and `steps` default to empty, unknown keys are errors.

| Field | Holds |
|---|---|
| `name` | the recipe; its steps are written to `out/sandbox/<name>/` |
| `kernel` | the kernel whose import (IMP1) the recipe starts from |
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
| `bind` | `node` (tasklet id), `brm`, `implementation`, `instance`, `params` (optional design params) | the tasklet and its spatial map become an accelerated node; the lanes param comes from the spatial bound; the tasklet's code on the port names must be the BRM's `function.code` |
| `unbind` | `node` (accelerated id) | the accelerated node becomes the subtree it `replaced` |
| `join_map` | `map` (id) | a temporal map with `loop.split` and its spatial map become the map they were split from |

The vecadd recipe: W lanes, one temporal map of N / W beats, bound to
`elementwise_add`. `recipes/vecadd_undo.json` (`unbind`, then `join_map`)
takes its last step back to the first.

<!-- snippet: recipes/vecadd.json -->
```json
 "params": {"W": 4},
 "symbols": {"N": 64},
 "steps": [
  {"transform": "split_map", "params": {"map": "add_map", "factor": "W"}},
```

## 13. Platform

The SNAX cluster around the accelerators (DP1a, D84): everything of the
cluster file that is not the accelerator. A platform file lives in
`platforms/` (`small16.json`, alu4's platform); it never names a recipe,
and a recipe never names it. `Platform.load(path)`; every field is written,
missing keys take their defaults (`l2` missing: no L2), unknown keys and
values of the wrong type are errors, and loading reports every problem at
once (section 14).

| Field | Holds |
|---|---|
| `name` | the platform (an identifier) |
| `base` | the platform this one was made from, or null for an original |
| `changes` | what differs from `base`: `--set` path -> value, in the order set |
| `l1`, `l2` | `L1Config`, `L2Config` (section 2); `l2` null for an L1-only cluster |
| `xbar` | `{"check_hold": bool}` |
| `dma` | `DmaConfig` (section 2), used when there is an L2 |
| `controller` | `ControllerConfig` (section 2) |
| `register_window` | registers per block, a power of two |
| `wait_mode` | `poll` or `signal`: how the controller waits for a task |
| `streamers` | the streamer shell: `default`, then an entry per streamer it changes |

**The streamer shell.** One streamer per port of every accelerated node's
instance, named `<instance>_<port>` (D75). The port decides `write` (its
direction is `out`), `n_ports` (its lanes) and, unless an entry gives them,
`spatial_bounds` (the port nest's spatial loops, fastest first); an entry
may not set `write` or `n_ports`. `default` holds `temporal_dims`,
`fifo_depth`, `addr_depth` and `prio`; an entry named after a streamer
overrides any of them, and `spatial_bounds`, for that streamer only.

<!-- snippet: platforms/small16.json -->
```json
 "register_window": 32,
 "wait_mode": "poll",
 "streamers": {"default": {"temporal_dims": 1, "fifo_depth": 2, "addr_depth": 8, "prio": 0}}
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

## 14. Design checks

Before the design step writes anything it runs every registered check on
the pairing of a bound graph and a platform (DP1a, D85) and reports every
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
| `platform.values` | a value its section rejects (`n_banks` 0, `dims` 0, a `wait_mode` or `register_window` that is not allowed) |
| `platform.streamer` | a streamer value below 1, spatial bounds that are not a non-empty list of ints >= 1 |
| `platform.l2` | L2 and L1 disagree on beat width (`beat_bits` vs `wide_bits`), word width, dtype or elements per word |
| `platform.banks` | with an L2, `n_banks` is not a multiple of the DMA's port group (`wide_bits / width_bits`) |
| `graph.load` | a graph file that does not load (section 11) |
| `graph.symbols` | a symbol still null: bind it in the recipe |
| `graph.unbound` | a tasklet or a map holding nothing to run outside every accelerated node, or no accelerated node: SNAX-MODEL has no core |
| `graph.brm` | an unknown BRM or implementation, design params that do not resolve, connectors that are not the BRM's ports, code that is not the BRM's |
| `graph.instance` | an instance named `xbar`, `dma` or `ctl` or after a streamer, or two streamers with one name |
| `connect.streamer_key` | a streamer entry that matches no port of the graph (the fix names the closest streamer) |
| `connect.derived` | a streamer entry that sets `write` or `n_ports` |
| `connect.lanes` | an entry's `spatial_bounds` that do not give the port's lanes or are not its nest's spatial loops |
| `connect.dtype` | a container whose dtype is not the L1's or not its port's, or an L1 with more than one element per word (open item 21) |
| `connect.temporal` | a map around an accelerated node that is not `temporal` (tiles are not lowered yet), a subset that is not affine in the firing loops, or a streamer with fewer `temporal_dims` than the node's firing loops (the fix names the count) |
| `connect.regmap` | a streamer, accelerator or DMA whose status and configuration registers do not fit `register_window` |
| `memory.pin` | a memory working copy that does not load; a `--set memory.` path that is not `<container>.<memory>.base` or `passes.<pass>`; a pin on an unknown container (the fix names the closest) or on a memory the container does not live in; an unknown pass. The plan is made after these pass |
| `memory.residency` | a layout in a memory the platform lacks or for a container the graph lacks; a non-transient container that lives nowhere; an accelerator operand without an L1 layout |
| `memory.layout` | a layout whose shape is not its container's |
| `memory.align` | a base or stride that is not a multiple of the word; a container the DMA moves (in L2 and L1) that is not contiguous, not whole wide beats (open item 38) or not on a beat |
| `memory.fit` | a layout that leaves its memory (the fix names the rows or L2 size that would hold it) |
| `memory.overlap` | two layouts in one memory that share bytes, naming the pins (the fix moves the latest pin past everything) |
| `point.keys` | a design point with unknown or missing keys or a malformed streamer entry |
| `point.streamers` | a design point whose `streamers` are not the platform's shell for its graph |

The model's own checks, when a cluster file is built, stay as a backstop.

## 15. Memory plan and design point

**Memory plan** (DP1b, D86, `snax_forge/design/memory.py`). A layout per
container and memory it lives in: `base` (byte address of index 0),
`shape` and one byte stride per dimension, the form of
`snax_forge/lower/layout.py`. Banks follow from addresses (word `i` in bank
`i % n_banks`, open item 18) and are not stated. Three passes make it, each
registered by name (`register_memory_pass(kind, name, fn)`):

| Pass | Decides | Default |
|---|---|---|
| `residency` | the memories of each container | `default`: not transient: L2 and L1 with an L2, else L1; transient: L1 |
| `layout` | shape and strides, base 0 | `contiguous`: row-major, one element per word |
| `placement` | the base per container and memory | `contiguous`: graph container order from the memory's first byte, no gaps; a layout in both L2 and L1 on a wide beat, others on a word; around the pins |

`--set memory.<container>.<memory>.base=N` pins a base; `--set
memory.passes.<pass>=NAME` picks a pass. Passes read a `MemoryContext`:
the graph with its symbols bound, the platform, shapes, words, memory
ranges and, for policies that avoid bank conflicts, each accelerator
port's element-index stream (`accesses`, beats x lanes x dimensions, from
its memlet over the maps around it) grouped by the node that runs them
together. The design step writes `out/design/<name>/memory.json`:
`passes`, `changes` (every memory path set, in order) and `layouts`;
`--memory` continues from it.

**Design point** (D74, D87, `snax_forge/design/point.py`). Written by the
design step to `out/design/<name>/design_point.json` once every check has
passed; not edited by hand. It is what SNAX-LOWER reads.

| Field | Holds |
|---|---|
| `name` | the design (the graph's sandbox folder, else its stem) |
| `graph_from` | the `.snaxdfg` it was made from |
| `graph` | that graph as read: instances, `code`, `replaced`, `loop.split` |
| `platform` | the platform working copy (section 13) |
| `streamers` | the resolved shell: `<instance>_<port>` -> `instance`, `port`, `write`, `n_ports`, `spatial_bounds`, `temporal_dims`, `fifo_depth`, `addr_depth`, `prio` |
| `memory` | the memory plan: `passes`, `changes`, `layouts` (container -> memory -> layout) |

There is no recipe in it: the graph holds every instance, and the sandbox
folder keeps the recipe. Loading one runs every check again on what it
holds (the stored layouts as they are) and requires `streamers` to be the
platform's shell for the graph.
