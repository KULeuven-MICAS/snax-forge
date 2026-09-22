# SNAX-MODEL Contracts (MOD10)

> What the artefacts around SNAX-MODEL *mean*. MOD9 fixed their shape; this
> file says what every field is, so SNAX-BRM, SNAX-DSE and SNAX-LOWER can
> produce them without reading `scenario.py`.
>
> Scope: the model side only (D26). The design point, the BRM and the
> task list are not here — they are M3's, and their open items stay open
> (BRM affine-nest notation: open item 1, M6; task-list format: open item
> 19, LOW1).
>
> Until the M6 freeze these are plain dataclasses and plain JSON; versioned
> schemas are F2's job (D26, F2). Every value marked **default** is a
> declared platform default, not a measurement (D51); nothing else about the
> format depends on it.
>
> Every fenced block below is copied verbatim out of a checked-in file or
> out of the output of `scenarios/reduce`, and
> `tests/snax_model/test_contracts.py` fails if one of them drifts. The
> `<!-- snippet: ... -->` line above each block names its source;
> `run:` means "produced by running that scenario".
>
> Layout: section 8 holds the rules a new block kind must follow. The
> reasoning behind each contract is in the module docstrings and in
> `docs/ARCHITECTURE.md` section 5.6; this file does not repeat it.

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
are design knobs for SNAX-DSE, not measurements. SNAX-LOWER will derive the
whole file from a design point; until then `scenarios/make.py` writes it.

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
   "attach": {"in": "ra", "out": "wr"}
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

A reader with `tstride[0] == 0` is rejected: the RTL repeats the beat
instead of re-reading (open item 5).

**Mapping a BRM's per-port affine nest onto these registers** is exactly
this: the nest's parallel innermost level becomes the spatial loops (its trip
count must equal the streamer's `n_ports`), every enclosing level becomes a
temporal loop with the same bound and its stride in bytes, innermost first,
and the nest's first element address becomes `base`. The nest notation
itself is open (open item 1, closed in M6); this register side is not.

<!-- snippet: scenarios/vecadd/scenario.json -->
```json
  {"op": "csr_write", "reg": "ra.base", "value": 0},
  {"op": "csr_write", "reg": "ra.tbound[0]", "value": 16},
  {"op": "csr_write", "reg": "ra.tstride[0]", "value": 32},
  {"op": "csr_write", "reg": "ra.sstride[0]", "value": 8},
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
  {"op": "csr_write", "reg": "ra.start", "value": 1},
  {"op": "csr_write", "reg": "wr.start", "value": 1},
  {"op": "csr_write", "reg": "acc.start", "value": 1},
  {"op": "wait", "block": "wr", "mode": "signal"},
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
 "memory": [{"mem": "l2", "addr": 0, "npy": "a.npy"}, {"mem": "l2", "addr": 1024, "npy": "b.npy"}],
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
(`busy / total`, which is the firing rate — a busy cycle is a firing cycle,
so there is no separate firing count) and beats per port; a streamer gives
cycles per class, its xbar ports and its FIFO occupancy (max, time-weighted
mean and a histogram per lane, D40); a DMA gives cycles per class, beats and
bytes each way and the peak buffer; `banks` gives per-bank reads, writes,
grants, conflicts, stalls and cycles blocked by a wider grant; `ports` gives
per xbar port its owner, width, grants, stalls and the stalls caused by a
wider grant. `functional_check` stays empty until the reference executor
exists (E2E1, open item 13).

Cycle classes, one per cycle per component, in this order:

| Component | Classes |
|---|---|
| accelerator | `busy` (a firing) > `stall_out` (frozen on a full output) > `idle` (II gap) > `stall_in` (a due input empty) > `idle` |
| streamer | `busy` (a port granted) > `stall_xbar` (requested, none granted) > `stall_fifo` (an address but no credit or no data) > `idle` |
| DMA | `busy` (a beat moved) > `stall_l1` (an L1 request refused) > `idle` (bandwidth gap) > `stall_mem` (waiting for memory) > `idle` |
| controller | `command` (control overhead) / `wait` / `idle` |

**Trace** (`trace.py`): an independent event log, written from each cycle's
final wires in commit. Levels: `off`; `task` = controller commands, block
starts and dones, and the class intervals; `beat` = adds xbar grants and
stalls (these *are* the L1 accesses: there is no separate access event),
accelerator firings, DMA beats, controller polls and FIFO count changes.
Every event is a flat dict of `t`, `k`, `src` and the kind's own fields:

<!-- snippet: snax_forge/snax_model/trace.py -->
```
    task  cmd       pc, op, reg | block, value, mode, done, last  (t = first cycle)
    task  start     (t = cycle the start landed; busy from t + 1)
    task  done      (t = done_cycle: first cycle busy reads 0)
    beat  grant     port, mem, w, addr, banks, row
    beat  stall     port, mem, w, addr, banks, row, wider
    beat  fire      n (firing index)
    beat  dma_beat  side (src/dst), i (beat index), mem (l1/l2), addr
    beat  poll      block, value (busy as sampled)
    beat  fifo      lane, count (t = first cycle the count holds)
```

An action event carries the cycle it happens in; a state change (`done`,
`fifo`) the first cycle its new state is visible. Inside a cycle the order is
state changes, then phase, then source in registration order, then emission
order — identical with skipping on and off.

<!-- snippet: run:reduce/trace.jsonl -->
```json
{"t": 0, "k": "cmd", "src": "ctl", "pc": 0, "op": "csr_write", "last": 0, "reg": "ra.base", "value": 0}
{"t": 14, "k": "fifo", "src": "ra.fifo", "lane": 0, "count": 1}
```

**Class intervals** are not events: `on_gap` reports a skipped range only
when the component wakes, after other components' later events. They are
kept per component and written to `trace_meta.json` as half-open runs
`[class, start, stop)` covering `[0, total)`:

<!-- snippet: run:reduce/trace_meta.json -->
```json
  "ra": [["idle", 0, 12], ["busy", 12, 28], ["idle", 28, 37]],
```

Two things need the intervals rather than the totals, so they need at least a
task-level trace: the anchor report's overlap of accelerator-active phases
with control overhead (section 7 of ARCHITECTURE.md, deferred) and FIFO occupancy
over the owner's busy window (VIS3). Both are open item 11.

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
so pick them as SNAX-LOWER's C backend will want to read them.
