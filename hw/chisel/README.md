# SNAX-FORGE Chisel datapaths

Hardware side of SNAX-FORGE: a hand-written library of datapaths and the group control unit, their chiseltest specs, and `Emit`, which elaborates a catalogue of configurations to SystemVerilog and a design point's accelerator from its `accelerator.json` (C§17). It is a project of its own: nothing here imports `snax_forge/`, and the Python side calls it only through `pixi run gen-rtl`. The datapaths are the reference implementations of the BRMs (`simd` names `SimdLanes`, `accumulate` `TreeAccumulate`, `lane_accumulate` `LaneAccumulate`).

## Why these versions

Scala 2.13.14, Chisel 6.4.0, chiseltest 6.0.0 — identical to `snax_cluster/hw/chisel`. Generated datapaths are meant to sit inside a SNAX shell next to the hand-written streamer, so a Chisel version skew between the two trees buys nothing and costs a bad afternoon.

The JVM comes from pixi (`sbt` on conda-forge drags in an openjdk). Chisel and FIRRTL jars are *not* conda packages: sbt resolves them from Maven Central on first build, and Chisel 6 additionally downloads the `firtool` binary through `firtool-resolver`. Both land in the Coursier cache under `~/.cache/coursier`, so the first `chisel-test` takes minutes and later ones take seconds. That is expected, not a hang.

## Layout

```
hw/chisel/
├── build.sbt
├── project/{build.properties,plugins.sbt}
├── .scalafmt.conf                      # copied verbatim from snax_cluster
├── blocks.json                         # the datapath catalogue (D130, D147)
└── src/
    ├── main/scala/snax/forge/
    │   ├── Emit.scala                  # elaboration entry point and catalogue; --describe, --spec, --block
    │   ├── Datapaths.scala             # the datapath registry (blocks.json)
    │   ├── IntType.scala               # NumPy's integer types as Chisel lanes (D143)
    │   ├── ctrl/                       # GroupCtrl and its control model GroupRule
    │   ├── datapath/                   # SimdLanes, TreeAccumulate, Mac, SerialAccumulate, LaneAccumulate, LinkBuffer
    │   ├── shell/                      # accelerator.json, <tag>_csr, _ctrl, _datapath, the wrapper
    │   ├── RepoPaths.scala             # out/hw and repository paths
    │   ├── elementwise/                # the typed ALU lane, the ops
    │   └── stream/                     # FifoSlicer, FifoPacker (parked, D142)
    └── test/
        ├── scala/snax/forge/           # one spec per module, plus a model; shell/ for the accelerator
        └── resources/
            └── rtl/<design>/           # the fixtures' vectors.json (accelerator.json from forge/)
```

## A design point's accelerator (D130)

The boundary with the Python side is two files. `blocks.json`, checked in, is the datapath catalogue: per datapath its params with types and limits, its ports, latency, span, register bits (`flops`), pins and the rule that fills the params from the unit's ports (C§17); the registry `Datapaths` holds the same and builds them, and `DatapathsSpec` checks `Emit --describe` equal to it. `accelerator.json`, per design point, is written by `snax_forge/gen/` and checked by `gen/rtl.py` against `blocks.json`; `Emit --spec accelerator.json [--out DIR]` reads the file's accelerators ⊃ groups ⊃ units as one shell, the groups in CSR order across the accelerators and the outer ports set after set (`AcceleratorSpec`, D157), and builds `<tag>_shell_wrapper` (`shell/`, the tag the entry's `<accelerator_slot>_<design>`) into `DIR/sv/`, one file per module, with `DIR/emitted.json`:

- `<tag>_csr` (`AccCsr`): the file's CSRs by offset, snax_cluster's ReqRspManager protocol (D130) as the core reaches it after `snax_csr_mux_demux` with snax_cluster's manager off; every CSR at its width (D152); START a strobe that reads 0, a write with bit 0 set its group's start, a bare strobe to its control unit; per group JOBS and each JOB_BEATS in unit order as wires to its control unit, which reads them live with no copy (D168); a write to a busy group's read-write CSRs (START, JOBS, each JOB_BEATS, each scalar) held until E + 1, so they stay constant through the run, while another group's writes and every read go through; per group STATUS's sticky done from its control unit's busy and done pulse, and BUSY_COUNTER (saturating) in its submodule `<tag>_peripherals` (`AccPeripherals`): the interface; a unit's scalar CSRs (`K1` after its JOB_BEATS, or `K1_LO` and `K1_HI` at 64 bits, each with `pin` k1 in `accelerator.json`) are registers like JOBS, and drive the datapath's pin through the wrapper (the high word over the low);
- `<tag>_ctrl` (`AccCtrl`): per group `<tag>_<group>_ctrl` (`GroupCtrl`), generated from the group's units and links: the start, the empty run, busy, the done pulse and the done pin around its control model `GroupRule`, built inside the module with each unit's signals named after the unit. `GroupRule` is snax-gvsoc's SnaxGroupRule (D147, C§17) over the group's edges, each link `{from, to, to_input, depth}`, the units in topological order (a fork, a join, a cycle or two chains refused until tier A), with nested counters and no product (D153): a register only for what a firing changes, JOBS and each JOB_BEATS read live from the csr with no copy (D168), a test at its last value as d + 1 = T sharing the counter's increment (a fixed T: d = T − 1), a unit at L 0 on a wire borrowing its consumer's test, any other counting its results in mixed radix over the Ts after it and JOBS, no count of the sink's pushes;
- `<tag>_datapath` (`AccDatapath`): per unit its datapath in a module `<tag>_<unit>`, per buffer a `LinkBuffer` in a module `<tag>_<link>`, wired by the links (a wire, or through the buffer); data and the control units' pins only.

Only the outer ports reach the wrapper, which splits each Decoupled port: its data to its unit's datapath, its valid and ready to its group's control unit. `ShellEmitSpec` checks the emitted hierarchy against `bundle.json`'s `modules` (C§18) on every fixture. `pixi run gen-rtl DESIGN_POINT` does the check and the elaboration (`docs/RUN.md` section 7), running `Emit` with java on hw/chisel's runtime classpath, which sbt exports into `target/emit-classpath.txt` on the first run and again only after a change under `src/main/` or to `build.sbt`. The fixtures in `src/test/resources/rtl/<design>/` hold `vectors.json`; their `accelerator.json` and streamer hjsons (`<tag>_streamer_<set>.hjson`) are read from snax-gvsoc's `forge/<design>/` (the submodule; a spec that needs one cancels without it), but for vsum_tree_w8 and dot_5_w1 (dot 5 at W = 1: SerialAccumulate at lanes 1), which `forge/` does not hold and which keep the two beside their vectors. The Python side writes and checks them byte for byte (`tests/gen/test_rtl.py`); refresh one with

    pixi run sandbox recipes/reduce/vsum_tree.json --set W=8 --out out/sandbox/vsum_tree_w8
    pixi run design out/sandbox/vsum_tree_w8/2_bind.snaxdfg --platform platforms/snax_forge.json --out out/design/vsum_tree_w8
    pixi run gen-rtl out/design/vsum_tree_w8/design_point.json --fixture hw/chisel/src/test/resources/rtl/vsum_tree_w8

(the others from their recipe as it is, named after it; dot_2 to dot_6 with `--set N=16`, matvec_4 to matvec_6 with `--set M=2 --set N=16`). The specs on them: `ShellEmitSpec` (one emit each: the wrapper's ports are the template's for its streamer hjson, plus the group pins; the hierarchy is `bundle.json`'s; `emitted.json` lists every file), `ShellCasesSpec` (vecadd's and vsum's, through snax-gvsoc's shell cases, below), `ShellEndToEndSpec` (the CSR writes of `main.c`, an always-ready streamer: the goldens and the counter of `vectors.json`; each linked variant, dot 2–6 and matvec 4–6, at N 8 and 16 on random data against NumPy's results and the cycles of snax-gvsoc's matching group case, and matvec 4–6 at int32 through the wrapping product q = (−2147478991, 4)), `ShellRandomSpec` and `CtrlRegistersSpec` (one emit each: every `<tag>_<group>_ctrl`'s registers, by name and bits, are the fixture's in `ctrl_registers.json` beside them, `snax_forge/estimate/bits.py`'s `ctrl_registers` on its `accelerator.json`, which `pixi run python util/ctrl_regs.py --json` writes and `tests/estimate/test_bits.py` checks is current; firtool's `io_done_REG` is the estimate's `done`, its `<u>_v_0` … one `<u>_v`). `TwoGroupsSpec` runs each of dot 1's groups (`mul` and `sum`, `forge/dot_1/`) on its own.

One catalogue block alone (CHAR1, C§17): `Emit --block NAME --params JSON --out DIR` checks the params (`Datapath.check`) and builds `make(params)` as the top module with an asynchronous reset, as inside the wrapper, into `DIR/sv/` with `DIR/emitted.json`; `pixi run gen-block LIST.json` runs a list of them in one java call. The reset comes from `Module.ResetType`: `DatapathModule` takes its `resetType` from `DatapathModule.asyncTop`, which `emitBlock` sets around the construction, so no block, no builder and no wrapper changes, and every other construction keeps Chisel's default. `BlockEmitSpec` emits a block of every datapath (Mac in its three forms, SerialAccumulate at lanes 1 and 8, LinkBuffer at depth 1 and 2) and checks its ports against `blocks.json`'s live ports and pins (a pin of no bits is absent: LinkBuffer's slots at depth 1, SerialAccumulate's `lane` at lanes 1) and every flop on `posedge clock or posedge reset`; `BlockInShellSpec` checks the block alone byte for byte equal to the module inside the accelerator, vsum_tree_w8's TreeAccumulate and dot_5_w1's Mac, and reports dot_5_w1's SerialAccumulate, whose constant `first` firtool folds into the module inside. `BlockFlopsSpec` emits a block of every datapath at random params per seed (the Mac in its three forms) and checks the register bits its SV declares equal to its `flops` on the params.

## The shared cases

The specs read snax-gvsoc's cases from the folders four variables name; without one, its spec is canceled. CI's chisel job sets all four to the submodule's folders, and a local run does the same:

    SNAX_GVSOC_UNIT_CASES=$PWD/gvsoc/snax-gvsoc/tests/snax/unit/cases \
    SNAX_GVSOC_GROUP_CASES=$PWD/gvsoc/snax-gvsoc/tests/snax/group/cases \
    SNAX_GVSOC_DATAPATH_VECTORS=$PWD/gvsoc/snax-gvsoc/tests/snax/datapath \
    SNAX_GVSOC_SHELL_CASES=$PWD/gvsoc/snax-gvsoc/tests/snax/shell/cases \
    pixi run -e hw chisel-test

- `UnitCasesSpec` (`SnaxUnitCases`): a group controller with its datapath through the one-unit cases: the start line's JOBS and JOB_BEATS, every cycle's input valids and output ready, the run lines and every result (axpy's cases at op axpby, k1 and k2 baked or on their pins, a CSR's value; a case with `fold lanes` on `LaneAccumulate`, each lane its own sum, rowsum 2's `lane_fold_rowsum`); a case whose L and II no catalogue datapath has runs the controller alone.
- `GroupCasesSpec` (`SnaxGroupCases`): `GroupCtrl` alone through the group cases (dot 2–6 at C§17's formulas, backpressure, an input gap, JOBS 0, the fold's seed: two folds unseeded, the first or both seeded, a late seed on a fold and on a SerialAccumulate): every cycle's outer valids and ready, each input's ready, each unit's fire and give, each buffer's occupancy, busy, and the run lines; each case twice, its Ts fixed and from JOB_BEATS (wires held through the run).
- `DatapathVectorsSpec` (`SnaxDatapathVectors`): every catalogue datapath on the NumPy vectors of `vectors.txt`, as snax-gvsoc's `tb.cpp` reads them; a `set` line pokes a CSR slot's pin (SimdLanes' `k`, `k1`, `k2`) before its step.
- `ShellCasesSpec` (`SnaxShellCases`): vecadd's and vsum's `<tag>_shell_wrapper` through the `SnaxShell` cases, which snax-gvsoc's `tb.cpp` checks `snax_shell.hpp`'s rule against: every access's result (a write's cycle, a read's value) and, every cycle, the group's STATUS from the busy and done pins; a case with JOB_BEATS runs on the fold, one with a unit's scalar CSRs (`K1:32`, `K1_LO K1_HI`) on vecadd's wrapper turned into axpy's unit, those CSRs after JOBS (the `words` lines are snax-gvsoc's alone).

## Tests

Every module has a spec; the random ones drive it with seeded random data, configurations, Decoupled gaps and stalls (`Source` and `Sink` hold valid and bits until fire) and compare it every cycle with a Scala model written from the documented behaviour, never from the Chisel code:

| Spec | Random | Model |
|---|---|---|
| `SimdLanesSpec` | in, op and out types, lanes, every op, data; one input at int32 and int64, add, sub, mul, min and max, k random and at the edges (0, -1, the type's min and max), or from its pin, the out type; axpby at int32, int64 and int64 with an int16 `b`, k1 from its pin or baked, k2 from its pin or unset, lanes, data | `ElementwiseModel`, NumPy's rule (D143), b the constant k; axpby's `k1 * a + k2 * b` at the op type, wrapping |
| `TreeAccumulateSpec` | in and out types, lanes, data, `fire` and `first`; seeded (D163) at 1, 2 and 4 lanes, a seed each cycle | lane sums from each `first`, each lane cast to the out type (the docstring); seeded, a `first` fire starts from seed + the beat's sum; matvec_acc's worked example, 6 then 10 |
| `MacSpec` | the three forms (accumulating, plain, plain with `regOut`), 2, 4, 8 lanes, int64/int32 signed and unsigned, a wider out type, data and enables | per lane p = cast_out(cast_op(a)·cast_op(b)), s ← first ? p : s + p, or p + c (C§17) |
| `LaneAccumulateSpec` | 1, 4, 8 lanes, int32 → int32, int32 → int64, int64 → int64, data, jobs of random T, fires spread over random gaps (no gaps, 30 % or 70 %) | per lane x = cast_out(a), s ← first ? x : s + x on fire, held otherwise (C§17); each job's sums after its last fire |
| `SerialAccumulateSpec` | 1, 2, 4, 8 lanes, in and out types, data, `read`, `lane`, `first`; seeded (D163) at 1, 2 and 4 lanes, jobs of random T and gaps, a seed each | acc ← (first ∧ lane 0) ? x_lane : acc + x_lane (C§17); seeded, the job's first beat's last lane adds the seed |
| `LinkBufferSpec` | depth 1, 2, 4, lanes, types, data, loads and takes as a ring | slot wslot loaded, slot rslot out (C§17) |
| `GroupCtrlSpec` | chains of 1, 2 and 3 units, half in a shuffled unit order, L ∈ {0, 1, 2}, II ∈ {1, 2}, span ∈ {1, 2, 4}, T ∈ {1..4} fixed or from JOB_BEATS, 0–2 outer inputs, depth ∈ {0, 1, 2}, J ∈ {0, 1, 3}, T 0 (JOBS and the Ts on their wires, drawn again only while idle), starts while busy, input valids, output stalls, job_width 3 or 32; again with seeded units (D163), a seed's valid at random | `GroupModel`, SnaxGroupRule (`snax_group.hpp`): a seeded unit's first beat waits for its seed, which its firing takes |
| `GroupFoldSpec` | chains of 1–3 units, half in a shuffled unit order, L ∈ {0, 1, 2}, T ∈ {1, 2, 3, 4, 8} fixed or from JOB_BEATS, depth ∈ {0, 1, 2}, back-to-back runs of random JOBS (held on the wires through a run), input valids, output stalls, job_width 4 or 32 | `GroupModel`'s done cycle, pushes and class: the nested counters (`GroupRule`'s rules) |
| `GroupRuleSettingsSpec` | chains of 1–3 units, L ∈ {0, 1, 2}, T fixed or from JOB_BEATS, depth ∈ {0, 1, 2}, JOBS and JOB_BEATS from {0, 1, 2, 3, 5} on their wires, held through a run, job_width 3, 8 or 32; JOBS 255 of T 255 at job_width 8 (a unit at L 0, one at L 1, a head counting over its consumer's T) | `GroupModel` cycle by cycle: the counters' shared `+ 1` at its top value, the empty run tested on the wires |
| `GroupDescSpec` | a chain listed out of order; a fork, a join, a cycle, two chains, a link to no unit, on an outer input or on a fed one, an unfed input | the order, head and owner, and SnaxGroupRule's reasons (`configure()`) |
| `AccCsrRandomSpec` | 1 to 3 groups of 0, 1 or 2 folds (a JOB_BEATS each), each unit with no scalar, a K1 of 8, 16 or 32 bits or K1_LO and K1_HI, job_width and busy_counter_width from 1 to 32; reads, writes and starts in and out of range, `rsp` and each start's ready, each group's busy and done pulse | the protocol in `AccCsr`'s docstring: a busy group's writes held until idle, JOBS and each JOB_BEATS on `settings`, each scalar's register on `scalars` |
| `AccPeripheralsSpec` | 1 to 3 groups, BUSY_COUNTER of 1, 2, 3 or 8 bits, starts and busy runs of 0 to 20 cycles as a control unit gives them | cleared by the accepted start, + 1 per busy cycle, saturating: E − a after a run |
| `ShellScalarSpec` | axpy's shell at int32 and int64, k1 (and k2) from its CSRs: a random k per run written as `main.c` does, then START, random data | K1's offset and width (K1_LO and K1_HI at 64 bits); NumPy's `k1 * a + k2 * b` at the k last written |
| `ShellRandomSpec` | per fixture: J, T, empty runs, data at its types, streamer gaps and stalls, polled or held starts | the datapath models; STATUS and BUSY_COUNTER as C§17 states them |
| `FifoSlicerSpec`, `FifoPackerSpec` | types, words, lanes, data, input gaps, output stalls; the slicer into the packer | the element order of D142 (the docstrings) |

The random specs run one test per seed of `Seeds` (scalatest names it "seed N"); `SNAX_FORGE_SEED=N pixi run -e hw chisel-test` runs only seed N. Each suite runs in a forked JVM of its own, a fork per CPU and at most 8 at a time (`build.sbt`), so the slowest suite (ShellRandomSpec) bounds the run.

Waveforms: `SNAX_FORGE_VCD=1 pixi run -e hw chisel-test` writes each test's to `hw/chisel/test_run_dir/<test name>/<module>.vcd`, e.g. vecadd's end to end at `test_run_dir/the_shell_wrapper_end_to_end_should_give_vecadds_goldens_and_its_counter/ShellHarness.vcd`.

Package root is `snax.forge`, matching `snax.streamer` / `snax.xdma` in snax_cluster.

Elaborated SystemVerilog goes to `out/hw/` (gitignored). RTL is never committed — the Scala is the source of truth here, as the SDFG is on the Python side.

## Tasks

| Command | Does |
|---|---|
| `pixi run -e hw chisel-compile` | compile Scala only |
| `pixi run -e hw chisel-test` | run the chiseltest specs (with the case folders above) |
| `pixi run -e hw chisel-gen` | elaborate everything to `out/hw/*.sv`; `-- --describe` prints the catalogue, `-- --spec accelerator.json` builds an accelerator, `-- --block NAME --params JSON --out DIR` one catalogue block alone |
| `pixi run -e hw chisel-fmt` | scalafmt in place |
| `pixi run -e hw chisel-check` | fail if unformatted (CI uses this) |
| `pixi run -e hw chisel-clean` | drop build trees, keep the Coursier cache |

## The datapaths

All are data and enables only, under their group's control unit (D147); `blocks.json` gives each one's ports, timing, register bits and pins.

- **SIMD (`SimdLanes`, vecadd's, relu's and axpy's).** A beat in, a beat out, L = 0, II = 1, no pins but its scalars'; a typed ALU per lane. Two inputs (`streams` 2), or one with the constant `k` as every lane's second operand, baked in at `opType` (`streams` 1, the `b` port absent, the module named `..._k<k>`, `_km3` for -3), or from its pin `k` (`kCsr`, `_kr`); `k` is a `const` param, null at two streams, read from `accelerator.json` as a JSON number, so exact within ±2^53 (`Datapaths.const`). axpby (two inputs) is `k1 * a + k2 * b` at `opType` (`+ b` with no k2), each of `k1` and `k2` a `const` baked in (`_k13`) or its pin (`k1Csr`, `k2Csr`: the unit's `scalars`, its K1 or K2 CSR; `_k1r`), the module named `..._axpby` and its slots'.
- **Folds (`TreeAccumulate`, `SerialAccumulate`, an accumulating `Mac`, `LaneAccumulate`).** T beats in, one result out (JOB_BEATS = T), L = 1: `TreeAccumulate` sums a beat's lanes in a tree, `SerialAccumulate` reads them one per cycle in place (span = lanes), `Mac` keeps a lane-wise product sum, `LaneAccumulate` a lane-wise sum of its input (pins `fire` and `first`, s ← first ? a : s + a at the output type; rowsum 2's W rows on W lanes, a column a beat). A fold is a chaining barrier: no output exists until its last input is in.
- **`LinkBuffer`.** A link of depth D ≥ 1 between two units of a group: a ring of D beats, loaded by the producer and read by the consumer.

dot and matvec are compositions of these, one group of linked units each (dot 2–6, matvec 4–6).

## Conventions

- **Random tests for every module.** Each Chisel module gets a seeded random spec against a Scala model (Tests above): not necessarily comprehensive, but its function under random data and random stalls.
- **Elastic at the wrapper.** Every outer port of `<tag>_shell_wrapper` is a valid/ready handshake, which the group's control unit answers; the datapaths take enables, not handshakes. Because the SDFG has already fixed allocation, schedule and binding, latency balancing is the only residual problem, and ready/valid turns it from a correctness obligation into a throughput knob. A group may stall; it may never produce a wrong answer because it stalled.
- **`valid` never depends on `ready`.** `ready` may depend on other channels' `valid` (that is how a join works). The reverse deadlocks. `GroupCtrlSpec` checks it every cycle, for every group it draws.
- **No internal storage, no control in a datapath.** Datapaths see data in flight only; address generation and buffering belong to the streamer, and valid, ready and counting to the group's control unit. Reduction accumulators are the sanctioned exception to storage — they hold a scalar, not a tile.
- **Typed lanes, a `UInt` ALU.** A block's ports are lanes of NumPy's types (`IntType`: `SInt` signed, `UInt` unsigned), fixed at elaboration. Each operand is extended to the op type by its own signedness, the op runs on `UInt` bits at that width (two's complement wraps alike; min and max compare by the op type's signedness), and the result is extended or wrapped into the output type (D143).
- **Width and lane count come from the caller** (`accelerator.json` through the catalogue's fill rule), never from a default. The defaults in these constructors exist so the specs can run standalone.

## Not here yet

A skid buffer and lane replication.
