# SNAX-FORGE Status

Build order and rationale: `docs/ARCHITECTURE.md` section 8, D24, D51 and D54
(order M1, M4a, M3, M4b, M5–M10, then M2); inside M3, LOW1b comes before
LOW1a (D63).
What the model-side artefacts mean: `docs/CONTRACTS.md` (MOD10).
Status values: `todo`, `brief` (brief written), `wip`, `done`, `deferred`.

## Existing Code

- Code from before the v1.1 plan: SDFG ingest, patterns, libnodes, descriptors and a direct ChiselHwGen. It is kept and reused in M7 (SDFG front end) and M10 (HW generator). It is not part of M1–M6.
- SNAX-MODEL (M1) lives in snax_forge/snax_model/, tests in tests/snax_model/ (shared test helpers in tests/snax_model/helpers.py). Scenarios (MOD9) live in scenarios/, one folder each with the scenario.py that makes it and its hand-written tasks.json; scenarios/make.py writes the scenario files, data and cluster files, which are generated and not in git (D65–D67; `pixi run scenarios`, and the test session writes them first). The contracts (MOD10) are docs/CONTRACTS.md; the configuration classes and the JSON writer they describe are snax_forge/snax_model/config.py.
- The visualiser (M4a, D55) lives in snax_forge/viz/ (server, API, static viewer), tests in tests/viz/ (a package, so its helpers.py does not clash with snax_model's).
- SNAX-BRM (M3, D68) lives in snax_forge/brm/ (the BRM dataclasses and their validation, value expressions, the registry of dataflow notations, instances and their accelerator entry, the `affine` dataflow notation), tests in tests/brm/ (a package, like tests/lower). The library of hand-written BRMs, snax_forge/brm/library/, gets its first file with BRM3.
- SNAX-LOWER (M3, D64) lives in snax_forge/lower/ (task list, per-type values, lowering to commands, the `Program` command builder, and since BRM2 the provisional buffer `Layout` and the nest-to-streamer mapping, D70), tests in tests/lower/ (a package, like tests/viz). Every scenario has a hand-written `tasks.json` in its folder, which its scenario.py lowers into the program of its `scenario.json` (D65, D66).

## Milestones

| Milestone | Content | Status |
|---|---|---|
| M1 | SNAX-MODEL, kernel-agnostic | `done` |
| M4a | Run views: profile report, schedule, cluster view | `done` |
| M3 | Build backwards to close `vecadd` | `wip` |
| M4b | Remaining views and first manual loop | todo |
| M5 | `dot` | todo |
| M6 | Contract freeze | todo |
| M7 | SDFG front end | todo |
| M8 | DSE via config | todo |
| M9 | `jacobi1d` | todo |
| M10 | Outer path (independent of M3–M9, D52) | todo |
| M2 | Anchor against SNAX RTL (after M10, D51) | `deferred` |

## Task Breakdown

### M1: SNAX-MODEL (kernel-agnostic)

All tests use synthetic traffic and hand-written scenarios.

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| MOD1 | Event-driven scheduler; ticks only components with pending work, skips idle cycle ranges; struct-of-arrays state (D10, D21) | none | Toy components give identical results with skipping on and off; two runs are identical | `done` |
| MOD2 | L1 banks: count, width, read latency, one access per bank per cycle; element type and elements per word (D13) | MOD1 | Read latency is exact; a second access to the same bank in the same cycle is refused | `done` |
| MOD3 | TCDM interconnect: round-robin arbitration, conflicts and stalls recorded | MOD2 | Grant sequences for 2–3 masters on one bank match hand-worked tables; distinct banks proceed in parallel | `done` |
| MOD4 | Streamer from raw registers (base, bounds and strides per loop), FIFO depth, valid/ready, configurable ports (D12) | MOD3 | Address streams equal a NumPy enumeration for 1D, 2D and strided nests; a full FIFO causes stalls; conflict-free throughput equals the port count per cycle | `done` |
| MOD5 | Accelerator interface: ports with per-port element rate, `L`, `II`, Python function; elementwise (N→1) and reduce (T→1) stubs (D25) | MOD4 | With ideal streams, both stubs hit their cycle formulas; the reduce stub proves unequal port rates work | `done` |
| MOD6 | L2 and DMA with a wide port with per-cycle superbank priority (D33, D34) | MOD3 | DMA bandwidth test; DMA–streamer contention shows up in the trace | `done` |
| MOD7 | Uniform register interface and controller executing `csr_write`, `csr_read`, `wait` (poll, signal) (D11, D36, D37) | MOD5, MOD6 | Poll and signal give the same output data; cycle counts differ only by the expected control overhead | `done` |
| MOD8 | Profile and JSON trace | MOD7 | Per accelerator, busy + idle + stalled = total; per-bank access counts equal the bank coverage of the trace's grant events (D38, D39) | `done` |
| MOD9 | Scenario runner: JSON with cluster config, initial memory, command list; dumps profile, trace, final memory | MOD8 | Elementwise, reduce and DMA scenarios run from the CLI; final memory checked against NumPy | `done` |
| MOD10 | Write down the model-side contracts in docs/CONTRACTS.md; configs carry their own to_dict / from_dict; gap rules, trace filter and housekeeping (D26, D46-D50) | MOD9 | Every snippet in the document is checked against its file and the register names against the adapters (test_contracts.py); the gap rules have tests (test_gaps.py) | `done` |

### M4a: Run views (before M3, D54)

Everything here reads only a model run's output directory, through the local
server and viewer of D55 (`pixi run view DIR [DIR ...]`); tested on the M1
scenarios, `scenarios/vecadd_conflict`, `scenarios/vecadd_tiled`
(3 tiles, 471 cycles, for a longer schedule) and `scenarios/fmul` (5 tiles
on `clusters/mul1.json`, double buffered: the DMA works behind a
multi-cycle multiplier, 525 cycles).

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| VIS1 | Server, CLI, viewer shell and profile report (D55, D56): `python -m snax_forge.viz DIR [DIR ...]`, JSON API over the run directories, report of cycles per class, accelerators, streamers and FIFOs (with the busy window), memory, DMA and L2, controller, cluster configuration | MOD10 | API tests pass (tests/viz); report checked by eye on `scenarios/vecadd` and `scenarios/vecadd_conflict` (a and b in the same banks) | `done` |
| VIS2 | Schedule view, HLS-schedule style (D57, D58, D60): per component its class runs, tasks and commands over a cycle window, beat-level detail rows (ports, FIFO, firings, DMA beats, polls), a selected cycle with everything that happened in it | VIS1 | Kind filter of the events route tested (tests/viz); schedule checked by eye on `vecadd`, `vecadd_conflict`, `reduce` (task trace) and `dma` (filtered beat trace) | `done` |
| VIS3 | Cluster view (D61): banks, interconnect, streamers, accelerator, DMA and controller at the schedule's selected cycle, under the schedule on the same page; requests (teal) and read data coming back (green) as separate lanes (D62), conflicts (list in the interconnect box, stalled side red), FIFO fill per lane, firings, DMA and L2 requests and responses, no data values (open item 23); layout built from the cluster file | VIS1, VIS2 | Checked by eye on `vecadd_conflict` (ra and rb on banks 8–11) and `fmul` (DMA in one superbank while the streamers use others, e.g. cycle 65); request / response timing on `vecadd_conflict` cycles 41–44 and 13–15; `resp` checked against the grants in test_profile | `done` |

### M3: Build backwards to close `vecadd`

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| BRM1 | BRM format (D68): shared part (interface, function, dataflow, pattern) and implementations (source, supports, timing, optional binding); design and runtime params; value expressions; dataflow as a registered notation; then the link to the accelerator entry of the cluster file through the registered accel kind (D43) | MOD10 | Missing required part is rejected; no binding is accepted; round trip writes every field; the entry resolved from a BRM is checked against the AccelConfig its kind builds | `done` |
| BRM2 | First affine nest notation and enumerator (D70), mapped onto the MOD10 streamer register layout through a provisional buffer layout in SNAX-LOWER | BRM1 | Enumeration equals hand-written index lists for 1D, 2D and strided cases (tests/brm/test_affine.py); mapped registers reproduce the enumerated addresses, and vecadd's nests give the streamer values of `scenarios/vecadd/tasks.json` (tests/lower/test_streams.py) | `done` |
| BRM3 | Elementwise-add BRM: lanes `W`, per-port nests, `L`/`II`, function, pattern | BRM1, BRM2 | In the model, gives the same cycles and data as the elementwise stub | todo |
| DP1 | Design point structure and hand-written `vecadd` design point; an instance names BRM, implementation and design params (D68); buffer layout per open item 29 | BRM3 | Validation catches overlapping buffers, out-of-range banks, unknown BRMs and implementations | todo |
| LOW1a | Lowering from design point to an ordered task list (D45) in the format of D64 | DP1, LOW1b | Task list for the `vecadd` design point equals `scenarios/vecadd/tasks.json` | todo |
| LOW1b | Task-list format (D64, closes open item 19) and task list → plain command list through the model's adapters (D36, D45); built before LOW1a on hand-written task lists (D63) | MOD10 | Program lowered from `scenarios/vecadd/tasks.json` equals vecadd's hand-scheduled program (written out in tests/lower, D67); every scenario keeps its cycle count (tests/lower) | `done` |
| LOW1c | Design point + BRMs → cluster file (D53): accelerator entries from BRM interface and timing, one streamer per port with `n_ports` = lanes, platform parts from the cluster configuration | DP1, BRM3 | Cluster file for the `vecadd` design point equals `scenarios/clusters/alu4.json` | todo |
| DFG1 | Minimal SNAX-DFG: data container, tasklet, map scope with symbolic range, memlet | none | Hand-built `vecadd` DFG round-trips | todo |
| DFG2 | Accelerated node referencing a BRM instance, nesting allowed | DFG1, BRM3 | `vecadd` with its map replaced validates; nested case validates | todo |
| REF1 | NumPy reference executor for DFG1 kinds | DFG1 | `vecadd` equals `a+b` over random N, including N not divisible by the lane count | todo |
| E2E1 | Full `vecadd` path from DFG to profile | all of the above | Output matches REF1 exactly; cycles equal a run of `scenarios/vecadd` | todo |

### M4b: Remaining views and first manual loop (after M3, D54)

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| VIS4 | Design point view: memory map, accelerator instances and parameters | DP1, VIS1 | Every buffer and instance appears with correct addresses and banks | todo |
| VIS5 | DFG view: original and with accelerated nodes | DFG2, VIS1 | Node and edge counts match; replaced subgraphs are marked | todo |
| VIS6 | Diff between two runs: design point fields, profile metrics, timelines side by side | VIS2–VIS4 | For two `vecadd` runs differing only in lanes, exactly that field and its effects are flagged | todo |
| VIS7 | Compressed trace summary for LLM use | MOD8 | Under a size limit; numbers equal the profile | todo |
| LOOP1 | One documented iteration: run, read views, edit design point, rerun, diff (D27) | VIS6, VIS7 | Checked-in example with both design points and the diff page; cycle change matches what the views predicted | todo |

### M5: `dot`

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| DFG3 | Reduction in SNAX-DFG and REF | DFG1, REF1 | REF `dot` equals `np.dot` on integer inputs (D28) | todo |
| BRM4 | Accumulator BRM, based on the Chisel accumulator | BRM1, BRM2 | Same cycles and data as the reduce stub | todo |
| LOW2 | Chaining multiply → accumulate, waits at accelerator boundaries | LOW1b, BRM4 | Waits appear only at dependencies crossing a boundary | todo |
| LOW3 | DMA insertion from L2 addresses in the memory plan | LOW1b | DMA list covers exactly the buffers in L2, no duplicates | todo |
| E2E2 | `dot` end to end, timeline check | all of the above, VIS2 | Output matches exactly; timeline shows the chaining wait | todo |

### M6: Contract freeze

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| F1 | Package layout, CI, CLAUDE.md conventions | M5 | CI runs green | todo |
| F2 | Versioned JSON schemas for SNAX-DFG, BRM, design point, control program, scenario, profile, trace; close open item 1 | M5 | Every existing scenario and fixture validates; an old version is rejected | todo |
| F3 | Registries and namespaced attributes (D19) | F2 | A new kind can be added from outside the core; unknown attrs survive a round trip | todo |

### M7: SDFG front end

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| FE1 | SDFG → SNAX-DFG translator, reusing the existing ingest | DFG3, F2 | DaCe SDFGs for `vecadd`, `dot`, `jacobi1d` translate; REF output equals NumPy | todo |
| FE2 | Named errors for unsupported SDFG constructs | FE1 | Each unsupported construct in the fixtures gets a named error | todo |

### M8: DSE via config

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| DSE1 | Config format and schema; single point or sweep (open item 2) | F2 | Decision logged; schema tests | todo |
| DSE2 | Pattern-based replacement | FE1, BRM3, BRM4 | Auto-replaced `vecadd` is structurally equal to DP1 | todo |
| DSE3 | Parameter choices: lanes, tiling, instance count | DSE2 | Each config value appears in the design point | todo |
| DSE4 | Memory planner: bank placement and alignment | DP1 | No overlaps; each policy gives the expected bank map | todo |
| DSE5 | Sweep runner writing a results table | DSE1–DSE4 | Lanes × bank-count sweep is reproducible and shows hand-checked trends | todo |
| VIS8 | Diff view shows config changes alongside design point changes | DSE1, VIS6 | A one-field config change is shown with the design point fields it caused | todo |

### M9: `jacobi1d`

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| J1 | Stencil BRM with reuse, DFG support, double buffering in DSE and SNAX-LOWER | M5, M8 | Output matches exactly; trace shows DMA overlapping compute, with fewer cycles than single buffering | todo |

### M10: Outer path

Independent of M3–M9 (D52): cosim checks an accelerator's RTL against its
model, not the platform.

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| GEN1 | Chisel generation through BRM bindings into one accelerator top (reuse ChiselHwGen) | BRM1 | Elaborates for the BRM3 and BRM4 parameter sets | todo |
| GEN2 | C backend for SNAX-LOWER | LOW3 | C kernel command sequence equals the JSON program (D18); builds with the SNAX toolchain | todo |
| COS1 | cocotb bridge replacing accelerator models with RTL | GEN1, MOD7 | `vecadd` and `dot` outputs match the model's | todo |
| COS2 | Mismatch report per accelerator (D52) | COS1 | Deviation of each accelerator's RTL from its declared `latency` and `ii` reported per BRM | todo |

### M2: Anchor (deferred until after M10, D51)

Checks the platform model against the real SNAX cluster RTL (ARCHITECTURE.md
section 7). Until then the platform values are declared defaults and model
cycles compare design points only.

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| ANC1 | Run `vecadd` in the SNAX RTL flow, record cycles per phase | M10 | Numbers checked in | `deferred` |
| ANC2 | Matching hand-written scenario, deviation report, error target (open item 3) | MOD9, ANC1 | Deviation explained per phase, CPU-side CSR programming kept separate; target in the Decision Log | `deferred` |
| ANC3 | Regression test locking the anchor numbers | ANC2 | CI fails when model cycles drift outside the target | `deferred` |

## Next Up

M1 is done: the model runs scenarios, writes profiles and traces, and its
contracts are written down (`docs/CONTRACTS.md`). The anchor (M2) is
deferred until after M10 (D51): the user supplies the accelerator's entry in
the cluster file, and the rest of the cluster keeps its declared defaults.

M4a is done: `pixi run view DIR [DIR ...]` serves the profile report, the
schedule and, under it, the cluster view of model runs (D55–D62). Data
values in the trace (open item 23) and a push event (open item 25) are
decided once the views have been used for a while.

M3 closes `vecadd` end to end, building backwards from `scenarios/vecadd`.
LOW1b is done (D63, D64): a task list (`configure`, `start`, `sync`, `read`)
is lowered to the model's command list, and `scenarios/vecadd/tasks.json`
gives exactly vecadd's hand-scheduled program. Scenarios live one folder
each, every one with a hand-written task list, fmul included (D65, D66); the
scenario files are generated and not in git (D67).

BRM1 and BRM2 are done (D68, D70): a BRM is a hand-written JSON file with a
shared part and a map of implementations; `Brm.resolve` turns one
implementation and its design params into the accelerator entry of the
cluster file, checked against the model's registered kind; each port's
`affine` nest is enumerated per task and mapped through a buffer layout
onto streamer values. The model gained the reader repeat on temporal
stride 0 (D69) on the way. Next is BRM3 (`elementwise_add`, the library's
first file), then LOW1c, the cluster file from a design point and BRMs
(D53), accepted against `scenarios/clusters/alu4.json`, with DP1; LOW1a
then produces `scenarios/vecadd/tasks.json` from the design point. M4b
follows.

## Sync Reminders

- After every new update, PR, commit, or new task done with Claude, synchronise
  `./docs`: update task status here, and log any design change in the
  ARCHITECTURE.md Decision Log.