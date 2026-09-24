# SNAX-FORGE Status

Build order and rationale: `docs/ARCHITECTURE.md` section 8, D24, D51 and D54
(order M1, M4a, M3, M4b, M5–M10, then M2); inside M3, LOW1b comes before
LOW1a (D63), and since D76 M3 closes `vecadd` from the kernel forwards.
What the model-side artefacts mean: `docs/CONTRACTS.md` (MOD10).
Status values: `todo`, `brief` (brief written), `wip`, `done`, `deferred`.

## Existing Code

- Code from before the v1.1 plan: SDFG ingest, patterns, libnodes, descriptors and a direct ChiselHwGen. The SDFG ingest (snax_forge/sdfg/, `pixi run forge <kernel>` writes out/sdfg/<kernel>.raw.sdfg and .simplified.sdfg) is reused by the SNAX-DFG importer from M3 on (IMP1, D71), and the SDFG recipes in transforms/ are the model for SNAX-SANDBOX's recipes (D72); ChiselHwGen is reused in M10 (HW generator). Patterns, libnodes and descriptors are not used yet.
- SNAX-MODEL (M1) lives in snax_forge/snax_model/, tests in tests/snax_model/ (shared test helpers in tests/snax_model/helpers.py). Scenarios (MOD9) live in scenarios/, one folder each with the scenario.py that makes it and its hand-written tasks.json; scenarios/make.py writes the scenario files, data and cluster files, which are generated and not in git (D65–D67; `pixi run scenarios`, and the test session writes them first). The contracts (MOD10) are docs/CONTRACTS.md; the configuration classes and the JSON writer they describe are snax_forge/snax_model/config.py.
- The visualiser (M4a, D55) lives in snax_forge/viz/ (server, API, static viewer), tests in tests/viz/ (a package, so its helpers.py does not clash with snax_model's).
- SNAX-BRM (M3, D68) lives in snax_forge/brm/ (the BRM dataclasses and their validation, value expressions, the registry of dataflow notations, instances and their accelerator entry, the `affine` dataflow notation, the library loader), tests in tests/brm/ (a package, like tests/lower). The library of hand-written BRMs is snax_forge/brm/library/, one JSON file per BRM (`elementwise_add` so far).
- SNAX-LOWER (M3, D64) lives in snax_forge/lower/ (task list, per-type values, lowering to commands, the `Program` command builder, and since BRM2 the provisional buffer `Layout` and the nest-to-streamer mapping, D70), tests in tests/lower/ (a package, like tests/viz). Every scenario has a hand-written `tasks.json` in its folder, which its scenario.py lowers into the program of its `scenario.json` (D65, D66).
- Planned in M3 (D71–D76): snax_forge/dfg/ (the `.snaxdfg` format, importers, the reference executor), snax_forge/sandbox/ (transforms and recipes), and the DFG viewer as a second mode of snax_forge/viz/ (`python -m snax_forge.viz.dfg`, pixi `view-dfg`). Generated `.snaxdfg` files and design points go under out/, not in git.

## Milestones

| Milestone | Content | Status |
|---|---|---|
| M1 | SNAX-MODEL, kernel-agnostic | `done` |
| M4a | Run views: profile report, schedule, cluster view | `done` |
| M3 | Close `vecadd` end to end, from the kernel (D76) | `wip` |
| M4b | Remaining views and first manual loop | todo |
| M5 | `dot` | todo |
| M6 | Contract freeze | todo |
| M7 | Remaining front ends | todo |
| M8 | Automated DSE over recipes | todo |
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

### M3: Close `vecadd` end to end (D76)

From the kernel forwards: `kernels/polybench/vecadd.py` → simplified SDFG
(`pixi run forge vecadd`) → `vecadd.snaxdfg` → recipe in SNAX-SANDBOX →
design point → cluster file and task list → scenario → SNAX-MODEL. The BRM
and task-list work below is done; the rest follows in table order.

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| BRM1 | BRM format (D68): shared part (interface, function, dataflow, pattern) and implementations (source, supports, timing, optional binding); design and runtime params; value expressions; dataflow as a registered notation; then the link to the accelerator entry of the cluster file through the registered accel kind (D43) | MOD10 | Missing required part is rejected; no binding is accepted; round trip writes every field; the entry resolved from a BRM is checked against the AccelConfig its kind builds | `done` |
| BRM2 | First affine nest notation and enumerator (D70), mapped onto the MOD10 streamer register layout through a provisional buffer layout in SNAX-LOWER | BRM1 | Enumeration equals hand-written index lists for 1D, 2D and strided cases (tests/brm/test_affine.py); mapped registers reproduce the enumerated addresses, and vecadd's nests give the streamer values of `scenarios/vecadd/tasks.json` (tests/lower/test_streams.py) | `done` |
| BRM3 | Elementwise-add BRM, the library's first file (`snax_forge/brm/library/elementwise_add.json`): lanes `W` (default 4), per-port affine nests, one Chisel implementation (L = 0, II = 1), function, pattern; `load_brm` | BRM1, BRM2 | Resolves to exactly alu4's `acc` entry; vecadd run with it gives the same cycles, profile and data as the elementwise stub, and its nests give vecadd's streamer values (tests/brm/test_library.py, tests/lower/test_streams.py) | `done` |
| LOW1b | Task-list format (D64, closes open item 19) and task list → plain command list through the model's adapters (D36, D45); built before LOW1a on hand-written task lists (D63) | MOD10 | Program lowered from `scenarios/vecadd/tasks.json` equals vecadd's hand-scheduled program (written out in tests/lower, D67); every scenario keeps its cycle count (tests/lower) | `done` |
| DFG1 | SNAX-DFG format (D71): `.snaxdfg` JSON with containers, tasklets, map scopes with symbolic ranges and `loop.kind`, connectors, memlets, the accelerated node (nesting allowed); registered kinds and namespaced attrs (D19); `to_dict` / `from_dict` | none | A hand-built vecadd, plain and accelerated, round-trips with every field written; unknown kinds and dangling references are rejected by name | todo |
| IMP1 | SDFG importer for vecadd (D71): reads the simplified SDFG of `pixi run forge vecadd`, folds the transient copy, keeps `N` symbolic; settles the kernel dtype (open item 30) | DFG1 | The import equals a checked-in `vecadd.snaxdfg` fixture, with no transient and no copy left | todo |
| REF1 | NumPy reference executor on `.snaxdfg` (D20), symbols bound; an accelerated node runs through its BRM's function | DFG1, BRM3 | The imported vecadd equals the kernel's `reference` on `make_inputs`, for N a multiple of the lane count and not; plain, split and accelerated graphs agree | todo |
| VIS5 | DFG viewer (D76): `python -m snax_forge.viz.dfg FILE [FILE ...]`, pixi `view-dfg`; several files side by side, Reload; containers, maps as nested boxes by loop kind, tasklets, accelerated nodes, memlets with subsets | DFG1, VIS1 | API tests (tests/viz); `vecadd.snaxdfg` and `vecadd_accelerated.snaxdfg` checked by eye side by side | todo |
| SBX1 | SNAX-SANDBOX (D72, D73): registered transforms `split_map` and `bind` (absorbs DFG2: the pattern predicate and design-param extraction), the recipe format with symbol bindings, the reference check after every step, a CLI writing each step's `.snaxdfg`; settles open item 35 or leaves it to DP1 | DFG1, REF1, BRM3 | The vecadd recipe gives `vecadd_accelerated.snaxdfg`: a temporal loop of N / W and a spatial loop of W, bound to `elementwise_add`; W = 4 and W = 8 both pass the reference check; a bound that is not a multiple of W (open item 31) and a W the BRM does not allow are rejected | todo |
| DP1 | Design point (D74): written by the sandbox from a recipe; the mapped DFG, the memory plan (a layout per container and memory, set by a place step), the instances, the platform; closes open item 29 | SBX1 | vecadd's design point places a, b and c where `scenarios/vecadd` has them; validation catches overlapping containers, out-of-range addresses, unknown BRMs and implementations | todo |
| NAME1 | Derived names (D75): streamers `<instance>_<port>` and task names `<node>_<component>`, `load_<container>`, `store_<container>` (vecadd's as the imported graph names them) in the cluster builders, every `tasks.json`, the tests and the CONTRACTS.md snippets | LOW1b, IMP1 | Every scenario keeps its cycle count (tests/lower) | todo |
| LOW1c | Design point + BRMs → cluster file (D53): accelerator entries from BRM interface and timing, one streamer per port named `<instance>_<port>` with `n_ports` = lanes, platform parts from the design point | DP1, NAME1 | Cluster file for the `vecadd` design point equals `scenarios/clusters/alu4.json` | todo |
| LOW1a | Design point → ordered task list (D45) in the format of D64: order from the mapped DFG, sequential per tile; loads and stores from the memory plan; streamer values from memlets through layouts (D73); `after` from the memlets; derived task names (D75) | DP1, LOW1b, NAME1 | Task list for the `vecadd` design point equals `scenarios/vecadd/tasks.json` | todo |
| E2E1 | Full `vecadd` path: kernel → import → recipe → design point → cluster file and task list → scenario (inputs from `make_inputs`) → run | all of the above | Output equals the kernel's reference and REF1 exactly; cycles equal a run of `scenarios/vecadd` | todo |

### M4b: Remaining views and first manual loop (after M3, D54)

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| VIS4 | Design point view: memory map, accelerator instances and parameters | DP1, VIS1 | Every buffer and instance appears with correct addresses and banks | todo |
| VIS6 | Diff between two runs: design point fields, profile metrics, timelines side by side | VIS2–VIS4 | For two `vecadd` runs differing only in lanes, exactly that field and its effects are flagged | todo |
| VIS7 | Compressed trace summary for LLM use | MOD8 | Under a size limit; numbers equal the profile | todo |
| LOOP1 | One documented iteration: run, read views, edit the recipe, rerun, diff (D72) | VIS6, VIS7 | Checked-in example with both recipes, their design points and the diff page; cycle change matches what the views predicted | todo |

### M5: `dot`

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| DFG3 | Reduction in SNAX-DFG, its import from `kernels/polybench/dot.py` and REF | IMP1, REF1 | REF on the imported `dot` equals `np.dot` on integer inputs (D28) | todo |
| BRM4 | Accumulator BRM, based on the Chisel accumulator | BRM1, BRM2 | Same cycles and data as the reduce stub | todo |
| LOW2 | Chaining multiply → accumulate, waits at accelerator boundaries | LOW1b, BRM4 | Waits appear only at dependencies crossing a boundary | todo |
| LOW3 | General DMA insertion from the memory plan: several containers, an L2 slice per tile (the simple case is LOW1a's) | LOW1a | DMA list covers exactly the containers with an L2 layout, per tile, no duplicates | todo |
| E2E2 | `dot` end to end, timeline check | all of the above, VIS2 | Output matches exactly; timeline shows the chaining wait | todo |

### M6: Contract freeze

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| F1 | Package layout, CI, CLAUDE.md conventions | M5 | CI runs green | todo |
| F2 | Versioned JSON schemas for SNAX-DFG, BRM, recipe, design point, task list, control program, scenario, profile, trace; close open item 1 | M5 | Every existing scenario and fixture validates; an old version is rejected | todo |
| F3 | Registries and namespaced attributes (D19) | F2 | A new kind can be added from outside the core; unknown attrs survive a round trip | todo |

### M7: Remaining front ends

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| FE1 | SDFG importer for the remaining constructs (jacobi1d's stencil), reusing the existing ingest; vecadd and dot are imported in M3 and M5 (D76) | DFG3, F2 | `jacobi1d` imports; REF output equals the kernel's reference | todo |
| FE2 | Named errors for unsupported SDFG constructs | FE1 | Each unsupported construct in the fixtures gets a named error | todo |

### M8: Automated DSE over recipes

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| DSE1 | Search and sweep settings for automated DSE, and their schema (the rest of open item 2); recipes are SBX1's | F2, SBX1 | Decision logged; schema tests | todo |
| DSE2 | Pattern-based replacement across the BRM library: every matching BRM and implementation, not only the one a recipe names | SBX1, BRM4 | Auto-replaced `vecadd` equals the design point of its hand-written recipe (DP1) | todo |
| DSE3 | Parameter choices (lanes, tiling, instance count) made by search and written as recipes | DSE2 | Each chosen value appears in the recipe and the design point | todo |
| DSE4 | Memory planner policies for the place step: bank placement and alignment | DP1 | No overlaps; each policy gives the expected bank map | todo |
| DSE5 | Sweep runner writing a results table | DSE1–DSE4 | Lanes × bank-count sweep is reproducible and shows hand-checked trends | todo |
| VIS8 | Diff view shows recipe changes alongside design point changes | DSE1, VIS6 | A one-parameter recipe change is shown with the design point fields it caused | todo |

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

M3 closes `vecadd` end to end. LOW1b is done (D63, D64): a task list (`configure`, `start`, `sync`, `read`)
is lowered to the model's command list, and `scenarios/vecadd/tasks.json`
gives exactly vecadd's hand-scheduled program. Scenarios live one folder
each, every one with a hand-written task list, fmul included (D65, D66); the
scenario files are generated and not in git (D67).

BRM1–BRM3 are done (D68, D70): a BRM is a hand-written JSON file with a
shared part and a map of implementations; `Brm.resolve` turns one
implementation and its design params into the accelerator entry of the
cluster file, checked against the model's registered kind; each port's
`affine` nest is enumerated per task and mapped through a buffer layout
onto streamer values. The library's first file, `elementwise_add`,
resolves to exactly alu4's adder and gives vecadd's streamer values. The
model gained the reader repeat on temporal stride 0 (D69) on the way.

The plan for the rest of M3 changed with D71–D76: `vecadd` is now closed
from the kernel forwards, and SNAX-DSE starts as SNAX-SANDBOX. Next is the
SNAX-DFG: DFG1 fixes the `.snaxdfg` format, IMP1 imports vecadd's simplified
SDFG into it, REF1 runs it in NumPy and VIS5 shows it in the browser. SBX1
adds SNAX-SANDBOX: `split_map` and `bind` turn `vecadd.snaxdfg` into
`vecadd_accelerated.snaxdfg` from a recipe. DP1 writes the design point,
NAME1 renames the streamers, LOW1c and LOW1a derive `alu4.json` and
`vecadd/tasks.json` from the design point, and E2E1 runs the whole path. A
sweep over W is then one recipe with W as a parameter. M4b follows.

## Sync Reminders

- After every new update, PR, commit, or new task done with Claude, synchronise
  `./docs`: update task status here, and log any design change in the
  ARCHITECTURE.md Decision Log.