# SNAX-FORGE Status

Open work: milestones, tasks and open items. What is built:
`docs/ARCHITECTURE.md`; why: `docs/DECISIONS.md`; the artefacts:
`docs/CONTRACTS.md`. Done tasks are removed when their milestone closes
(D117); their tables are in git history.

Status values: `todo`, `brief` (brief written), `wip`, `done`, `deferred`.

## Where the Code Lives

| Component | Code | Tests | Data |
|---|---|---|---|
| SDFG ingest | `snax_forge/sdfg/` | `tests/dfg/test_expand.py` (and through `tests/dfg`) | `kernels/` |
| SNAX-DFG, importer, reference executor | `snax_forge/dfg/`, `snax_forge/expr.py` | `tests/dfg/`, `tests/test_expr.py` | `tests/dfg/fixtures/` |
| SNAX-BRM | `snax_forge/brm/` | `tests/brm/` | `snax_forge/brm/library/` |
| SNAX-SANDBOX | `snax_forge/sandbox/` | `tests/sandbox/` | `recipes/` |
| Design step | `snax_forge/design/` | `tests/design/` | `platforms/` |
| Generator (`accelerator.json`, the hjson files, headers, `data.h`, the bundle; `main.c` from SNAX-LOWER's `lower/c.py`) | `snax_forge/gen/` | `tests/gen/` | `platforms/snax_forge_cluster.hjson` |
| SNAX-LOWER | `snax_forge/lower/` | `tests/lower/` | |
| Flow and `pixi run forge` | `snax_forge/flow/` | `tests/flow/` | |
| A flow run's profile and trace (the events converted, D137) | `snax_forge/run/` | `tests/run/` | `tests/run/fixtures/` |
| Viewers | `snax_forge/viz/` | `tests/viz/` | |
| Reports | `snax_forge/report/` | `tests/report/` | |
| Examples | `examples/` | `tests/flow/test_loop1.py` (on SNAX-GVSoC, marker `gvsoc`), `tests/flow/test_dot_example.py` (skipped while dot is parked) | |
| Chisel datapaths and controllers; a design point's accelerator RTL (from `accelerator.json`, the shell wrapper) | `hw/chisel/`, `snax_forge/gen/rtl.py` | `hw/chisel/src/test/`, `tests/gen/test_rtl.py` | `hw/chisel/blocks.json`, `hw/chisel/src/test/resources/rtl/` |
| SNAX-GVSoC | the submodule `gvsoc/snax-gvsoc` (its own KULeuven-MICAS repository) | in that repository (`make regress`) | its `forge/` fixtures |
| SNAX-GVSoC from snax-forge | `gvsoc/Makefile` | | |
| Kernels (KER0) | `kernels/` | `tests/kernels/` | |
| Utilities (clean, environment check, kernel list) | `util/` | | |
| Development image (D118) | `container/` | | `container/toolchain/` |

Generated files (`.snaxdfg`, design points, runs) go under `out/`. GVSoC was learnt
in `KULeuven-MICAS/snax-gvsoc-experiment` (X1), which stays a playground.

## Milestones

Done: M1, M4a, M3, M4b, M5, X1, M16 (SNAX-GVSoC checked against
snax_cluster's RTL, D123; its tasks in snax-gvsoc's `docs/plans/`), M14
(generation to SNAX-GVSoC and snax_cluster, D111; closed by CLN1). Open, in
order (D116):

| Milestone | Content | Status |
|---|---|---|
| M13 | The block library (D141): basic blocks, their composition (D112) and dot's six variants; M7's tier A; BASE1's forge side | `wip` |
| M7 | Front ends: NPBench's DaCe sources in tiers, integers first (D113); tier A is in M13 | todo |
| M15 | Tiling: a DMA per tile | todo |
| M9 | `jacobi1d` | todo |
| M11 | Layers: conv2d, softmax and attention, HDC operations (D100) | todo |
| M12 | Blocks and models: ResNet, Transformer, NVSA (D100) | todo |
| P1 | Paper plan, alongside (D100) | `wip` |
| P2 | Paper draft (D100) | todo |
| M6 | Contract freeze, when the contracts call for it | todo |
| M8 | Automated DSE over recipes, when sweeps call for it | todo |
| M10 | Cosim of one accelerator (D52) | todo |

M7's tiers interleave with the rest (ARCHITECTURE §8): tier A inside M13,
then tiers B, C and P, M15, tier D with the GeMM (M9 inside it), then tier
E.

## Later Milestones

The tasks are a first cut; each milestone is planned in a round of its own.

| Milestone | Tasks | Content |
|---|---|---|
| P1 | PAP2 (`brief`), BASE1, BASE2, PAP3 | Related work after the reading; the SNAX RTL flow logged step by step by hand (BASE1; its forge side, one `pixi run forge` per dot variant, is in M13); SNAX-GVSoC's speed on large runs (BASE2); the evaluation plan mapped onto the claims (PAP3) |
| M13 | B0 (`done`), B1 (`done`), B1x, clean-up, VIEW2, SYN1, HWCOST, B2, tier A; RAM1 alongside | **B0** flow pieces: CLN1's leftovers; the runtime's cluster headers per design point (open item 68); the importer's scalars, views, `max`, `min`, constant casts and maps over several parameters (open item 37); the MatMul and Transpose pass; ND1's slice: 2-D containers, a tag without a split, the streamer from the temporal loops (open items 45, 46, 48, 78); a restart every T < n (open item 81). **B1** one description (D145–D148): `accelerator.json` per design point (C§17), written once by snax-forge, read by Emit, the catalogue check and SNAX-GVSoC, in place of `rtl.json` and `<slot>_acc.hjson`; groups of linked units, each with one JOBS and its own START, JOBS, STATUS, COUNTER and busy and done pins, an empty run ending in a done; every block a pure datapath (`ElementwiseLanes`, `TreeAccumulate`) under one nest controller per group, in three modules (`<tag>_csr`, `<tag>_ctrl`, `<tag>_datapath`), `blocks.json` the datapath catalogue and the binding `{datapath, args, controller, ports}`; accelerator ⊃ groups ⊃ units in SNAX-DFG and `encapsulate`, so dot 1 is one accelerator of groups `mul` and `sum`; snax-gvsoc's default config and `test_snax_cfg` on `forge/vecadd/`. Rounds: snax-forge R0–R4 (with R1b), snax-gvsoc R1–R3. With R1's regenerated `main.c` rowsum_serial's counter moved 67 → 65 and the forge cycle counts by −12 to +17: the start writes shifted across an I-cache line (open item 65; the accelerator is unchanged). **B1x** dot 2–6 at N = 512 and matvec 1–6 (the row form; matvec 1 at 8 × 512, open item 97, matvec 2–6 at 16 × 512), W = 4 and 8, each one accelerator on one description (D112): links of a depth, one generated control unit per group (D147), the datapaths `Mac`, `SerialAccumulate` and `LinkBuffer` (D149), the transforms `split_reduce`, `fuse_map` and `link` with one recipe per variant (D150), a unit's ports with a `dir` in `accelerator.json` (snax-gvsoc takes it from `inputs`/`outputs` until then). Rounds: F0 the docs (C§10–C§12, C§17, C§18, `done`), F1 Python (`done`: the description with links, `dir`, `span` and `params.T`; the catalogue's `Mac`, `SerialAccumulate`, `LinkBuffer` and `when`; the BRM `mac` and `chisel_serial`; the three transforms; `kernels/blas/matvec.py` and the eleven recipes; design, lowering and headers for linked groups; the viewer; `bundle.json`'s `top` and `modules`; until G1 snax-gvsoc's `forge/` fixtures lack `dir` and `span`, so `tests/gen/test_fixture.py` fails, and until F2 Chisel's `DatapathsSpec` differs from `blocks.json`), G1 snax-gvsoc's model (`done`: SnaxGroup, one component per group, its control model SnaxGroupRule with SnaxUnitRule its one-unit case, a registry of datapath classes, the shared group cases of dot 2–6 at the C§17 formulas, the shell's ctrl write of JOBS then each JOB_BEATS, `forge/` with `dir` and `span`; its choices: chain-only groups, span > 1 only behind a link, a frozen unit reads no lanes, busy counts lane reads, a linked group traced at `snax_acc<i>/group_<group>`), F1b (`done`: the submodule at G1, the trace path in the manifest and `run/`, `gen.chain` and `gen.span` with the sandbox's `link.chain`), F2 Chisel (steps 0–2: the per-unit join in C§17, Scala's catalogue as `blocks.json`; `Mac`, `SerialAccumulate`, `LinkBuffer` on snax-gvsoc's vectors; `GroupCtrl` generated from a group's units and links with its control model `GroupRule` inside, `NestCtrl` its one-unit case, all 23 group cases cycle-exact; every existing design's control SV the same ports and registers, `NestCtrl_*` folded into `<tag>_<group>_ctrl`), F4a (#38's submodule at snax-gvsoc main `d7ddfda9`, #37 with G1 and G2; `tests/gen/test_fixture.py` reproduces its 28 `forge/` folders, B1x's 11 among them), F3 the flow on SNAX-GVSoC, then C1, a local vsim check on snax_cluster with no snax_cluster PR. The converter (`run/`) reads several groups' units (`snax_acc<i>/unit_<unit>`) at task level since B1's R3; beat-level conversion and the run viewer's multi-unit view are open item 96. **clean-up** snax_cluster's CI time (agreed with the cluster's maintainers first), and whatever B1x leaves. **VIEW2** the interactive SNAX-DFG viewer: select a part of the graph, apply a transform from a side list, a step history with undo, save a recipe, a first-order profile; its decisions, D55, D73, D23 and open items 4 and 49, are amended when VIEW2 runs. **SYN1** synthesis of the basic blocks and kernels in KULeuven-MICAS/snax-forge-experiments (PAPER §7). **HWCOST** a first-order area and energy estimate from SYN1. **B2** (shrunk): the 1-stream SIMD, axpy's CSR scalar, the lane accumulator without an op, the fold's `init` port, other group binds. **tier A** (B3 and B4 merged): B3's fork, join, broadcast, hold and fragments (applied by output name, `augment_init`), with B4's bicg, mvt (LOW4, open item 47), gesummv, atax. **RAM1** alongside: OpenRAM macros sized for the cluster, in snax-forge-experiments, so snax_cluster and snitch_cluster synthesize on the same macros. `recipes/dot_1.json` runs dot 1 (B1's R3); `examples/dot/` stays parked until B1x. |
| M7 | KER0 (`done`), FE1b, FE1c, PACK0 (`done`), PACK1, FE1d, FE1e, FE2, FE3 | The tiers of PAPER §6.1 after tier A (D113), which M13 does. KER0 staged every target kernel in `kernels/`, not linked to the flow (`kernels/README.md`, `pixi run kernels`). FE1b is tier B; FE1c tier C; PACK0 packed int32 and int8 into 64-bit words with NumPy's arithmetic (D142–D144); PACK1 keeps tier P: the exact products, requantization and the int8 kernels (the reference executor accumulates as the hardware does); FE1d tier D; FE1e tier E. FE3 is C2's coverage table |
| M15 | LOW3b | A tile transform and a DMA per tile, an L2 slice per tile, reductions over tiles (open item 44) |
| M9 | J1 | Stencil BRM with reuse and double buffering (see open item 76) |
| M11 | LAY1–LAY3 | conv2d tiled on a MAC-array BRM; softmax and attention with several BRMs; HDC bind, bundle and similarity |
| M12 | BLK2–BLK4 | A ResNet basic block, a Transformer encoder block, an NVSA pipeline; JEPA if it adds a case |
| P2 | PAP4 | The paper, every number from a checked-in command and a test |
| M6 | F1–F3 | Package layout and CI; versioned schemas for every contract (closes open item 1); registries and namespaced attrs |
| M8 | DSE0–DSE5 | The engine-alone number in `design.md` (open item 49); search over recipes; replacement across the BRM library; parameter choices; memory-plan policies (D86); a sweep runner |
| M10 | COS1, COS2 | One accelerator's RTL in place of its model, and the mismatch of its latency and II per BRM (D52) |

## Next Up

M14 is done, closed by CLN1: a recipe runs to a checked SNAX-GVSoC run
(`pixi run flow`) and to a bundle whose ELF runs on snax_cluster's RTL
(`pixi run forge`, D123). M13's B0 (D141) is done: CLN1's leftovers, the
cluster headers, the importer, the MatMul and Transpose pass, ND1's slice
and the restart every T < n (rowsum). vecadd, vecmul, vsum and dot behave as
before; rowsum, axpy, relu, bicg, mvt, gesummv and atax import and pass the
reference check. PACK0 is done (D142–D144): int32 and int8 packed into the
64-bit words, the elementwise op at NumPy's result type and a fold's lanes
cast to its output's, a tail padded in the data; on SNAX-GVSoC (snax-gvsoc
#32, #33, its sixteen forge/ fixtures), in the Chisel blocks and on
snax_cluster's Verilator CI (#663, five fixtures, counters equal to
SNAX-GVSoC's). The views label a port's packing and show padding apart; the
FIFO slicer and packer are built and parked (open item 93). M13's B1 is
done (D145–D148): `accelerator.json` as the one description, per-group
CSRs and pins, the nest controller over pure datapaths in three modules,
and dot 1 as one accelerator of two groups; counters equal on SNAX-GVSoC,
Verilator and Questa (snax_cluster #664, six fixtures). Next up: M13's B1x,
dot 2–6 and the row form with links (the M13 row), F0 to F3 and C1.

## Open Items

Questions not decided yet. Numbers are never reused; a closed item is
removed, and where it closed is in git history. Closed in CLN1: 63 (the ALU
left snax-gvsoc; generated accelerators follow D130). Closed in M13 B0: 45, 46, 48, 81.
Closed in PACK0: 20 (a port's width: its num_ch, D142), 21 (narrow elements: D142 and D143
on SNAX-GVSoC, in Chisel and on snax_cluster's RTL CI) and 31 (tails: padding, D144).
Closed in M13 B1: 79 (several accelerators on one core: dot 1 is one accelerator of two
groups, D148) and 83 (control alignment with snax_cluster: an empty run ends in a done,
STATUS's sticky done, COUNTER per group, the nest controller's Decoupled config, D146,
D147). Next free number: 98.

1. BRM dataflow notation: `affine` and `stream` (D70), frozen in M6.
2. What automated search reads and how a sweep is configured (DSE1, M8).
4. Stream as a proposer for DNN-shaped workloads, and an analytical
   estimator of our own (the engine alone, open item 49, plus bank conflicts
   from the memory passes, D86). Parked (D23).
18. The L1 address map is fixed and word-interleaved, so the design point
    states no banks; a registered map kind if snax_cluster's differs.
27. Generating a BRM from Chisel, SystemVerilog or HLS sources. Far future.
28. Implementations whose `source` is SystemVerilog or HLS: only `chisel`
    until one is needed.
32. The DFG viewer's layout (D55), revisited if graphs outgrow it.
33. An MLIR importer that writes `.snaxdfg` directly.
34. A hand-edited `.snaxdfg` runs through the flow (D108) but cannot be
    replayed or swept, and the design checks' fixes speak of the recipe.
36. Control flow in SNAX-DFG: a sequential `loop` kind and a `branch` kind,
    registered when a kernel needs them (M9).
37. The expression grammar has `+ - * //`, `max` and `min`, and constant
    casts to the output's type: branches need comparisons, jacobi1d a cast
    of a value.
38. The DMA moves a container as contiguous whole beats (D105); strided or
    2-D transfers when a kernel needs them.
40. Switching runs in the viewer clears the selected cycle.
41. A report glossary, a "where the cycles went" ranking and the knobs'
    `--set` paths: on hold.
44. Reductions over tiles need partial sums and a final combine; until then
    no tile loop sits around a `reduce` memlet (LOW3b).
47. An in-place update (`y += x`) binds but does not lower: the writer waits
    for its own node's reader (LOW4).
49. The engine-alone number C3 compares against (firings × II + latency) is
    computed nowhere; it belongs in `design.md` (DSE0).
50. SNAX-GVSoC's speed on large runs, measured in BASE2; timing-only runs
    and extrapolation only if it is too slow for C1.
53. A link between two accelerators: amends D12 and D51 and the SNAX
    shell. Parked (D112).
54. Which kernels take PolyBench's SMALL in a 128 KiB L1 (M13 tier A); the NumPy
    tolerance for float sums waits for the float pass (D113).
65. Instruction fetch: snax_cluster's L0 prefetches the next line and taken
    branches' targets, GVSoC's I-cache does not, so SNAX-GVSoC misses more
    lines for fewer cycles each (tens of cycles in fixed-cost phases). A
    forge accelerator's counter depends on the core's timing between the
    streamer's and the accelerator's start writes, so GEN2b compares it with
    care (D123). Modelled when a kernel's numbers need it.
66. Atomics to DRAM are one access on SNAX-GVSoC, a read-modify-write on the
    RTL testbench, so the exit's barrier is about 50 cycles shorter here.
    Outside the cluster.
67. The core's own timing beyond its loads (D125): `dmstati` polled every 5
    cycles on the RTL against 2, a SNAX CSR poll 5 against 6, the DMA core's
    copies a cycle or more faster (snax-gvsoc `SG7.md`). Kept as is (D123).
68. Generation is spread out: snaxgen and Chisel in snax_cluster (the
    wrappers, `streamer_csr_addr_map.h`), clustergen and regtool outputs
    copied into snax-gvsoc's `sw/` (D126), `snax_hjson` and its timing
    constants, `snax_forge/gen` with the bundle, which writes the runtime's
    cluster headers per design point since M13's B0. One generator for the
    rest when a design needs it.
69. The SNAX crossbar's AXI input is wired but no program sends through it,
    on either side (snax-gvsoc `SG7.md`).
70. On SNAX-GVSoC a store or AMO after a pending L1 load waits for the
    load's data; the RTL sends it. No measured loop has one.
71. An L1 load after a DRAM load reaches the crossbar up to 13 cycles early;
    the core's timing is right, and no streamer request meets it yet.
72. An L1-to-L1 DMA copy's read and a delayed write reach the crossbar in one
    cycle, where the RTL's `axi_to_mem` takes one a cycle; no program makes
    one.
73. The DMA reading L1 was not split against the RTL (D121's read path); no
    program has it read L1 against a narrow request.
74. The iDMA through the sparse interconnect as ports, instead of the
    superbank mux: a snax_cluster design change, for exploration.
75. Sparse granularity above 1: the crossbar supports it, checked against
    an elaborated RTL module, but no target uses it.
76. Where reuse lives: each loop in the streamer, the accelerator's state or
    its lanes, every unit described to the level above by r, ports,
    throughput, latency, state and area. A research direction (PAPER §10);
    J1 may become jacobi1d in three placements, with jacobi2d.
78. The streamer as an input of its own (amends D84 and D134): the remaining
    knobs and per-streamer overrides out of the platform, a frozen streamer
    file as input; `temporal_dim` comes from the graph since M13's B0.
80. One Snitch core driving both the DMA and the accelerator: the C backend
    takes core roles from the cluster file, `snax_hjson` wants xdma on the
    last core. Never run.
82. SNAX-GVSoC dumps the output regions at exit, so the flow could compare
    them itself; until then the check is the program's own (D136).
86. A custom tag (`--set accelerator_slot=snax_forge_<x>`) is untested past
    the generator's file names, nothing checks that a tag starts with
    `snax_forge`, and several generated accelerators per cluster wait for
    M13.
88. snax_cluster tests `snax_acc_cfg[0]`'s truthiness for an accelerator and
    clustergen writes schema defaults into the config, so the schema's
    `snax_acc_cfg` items carry no defaults.
89. snax_cluster's testharness leaves the cluster's CSR port undriven, so a
    task-queue CSR (`0x5FE`, `0x5FF`) would hang; its own PR.
90. snax_cluster's generated SV is not reproducible (`ComplexQueue`'s
    unseeded queue names, the streamer header's timestamp); its own PR.
91. vsum's window over N (64 to 4096) has no per-element gap: SNAX-GVSoC
    and the RTL differ by at most 16 cycles (2.2%), their slopes' ratio
    0.996, with `forge counter` equal. Kept as is (D123).
92. The per-group busy and done pins (`<group>_busy_o`, `<group>_done_o`,
    D146), which snax_cluster leaves unconnected, into a hardware barrier in
    the cluster, like `snitch_hw_barrier`, so a core waits on the groups in
    the barrier instead of polling their STATUS: a snax_cluster template
    change, then SNAX-GVSoC to match. Parked: B1's six snax_cluster fixtures
    leave them unconnected (Verilator `PINMISSING`, Questa `vopt-2718`).
93. A port of fewer lanes than its beat's elements (rho ≠ 1, D142): the flow
    places only the wire slice. The FIFO slicer and packer are built and
    tested (`hw/chisel/.../stream/`, `snax.forge.stream`) and parked: placing
    them in the flow when a design needs it.
94. A masked last beat, the hardware alternative to padding a tail (D144): a
    lane-valid mask from a CSR, so a tail needs no pads in the data.
95. The cycle gap between SNAX-GVSoC and snax_cluster's RTL on B1's six
    snax_cluster fixtures (#664), `forge counter` equal on all (D123), for
    BASE2: SNAX-GVSoC's window is −9 to +26 cycles off Verilator's (vecadd
    214 vs 223, vecadd_i32 228 vs 202); Questa equals Verilator. Earlier
    tracing put one such gap in core 1's instruction fetch, not the
    accelerator (open item 65).

    | Design | Verilator | Questa | SNAX-GVSoC |
    |---|---|---|---|
    | vecadd | 223 | 223 | 214 |
    | vsum | 246 | – | 254 |
    | vecadd_i32 | 202 | – | 228 |
    | vsum_tree_i8 | 185 | – | 192 |
    | vecmul_i32_i64 | 218 | – | 214 |
    | dot_1 | 302 | 302 | 304 |

96. B1x's measurements, parked to SYN1: per-unit and per-link events, the
    converter at beat level for several units, the run viewer's multi-unit
    view, bank and port statistics, the export CSV, the area × cycles
    metric.
97. dot 1's row form at 16 × 512 does not fit: tmp0 needs 132 KiB of the
    128 KiB L1, so it runs at 8 × 512.

## Sync Reminders

- After every change done with Claude, update this file and log any design
  change in `docs/DECISIONS.md` (rules at its top).
- When a milestone closes, compact `docs/DECISIONS.md` and remove the
  milestone's done tasks from this file (D117).
