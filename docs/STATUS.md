# SNAX-FORGE Status

Open work: milestones, tasks and open items. What is built, and where its code lives: `docs/ARCHITECTURE.md`; why: `docs/DECISIONS.md`; the artefacts: `docs/CONTRACTS.md`. Done tasks are removed when their milestone closes (D117); their tables are in git history.

Status values: `todo`, `brief` (brief written), `wip`, `done`, `deferred`.

## Milestones

Done: M1, M4a, M3, M4b, M5, X1 (GVSoC learnt in `KULeuven-MICAS/snax-gvsoc-experiment`, which stays a playground), M16 (SNAX-GVSoC checked against snax_cluster's RTL, D123; what it measured in snax-gvsoc's `docs/notes/NOTES.md`), M14 (generation to SNAX-GVSoC and snax_cluster, D111; closed by CLN1). Open, in order (D116):

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

M7's tiers interleave with the rest (ARCHITECTURE §8): tier A inside M13, then tiers B, C and P, M15, tier D with the GeMM (M9 inside it), then tier E.

## Later Milestones

The tasks are a first cut; each milestone is planned in a round of its own.

| Milestone | Tasks | Content |
|---|---|---|
| P1 | PAP2 (`brief`), BASE1, BASE2, PAP3 | Related work after the reading; the SNAX RTL flow logged step by step by hand (BASE1; its forge side, one `pixi run forge` per dot variant, is in M13); SNAX-GVSoC's speed on large runs (BASE2); the evaluation plan mapped onto the claims (PAP3) |
| M13 | B0, B1, B1x, CLN2 (`done`); VIEW2 (`wip`); SYN1, HWCOST, B2, tier A; RAM1 alongside | **B0** (`done`, D141, #33): the block library's flow pieces: the runtime's cluster headers per design point, the importer's scalars, views, `max`, `min`, constant casts and maps over several parameters, the MatMul and Transpose pass, ND1's 2-D slice, a restart every T < n. **B1** (`done`, D145–D148, #35–#37, snax-gvsoc #34): `accelerator.json` as the one description (C§17), per-group CSRs and busy and done pins, pure datapaths under a control unit per group in three modules, dot 1 as one accelerator of two groups. **B1x** (`done`, D112, D147, D149, D150, #38, #39, snax-gvsoc #37): dot 2–6 and matvec 1–6 as one accelerator each, a group of linked units under one generated control unit (SnaxGroupRule and `GroupCtrl`, 23 shared group cases); the datapaths `Mac`, `SerialAccumulate` and `LinkBuffer`; the transforms `split_reduce`, `fuse_map` and `link`; every group's COUNTER C§17's cycles plus a fill of 3 (2 for a one-input unit). **CLN2** (`done`, #40–#43, snax-gvsoc #38, #39, snax_cluster #666): the clean-up after B1x: Python and tests cleaned, flow outputs trimmed, the group rule over edges on all three sides with the counter fold, the fixture counter from the rule's model (open item 98 closed), and bundle sets checked locally on vsim (`pixi run forge-set`, snax_cluster's `snax-forge-check`) in place of snax_cluster's CI. **VIEW2** (`wip`, D55, D72, D73, D151; branch `ry/view2`, snax-forge only): the SNAX-DFG viewer becomes an editor, one page in two modes (`view-dfg`, `edit-dfg`): select a node, apply a sandbox transform from a side list, a step history with undo, save a recipe once the sequence looks right, and the ideal estimate of what the mapping buys. Rounds: R0 the docs, R1 the sandbox (`site`, `split`), R2 the estimate (`estimate/`, C§19, in `design.md` and `run.md`; `gen.instances` a design check) R3 the session (`sandbox/session.py`, no HTTP) and R4 the server (`viz/dfg/server.py` in both modes, `pixi run edit-dfg`, JSON-only POSTs) done; R5a–c the page; R6 the close. **SYN1** synthesis of the basic blocks and kernels in KULeuven-MICAS/snax-forge-experiments (PAPER §7). **HWCOST** a first-order area and energy estimate from SYN1. **B2** (shrunk): the 1-stream SIMD, axpy's CSR scalar, the lane accumulator without an op, the fold's `init` port, other group binds; it lifts `gen.span` (a unit of span > 1 reading from a link). **tier A** (B3 and B4 merged): B3's fork, join, broadcast, hold and fragments (applied by output name, `augment_init`), with B4's bicg, mvt (LOW4, open item 47), gesummv, atax; it lifts `gen.chain` (a group is a chain). **RAM1** alongside: OpenRAM macros sized for the cluster, in snax-forge-experiments, so snax_cluster and snitch_cluster synthesize on the same macros. |
| M7 | KER0 (`done`), FE1b, FE1c, PACK0 (`done`), PACK1, FE1d, FE1e, FE2, FE3 | The tiers of PAPER §6.1 after tier A (D113), which M13 does. KER0 staged every target kernel in `kernels/`, not linked to the flow (`kernels/README.md`, `pixi run kernels`). FE1b is tier B; FE1c tier C; PACK0 packed int32 and int8 into 64-bit words with NumPy's arithmetic (D142–D144); PACK1 keeps tier P: the exact products, requantization and the int8 kernels (the reference executor accumulates as the hardware does); FE1d tier D; FE1e tier E. FE3 is C2's coverage table |
| M15 | LOW3b | A tile transform and a DMA per tile, an L2 slice per tile, reductions over tiles (open item 44) |
| M9 | J1 | Stencil BRM with reuse and double buffering (see open item 76) |
| M11 | LAY1–LAY3 | conv2d tiled on a MAC-array BRM; softmax and attention with several BRMs; HDC bind, bundle and similarity |
| M12 | BLK2–BLK4 | A ResNet basic block, a Transformer encoder block, an NVSA pipeline; JEPA if it adds a case |
| P2 | PAP4 | The paper, every number from a checked-in command and a test |
| M6 | F1–F3 | Package layout and CI; versioned schemas for every contract (closes open item 1); registries and namespaced attrs |
| M8 | DSE1–DSE5 | Search over recipes; replacement across the BRM library; parameter choices; memory-plan policies (D86); a sweep runner |
| M10 | COS1, COS2 | One accelerator's RTL in place of its model, and the mismatch of its latency and II per BRM (D52) |

## Next Up

M14 is done (CLN1): a recipe runs to a checked SNAX-GVSoC run (`pixi run flow`) and to a bundle whose ELF runs on snax_cluster's RTL (`pixi run forge`, D123). In M13, B0, B1, B1x and CLN2 are done (the M13 row above), and M7's PACK0 (D142–D144). CLN2 ended with the local RTL check: `pixi run forge-set quick|full`, then the printed line in snax_cluster (`docs/RUN.md` §8, CLAUDE.md's rule), the full set's numbers in open item 95.

Next up: VIEW2, the interactive SNAX-DFG viewer; then SYN1, HWCOST, B2 and tier A (D141).

## Open Items

Questions not decided yet. Numbers are never reused; a closed item is removed, and where it closed is in git history. Next free number: 101.

1. BRM dataflow notation: `affine` and `stream` (D70), frozen in M6.
2. What automated search reads and how a sweep is configured (DSE1, M8).
4. Stream as a proposer for DNN-shaped workloads, and bank conflicts from the memory passes (D86) in an estimate of our own, on top of the ideal one (D151). Parked (D23).
18. The L1 address map is fixed and word-interleaved, so the design point states no banks; a registered map kind if snax_cluster's differs.
27. Generating a BRM from Chisel, SystemVerilog or HLS sources. Far future.
28. Implementations whose `source` is SystemVerilog or HLS: only `chisel` until one is needed.
32. The DFG viewer's layout (D55), revisited if graphs outgrow it.
33. An MLIR importer that writes `.snaxdfg` directly.
34. A hand-edited `.snaxdfg` runs through the flow (D108) but cannot be replayed or swept, and the design checks' fixes speak of the recipe.
36. Control flow in SNAX-DFG: a sequential `loop` kind and a `branch` kind, registered when a kernel needs them (M9).
37. The expression grammar has `+ - * //`, `max` and `min`, and constant casts to the output's type: branches need comparisons, jacobi1d a cast of a value.
38. The DMA moves a container as contiguous whole beats (D105); strided or 2-D transfers when a kernel needs them.
40. Switching runs in the viewer clears the selected cycle.
41. A report glossary, a "where the cycles went" ranking and the knobs' `--set` paths: on hold.
44. Reductions over tiles need partial sums and a final combine; until then no tile loop sits around a `reduce` memlet (LOW3b).
47. An in-place update (`y += x`) binds but does not lower: the writer waits for its own node's reader (LOW4).
50. SNAX-GVSoC's speed on large runs, measured in BASE2; timing-only runs and extrapolation only if it is too slow for C1.
53. A link between two accelerators: amends D12 and D51 and the SNAX shell. Parked (D112).
54. Which kernels take PolyBench's SMALL in a 128 KiB L1 (M13 tier A); the NumPy tolerance for float sums waits for the float pass (D113).
65. Instruction fetch: snax_cluster's L0 prefetches the next line and taken branches' targets, GVSoC's I-cache does not, so SNAX-GVSoC misses more lines for fewer cycles each (tens of cycles in fixed-cost phases). A forge accelerator's counter depends on the core's timing between the streamer's and the accelerator's start writes, so GEN2b compares it with care (D123). Modelled when a kernel's numbers need it.
66. Atomics to DRAM are one access on SNAX-GVSoC, a read-modify-write on the RTL testbench, so the exit's barrier is about 50 cycles shorter here. Outside the cluster.
67. The core's own timing beyond its loads (D125): `dmstati` polled every 5 cycles on the RTL against 2, a SNAX CSR poll 5 against 6, the DMA core's copies a cycle or more faster (snax-gvsoc NOTES.md, "The core and its timing"). Kept as is (D123).
68. Generation is spread out: snaxgen and Chisel in snax_cluster (the wrappers, `streamer_csr_addr_map.h`), clustergen and regtool outputs copied into snax-gvsoc's `sw/` (D126), `snax_hjson` and its timing constants, `snax_forge/gen` with the bundle, which writes the runtime's cluster headers per design point since M13's B0. One generator for the rest when a design needs it.
69. The SNAX crossbar's AXI input is wired but no program sends through it, on either side (snax-gvsoc NOTES.md, "The crossbar and the sparse interconnect").
70. On SNAX-GVSoC a store or AMO after a pending L1 load waits for the load's data; the RTL sends it. No measured loop has one.
71. An L1 load after a DRAM load reaches the crossbar up to 13 cycles early; the core's timing is right, and no streamer request meets it yet.
72. An L1-to-L1 DMA copy's read and a delayed write reach the crossbar in one cycle, where the RTL's `axi_to_mem` takes one a cycle; no program makes one.
73. The DMA reading L1 was not split against the RTL (D121's read path); no program has it read L1 against a narrow request.
74. The iDMA through the sparse interconnect as ports, instead of the superbank mux: a snax_cluster design change, for exploration.
75. Sparse granularity above 1: the crossbar supports it, checked against an elaborated RTL module, but no target uses it.
76. Where reuse lives: each loop in the streamer, the accelerator's state or its lanes, every unit described to the level above by r, ports, throughput, latency, state and area. A research direction (PAPER §10); J1 may become jacobi1d in three placements, with jacobi2d.
78. The streamer as an input of its own (amends D84 and D134): the remaining knobs and per-streamer overrides out of the platform, a frozen streamer file as input; `temporal_dim` comes from the graph since M13's B0.
80. One Snitch core driving both the DMA and the accelerator: the C backend takes core roles from the cluster file, `snax_hjson` wants xdma on the last core. Never run.
82. SNAX-GVSoC dumps the output regions at exit, so the flow could compare them itself; until then the check is the program's own (D136).
86. A custom tag (`--set accelerator_slot=snax_forge_<x>`) is untested past the generator's file names, nothing checks that a tag starts with `snax_forge`, and several generated accelerators per cluster wait for M13.
88. snax_cluster tests `snax_acc_cfg[0]`'s truthiness for an accelerator and clustergen writes schema defaults into the config, so the schema's `snax_acc_cfg` items carry no defaults.
89. snax_cluster's testharness leaves the cluster's CSR port undriven, so a task-queue CSR (`0x5FE`, `0x5FF`) would hang; its own PR.
90. snax_cluster's generated SV is not reproducible (`ComplexQueue`'s unseeded queue names, the streamer header's timestamp); its own PR.
91. vsum's window over N (64 to 4096) has no per-element gap: SNAX-GVSoC and the RTL differ by at most 16 cycles (2.2%), their slopes' ratio 0.996, with `forge counter` equal. Kept as is (D123).
92. The per-group busy and done pins (`<group>_busy_o`, `<group>_done_o`, D146), which snax_cluster leaves unconnected, into a hardware barrier in the cluster, like `snitch_hw_barrier`, so a core waits on the groups in the barrier instead of polling their STATUS: a snax_cluster template change, then SNAX-GVSoC to match. Parked: snax_cluster leaves them unconnected (Verilator `PINMISSING`, Questa `vopt-2718`).
93. A port of fewer lanes than its beat's elements (rho ≠ 1, D142): the flow places only the wire slice. The FIFO slicer and packer are built and tested (`hw/chisel/.../stream/`, `snax.forge.stream`) and parked: placing them in the flow when a design needs it.
94. A masked last beat, the hardware alternative to padding a tail (D144): a lane-valid mask from a CSR, so a tail needs no pads in the data.
95. The cycle gap between SNAX-GVSoC and snax_cluster's RTL, for BASE2 (D123): on the full set, measured on vsim with snax_cluster at #666 and snax-forge at #43, every `forge counter` is equal and every run has `forge data: 0 errors`; the windows are within −4.5% to +8.6%. The gaps are the core's timing (open items 65, 67; snax-gvsoc NOTES.md, "The comparison with snax_cluster's RTL").

    | Design | `forge counter` (both) | vsim window | SNAX-GVSoC window |
    |---|---|---|---|
    | vecadd_i32 | 19 | 210 | 228 |
    | vsum_tree_i8 | 11 | 197 | 192 |
    | dot_1 | 19, 19 | 300 | 304 |
    | dot_4 | 133 | 443 | 437 |
    | vecadd | 19 | 210 | 214 |
    | vsum | 67 | 266 | 254 |
    | vecmul_i32_i64 | 19 | 210 | 214 |
    | dot_2 | 133 | 443 | 437 |
    | dot_3 | 132 | 443 | 434 |
    | dot_5 | 136 | 448 | 440 |
    | dot_6 | 137 | 448 | 440 |
    | matvec_5 | 2101 | 3384 | 3375 |
    | matvec_6 | 2057 | 3339 | 3330 |

96. B1x's measurements, parked to SYN1: per-unit and per-link events, the converter at beat level for several units, the run viewer's multi-unit view, bank and port statistics, the export CSV, the area × cycles metric. F3's bank conflicts (SNAX-GVSoC, the plan as it is): every two-input variant's second reader (`mul_b`, `mac_b`) waits 1 cycle on the first beat, A and B a whole number of bank rows apart, the third cycle of every counter's fill; matvec 5's `mac_b` 15 more (W 4) and 16 (W 8) while the MAC waits for the serial sum, and the writer `sum_out` 3 to 22 in matvec 1–6 (most at W 8), both inside the FIFOs: no cycle over C§17 + the fill.
97. dot 1's row form at 16 × 512 does not fit: tmp0 needs 132 KiB of the 128 KiB L1, so it runs at 8 × 512.
99. L2 in SNAX-DFG: containers get a level (L2 or L1), and an L2-to-L1 move becomes a DMA copy node placed by a transform, so the editor shows its cost; today lowering derives the DMAs from the memory plan (D64, D86). It belongs with M15's LOW3b.
100. Unmapped work on a core: a node no accelerator runs is run by a core, its tasklet turned into C; this needs `graph.unbound` (C§14) relaxed and lowering to emit core code. Until then the estimate shows a not-mapped count (D151).

## Sync Reminders

- After every change done with Claude, update this file and log any design change in `docs/DECISIONS.md` (rules at its top).
- When a milestone closes, compact `docs/DECISIONS.md` and remove the milestone's done tasks from this file (D117).
