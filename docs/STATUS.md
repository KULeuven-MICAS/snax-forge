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
| M13 | B0, B1, B1x, CLN2, VIEW2, SPAN1, SYN1, CTRL1, CHAR1, CHAR1b, DRY1, DRY1-2, SLOT1, DRY1-3 (`done`); HWCOST, B2, tier A; RAM1 alongside | Its tasks, below (M13) |
| M7 | KER0 (`done`), FE1b, FE1c, PACK0 (`done`), PACK1, FE1d, FE1e, FE2, FE3 | The tiers of PAPER §6.1 after tier A (D113), which M13 does. KER0 staged every target kernel in `kernels/`, not linked to the flow (`kernels/README.md`, `pixi run kernels`). FE1b is tier B; FE1c tier C; PACK0 packed int32 and int8 into 64-bit words with NumPy's arithmetic (D142–D144); PACK1 keeps tier P: the exact products, requantization and the int8 kernels (the reference executor accumulates as the hardware does); FE1d tier D; FE1e tier E. FE3 is C2's coverage table |
| M15 | LOW3b | A tile transform and a DMA per tile, an L2 slice per tile, reductions over tiles (open item 44) |
| M9 | J1 | Stencil BRM with reuse and double buffering (see open item 76) |
| M11 | LAY1–LAY3 | conv2d tiled on a MAC-array BRM; softmax and attention with several BRMs; HDC bind, bundle and similarity |
| M12 | BLK2–BLK4 | A ResNet basic block, a Transformer encoder block, an NVSA pipeline; JEPA if it adds a case |
| P2 | PAP4 | The paper, every number from a checked-in command and a test |
| M6 | F1–F3 | Package layout and CI; versioned schemas for every contract (closes open item 1); registries and namespaced attrs |
| M8 | DSE1–DSE5 | Search over recipes; replacement across the BRM library; parameter choices; memory-plan policies (D86); a sweep runner |
| M10 | COS1, COS2 | One accelerator's RTL in place of its model, and the mismatch of its latency and II per BRM (D52) |

## M13

Done, each in its PRs and git history:

| Task | PRs | What it gave |
|---|---|---|
| B0 | #33 | The block library's flow pieces: cluster headers per design point, the importer's scalars, views, `max`, `min`, constant casts and maps over several parameters, MatMul and Transpose, ND1's 2-D slice (D141) |
| B1 | #35–#37, snax-gvsoc #34 | `accelerator.json` as the one description (C§17), per-group CSRs and pins, datapaths under a control unit per group (D145–D148) |
| B1x | #38, #39, snax-gvsoc #37 | dot 2–6 and matvec 1–6 as one accelerator each: linked units under one control unit, `Mac`, `SerialAccumulate`, `LinkBuffer`, `split_reduce`, `fuse_map`, `link` (D112, D147, D149, D150) |
| CLN2 | #40–#44, snax-gvsoc #38, #39, snax_cluster #666 | The clean-up after B1x; the local RTL check (`forge-set`) |
| VIEW2 | #45 | The SNAX-DFG editor (`edit-dfg`) and the ideal estimate (D55, D72, D73, D155; C§12, C§19) |
| SPAN1 | #46, snax-gvsoc #41 | A unit reads at every span (dot 5 and 6 at W = 1); the design note `design.serial_one_lane` |
| SYN1 | #47, snax-forge-experiments | forge's 48 points on Nangate 45 and 15 nm beside hls-exercises' tier done (`results/forge_vs_hls.md`, PAPER §7) |
| CTRL1 | #48, snax-gvsoc #42 | The csr as the interface; nested group counting with no product (D152, D153) |
| CHAR1 | #49, snax-forge-experiments #2 | Each block alone emitted (`Emit --block`, `gen-block`) and characterized; the model (D154) |
| CHAR1b | #50 | The model in snax-forge: `flops`, `ctrl_bits`, `snax_forge/char/<tech>.json`, the cost in the estimate, the viewer and the editor (C§19, C§20) |
| DRY1 | #51 | `examples/dryrun/1_vecmul.md` and the editor fixes it found |
| DRY1-2 | #52 | `examples/dryrun/2_dot_1.md` and its fixes |
| SLOT1 | #53, #54, snax_cluster #679, #680, snax-gvsoc #43 | A design's accelerators as streamer sets of one snax_cluster entry, run in series (D157, D158; D123 amended); `dot_1` two accelerators, `dot_1_enc` one; plan in the project's `claude/slot1-plan-2026-11-04.md` |
| DRY1-3 | #55 | `examples/dryrun/3_dot_2_6.md`, dot 2–6 against SYN2 (cycles exact, area within 0.6 %, kernel energy 1.01–1.08); a buffer link loads once per output of its producer; a fused nest's local value leaves the graph at its bind and is drawn as a wire; the bind form lists only what fits; the dry-run pages name no commits (D98) |

To do, in order:

- **HWCOST**: the costs CHAR1b left as placeholders (the estimate already sums the blocks, and DRY1-3 checked it against dot 2–6 synthesized whole): the streamers with the memory and the interconnect (open item 110), in place of D154's two placeholders (an L1 word at 4 × e_load per bit, a control unit's comb area equal to its flops'); dot 2's buffer (open item 112); energy on SNAX-GVSoC from what ran, each unit's active cycles times its logic and register energy.
- **B2** (shrunk): the 1-stream SIMD, axpy's CSR scalar, the lane accumulator without an op, the fold's `init` port, other group binds; it lifts `gen.span` (a unit of span > 1 reading from a link).
- **tier A** (B3 and B4 merged): B3's fork, join, broadcast, hold and fragments (applied by output name, `augment_init`), with B4's bicg, mvt, gesummv and atax; it lifts `gen.chain` (a group is a chain). With open items 47 (mvt's in-place update, LOW4), 109 (overlap of independent accelerators, mvt first), 54 (SMALL in a 128 KiB L1) and 113 (one unit across DFG regions, atax).
- **RAM1**, alongside: OpenRAM macros sized for the cluster, in snax-forge-experiments, so snax_cluster and snitch_cluster synthesize on the same macros; HWCOST's memory costs come from them.

## Next Up

HWCOST, then B2 and tier A (M13), then M7's tiers (ARCHITECTURE §8). The paper targets ISPASS 2027, the abstract on 11 Dec 2026 and the paper on 18 Dec (D100, PAPER §1); which experiments fit by then is PAP3's choice (P1).

## Open Items

Questions not decided yet. Numbers are never reused; a closed item is removed, and where it closed is in git history. Next free number: 114.

1. BRM dataflow notation: `affine` and `stream` (D70), frozen in M6.
2. What automated search reads and how a sweep is configured (DSE1, M8).
4. Stream as a proposer for DNN-shaped workloads, and bank conflicts from the memory passes (D86) in an estimate of our own, on top of the ideal one (D155). Parked (D23).
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
86. A custom prefix (`--set accelerator_slot=<x>`) is untested past the generator's file names, and nothing checks that the entry's name starts with `snax_forge`; several entries on one core are open item 108.
88. snax_cluster tests `snax_acc_cfg[0]`'s truthiness for an accelerator and clustergen writes schema defaults into the config, so the schema's `snax_acc_cfg` items carry no defaults.
89. snax_cluster's testharness leaves the cluster's CSR port undriven, so a task-queue CSR (`0x5FE`, `0x5FF`) would hang; its own PR.
90. snax_cluster's generated SV is not reproducible (`ComplexQueue`'s unseeded queue names, the streamer header's timestamp); its own PR.
91. vsum's window over N (64 to 4096) has no per-element gap: SNAX-GVSoC and the RTL differ by at most 16 cycles (2.2%), their slopes' ratio 0.996, with `forge counter` equal. Kept as is (D123).
92. The per-group busy and done pins (`<group>_busy_o`, `<group>_done_o`, D146), which snax_cluster leaves unconnected, into a hardware barrier in the cluster, like `snitch_hw_barrier`, so a core waits on the groups in the barrier instead of polling their STATUS: a snax_cluster template change, then SNAX-GVSoC to match. Parked: snax_cluster leaves them unconnected (Verilator `PINMISSING`, Questa `vopt-2718`).
93. A port of fewer lanes than its beat's elements (rho ≠ 1, D142): the flow places only the wire slice. The FIFO slicer and packer are built and tested (`hw/chisel/.../stream/`, `snax.forge.stream`) and parked: placing them in the flow when a design needs it.
94. A masked last beat, the hardware alternative to padding a tail (D144): a lane-valid mask from a CSR, so a tail needs no pads in the data.
95. The cycle gap between SNAX-GVSoC and snax_cluster's RTL, for BASE2 (D123): on the full set, measured on vsim with snax_cluster at #666 and snax-forge at #43, every `forge counter` is equal and every run has `forge data: 0 errors`; the windows are within −4.5% to +8.6%. The quick set measured again in SLOT1 C1 (snax_cluster `ry/slot1` at `5b59191e`, tag `slot1-s3`, snax-forge `slot1-f3`, SNAX-GVSoC `1e988679`, image `d3d8f5572ff4`): every `forge counter` equal and `forge data: 0 errors` again, the windows within −4.8% to +6.8%; dot 1's two streamer sets (`dot_1`) and its encapsulated form (`dot_1_enc`, the earlier `dot_1` row's shape) are its new rows. The gaps are the core's timing (open items 65, 67; snax-gvsoc NOTES.md, "The comparison with snax_cluster's RTL").

    | Design | `forge counter` (both) | vsim window | SNAX-GVSoC window |
    |---|---|---|---|
    | vecadd_i32 | 19 | 217 | 228 |
    | vsum_tree_i8 | 11 | 196 | 192 |
    | dot_1 | 19, 19 | 282 | 264 |
    | dot_1_enc | 19, 19 | 302 | 304 |
    | dot_4 | 133 | 449 | 437 |
    | vecadd | 19 | 210 | 214 |
    | vsum | 67 | 266 | 254 |
    | vecmul_i32_i64 | 19 | 210 | 214 |
    | dot_2 | 133 | 443 | 437 |
    | dot_3 | 132 | 443 | 434 |
    | dot_5 | 136 | 448 | 440 |
    | dot_6 | 137 | 448 | 440 |
    | matvec_5 | 2101 | 3384 | 3375 |
    | matvec_6 | 2057 | 3339 | 3330 |

96. The measurements of a linked group: per-unit and per-link events, the converter at beat level for a group of several linked units (`convert.level` refuses it; an accelerator of several one-unit groups, dot 1, is done) and the run viewer's view of it; bank and port statistics, the export CSV, the area × cycles metric. The bank conflicts measured so far (F3, SNAX-GVSoC) stay inside C§17's cycles plus the fill.
97. dot 1's row form at 16 × 512 does not fit: tmp0 needs 132 KiB of the 128 KiB L1, so it runs at 8 × 512.
99. L2 in SNAX-DFG: containers get a level (L2 or L1), and an L2-to-L1 move becomes a DMA copy node placed by a transform, so the editor shows its cost; today lowering derives the DMAs from the memory plan (D64, D86). It belongs with M15's LOW3b.
100. Unmapped work on a core: a node no accelerator runs is run by a core, its tasklet turned into C; this needs `graph.unbound` (C§14) relaxed and lowering to emit core code. Until then the estimate counts it at one cycle per iteration, an assumption (D155, `UNMAPPED_CYCLES_PER_ITERATION`), which a measured core value could replace.
101. The graph page's edges into a node lower in a map pass behind the nodes above it (`viz/dfg/api.py`'s layout, `static/dfg.js`): a route around them, or the rows inside a map spaced for it.
102. `gen/accelerator.py`'s `_check_chain` takes a `ports` argument it never reads (from B1x).
103. A unit with L ≥ 1 on a wire into a consumer s with T = 1 could borrow s's test as an L-0 unit does, with g_s + popcount(v_k) ≠ R_s (its results in flight counted), dropping `mac_jobs_counter` in dot 4–6 (from CTRL1).
105. SNAX-GVSoC's TCDM crossbar takes at most 64 inputs (`snax/snax_tcdm_xbar.cpp`), so `connect.xbar_inputs` refuses a design above it (vecmul at W = 32: 99). Raising it is a snax-gvsoc change, then the platforms' `l1.max_xbar_inputs` (DRY1).
106. A fold reading through a tail a container whose pad words its producer never writes (dot 1 at W_mul = 8, W_sum = 10: tmp0's 2048 words written, 2050 read) needs the program to write the fold's identity there before the fold's task; `connect.pad` refuses it until then (D144, DRY1-2).
108. Several `snax_acc_cfg` entries on one core: the range decoder between entries in place of `snax_acc_mux_demux` and 0x7C6, and `snaxgen.py:620` building every entry; untested since SLOT1 drops the multi-ALU cluster.
109. Overlap of independent accelerators: D158's serial rule dropped for data-only waits; first case PolyBench mvt, run serial as the baseline; `accelerator_cycles` and C_acc change with it.
110. Characterize the streamer, with the memory and the interconnect, so a streamer set has a cost.
112. A buffer link's energy against SYN2 (DRY1-3): dot 2's `tmp0` buffer, which loads every beat, is 1.59× (nangate45) and 1.49× (nangate15) high against CHAR1's `link/d1_w8` stream point, not explained. dot 6's `out_partial` was 2.65× and 3.13× high while a buffer loaded once per firing of its producer; it now loads once per output (C§20), once per job behind the Mac.
113. One unit across several DFG regions (atax's A and Aᵀ on one GeMM): a unit bound at several nodes, trading throughput for area; amends D112 (Ryan, 27 Oct 2026, for the paper). tier A's atax is the first case.

## Sync Reminders

- After every change done with Claude, update this file and log any design change in `docs/DECISIONS.md` (rules at its top).
- When a milestone closes, compact `docs/DECISIONS.md` and remove the milestone's done tasks from this file (D117).
