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
| P1 | PAP3 (`done`), PAP2 (`brief`), BASE1, BASE2 | PAP3 chose the ISPASS 2027 paper's experiments and the order of the work (PAPER §6, Next Up; D100, D141, D159). PAP2: related work after the reading, DSAGEN and OverGen first (PAPER §8.4). BASE1 and BASE2 are E2: the vsim path timed step by step against SNAX-GVSoC's, and both simulators' cycles per second, logged on E1's runs |
| M13 | B0, B1, B1x, CLN2, VIEW2, SPAN1, SYN1, CTRL1, CHAR1, CHAR1b, DRY1, DRY1-2, SLOT1, DRY1-3, RAM1, B2 (`done`); track A: MVT1, L2V1, XBAR1, tier A, PORT1; track B: the streamer and interconnect sweep, DaCe's dot, HWCOST | Its tasks, below (M13) |
| M7 | KER0 (`done`), FE1b, FE1c, PACK0 (`done`), PACK1, FE1d, FE1e, FE2, FE3 | The tiers of PAPER §6.1 after tier A (D113), which M13 does. KER0 staged every target kernel in `kernels/`, not linked to the flow (`kernels/README.md`, `pixi run kernels`). FE1b is tier B; FE1c tier C; PACK0 packed int32 and int8 into 64-bit words with NumPy's arithmetic (D142–D144); PACK1 keeps tier P: the exact products, requantization and the int8 kernels (the reference executor accumulates as the hardware does); FE1d tier D; FE1e tier E. FE3 is C2's coverage table |
| M15 | LOW3b | A tile transform and a DMA per tile, an L2 slice per tile, reductions over tiles (open item 44) |
| M9 | J1 | Stencil BRM with reuse and double buffering (see open item 76) |
| M11 | LAY1–LAY3 | conv2d tiled on a MAC-array BRM; softmax and attention with several BRMs; HDC bind, bundle and similarity |
| M12 | BLK2–BLK4 | A ResNet basic block, a Transformer encoder block, an NVSA pipeline; JEPA if it adds a case |
| P2 | PAP4 | The paper, every number from a checked-in command; the tests checking them follow after the submission |
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
| B2 | #58, #59, snax-gvsoc #44, #45, snax_cluster #688 | relu, axpy, rowsum 2 and the fold's seed: one `simd` BRM over `SimdLanes` (its code a table over `op`, `streams` and scalar slots), CSR scalars and `bake_scalar`, the in-place write, `interchange_map`, `LaneAccumulate`, the `padded_rows` layout, `fuse_seed` on `TreeAccumulate` and `SerialAccumulate`, `chisel_accumulator` removed; the editor's partner highlight; StreamerGen's prefix fix (D82, D68, D86, D102, D149, D150, D161–D163; plan in the project's `claude/b2-plan-2026-11-06.md`) |
| RAM1 | snax-forge-experiments #4 | SRAM area and read and write energy per access from CACTI 7 at 45 nm (OpenRAM probed and dropped), a lookup table over depth 32–4096 and width 8–256, `.lib` and `.v` macros generated from it, memory energy as counted accesses times the table (D159, D160) |

To do, in two tracks (D141), each task in its own PR:

- **Track A, the code** (snax-forge, snax-gvsoc where a task changes the model):
  1. **MVT1** (`wip`, branch `ry/mvt1`): the DFG view in columns, one per independent chain, and `mark_transpose` on half 2's read of A, drawn as one A with `Aᵀ` (done, `mvt1-r1a`, D164); one instance shared by several nodes, chosen in `bind`'s form (done, D165), and a linked group shared by `share` once each half is linked (done, D165); variant 1 as `recipes/mvt/mvt_1.json` (both halves on one `vmul`, `vsum`, `vadd`) and `mvt_1_dot2.json` (one `vmul → link → vsum` group run twice), variant 2 as `mvt_2.json` (a dot group and a `vadd` per half) (Ryan's numbering: 1 the shared set, 2 a set per half, then the link and the seed); then mvt in the row form on both sides, $x_1 \mathrel{+}= A y_1$ as dots per row and $x_2 \mathrel{+}= A^T y_2$ by the transposed read (bicg 1), at MINI sizes (N = 40): open item 117 first, then its variants one at a time, recipe against recipe (Claude's from the hw-tier notes, Ryan's from `edit-dfg`), in `examples/dryrun/4_mvt.md`, each on SNAX-GVSoC and snax_cluster; the column form, the overlap (open item 109) and the reuse variants stay in tier A; the full set and open item 95 at its close. Split from B2 (its R6; the project's `claude/b2-plan-2026-11-06.md`, topics 1 and 5).
  2. **L2V1**: each container the DMA moves drawn in the DFG views as its L2 copy with a DMA edge into L1, and a DMA row in the estimate from the memory plan, $\sum_c (\lceil B_c / 64 \rceil + t_0)$ for a 512-bit DMA and a fixed cost $t_0$ per transfer read from a SNAX-GVSoC run, added to the total (amends D155), checked against the run's DMA tasks. The graph-level copy node stays open item 99.
  3. **XBAR1**: an interconnect marker per running accelerator, $u = \sum_s \mathrm{ch}_s / n_{\mathrm{banks}}$ over its streamers' channels, green at $u \le 0.5$, orange up to 1, red above, the thresholds platform fields (D85), a design note (C§14) in the estimate, `design.md` and the editor (vecmul: W = 4 green, 8 orange, 16 red).
  4. **tier A**, in two halves: fork, join, broadcast, hold and fragments (applied by output name, `augment_init`) with bicg and gesummv; then mvt's other forms after MVT1 (open item 109) and atax (open item 113). It lifts `gen.chain` (a group is a chain). MINI first, SMALL only where a case needs it (open item 54).
  5. **PORT1**, beside open item 109: a design's streamer sets, which run in series (D158), share their crossbar ports through a mux selected by the busy set, so the ports are the most any set needs, not their sum (amends D157 and C§16); on SNAX-GVSoC and in HWCOST's interconnect cost first, its RTL in snax_cluster when open item 114 calls for it.
- **Track B, the server** (snax-forge-experiments, Nangate 45 nm only, D159):
  1. **RAM1** (done, above).
  2. The streamer and the TCDM interconnect synthesized over their parameters (channels, temporal dims; inputs against banks), for open item 110.
  3. forge's untiered points synthesized and powered at Vitis's power sizes, after B2, for E7; DaCe's dot at n2048 with its buffer as a generated macro (D160), its size read from Vitis's RTL.
- **HWCOST**, where the tracks join (snax-forge): RAM1's table (snax-forge-experiments' `results/ram/nangate45/ram_table.csv`, D160) and the sweeps' costs into `snax_forge/char/nangate45.json` in place of D154's two placeholders, the estimate, the viewer and the editor; open item 112 stated as a known error. Energy on SNAX-GVSoC from what ran moves after the paper.

## Next Up

The ISPASS 2027 plan (PAP3; D100, PAPER §6): M13's two tracks above, run at once, track B starting now so the server never waits; HSYN1 finishes in snax-forge-experiments on both technologies. Then, as each lands:

| Step | Waits for | Gives |
|---|---|---|
| E4: the composition sweeps (a sweep script over `pixi run flow --set`, no DSE5) | tier A, PORT1, HWCOST | C3 |
| E7: forge against Vitis opt and dace, Bambu and XLS on tier done and untiered, W ≤ 16 | HSYN1, track B's step 3 | the blocks against HLS |
| E1: mvt and atax (gesummv if time) on vsim, 3–5 compositions each; E2's times logged on the same runs | tier A | C4, C1 |
| The cluster synthesis (2–3 composites) and E11, the homogeneous cluster (snax_cluster's cores, no accelerator, hand-written C) | E1 | cluster PPA, the Snitch baseline |
| E5: Transformer decode, one head (q·Kᵀ, softmax, ·V) at a short context in L1 | tier A; `exp`, `max` and division (M7 tiers B and C) | the end-to-end case |

Dates: 27 Nov, E5's cut-off (softmax not running on SNAX-GVSoC moves E5, and E6 with it, to future work); 4 Dec, the abstract's result sentence from E4; 11 Dec, the abstract and the data freeze (figure fixes only after it); 12–18 Dec, writing. PAPER §2 and §3 are written alongside from now.

## Open Items

Questions not decided yet. Numbers are never reused; a closed item is removed, and where it closed is in git history. Next free number: 122.

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
50. SNAX-GVSoC's speed on large runs, measured in E2 on E1's runs (BASE2); timing-only runs and extrapolation only if it is too slow for C1.
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
95. The cycle gap between SNAX-GVSoC and snax_cluster's RTL, for BASE2 (D123): on the full set, measured on vsim with snax_cluster at #666 and snax-forge at #43, every `forge counter` is equal and every run has `forge data: 0 errors`; the windows are within −4.5% to +8.6%. The quick set measured again in SLOT1 C1 (snax_cluster `ry/slot1` at `5b59191e`, tag `slot1-s3`, snax-forge `slot1-f3`, SNAX-GVSoC `1e988679`, image `d3d8f5572ff4`): every `forge counter` equal and `forge data: 0 errors` again, the windows within −4.8% to +6.8%; dot 1's two streamer sets (`dot_1`) and its encapsulated form (`dot_1_enc`, the earlier `dot_1` row's shape) are its new rows. relu's row is from B2 R2 (snax_cluster main at `ea9c837b`, tag `b2-r2`), axpy's and axpy_baked's from R3b (tag `b2-r3b`), where the quick set measured the same again each time. axpy's K1 write costs SNAX-GVSoC 11 cycles over the baked design and the RTL 4. rowsum_2's two rows are from R4 (tag `b2-r4`): the four lanes of the contiguous layout (rows 64 words apart, all starting in bank 0) collide on their first reads only; each lane is its own channel with its own FIFO, so they then run a cycle apart and never meet again, and padding saves those 3 cycles, the same on the model and the RTL, far below the 4-way conflict a static bank map predicts. R5's rows are from tag `b2-r5` (snax_cluster main at `cad966bf`, SNAX-GVSoC `356c659`, image `d3d8f5572ff4`), where the quick set measured the same again: vsum, now `chisel_adder_tree` (B0), keeps its counter and runs 271 on vsim against 266; matvec_seed's q += A p has matvec_1's counters on SNAX-GVSoC (1027, 1027) and a window of 2842 against its 2845, so the seed's add costs nothing (matvec_1's row needs snax_cluster #688); matvec_seed_serial's sum is 4 · 1024 + 3, one lane a cycle. matvec_seed and rowsum_serial are the first rows whose counters differ, each within its fill (C§17): matvec_seed's 1028 and 1025 on vsim, rowsum_serial's 67 against 65. The windows run from −5.9% (rowsum_serial) to +6.8%. The gaps are the core's timing (open items 65, 67; snax-gvsoc NOTES.md, "The comparison with snax_cluster's RTL").

    | Design | `forge counter` (both) | vsim window | SNAX-GVSoC window |
    |---|---|---|---|
    | vecadd_i32 | 19 | 217 | 228 |
    | vsum_tree_i8 | 11 | 196 | 192 |
    | dot_1 | 19, 19 | 282 | 264 |
    | dot_1_enc | 19, 19 | 302 | 304 |
    | dot_4 | 133 | 449 | 437 |
    | vecadd | 19 | 210 | 214 |
    | vsum | 67 | 271 | 254 |
    | vecmul_i32_i64 | 19 | 210 | 214 |
    | dot_2 | 133 | 443 | 437 |
    | dot_3 | 132 | 443 | 434 |
    | dot_5 | 136 | 448 | 440 |
    | dot_6 | 137 | 448 | 440 |
    | matvec_5 | 2101 | 3384 | 3375 |
    | matvec_6 | 2057 | 3339 | 3330 |
    | relu | 18 | 204 | 224 |
    | axpy | 19 | 210 | 217 |
    | axpy_baked | 19 | 206 | 206 |
    | rowsum_2 | 134 | 369 | 375 |
    | rowsum_2 padded_rows | 131 | 370 | 369 |
    | matvec_1 (snax_cluster #688) | 1027, 1027 | 2858 | 2845 |
    | matvec_seed | 1027, 1027 (vsim 1028, 1025) | 2856 | 2842 |
    | matvec_seed_serial | 1027, 4099 | 5927 | 5917 |
    | dot_serial | 19, 67 | 311 | 307 |
    | rowsum_serial | 65 (vsim 67) | 256 | 272 |

96. The measurements of a linked group: per-unit and per-link events, the converter at beat level for a group of several linked units (`convert.level` refuses it; an accelerator of several one-unit groups, dot 1, is done) and the run viewer's view of it; bank and port statistics, the export CSV, the area × cycles metric. The bank conflicts measured so far (F3, SNAX-GVSoC) stay inside C§17's cycles plus the fill.
97. dot 1's row form at 16 × 512 does not fit: tmp0 needs 132 KiB of the 128 KiB L1, so it runs at 8 × 512.
99. L2 in SNAX-DFG: containers get a level (L2 or L1), and an L2-to-L1 move becomes a DMA copy node placed by a transform, so tiling can choose what moves per tile; today lowering derives the DMAs from the memory plan (D64, D86), and L2V1 only draws them and counts their cycles. It belongs with M15's LOW3b.
100. Unmapped work on a core: a node no accelerator runs is run by a core, its tasklet turned into C; this needs `graph.unbound` (C§14) relaxed and lowering to emit core code. Until then the estimate counts it at one cycle per iteration, an assumption (D155, `UNMAPPED_CYCLES_PER_ITERATION`), which a measured core value could replace.
101. The graph page's edges into a node lower in a map pass behind the nodes above it (`viz/dfg/api.py`'s layout, `static/dfg.js`): a route around them, or the rows inside a map spaced for it.
102. `gen/accelerator.py`'s `_check_chain` takes a `ports` argument it never reads (from B1x).
103. A unit with L ≥ 1 on a wire into a consumer s with T = 1 could borrow s's test as an L-0 unit does, with g_s + popcount(v_k) ≠ R_s (its results in flight counted), dropping `mac_jobs_counter` in dot 4–6 (from CTRL1).
105. SNAX-GVSoC's TCDM crossbar takes at most 64 inputs (`snax/snax_tcdm_xbar.cpp`), so `connect.xbar_inputs` refuses a design above it (vecmul at W = 32: 99). Raising it is a snax-gvsoc change, then the platforms' `l1.max_xbar_inputs` (DRY1).
106. A fold reading through a tail a container whose pad words its producer never writes (dot 1 at W_mul = 8, W_sum = 10: tmp0's 2048 words written, 2050 read) needs the program to write the fold's identity there before the fold's task; `connect.pad` refuses it until then (D144, DRY1-2).
108. Several `snax_acc_cfg` entries on one core: the range decoder between entries in place of `snax_acc_mux_demux` and 0x7C6, and `snaxgen.py:620` building every entry; untested since SLOT1 drops the multi-ALU cluster.
109. Overlap of independent accelerators: D158's serial rule dropped for data-only waits; first case PolyBench mvt, run serial as the baseline; `accelerator_cycles` and C_acc change with it.
110. Characterize the streamer, with the memory and the interconnect, so a streamer set has a cost: track B's step 2 (the memory is done, D160), on Nangate 45 nm only (D159), into HWCOST.
112. A buffer link's energy against SYN2 (DRY1-3): dot 2's `tmp0` buffer, which loads every beat, is 1.59× (nangate45) and 1.49× (nangate15) high against CHAR1's `link/d1_w8` stream point, not explained. dot 6's `out_partial` was 2.65× and 3.13× high while a buffer loaded once per firing of its producer; it now loads once per output (C§20), once per job behind the Mac.
113. One unit across several DFG regions (atax's A and Aᵀ on one GeMM): a unit bound at several nodes, trading throughput for area; amends D112 (Ryan, 27 Oct 2026, for the paper). tier A's atax is the first case.
114. PORT1's shared crossbar ports exist on SNAX-GVSoC only: snax_cluster's mux follows when a shared-port design is in E4's result, in E1 or in the cluster synthesis, before E1 runs it; until then those runs are marked model only, outside D123's check.
115. An ALU SIMD: the `simd` BRM's `op` as an R/W CSR, its op set a design param, so a design pays only for the ops it can select; worth it once one unit runs several ops, a unit bound at several nodes (open item 113).
116. `gen.span` refuses a unit of span > 1 reading an outer input (a serial unit straight from its streamer: vsum or rowsum 1 on `chisel_serial` at W > 1, the beat held for span cycles); lifted when E7 or a kernel needs such a point.
117. Reusing a composite's recipe: select a region shaped like a saved recipe's kernel (a dot inside each row of a mat-vec) and apply the recipe (`dot_3`, say) in place of its steps by hand, drawn as one bound accelerator labelled with its recipe; the recipe's map ids, variables, containers and symbols mapped onto the region's, a marked read (D164) matched through its permutation, extra outer maps allowed. Its own PR, after MVT1's variants 1 and 2. REUSE1 (`wip`, branch `ry/reuse1`): round 1 done, the matcher, `lift`/`unlift` (blocks nested in either order) and `apply_recipe` (a sub-recipe's steps on a region, a fuse taking the rows' loop with it), `sandbox/blocks.py`; `recipes/matvec/matvec_dot3.json`, and matvec_dotK equals matvec_K for K = 2 to 6 (`tests/sandbox/test_blocks.py`); round 2 done, the editor: the sites of `lift` (at the selected node's region, or every region), `unlift` and `apply_recipe` (a recipe of the block's kernel, its params as the session's), a block drawn as one box inside its firing loops (the outer maps stay, the nests' alike ones as one, each port labelled with the slice one firing takes, `A[m_t, 0:N]`, half 2's `A[0:M, n_t]`) until expanded (double-click, or the Blocks section between the transforms and the Containers), an expanded one a dotted outline, halves sharing units expanding together (`viz/dfg/api.py`'s `fold`, `tests/viz/test_blocks_view.py`); round 3 done: `unapply` (the graph before the recipe, refused once a later step changed what it made), `name_block` (a region named by hand), a sub-recipe's own blocks followed by name, `matvec_dot1`–`matvec_dot6` and mvt's three variants on them (`recipes/mvt/*_reuse.json`, the hand-written graphs), the open blocks in the hash, D166 and C§12. Before its PR: the MVT1 run below, then one PR for `ry/mvt1` and `ry/reuse1`. The first MVT1 run stopped at `make sw`: mvt's `y1` array clashed with math.h's `y1` (the runtime's `sync.h` includes it); a container named after a C, libc or runtime symbol now gets `<name>_l2` in data.h, main.c and the run's regions (C§ data.h), and every forge-set recipe and mvt's six are compiled for riscv32 in `tests/gen/test_compiles.py` (syntax only; skipped without such a clang).
118. The run viewer (`pixi run view`: banks, interconnect, transactions) is kept working but gets no new features until after the paper (Ryan, 6 Nov 2026); the DFG views (`view-dfg`, `edit-dfg`) stay first.
119. The cost estimate reads the graph only, so a layout pass's padding (`padded_rows`: rowsum 2's 8 × 65 words against 8 × 64) is not in its memory bytes; the design report and the memory plan do count it (B2 R4).

## Sync Reminders

- After every change done with Claude, update this file and log any design change in `docs/DECISIONS.md` (rules at its top).
- When a milestone closes, compact `docs/DECISIONS.md` and remove the milestone's done tasks from this file (D117).
