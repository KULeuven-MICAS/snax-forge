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
| M13 | B0, B1, B1x, CLN2, VIEW2, SPAN1, SYN1, CTRL1, CHAR1, CHAR1b, DRY1, DRY1-2, SLOT1, DRY1-3, RAM1, B2, REUSE1, STRM1 (`done`); track A: MVT1, PORT1, PACK1a, L2V1, XBAR1, tier A; track B: E7's untiered points, DaCe's dot, CSYN1; HWCOST; CTRL2 | Its tasks, below (M13) |
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
| STRM1 | snax-forge-experiments #6 | The streamer's reader and the TCDM interconnect synthesized and powered over their parameters (depth, loops, channels; inputs against banks), the tables and their interpolation in its `docs/cost_tables.md` (X52–X54) |
| REUSE1 | #60 | A recipe as the unit of reuse (D166): `lift`, `unlift`, `apply_recipe`, `unapply`, `name_block` (`sandbox/blocks.py`), `matvec_dot1`–`matvec_dot6` and mvt's `_reuse` recipes, blocks in the editor; with MVT1's part 1 (D164, D165) and the `<name>_l2` rename of a container named after a C symbol |

To do, in two tracks (D141), each task in its own PR:

- **Track A, the code** (snax-forge, snax-gvsoc where a task changes the model):
  1. **MVT1** (`wip`, paused at step 1 on 10 Nov, branch `ry/mvt1-2`; part 1 in #60): the DFG view in columns and `mark_transpose` (D164), one instance or linked group shared by both halves (D165), variant 1 as `recipes/mvt/mvt_1.json` (both halves on one `vmul`, `vsum`, `vadd`) and `mvt_1_dot2.json` (one `vmul → link → vsum` group run twice), variant 2 as `mvt_2.json` (a dot group and a `vadd` per half), each with its `_reuse` twin (REUSE1), at M = N = 40 since MVT1-2 step 1 (8 × 16 before; Ryan's numbering: 1 the shared set, 2 a set per half, then the link and the seed). Next, mvt in the row form on both sides, $x_1 \mathrel{+}= A y_1$ as dots per row and $x_2 \mathrel{+}= A^T y_2$ by the transposed read (bicg 1), at MINI sizes (M = N = 40), one step at a time, recipe against recipe (Claude's from the hw-tier notes, Ryan's from `edit-dfg`), each on SNAX-GVSoC and snax_cluster, written up in `examples/dryrun/4_mvt.md`: (1) the six recipes at 40 × 40, W = 4 and 8, `contiguous` and `padded_rows`; (2) variant 3, a dot → link → `vadd` at W = 1 per half (hw-tier-a form 2; no recipe links a fold into a SIMD yet); (3) variant 4, the seed per half (`fuse_seed`, D163, from `matvec_seed`); the shared forms of 3 and 4 only if a point leaves an area question. The layout stays D86's (`contiguous`, `padded_rows` by `--set`), not chosen by the transpose mark: W lanes down a column of pitch p words meet $n_\text{banks} / \gcd(p, n_\text{banks})$ banks, so at 40 × 40 on 32 banks W = 4 has no conflict and W = 8 two-way; a pass padding only where W exceeds that is a new decision if the runs follow it. The column form, the overlap (open item 109) and the reuse variants stay in tier A; the full set and open item 95 at its close. Split from B2 (its R6; the project's `claude/b2-plan-2026-11-06.md`, topics 1 and 5). Paused at step 1 (the six recipes at 40 × 40, on SNAX-GVSoC and vsim, open items 95 and 120; mvt_2 at W = 8 dropped, 85 crossbar inputs, variant 2's port cost); variants 3 and 4 resume at int64 after PORT1 (10 Nov): mvt stays int64 until tier A's column form, as its transposed read cannot pack (D142), and PORT1 changes mvt_2's crossbar inputs.
  2. **PORT1** (by 13 Nov), beside open item 109: a design's streamer sets, which run in series (D158), share their crossbar ports through a mux selected by the busy set, so the ports are the most any set needs, not their sum (amends D157 and C§16); on SNAX-GVSoC and in HWCOST's interconnect cost first, its RTL in snax_cluster when open item 114 calls for it. With it, the crossbar limit (amends D85, closes open item 105): no refusal from 64 to 128 inputs, a note above 64 (its cost extrapolated beyond the tables, and beyond SNAX's own crossbar), a hard maximum at 128 (`l1.max_xbar_inputs` 128, a new platform field for the 64), SNAX-GVSoC's crossbar masks widened from 64 to 128 bits; mvt_2 at W = 8 (85 inputs) then runs, and PORT1 brings it to 44.
  3. **PACK1a** (after MVT1's variants 3 and 4): int8 on the existing row-form kernels whose every L1 stream packs (dot, matvec, the elementwise and reduce kernels), W lanes packed into W / 8 channels (D142), exact products by a cast in the kernel (open item 37). Only the 64-bit word and interconnect are fixed (D142): a port of fewer lanes than its word's elements (a fold's one-lane int32 output, a serial sum's read of an int32 transient) gets the packer or the FIFO slicer placed by the flow (open item 93), so narrow elements still fill whole words. mvt, atax and bicg stay int64 in the row form until tier A's column form packs their transposed read; E7 stays int64 (HSYN1). A decision at its kickoff amends D113 (int8 before tier P) and PAPER's table (packing before the paper); PACK1's requantization and tier P stay in M7.
  4. **L2V1**: each container the DMA moves drawn in the DFG views as its L2 copy with a DMA edge into L1, and a DMA row in the estimate from the memory plan, $\sum_c (\lceil B_c / 64 \rceil + t_0)$ for a 512-bit DMA and a fixed cost $t_0$ per transfer read from a SNAX-GVSoC run, added to the total (amends D155), checked against the run's DMA tasks. The graph-level copy node stays open item 99.
  5. **XBAR1**: an interconnect marker per running accelerator, $u = \sum_s \mathrm{ch}_s / n_{\mathrm{banks}}$ over its streamers' channels, green at $u \le 0.5$, orange up to 1, red above, the thresholds platform fields (D85), a design note (C§14) in the estimate, `design.md` and the editor (vecmul: W = 4 green, 8 orange, 16 red).
  6. **tier A**, in two halves: fork, join, broadcast, hold and fragments (applied by output name, `augment_init`) with bicg and gesummv; then mvt's other forms after MVT1 (open item 109) and atax (open item 113). It lifts `gen.chain` (a group is a chain). MINI first, SMALL only where a case needs it (open item 54).
- **Track B, the server** (snax-forge-experiments, Nangate 45 nm only, D159):
  1. **RAM1** (done, above).
  2. **STRM1** (done, above).
  3. forge's untiered points synthesized and powered at Vitis's power sizes, after B2, for E7 (run with CSYN1, from 10 Nov); DaCe's dot at n2048 with its buffer as a generated macro (D160), its size read from Vitis's RTL.
  4. **CSYN1** (from 10 Nov, a provisional name): the cluster-level synthesis for C4's cost check and E11, on RAM1's macros (X49), a probe first (one existing bundle, dot_4 or mvt_1_dot2, and snax_cluster with its cores only), so the flow is ready before E1 picks the 2–3 composites.
- **CTRL2** (`wip`, branch `ry/ctrl2` in snax-gvsoc, snax-forge and snax-forge-experiments, rounds R0–R5; the project's `claude/ctrl2-plan-2026-11-10.md`), urgent, before CSYN1 Part B: a register goes where its writer and reader are (D168). The settings stay in `<tag>_csr`, which holds a write to a busy group's START or settings until the group is idle, and the control unit reads them live with no copy, counting up with its tests sharing the counters' `+ 1` (D153's form, every cycle count unchanged); COUNTER becomes BUSY_COUNTER (`counter_width` `busy_counter_width`, its CSR slot kept) in the csr's submodule `<tag>_peripherals`, always present and left out of the reported csr (D152). Rounds: R0 the docs, written as they stand after R3 (done); R1 snax-gvsoc's `snax_shell` (done, snax-gvsoc #46: the hold, the shared case `settings_held`, the rename, its `forge/` written again by this branch, the rename alone); R2 the rename in snax-forge, no behaviour change (done: Python, schema, headers, fixtures, docs and the Scala text; its compile and the Chisel specs run in R3); R3 the control (done: `AccCsr` holds a busy group's writes and sends a bare start strobe and JOBS and JOB_BEATS as wires, `<tag>_peripherals` holds BUSY_COUNTER, `GroupCtrl` and `GroupRule` read the settings live with no copy, `estimate/bits.py` without the copies; the emitted ctrl loses exactly them, 32 bits elementwise and 64 per fold group, every Chisel spec passing, `settings_held` too); R4 vsim on main and the branch (the quick set, dot 2, dot 3: data exact, BUSY_COUNTER and each program's total cycles equal to main's); R5 CSYN1 Part A's points resynthesized with power in snax-forge-experiments, the csr reported without `<tag>_peripherals`, the X64 pages overwritten and a before-and-after page. Merge order: snax-gvsoc, snax-forge, then snax-forge-experiments.
- **HWCOST** (`wip`, branch `ry/hwcost`, rounds R0–R2 and R3's docs; the project's `claude/hwcost-plan-2026-11-10.md`), where the tracks join (snax-forge): a design's cost in five parts, datapath, control, streamers, crossbar and memory (D167), the streamers, crossbar and banks priced from snax-forge-experiments' tables copied into `snax_forge/char/nangate45/` (`pixi run import-costs`), in place of D154's L1 placeholder; the parts with their shares in `design.md` and as two bars in the viewer and the editor; outside the tables' grid extrapolated and noted (`cost.extrapolated`); nangate45 only (nangate15 kept as data). Every count read where it is made, so PORT1's crossbar inputs and PACK1a's channels change the cost without touching it. A streamer is priced per beat and a crossbar access at one price per bank count (a full crossbar's), so the energy follows the accesses whatever the inputs. Energy on SNAX-GVSoC from what ran, with the streamers' waiting clocks, the crossbar's per-cycle part and its conflicts, moves after the paper; open item 112 stays.

## Next Up

The ISPASS 2027 plan (PAP3; D100, PAPER §6): M13's two tracks above, run at once, track B starting now so the server never waits; HSYN1 finishes in snax-forge-experiments on both technologies. The week of 10 Nov: CTRL2 first (CSYN1 Part B waits for it), then at once HWCOST, and CSYN1 (the cluster-level synthesis for E11 and C4's cost check) with forge's untiered rows for E7 on the server; PORT1 by 13 Nov; then MVT1's variants 3 and 4, PACK1a and tier A. Then, as each lands:

| Step | Waits for | Gives |
|---|---|---|
| E4: the composition sweeps (a sweep script over `pixi run flow --set`, no DSE5) | tier A, PORT1, HWCOST | C3 |
| E7: forge against Vitis opt and dace, Bambu and XLS on tier done and untiered, W ≤ 16 | HSYN1, track B's step 3 | the blocks against HLS |
| E1: mvt and atax (gesummv if time) on vsim, 3–5 compositions each; E2's times logged on the same runs | tier A | C4, C1 |
| The cluster synthesis (2–3 composites) and E11, the homogeneous cluster (snax_cluster's cores, no accelerator, hand-written C) | E1 for the composites; CSYN1 sets up the flow before it | cluster PPA, the Snitch baseline |
| E5: Transformer decode, one head (q·Kᵀ, softmax, ·V) at a short context in L1 | tier A; `exp`, `max` and division (M7 tiers B and C) | the end-to-end case |

Dates: 27 Nov, E5's cut-off (softmax not running on SNAX-GVSoC moves E5, and E6 with it, to future work); 4 Dec, the abstract's result sentence from E4; 11 Dec, the abstract and the data freeze (figure fixes only after it); 12–18 Dec, writing. PAPER §2 and §3 are written alongside from now.

## Open Items

Questions not decided yet. Numbers are never reused; a closed item is removed, and where it closed is in git history. Next free number: 126.

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
95. The cycle gap between SNAX-GVSoC and snax_cluster's RTL, for BASE2 (D123): on the full set, measured on vsim with snax_cluster at #666 and snax-forge at #43, every `forge counter` is equal and every run has `forge data: 0 errors`; the windows are within −4.5% to +8.6%. The quick set measured again in SLOT1 C1 (snax_cluster `ry/slot1` at `5b59191e`, tag `slot1-s3`, snax-forge `slot1-f3`, SNAX-GVSoC `1e988679`, image `d3d8f5572ff4`): every `forge counter` equal and `forge data: 0 errors` again, the windows within −4.8% to +6.8%; dot 1's two streamer sets (`dot_1`) and its encapsulated form (`dot_1_enc`, the earlier `dot_1` row's shape) are its new rows. relu's row is from B2 R2 (snax_cluster main at `ea9c837b`, tag `b2-r2`), axpy's and axpy_baked's from R3b (tag `b2-r3b`), where the quick set measured the same again each time. axpy's K1 write costs SNAX-GVSoC 11 cycles over the baked design and the RTL 4. rowsum_2's two rows are from R4 (tag `b2-r4`): the four lanes of the contiguous layout (rows 64 words apart, all starting in bank 0) collide on their first reads only; each lane is its own channel with its own FIFO, so they then run a cycle apart and never meet again, and padding saves those 3 cycles, the same on the model and the RTL, far below the 4-way conflict a static bank map predicts. R5's rows are from tag `b2-r5` (snax_cluster main at `cad966bf`, SNAX-GVSoC `356c659`, image `d3d8f5572ff4`), where the quick set measured the same again: vsum, now `chisel_adder_tree` (B0), keeps its counter and runs 271 on vsim against 266; matvec_seed's q += A p has matvec_1's counters on SNAX-GVSoC (1027, 1027) and a window of 2842 against its 2845, so the seed's add costs nothing (matvec_1's row needs snax_cluster #688); matvec_seed_serial's sum is 4 · 1024 + 3, one lane a cycle. matvec_seed and rowsum_serial are the first rows whose counters differ, each within its fill (C§17): matvec_seed's 1028 and 1025 on vsim, rowsum_serial's 67 against 65. The windows run from −5.9% (rowsum_serial) to +6.8%. MVT1 and REUSE1 (snax-forge `69c1a51`, snax_cluster main at `82defad8`, SNAX-GVSoC `a95b52bf`, image `d3d8f5572ff4`): the quick set measured the same again on vsim; mvt's rows on SNAX-GVSoC, contiguous and `padded_rows`, each `_reuse` recipe with its hand-written twin's counters (the units' names aside) and window, and padding shortens half 2's transposed read of A (51 → 36 on `mvt_1`'s shared `vmul`, 54 → 39 on the dot group). Their vsim column is the hand-written recipes' (snax-forge `381876d`, the RTL built in image `c5db84ad7bce`): `mvt_1`, `mvt_1_dot2`, `mvt_2` and `mvt_1_dot2` with `padded_rows` each PASS, `forge data: 0 errors`, two counters within their fill (`mvt_1`'s `sum_vsum` 33 against 35, fill 2; the padded `tmp2_mult_mvt` 38 against 39, fill 3), the windows +1.2% (`mvt_1`) to +5.3%; `mvt_1` and `mvt_2` with `padded_rows` are not measured on vsim. The 40 × 40 rows are MVT1-2 step 1's `pixi run forge-set mvt` on SNAX-GVSoC (snax-forge `cd52f3c`, SNAX-GVSoC `a95b52bf`); mvt_2 at `W_x1 = W_x2 = 8` is refused by the design check in both layouts (85 TCDM crossbar inputs, the platform takes 64), so the set stopped there. Half 2's transposed read (the fourth counter of `mvt_1`, the third of the others) against the bank rule (W lanes down a column of pitch p meet 32 / gcd(p, 32) banks): at W = 8 contiguous (p = 40, four banks, two-way) it is 361 and 347 against 232 and 217 padded (p = 41), half 1's own; at W = 4 the rule predicts no conflict, yet contiguous costs 481 against 416 and 452 against 410. MVT1-2 c2's stall traces of `mvt_1` (W = 4 and 8, both layouts) place it between ports: contiguous A stays on one set of banks down a column while y2 sweeps the banks, so A and y2 refuse each other on nearly every other firing all through the task (341 stall cycles at W = 4, 408 at W = 8, beside A's own 910 at W = 8); padded, A moves with y2 and they meet about once per column of A (52 and 152). Outside the tasks, padding leaves the DMA and the gaps between tasks within 16 cycles; `mvt_1` at W = 8 padded starts its first task 114 cycles later, as core0's setup waits twice (142 and 190 cycles) on what looks like instruction fetch behind the DMA's load of A, against once (141) contiguous, a code-placement effect and not the data layout's. The gaps are the core's timing (open items 65, 67; snax-gvsoc NOTES.md, "The comparison with snax_cluster's RTL"). Their vsim column is MVT1-2's (the same bundles, snax_cluster main, image `c5db84ad7bce`): nine PASS, every run `forge data: 0 errors (80 words)`; the two FAILs are half 2 at W = 8 with `padded_rows`, where the RTL runs 4 cycles under SNAX-GVSoC on one counter each (`sum_1_vsum` 201 against 205, fill 2; `tmp2_mult_mvt` 213 against 217, fill 3), the case where A, y2 and the output still meet on about 40% of firings (open item 120). The windows are within −4.0% to +4.8%. On the RTL `padded_rows` shortens every window; on SNAX-GVSoC it lengthens three of five (mvt_1 at W = 8, mvt_1_dot2 and mvt_2 at W = 4), so the layout's window ranking flips there, while the counters rank the same on both (open item 120).

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
    | mvt_1, mvt_1_reuse at 8 × 16 (estimate 136) | 34, 35, 4, 51, 35, 6 (vsim 34, 33, 4, 51, 35, 6) | 664 | 656 |
    | mvt_1, mvt_1_reuse at 8 × 16, padded_rows (estimate 136) | 35, 35, 4, 36, 36, 6 | — | 655 |
    | mvt_1_dot2, mvt_1_dot2_reuse at 8 × 16 (estimate 74) | 37, 4, 54, 6 | 533 | 506 |
    | mvt_1_dot2, mvt_1_dot2_reuse at 8 × 16, padded_rows (estimate 74) | 37, 4, 39, 6 (vsim 37, 4, 38, 6) | 520 | 500 |
    | mvt_2, mvt_2_reuse at 8 × 16 (estimate 74) | 37, 4, 54, 6 | 533 | 506 |
    | mvt_2, mvt_2_reuse at 8 × 16, padded_rows (estimate 74) | 37, 4, 39, 6 | — | 500 |
    | mvt_1 at 40 × 40, W = 4 (estimate 1622) | 403, 404, 10, 481, 403, 10 (vsim 403, 404, 13, 481, 404, 13) | 2382 | 2346 |
    | mvt_1 at 40 × 40, W = 4, padded_rows (estimate 1622) | 408, 404, 13, 416, 404, 13 | 2365 | 2312 |
    | mvt_1 at 40 × 40, W = 8 (estimate 812) | 220, 204, 5, 361, 204, 8 (vsim 220, 204, 8, 361, 204, 8) | 1692 | 1641 |
    | mvt_1 at 40 × 40, W = 8, padded_rows (estimate 812) | 231, 203, 8, 232, 205, 8 (vsim 231, 204, 8, 232, 201, 8: FAIL) | 1585 | 1651 |
    | mvt_1_dot2 at 40 × 40, W = 4 (estimate 824) | 408, 13, 452, 13 | 1521 | 1451 |
    | mvt_1_dot2 at 40 × 40, W = 4, padded_rows (estimate 824) | 410, 13, 410, 13 | 1467 | 1496 |
    | mvt_1_dot2 at 40 × 40, W = 8 (estimate 414) | 216, 8, 347, 8 (vsim 215, 8, 347, 8) | 1209 | 1241 |
    | mvt_1_dot2 at 40 × 40, W = 8, padded_rows (estimate 414) | 217, 8, 217, 8 (vsim 219, 8, 213, 8: FAIL) | 1069 | 1036 |
    | mvt_2, mvt_2_reuse at 40 × 40, W = 4 (estimate 824) | 408, 13, 452, 13 | 1521 | 1451 |
    | mvt_2 at 40 × 40, W = 4, padded_rows (estimate 824) | 410, 13, 410, 13 | 1467 | 1496 |

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
112. A buffer link's energy against SYN2 (DRY1-3): dot 2's `tmp0` buffer, which loads every beat, is 1.59× (nangate45) and 1.49× (nangate15) high against CHAR1's `link/d1_w8` stream point, not explained. dot 6's `out_partial` was 2.65× and 3.13× high while a buffer loaded once per firing of its producer; it now loads once per output (C§20), once per job behind the Mac.
113. One unit across several DFG regions (atax's A and Aᵀ on one GeMM): a unit bound at several nodes, trading throughput for area; amends D112 (Ryan, 27 Oct 2026, for the paper). tier A's atax is the first case.
114. PORT1's shared crossbar ports exist on SNAX-GVSoC only: snax_cluster's mux follows when a shared-port design is in E4's result, in E1 or in the cluster synthesis, before E1 runs it; until then those runs are marked model only, outside D123's check.
115. An ALU SIMD: the `simd` BRM's `op` as an R/W CSR, its op set a design param, so a design pays only for the ops it can select; worth it once one unit runs several ops, a unit bound at several nodes (open item 113).
116. `gen.span` refuses a unit of span > 1 reading an outer input (a serial unit straight from its streamer: vsum or rowsum 1 on `chisel_serial` at W > 1, the beat held for span cycles); lifted when E7 or a kernel needs such a point.
117. Reusing a composite's recipe, done in REUSE1 (#60, D166). Left: a block and a group of one name (`dot`) are selected together; a collapsed matvec in mvt has no outer loop, so both halves read `A[0:M, 0:N]` and the transpose shows one level down or by its mark; `name_block` is left out of the marks (it takes any node).
118. The run viewer (`pixi run view`: banks, interconnect, transactions) is kept working but gets no new features until after the paper (Ryan, 6 Nov 2026); the DFG views (`view-dfg`, `edit-dfg`) stay first.
119. The cost estimate reads the graph only, so a layout pass's padding (`padded_rows`: rowsum 2's 8 × 65 words against 8 × 64) is not in its memory bytes; the design report and the memory plan do count it (B2 R4).
120. MVT1-2's vsim check (open item 95): (1) under heavy bank conflicts (mvt's half 2 at W = 8, `padded_rows`) a counter on the RTL is 4 cycles under SNAX-GVSoC's, past its fill, likely the two crossbars resolving collisions differently; (2) the window differences between layouts are smaller than the model's window error (−4.0% to +4.8%), so three of five layout pairs rank the other way on the RTL, the model's head slower where core0's setup waits on instruction fetch behind the DMA (open item 65). E1 ranks on counters as well as windows, or compares designs further apart than that error; both looked at again if E1's mvt points depend on them.
122. A Mac expansion: W multipliers into a W → 1 adder tree and one accumulator, as one unit. Splitting both of dot's maps at W (`split_map`, no `split_reduce`) and then `fuse_map` gives one node, W products summed per firing and accumulated over the N / W firings, that nothing binds today: `mac` refuses it (it keeps one partial per lane, so `split_reduce` first, D150) and `accumulate` takes a reduce tasklet only. Its datapath is dot 3's, SimdLanes → wire → TreeAccumulate, which binds it as two units in one group; one unit from one fused node would drop the second unit's counters from the control unit, which its cost would show (Ryan, 10 Nov).
123. Programming the next task while one runs: a design-time option to latch a group's settings at its start, off by default (D168); if built, JOBS counts down so its copy is its counter, and one copy of T per unit whose T comes from a JOB_BEATS.
124. job_width sized by the flow from the largest JOBS and T of the target workload, a design-exploration knob; one job_width of 32 stays the default (D152), like for like with the HLS tools' u32.
125. `<tag>_peripherals` made optional for minimal-area points, the CSR slots kept and reading 0 (D168).

## Sync Reminders

- After every change done with Claude, update this file and log any design change in `docs/DECISIONS.md` (rules at its top).
- When a milestone closes, compact `docs/DECISIONS.md` and remove the milestone's done tasks from this file (D117).
