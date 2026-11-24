# SNAX-FORGE Status

Open work: milestones, tasks and open items. What is built, and where its code lives: `docs/ARCHITECTURE.md`; why: `docs/DECISIONS.md`; the artefacts: `docs/CONTRACTS.md`. Done tasks are removed when their milestone closes (D117); their tables are in git history.

Status values: `todo`, `brief` (brief written), `wip`, `done`, `deferred`.

## Milestones

Done: M1, M4a, M3, M4b, M5, X1 (GVSoC learnt in `KULeuven-MICAS/snax-gvsoc-experiment`, which stays a playground), M16 (SNAX-GVSoC checked against snax_cluster's RTL, D123; what it measured in snax-gvsoc's `docs/notes/NOTES.md`), M14 (generation to SNAX-GVSoC and snax_cluster, D111). Open, in order (D116):

| Milestone | Content | Status |
|---|---|---|
| M13 | The block library: basic blocks, their composition (D112) and dot's six variants; M7's tier A; BASE1's forge side | `wip` |
| M7 | Front ends: NPBench's DaCe sources in tiers, integers first (D113); tier A is in M13 | todo |
| M15 | Tiling: a DMA per tile | todo |
| M9 | `jacobi1d` | todo |
| M11 | Layers: conv2d, softmax and attention, HDC operations | todo |
| M12 | Blocks and models: ResNet, Transformer, NVSA | todo |
| P1 | Paper plan, alongside | `wip` |
| P2 | Paper draft | todo |
| M6 | Contract freeze, when the contracts call for it | todo |
| M8 | Automated DSE over recipes, when sweeps call for it | todo |
| M10 | Cosim of one accelerator (D52) | todo |

M7's tiers interleave with the rest (ARCHITECTURE §8): tier A inside M13, then tiers B, C and P, M15, tier D with the GeMM (M9 inside it), then tier E.

## Later Milestones

The tasks are a first cut; each milestone is planned in a round of its own.

| Milestone | Tasks | Content |
|---|---|---|
| P1 | PAP3 (`done`), PAP2 (`brief`), BASE1, BASE2 | PAP3 chose the ISPASS 2027 paper's experiments and the order of the work (PAPER §6, Next Up). PAP2: related work after the reading, DSAGEN and OverGen first (PAPER §8.4). BASE1 and BASE2 (E2) are not run as tasks: a sentence or two in PAPER §3 from the times the runs log anyway |
| M13 | B0 to VIEW3, TIERA1, TIERA2 (`done`, below); track A: TIERA3 (`wip`), E4S1; track C: SNB1; track D: E2E1–E2E5; after the paper: L2V1, XBAR1, DACE2 | Its tasks, below (M13) |
| M7 | KER0 (`done`), FE1b, FE1c, PACK0 (`done`), PACK1, FE1d, FE1e, FE2, FE3 | The tiers of PAPER §6.1 after tier A (D113), which M13 does. KER0 staged every target kernel in `kernels/`, not linked to the flow (`kernels/README.md`, `pixi run kernels`). FE1b is tier B; FE1c tier C; PACK0 packed int32 and int8 into 64-bit words with NumPy's arithmetic (D142–D144); PACK1 keeps tier P: requantization and the int8 kernels PACK1a (M13) leaves out; FE1d tier D; FE1e tier E. FE3 is C2's coverage table |
| M15 | LOW3b | A tile transform and a DMA per tile, an L2 slice per tile, reductions over tiles (open item 44) |
| M9 | J1 | Stencil BRM with reuse and double buffering (see open item 76) |
| M11 | LAY1–LAY3 | conv2d tiled on a MAC-array BRM; softmax and attention with several BRMs; HDC bind, bundle and similarity |
| M12 | BLK2–BLK4 | A ResNet basic block, a Transformer encoder block, an NVSA pipeline; JEPA if it adds a case |
| P2 | PAP4 | The paper, every number from a checked-in command; the tests checking them follow after the submission |
| M6 | F1–F3 | Package layout and CI; versioned schemas for every contract (closes open item 1); registries and namespaced attrs |
| M8 | DSE1–DSE5 | Search over recipes; replacement across the BRM library; parameter choices; memory-plan policies (D86); a sweep runner |
| M10 | COS1, COS2 | One accelerator's RTL in place of its model, and the mismatch of its latency and II per BRM (D52) |

## M13

Done, each in git history:

| Task | What it gave | D-numbers |
|---|---|---|
| B0 | The block library's flow pieces: cluster headers per design point, MatMul and Transpose, maps over several parameters | — |
| B1 | `accelerator.json` as the one description (C§10), per-group CSRs and pins | D145–D148 |
| B1x | dot 2–6 and matvec 1–6 as one accelerator each, linked units under one control unit | D112, D147, D149, D150 |
| CLN2 | The clean-up after B1x; the local RTL check (`forge-set`) | — |
| VIEW2 | The SNAX-DFG editor (`edit-dfg`) and the ideal estimate | D55, D72, D73, D155 |
| SPAN1 | A unit reads at every span; `design.serial_one_lane` | — |
| SYN1 | forge's 48 points on Nangate 45 and 15 nm beside hls-exercises (`results/forge_vs_hls.md`) | — |
| CTRL1 | The csr as the interface; nested group counting | D152, D153 |
| CHAR1 | Each block emitted alone (`Emit --block`, `gen-block`) and characterized; the cost model | D154 |
| CHAR1b | The cost model in snax-forge (`snax_forge/char/<tech>.json`), in the estimate and the views (C§13, C§14) | — |
| DRY1 | `examples/dryrun/1_vecmul.md` | — |
| DRY1-2 | `examples/dryrun/2_dot_1.md` | — |
| SLOT1 | A design's accelerators as streamer sets of one snax_cluster entry, run in series | D123, D157, D158 |
| DRY1-3 | `examples/dryrun/3_dot_2_6.md`, dot 2–6 against SYN2 | D98 |
| B2 | relu, axpy, rowsum 2: one `simd` BRM over `SimdLanes`, CSR scalars, the in-place write, `padded_rows`, `fuse_seed` | D68, D82, D86, D102, D149, D150, D161–D163 |
| RAM1 | SRAM area and energy per access from CACTI 7 at 45 nm, a lookup table and macros | — |
| STRM1 | The streamer's reader and the TCDM interconnect priced over their parameters | — |
| REUSE1 | A recipe as the unit of reuse: `lift`, `apply_recipe`, `name_block` | D164–D166 |
| CTRL2 | The settings in the csr, read live by the control unit | D168 |
| PORT1 | Shared movers behind a data-side mux (`share_ports`); `snax-forge-check`'s four verdicts | D85, D123, D169 |
| HWCOST | A design's cost in five parts from snax-forge-experiments' tables (`import-costs`) | D167 |
| CSYN1 | The HLS comparison (`results/compare/`); a design to a synthesized cluster (`pixi run cluster rtl\|syn\|gls`) | D123 |
| CORE1 | One Snitch core drives the DMA and the accelerator; run tests; the quiet build | D131, D135, D170, D171 |
| CLN3 | Housekeeping; TOOL.md as the command reference | — |
| PWR1 | The synthesized cluster's dynamic energy per component and section | D135 |
| PWR2 | dot_v3 at W = 2 to 32 through the cluster flow beside HWCOST | — |
| CG1 | Clock gating in the cluster's synthesis | — |
| CG2 | HWCOST's block tables from gated blocks | — |
| CG3 | The gated cluster over W; clock-gated synthesis the only basis | — |
| CG4 | HWCOST on the clock-gated tables | D154, D172 |
| PACK1a | int8, int16 and int32 presets; a narrow output packed by a `FifoPacker` | D113, D142, D143, D173–D175 |
| CG6 | The HLS points synthesized clock-gated | — |
| CG5 | forge's accelerators and `xbar_mix` gated; `char.json` the gated block table | — |
| CHAR2 | The char blocks at every width pair; the lookup at a block's width pair | D176 |
| GE1 | The dynamic energy (`--energy`); the crossbar priced by input kind | D177, D178 |
| VIEW3 | The DFG page as a flat graph laid out by ELK | D185 |
| TIERA1 | mvt's forms at 64 × 64 (`examples/dryrun/4_mvt.md`): schedules, the slicer, the DMA's transposed copy, the broadcast and seeded `Mac`, recipes named by version | D163–D165, D179, D180, D182–D184, D186–D188 |
| TIERA2 | bicg's forms and the one pass over A for bicg and mvt (`examples/dryrun/5_bicg.md`): the fork, the bank, the hold, carried and seeded seeds | D154, D172, D177, D189–D194 |
| CLN4 | A lean base before atax: snax-gvsoc's `forge/`, the RTL fixtures and the recorded runs one design per integration feature at their smallest sizes, `forge-set` quick and full by feature; composite tests per design, a refusal table per package, the rest cut to the coverage bar, with each composite's estimate and profile and every recipe's estimate totals checked against snapshots; dead code out; the documents cut to what holds; the markings guarded | D195 |


CLN4's sizes, before and after:

| What | Before | After |
|---|---:|---:|
| snax-gvsoc `forge/`, lines (folders) | 38 464 (48) | 6 942 (11) |
| recorded runs `tests/run/fixtures/`, lines (runs) | 14 635 (7) | 6 340 (4) |
| RTL fixtures, lines (designs) | 1 245 (35) | 571 (13) |
| `tests/**/*.py`, lines | 31 356 | 20 675 |
| `snax_forge/**/*.py`, lines | 37 326 | 37 205 |
| docs: STATUS, CONTRACTS, DECISIONS, TOOL, ARCHITECTURE, PAPER, characters | 121 043, 139 144, 55 476, 32 110, 17 787, 29 605 | 36 151, 92 567, 39 848, 24 517, 13 490, 29 256 |
| `examples/dryrun/`, characters | 118 125 | 84 994 |

To do, in four tracks, each task in its own thread, branch and PR (the ROAD1 plan, 16 Nov 2026). Each task runs in a clone set of its own, so tasks in parallel never share `out/`, a GVSoC build or snax_cluster's imported bundles:

- **Track A, tier A and E4** (snax-forge, snax-gvsoc where a task changes the model):
  1. **TIERA3** (`wip`, branch `ry/tiera3`): atax's forms at 64 × 64. The two-pass forms by reuse (`recipes/atax/`, `forge-set atax`; half 2 waits on half 1's writer of t under every schedule), the DMA copy at int8, then one group with t held over a link (D197) and A read once through a FIFO from the fork (D196), `atax_v3` and `atax_v4` (sandbox, LOWER, estimate and `accelerator.json` done; SNAX-GVSoC's `SnaxGroup` and the Chisel next); half 2's `Mac` multiplies int8 by int32 (X98).
  2. **E4S1** (kickoff from TIERA1's plan): E4's sweep script over `pixi run flow --set` (no DSE5) and its figure, first on what runs today (W on dot 3, matvec and mvt_v1; shared against dedicated ports), then on each tier A form as it lands. The bank-bound sweep at int64 and int32: a dot reads $2 W b$ bytes a cycle against $32 \cdot 8 = 256$, so it crosses at W = 16 (int64) and 32 (int32), and W stops at 32.
- **Track B, costs** (snax-forge-experiments, snax-forge's `char`; Nangate 45 nm only): E4's energy axis reads `--energy` or the static estimate; GE1's left-overs are open items 136 and 137.
- **Track C, the Snitch-only baseline** (snax_cluster, snax-forge-experiments; E11):
  1. **SNB1** (`brief`): the same kernels as hand-written C on `rv32ima` Snitch cores with no accelerator, every tier forge runs, one source per kernel, the type a typedef, the core count from the runtime, forge's inputs; three configs (one core, 1 DMA + 4 cores, 1 DMA + 8 cores) with forge's L1, banks and I-cache, synthesized gated through `pixi run cluster` (open item 88); area and energy per task in coarse parts (memory, interconnect, cores against forge's streamers and accelerators).
- **Track D, end to end** (snax-forge; after track A, best effort, cut from the bottom; each case sized to fit L1, shortcuts allowed for what is not built):
  1. **E2E1**: an MLP at batch 1 (matvec, bias add, relu, matvec).
  2. **E2E2**: a small conv2d.
  3. **E2E3**: a Transformer decode head (q·Kᵀ, softmax, ·V; E5): max as a fold, exp and the divide as the shortcuts its brainstorm chooses.
  4. **E2E4**: an image pipeline (grayscale, blur, threshold; E12).
  5. **E2E5**: a ResNet basic-block slice (conv, requantize, relu, skip add; E6).
- **After the paper**:
  - **L2V1**: each container the DMA moves drawn in the DFG views as its L2 copy with a DMA edge into L1, and a DMA row in the estimate from the memory plan, $\sum_c (\lceil B_c / 64 \rceil + t_0)$ for a 512-bit DMA and a fixed cost $t_0$ per transfer read from a SNAX-GVSoC run, added to the total (amends D155), checked against the run's DMA tasks. The graph-level copy node stays open item 99.
  - **XBAR1**: an interconnect marker per running accelerator, $u = \sum_s \mathrm{ch}_s / n_{\mathrm{banks}}$ over its streamers' channels, green at $u \le 0.5$, orange up to 1, red above, the thresholds platform fields (D85), a design note (C§6) in the estimate, `design.md` and the editor (vecmul: W = 4 green, 8 orange, 16 red).
  - **DACE2** (open item 130): a `dace_tuned` case in hls-exercises beside `dace`, tuned only with DaCe's own transforms and library expansions, then through Vitis, synthesis and power as the other HLS cases (E7).

## The local RTL check

The local RTL check (D123) runs one program on SNAX-GVSoC and, through snax_cluster's `snax-forge-check`, on vsim, and gives each bundle a verdict: PERFECT, NEAR (a counter's gap beyond its fill up to 10 % of its count), WARNING (up to 15 %) or FAIL. The full set as last measured, 26 bundles, every `forge data` exact, 17 PERFECT and 9 NEAR; cycles are the window, then the `forge counter`s in brackets:

| Bundle | SNAX-GVSoC cycles | vsim cycles | Verdict |
|---|---:|---:|---|
| vecadd_v1_1 | 216 (19) | 247 (19) | PERFECT |
| vsum_v1_1_2 | 156 (11) | 192 (11) | PERFECT |
| dot_v1 | 251 (19, 19) | 266 (16, 19) | NEAR, 0.0 % beyond the fill |
| dot_v1_1 | 303 (19, 19) | 300 (19, 17) | NEAR, 0.0 % beyond the fill |
| dot_v4 | 422 (133) | 445 (133) | PERFECT |
| axpy_v1 | 216 (19) | 247 (19) | PERFECT |
| vsum_v1_1 at i8, W = 8 (packed d_out 2) | 158 (11) | 180 (11) | PERFECT |
| matvec_v3 at i8, W = 8 (packed) | 1325 (1028) | 1345 (1028) | PERFECT |
| vecadd_v1_1 at W = 1 (sliced) | 247 (68) | 273 (68) | PERFECT |
| vecadd_v1 | 209 (19) | 232 (16) | NEAR, 0.0 % beyond the fill |
| vsum_v1 | 222 (67) | 263 (65) | NEAR, 0.0 % beyond the fill |
| rowsum_v3 | 223 (67) | 234 (67) | PERFECT |
| vecmul_v1_1_1 | 209 (19) | 232 (16) | NEAR, 0.0 % beyond the fill |
| dot_v2 | 422 (133) | 445 (133) | PERFECT |
| dot_v3 | 419 (132) | 439 (132) | PERFECT |
| dot_v5 | 425 (136) | 445 (136) | PERFECT |
| dot_v6 | 425 (137) | 445 (137) | PERFECT |
| dot_v7 | 300 (16, 67) | 321 (19, 65) | NEAR, 0.0 % beyond the fill |
| matvec_v1 | 2822 (1027, 1027) | 2866 (1024, 1027) | NEAR, 0.0 % beyond the fill |
| matvec_v5 | 3353 (2101) | 3397 (2101) | PERFECT |
| matvec_v6 | 3308 (2057) | 3349 (2057) | PERFECT |
| matvec_acc_v1 | 2835 (1024, 1027) | 2880 (1027, 1027) | NEAR, 0.0 % beyond the fill |
| matvec_acc_v2 | 5924 (1027, 4099) | 5944 (1027, 4099) | PERFECT |
| relu_v1 | 163 (18) | 191 (18) | PERFECT |
| rowsum_v2 | 357 (129) | 387 (134) | NEAR 2.3 % (open item 129) |
| rowsum_v2 padded_rows | 363 (131) | 376 (131) | PERFECT |

## Next Up

The ISPASS 2027 plan (PAP3, PAPER §6), as ROAD1 set it on 16 Nov: SNB1 and E4S1 now, track D after track A. Then, as each lands:

| Step | Waits for | Gives |
|---|---|---|
| E4: the composition sweeps (E4S1) | tier A; GE1's energy and CHAR2's tables for the final sweeps | C3 |
| E7: forge against Vitis opt and dace, Bambu and XLS on tier done and untiered, W ≤ 16 | HSYN1, CSYN1 Part A (both done) | the blocks against HLS |
| E1: mvt and atax (gesummv if time) on vsim, 3–5 compositions each, from tier A's rounds | tier A | C4, C5 |
| E11: the Snitch-only clusters against forge's (SNB1) | SNB1's flow; CSYN1's and PWR1's (done) | cluster PPA, the Snitch baseline |
| E5, E6 and E12: the end-to-end cases (E2E1–E2E5) | track A | where the cycles go per case |

Dates: 20 Nov, track A's target (fine if it slips); 27 Nov, the softmax gate (softmax not running on SNAX-GVSoC by then moves E2E3, E5, to future work); 4 Dec, the abstract's result sentence from E4; 11 Dec, the abstract and the data freeze (figure fixes only after it); 12–18 Dec, writing. PAPER §2 and §3 are written alongside.

## Open Items

Questions not decided yet. Numbers are never reused; a closed item is removed, and where it closed is in git history. Next free number: 165.

1. BRM dataflow notation: `affine` and `stream` (D70), frozen in M6.
2. What automated search reads and how a sweep is configured (DSE1, M8).
4. Stream as a proposer for DNN-shaped workloads, and bank conflicts from the memory passes (D86) in an estimate of our own, on top of the ideal one (D155). Parked.
18. The L1 address map is fixed and word-interleaved, so the design point states no banks; a registered map kind if snax_cluster's differs.
27. Generating a BRM from Chisel, SystemVerilog or HLS sources. Far future.
28. Implementations whose `source` is SystemVerilog or HLS: only `chisel` until one is needed.
33. An MLIR importer that writes `.snaxdfg` directly.
34. A hand-edited `.snaxdfg` runs through the flow (D108) but cannot be replayed or swept, and the design checks' fixes speak of the recipe.
36. Control flow in SNAX-DFG: a sequential `loop` kind and a `branch` kind, registered when a kernel needs them (M9).
37. The expression grammar has `+ - * //`, `max` and `min`, constant casts to the output's or the op's type and value casts (`int32(x)`, a kernel's `astype`, D143 · D175): branches need comparisons.
38. The DMA moves a container as contiguous whole beats (D105), and a `transpose` node's copy inside L1 one element a burst (D182); strided or 2-D transfers to and from L2 when a kernel needs them.
40. Switching runs in the viewer clears the selected cycle.
41. A report glossary, a "where the cycles went" ranking and the knobs' `--set` paths: on hold.
44. Reductions over tiles need partial sums and a final combine; until then no tile loop sits around a `reduce` memlet (LOW3b).
50. SNAX-GVSoC's speed on large runs, from the times the runs log (E2 is a sentence or two in PAPER §3); timing-only runs and extrapolation only if it is too slow for C1.
53. A link between two accelerators: amends D12 and D51 and the SNAX shell. Parked (D112).
54. Which kernels take PolyBench's SMALL in a 128 KiB L1 (M13 tier A); the NumPy tolerance for float sums waits for the float pass (D113).
65. Instruction fetch: snax_cluster's L0 prefetches the next line and taken branches' targets, GVSoC's I-cache does not, so SNAX-GVSoC misses more lines for fewer cycles each (tens of cycles in fixed-cost phases); a forge counter depends on the core's timing between the streamer's and the accelerator's start writes, so it is compared with care (D123). Modelled when a kernel's numbers need it.
66. Atomics to DRAM are one access on SNAX-GVSoC, a read-modify-write on the RTL testbench, so the exit's barrier is about 50 cycles shorter here. Outside the cluster.
67. The core's own timing beyond its loads (D125): `dmstati` polled every 5 cycles on the RTL against 2, a SNAX CSR poll 5 against 6, the DMA core's copies a cycle or more faster (snax-gvsoc NOTES.md, "The core and its timing"). Kept as is (D123).
68. Generation is spread out: snaxgen and Chisel in snax_cluster (the wrappers, `streamer_csr_addr_map.h`), clustergen and regtool outputs copied into snax-gvsoc's `sw/` (D126), `snax_hjson` and its timing constants, `snax_forge/gen` with the bundle and the runtime's cluster headers. One generator for the rest when a design needs it.
69. The SNAX crossbar's AXI input is wired but no program sends through it, on either side (snax-gvsoc NOTES.md, "The crossbar and the sparse interconnect").
70. On SNAX-GVSoC a store or AMO after a pending L1 load waits for the load's data; the RTL sends it. No measured loop has one.
71. An L1 load after a DRAM load reaches the crossbar up to 13 cycles early; the core's timing is right, and no streamer request meets it yet.
74. The iDMA through the sparse interconnect as ports, instead of the superbank mux: a snax_cluster design change, for exploration.
75. Sparse granularity above 1: the crossbar supports it, checked against an elaborated RTL module, but no target uses it.
76. Where reuse lives: each loop in the streamer, the accelerator's state or its lanes, every unit described to the level above by r, ports, throughput, latency, state and area. A research direction (PAPER §10); J1 may become jacobi1d in three placements, with jacobi2d.
78. The streamer as an input of its own (amends D84 and D134): the remaining knobs and per-streamer overrides out of the platform, a frozen streamer file as input.
82. SNAX-GVSoC dumps the output regions at exit, so the flow could compare them itself; until then the check is the program's own (D136).
86. A custom prefix (`--set accelerator_slot=<x>`) is untested past the generator's file names, and nothing checks that the entry's name starts with `snax_forge`; several entries on one core are open item 108.
88. snax_cluster tests `snax_acc_cfg[0]`'s truthiness for an accelerator and clustergen writes schema defaults into the config, so the schema's `snax_acc_cfg` items carry no defaults.
89. snax_cluster's testharness leaves the cluster's CSR port undriven, so a task-queue CSR (`0x5FE`, `0x5FF`) would hang; its own PR.
90. snax_cluster's generated SV is not reproducible (`ComplexQueue`'s unseeded queue names, the streamer header's timestamp); its own PR.
91. vsum's window over N (64 to 4096) has no per-element gap: SNAX-GVSoC and the RTL differ by at most 16 cycles (2.2%), their slopes' ratio 0.996, with `forge counter` equal. Kept as is (D123).
92. The per-group busy and done pins (`<group>_busy_o`, `<group>_done_o`, D146) into a hardware barrier in the cluster, like `snitch_hw_barrier`, so a core waits on the groups instead of polling their STATUS: a snax_cluster template change, then SNAX-GVSoC to match. Parked: snax_cluster leaves them unconnected (Verilator `PINMISSING`, Questa `vopt-2718`).
94. A masked last beat, the hardware alternative to padding a tail (D144): a lane-valid mask from a CSR, so a tail needs no pads in the data.
96. The measurements of a linked group: per-unit and per-link events (the converter reads a linked group at its boundary) and the run viewer's view of it (the cluster view's mover arrows find no unit, a mover's instance being a member and the answer's unit the group); bank and port statistics, the export CSV, the area × cycles metric.
97. dot 1's row form at 16 × 512 does not fit: tmp0 needs 132 KiB of the 128 KiB L1, so it runs at 8 × 512.
99. L2 in SNAX-DFG: containers get a level (L2 or L1), and an L2-to-L1 move becomes a DMA copy node placed by a transform, so tiling can choose what moves per tile; today lowering derives the DMAs from the memory plan (D64, D86). The copy inside L1 exists (`transpose`, D182); the L2 level and the load nodes remain, with M15's LOW3b.
100. Unmapped work on a core: a node no accelerator runs is run by a core, its tasklet turned into C, which needs `graph.unbound` (C§6) relaxed and lowering to emit core code. Until then the estimate counts it at one cycle per iteration (D155, `UNMAPPED_CYCLES_PER_ITERATION`), which a measured core value could replace.
103. A unit with L ≥ 1 on a wire into a consumer s with T = 1 could borrow s's test as an L-0 unit does, with g_s + popcount(v_k) ≠ R_s (its results in flight counted), dropping `mac_jobs_counter` in dot 4–6.
106. A fold reading through a tail a container whose pad words its producer never writes (dot 1 at W_mul = 8, W_sum = 10: tmp0's 2048 words written, 2050 read) needs the program to write the fold's identity there before the fold's task; `connect.pad` refuses it until then (D144).
108. Several `snax_acc_cfg` entries on one core: the range decoder between entries in place of `snax_acc_mux_demux` and 0x7C6, and `snaxgen.py:620` building every entry; untested since SLOT1 drops the multi-ALU cluster.
109. Overlap of independent accelerators, left after D179's schedules: gathering starts that are not adjacent in the derived order (an independent accelerator after a dependent one waits for the latter's data), and the quiet build's power windows of a pipelined run (open item 140).
112. A buffer link's energy against SYN2: dot 2's `tmp0` buffer, which loads every beat, is 1.59× (nangate45) and 1.49× (nangate15) high against CHAR1's `link/d1_w8` stream point, not explained.
115. An ALU SIMD: the `simd` BRM's `op` as an R/W CSR, its op set a design param, so a design pays only for the ops it can select; worth it once one unit runs several ops, a unit bound at several nodes (D165, `atax_v1_1`).
116. `gen.span` refuses a unit of span > 1 reading an outer input (a serial unit straight from its streamer: vsum or rowsum 1 on `chisel_serial` at W > 1, the beat held for span cycles); lifted when E7 or a kernel needs such a point.
117. Reuse's leftovers (D166): a block and a group of one name (`dot`) are selected together; a collapsed matvec in mvt has no outer loop, so both halves read `A[0:M, 0:N]` and the transpose shows one level down or by its mark; `name_block` is left out of the marks (it takes any node).
118. The run viewer (`pixi run view`: banks, interconnect, transactions) is kept working but gets no new features until after the paper (Ryan, 6 Nov 2026); the DFG views (`view-dfg`, `edit-dfg`) stay first.
119. The cost estimate reads the graph only, so a layout pass's padding (`padded_rows`: rowsum 2's 8 × 65 words against 8 × 64) is not in its memory bytes; the design report and the memory plan do count it.
120. mvt at 40 × 40 on vsim: under heavy bank conflicts (half 2 at W = 8, `padded_rows`) a counter on the RTL is 4 cycles under SNAX-GVSoC's, past its fill, likely the two crossbars resolving collisions differently; and the layouts' window differences are smaller than the model's window error (−4.0% to +4.8%), so three of five layout pairs rank the other way on the RTL (open item 65). E1 ranks on counters as well as windows, or compares designs further apart; both looked at again if E1's mvt points depend on them.
122. A Mac expansion: W multipliers into a W → 1 adder tree and one accumulator as one unit. Splitting both of dot's maps at W and `fuse_map` give one node that nothing binds today (`mac` keeps one partial per lane, D150; `accumulate` takes a reduce tasklet only); one unit would drop the second unit's counters from the control unit, which its cost would show (Ryan, 10 Nov).
123. Programming the next task while one runs: a design-time option to latch a group's settings at its start, off by default (D168); if built, JOBS counts down so its copy is its counter, and one copy of T per unit whose T comes from a JOB_BEATS.
124. job_width sized by the flow from the largest JOBS and T of the target workload, a design-exploration knob; one job_width of 32 stays the default (D152), like for like with the HLS tools' u32.
125. `<tag>_peripherals` made optional for minimal-area points, the CSR slots kept and reading 0 (D168).
126. `ReaderWriter` (snax_cluster's `readerWriter/ReaderWriter.scala`: a full Reader and Writer behind one set of TCDM channels, the writer first each cycle, no mode bit) for sharing one mover between a read port and a write port (D169 shares a role's movers only); needs a SNAX-GVSoC model (`UNMODELLED_MOVERS` in `snax_cfg.py`).
127. snax_cluster's `streamer/Streamer.scala` lines 188–189 build each `ReaderWriter` from `readerWriterParams(i / 2)` and `(i / 2 + 1)`, where `(2 * i)` and `(2 * i + 1)` are meant: a second pair would take the first pair's parameters (GeMMX has one pair, so nothing breaks today).
128. SNAX-GVSoC's shared movers (D169) run a dot task 6 to 19 cycles under the unshared twin, beyond the fill, while the RTL gives the shared and unshared designs the same counters within a cycle (mvt_v1_2_2 at W = 4 and 8, mvt_v2_2 at W = 8; at 64 × 64 also W = 4 `padded_rows`, half 1 1026 against 1032, and mvt_v1_2_2 at W = 8 contiguous, 3240 against 3249). Suspects: the masked writer and the crossbar component's per-cycle order of events; first check the beat and stall traces of a dot task, shared against unshared. Open item 139 is its reverse.
129. One core's program timing on SNAX-GVSoC against the RTL (D170): the windows of the short designs run 14 to 23 % longer on vsim (28 to 41 cycles), and SNAX-GVSoC starts a group 10 cycles after its streamer at the first start after `snrt_dma_wait_all()` where the RTL does not (rowsum_v2 contiguous: 129 against 134, NEAR 2.3 %). Likely the core's DMA instructions and waits (open item 67); the quiet build's sections (D135), run on both, compare them section by section.
130. A tuned DaCe case for E7: beside `dace`, a `dace_tuned` case built only from DaCe's own transforms and library expansions, no hand-written C++, pragmas or edits to generated code: dot by `MapReduceFusion` at W = 1 and a `Dot` node with `FPGA_Accumulate` at W > 1, axpy by fusion and an `Axpy` node, vsum and rowsum at W = 1 only (`Reduce`'s FPGA expansion refuses vectors), relu left out unless a NumPy rewrite vectorizes. Every tuned form computes the right result in C-sim at N = 2048; the lane forms need the kernel restated (`np.dot`, vector-typed arrays), which the paper says. Experiments in hls-exercises `bench/dace_tuned/`.
132. Tier A at int8, left: the fold type of `@` and `np.dot` (the operands', so a sum that wraps, D143 · D175), mvt's transposed read by the core (open item 143), the outputs the paper needs, mvt's reuse recipes off `matvec_v2_1` if matvec_v2 is ever removed, and E4's int8 W sweep (dot_v3 crosses the 32-bank bound at W = 128 at int8, 16 at int64: W up to 256 needs 67 crossbar inputs).
134. Whether `Mac` and `SimdLanes` take a narrow multiplier (w_a + w_b bits sign-extended to `opType`, bit-identical to NumPy's product at `opType`), the experiment ROAD1 left for later; the char table would then gain its pairs.
135. snax-forge-experiments' `cluster-breakdown`, `cluster-energy` and `cluster-sweep` no longer reproduce their files: they read the forge points' ungated `alone` rows, which CG5 removed; `breakdown.json` and `energy.json` hold the dot 3 hwcost from before CHAR2 and D178 (control 0.4 % low, the crossbar at the read-write price). `ge-check` reads `hwcost/` directly instead.
136. The dynamic energy prices a shared mover (D169) at its own channels, where the static estimate prices each user at its port's: the two agree only when every user takes all the mover's channels. A split per user needs the port mux's counts in the `snap` lines; first needed by E4's shared-port points.
137. The dynamic energy under contention (dot 3 at W = 32): its streamers are 1.26 of the gated cluster's (0.95 at W = 16), so a refused request seems to cost a streamer less than a granted beat; and the cluster's datapath is 7.7 nJ above W = 16 for the same fires (inputs that change without a fire). A refused request's own price, or a count of input changes per unit, if E4's figure needs it.
138. One simulator counts a task at exactly its ideal cycles (the group rule's, no fill) and the other 1 to 4 past the fill: mvt_v4 `padded_rows` (W = 8 half 1 521 on vsim against 514), mvt_1's `vsum` tasks, the shared dot forms at W = 8 `padded_rows`, mvt_v3_1 `padded_rows` serial (515 against 521), mvt_v4_1 and mvt_v4_1_1 contiguous half 1 (514 against 518); not the seeded fold, nor A's type. The question is what gives a task's streamers their head start before START on one simulator and not on the other; a stall trace of half 1 at W = 8 padded would tell.
139. A one-bank column read (D86, contiguous) or two overlapped reads of A ordered differently on the RTL: at W = 4 the unshared dot forms run 3807 on vsim against 3785; mvt_v3_1's half 2 serial at W = 8 2998 against 3249 (NEAR 7.6 %), where mvt_v3 on the shared set is 3249 on both; bicg_v1 contiguous `pipeline` 3309 against 3263, padded 557 and 534 against 550 and 559 (NEAR 3.9 %); bicg_v2 `sequential` 532 against 523; atax_v2 (contiguous) half 2 524 on vsim against 532, 5 beyond its fill (NEAR 0.9 %), `mac_a` refused 128 times on SNAX-GVSoC. The RTL's ordering depends on which crossbar inputs ask; first check the stall traces of half 2 at W = 4, mvt_v1_2 against mvt_v1_2_2, on both.
140. The quiet build (D135) waits on every start right after it, so it runs a `pipeline` or `pipeline_compute` program (D179) serially and a pipelined run has no gate-level power windows (`--energy` prices it in one window of the default build, D177); a quiet build of a pipelined program would mark one window per set of overlapping runs. The default build also waits on the DMA twice at the end: harmless, left until `main.c` changes for another reason.
142. The slicer and the packer in synthesized designs: cluster synthesis of `vecadd_v1_1` at W = 1 and the in-design check against HWCOST, later.
143. The core transpose, `data_transpose` by `core` (D182, refused today): 4 × 4 byte tiles transposed in RV32 registers, about 10 000–11 000 cycles at 64 × 64, against the DMA's 10 923; extra information for the paper.
144. A copy under `serial` and the pipelined schedules (D179) sits where a load of its destination would, but the core is held by the DMA's 3-deep request FIFO while it issues the copy's 64 two-dimensional copies (about 170 cycles each), so what overlaps the copy is not measured; the estimate adds `copy_cycles` to every schedule for now. With the trace off, `report/run.md` gives the DMA's total busy, not the copy's own cycles.
146. Short bursts from L1 into DRAM are paced by the testbench's DRAM: B returns 2 per 7 cycles and at most 8 writes wait for theirs, `s-l1l2 64x1 B p64` busy 211 on the RTL against 155 on SNAX-GVSoC (snax-gvsoc NOTES, "The DMA").
147. `mvt_v4_1_1` through the forge cluster flow (synthesis, gate level, power, snax-forge-experiments), once its design is settled; the copy alone is measured on the baseline netlist (X95).
152. i16 and i32 for the type-preset kernels of tier A: `mvt_types`, `bicg_types` and `atax_types` define the presets, only `i8` has run; their forms at `i16` and `i32` on SNAX-GVSoC and vsim, at the widths the bank bound allows.
153. A pattern library for `lift`: today `matches` searches the import of every kernel some recipe is written on (12, imported through DaCe once per process, 0.75 s) and finds neither bicg nor mvt on itself (`lift … at=` does). Proposed: stored `.snaxdfg` patterns in a base set and opt-in extension sets, matched bottom-up, a later chain joining through a container an earlier one reads; with the reuse path's leftovers (a sub-recipe's `types` dropped without a word, the span check skipping spans it cannot evaluate) and the bank form as a matvec recipe (seeded too, D194). Brainstormed after TIERA3, atax giving the matcher one more case.
155. The estimate's check on accelerators that run at once (D179's note) adds their words a cycle against the bank count, not which banks they reach: bicg_v2 at int64 under `pipeline` reads 25 words a cycle over 32 banks (no note) and loses 44–63 % on SNAX-GVSoC, the column form's strip of A on the same 8 banks for its 64 beats; padded, 618 and 692 (+20 %, +35 %), the RTL the same. A check by the banks each stream reaches (from its base, strides and the row pitch, as D86 counts a column's banks) would name such pairs.
156. A re-read input at N = W reads its one beat again every row: LOWER gives bicg's p the loops `[1, 64]` with strides `[32, 0]`, so the streamer's repeat (a temporal stride 0 on loop 0) does not apply and p's 4 words are read 64 times on banks 0–3, where A's rows 8k start, A waiting under the dynamic priority (bicg_v3 72 against the rule's 67, 78 on vsim). Dropping a loop of bound 1 ahead of the stride-0 one would read p once. Beside it: the expected BUSY_COUNTER's fill leaves out a sliced input's first take (D180).
157. The converter's queue counts of a sliced input at its unit's head (D180), at `--trace stall`: `_words` pops a reader's word at the last firing that names it, where the reader's queue pops when the slicer takes it, so bicg_v2_1 and bicg_v4 at `stall` stop at `convert.invariant` (`mul_a lane 0: fifo counts ...`). Meanwhile a stall trace leaves the sliced head out (`--trace stall:mul_a,mul_b,mac_b,sum_out,mac_out,dma` on bicg_v4).
158. The bank as a memory (D191): a `SyncReadMem` of R rows read one cycle early, the `Mac`'s latency 2, against the registers built now; a choice of the bind once a design needs R beyond what registers make cheap. Not started.
159. A carried seed's reader started late (D192) instead of reading the first strip and dropping it: started by the control unit about CARRY jobs into the run, it would save M seed reads (32 words, about 3 % of `bicg_v4_1`'s L1 reads). It needs a reader of a streamer set to start apart from the others, which neither snax_cluster's streamer nor SNAX-GVSoC's has; for when the streamer changes anyway.
160. `vectors.json` gives each port's array in memory order, which ShellEndToEndSpec streams beat by beat; a strided reader (`bicg_v4_1`'s strips, its seed read back through L1) needs each port's order too, or the spec the streamer's loops, before its goldens are checked on the wrapper.
161. bicg_v2_1 at W = 16 under `pipeline`: s's counter is 272 on vsim against 267 on SNAX-GVSoC, 2 beyond its fill of 3 (NEAR 0.7 %); on SNAX-GVSoC s's 10 stall_in cycles are where its reader of A is refused, 13 of 15 times a lane by q's reader of A. The RTL's 5 more cycles are not placed (no trace of its readers' refusals); next check a waveform of the two A readers' requests and grants on vsim.
162. The converter refuses two recorded designs at trace level `stall` while their `task` and `beat` levels convert: bicg_v4_1 (the fork, the bank and the carried seed) at every size tried, "mul_a lane 0: fifo counts ... its words visible minus popped give ..." (the forked readers' FIFO counts against their grants), and mvt_v5 sequential at 16 × 16 and 64 × 64, "mac_x2_seed's beat -1 granted as [None, ...]" (the seeded `Mac`'s seed reader). Their recorded runs (`tests/run/fixtures/`) are at `beat` and `task` until the stall check takes them.
163. `sandbox/blocks.py`'s `_outer_of` raises `_NoMatch`, a plain `Exception`, which nothing catches on `apply_recipe` → `_fuse_outer`: a sub-recipe's `fuse_map` naming a node outside the region would escape the session's refusal handling instead of coming back as a refusal (found by reading; not run).
164. A tailed split interchanged with the loop under it (`rowsum_v2` at M = 6, W = 4) fails the reference check at `interchange_map` ('y' differs); either `interchange_map` refuses a tailed split with its own message, or the pad lanes are placed so the check passes.
165. atax's one pass beyond the bank: strips of S = R·W columns of y, only the first a one pass (t needs whole rows), t written to L1 there and read back by the others, so A is read $MN(2 - 1/K)$ times for K strips; the first strip's FIFO takes only that strip's beats, a fork that passes part of the beats (D196 forks every beat). At one strip a design needs D ≥ N/W and R = N/W.
166. DATA1, every kernel's inputs uniform over the whole type in every experiment (Ryan's, not scheduled); forge's small per-kernel ranges until then. At the full int8 range `atax_types`' y wraps at int32 ($|y| \le MN \cdot 2^{21}$), exact modulo $2^{32}$ on NumPy, SNAX-GVSoC and the RTL alike.

## Sync Reminders

- After every change done with Claude, update this file and log any design change in `docs/DECISIONS.md` (rules at its top).
- When a milestone closes, compact `docs/DECISIONS.md` and remove the milestone's done tasks from this file (D117).
