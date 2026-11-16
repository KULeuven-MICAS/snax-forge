# SNAX-FORGE Status

Open work: milestones, tasks and open items. What is built, and where its code lives: `docs/ARCHITECTURE.md`; why: `docs/DECISIONS.md`; the artefacts: `docs/CONTRACTS.md`. Done tasks are removed when their milestone closes (D117); their tables are in git history.

Status values: `todo`, `brief` (brief written), `wip`, `done`, `deferred`.

## Milestones

Done: M1, M4a, M3, M4b, M5, X1 (GVSoC learnt in `KULeuven-MICAS/snax-gvsoc-experiment`, which stays a playground), M16 (SNAX-GVSoC checked against snax_cluster's RTL, D123; what it measured in snax-gvsoc's `docs/notes/NOTES.md`), M14 (generation to SNAX-GVSoC and snax_cluster, D111; closed by CLN1). Open, in order (D116):

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
| M13 | B0, B1, B1x, CLN2, VIEW2, SPAN1, SYN1, CTRL1, CHAR1, CHAR1b, DRY1, DRY1-2, SLOT1, DRY1-3, RAM1, B2, REUSE1, STRM1, CTRL2, PORT1, HWCOST, CSYN1, CORE1, CLN3, PWR1, PWR2, CG1, CG2, CG3, CG4, PACK1a, CG6, CG5, CHAR2 (`done`); track A: TIERA1 (MVT1's rest inside it), E4S1; track B: GE1; track C: SNB1; track D: E2E1–E2E5; after the paper: L2V1, XBAR1, DACE2 | Its tasks, below (M13) |
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

Done, each in its PRs and git history:

| Task | PRs | What it gave |
|---|---|---|
| B0 | #33 | The block library's flow pieces: cluster headers per design point, the importer's scalars, views, `max`, `min`, constant casts and maps over several parameters, MatMul and Transpose, ND1's 2-D slice |
| B1 | #35–#37, snax-gvsoc #34 | `accelerator.json` as the one description (C§10), per-group CSRs and pins, datapaths under a control unit per group (D145–D148) |
| B1x | #38, #39, snax-gvsoc #37 | dot 2–6 and matvec 1–6 as one accelerator each: linked units under one control unit, `Mac`, `SerialAccumulate`, `LinkBuffer`, `split_reduce`, `fuse_map`, `link` (D112, D147, D149, D150) |
| CLN2 | #40–#44, snax-gvsoc #38, #39, snax_cluster #666 | The clean-up after B1x; the local RTL check (`forge-set`) |
| VIEW2 | #45 | The SNAX-DFG editor (`edit-dfg`) and the ideal estimate (D55, D72, D73, D155; C§4, C§13) |
| SPAN1 | #46, snax-gvsoc #41 | A unit reads at every span (dot 5 and 6 at W = 1); the design note `design.serial_one_lane` |
| SYN1 | #47, snax-forge-experiments | forge's 48 points on Nangate 45 and 15 nm beside hls-exercises' tier done (`results/forge_vs_hls.md`, PAPER §7) |
| CTRL1 | #48, snax-gvsoc #42 | The csr as the interface; nested group counting with no product (D152, D153) |
| CHAR1 | #49, snax-forge-experiments #2 | Each block alone emitted (`Emit --block`, `gen-block`) and characterized; the model (D154) |
| CHAR1b | #50 | The model in snax-forge: `flops`, `ctrl_bits`, `snax_forge/char/<tech>.json`, the cost in the estimate, the viewer and the editor (C§13, C§14) |
| DRY1 | #51 | `examples/dryrun/1_vecmul.md` and the editor fixes it found |
| DRY1-2 | #52 | `examples/dryrun/2_dot_1.md` and its fixes |
| SLOT1 | #53, #54, snax_cluster #679, #680, snax-gvsoc #43 | A design's accelerators as streamer sets of one snax_cluster entry, run in series (D157, D158; D123 amended); `dot_1` two accelerators, `dot_1_enc` one |
| DRY1-3 | #55 | `examples/dryrun/3_dot_2_6.md`, dot 2–6 against SYN2 (cycles exact, area within 0.6 %, kernel energy 1.01–1.08); a buffer link loads once per output of its producer; a fused nest's local value leaves the graph at its bind and is drawn as a wire; the bind form lists only what fits; the dry-run pages name no commits (D98) |
| B2 | #58, #59, snax-gvsoc #44, #45, snax_cluster #688 | relu, axpy, rowsum 2 and the fold's seed: one `simd` BRM over `SimdLanes` (its code a table over `op`, `streams` and scalar slots), CSR scalars and `bake_scalar`, the in-place write, `interchange_map`, `LaneAccumulate`, the `padded_rows` layout, `fuse_seed` on `TreeAccumulate` and `SerialAccumulate`, `chisel_accumulator` removed; the editor's partner highlight; StreamerGen's prefix fix (D82, D68, D86, D102, D149, D150, D161–D163) |
| RAM1 | snax-forge-experiments #4 | SRAM area and read and write energy per access from CACTI 7 at 45 nm (OpenRAM probed and dropped), a lookup table over depth 32–4096 and width 8–256, `.lib` and `.v` macros generated from it, memory energy as counted accesses times the table |
| STRM1 | snax-forge-experiments #6 | The streamer's reader and the TCDM interconnect synthesized and powered over their parameters (depth, loops, channels; inputs against banks), the tables and their interpolation in its `docs/cost_tables.md` (X52–X54) |
| REUSE1 | #60 | A recipe as the unit of reuse (D166): `lift`, `unlift`, `apply_recipe`, `unapply`, `name_block` (`sandbox/blocks.py`), `matvec_dot1`–`matvec_dot6` and mvt's `_reuse` recipes, blocks in the editor; with MVT1's part 1 (D164, D165) and the `<name>_l2` rename of a container named after a C symbol |
| CTRL2 | #63, snax-gvsoc #46, snax-forge-experiments `ry/ctrl2` | A register goes where its writer and reader are (D168): the settings live in the csr, which holds a busy group's writes, and the control unit reads them live (an elementwise ctrl 66 → 34 flops, each fold group 64 fewer, every cycle count unchanged); BUSY_COUNTER in `<tag>_peripherals`; CSYN1 Part A's 70 points resynthesized, 33 verdicts turned toward forge (X65) |
| PORT1 | #65–#67, snax-gvsoc #47, #48 | Two accelerators that hand data over through L1 share one streamer's movers behind a data-side mux (D169: `share_ports`, `PORT_MUX_SEL`, `configurable_channel`); the crossbar limit 128 with a note above 64 (D85); the shared designs on vsim with data exact (dot 1 PASS, three mvt designs NEAR, open item 128); `snax-forge-check`'s four verdicts (D123) |
| HWCOST | #62, #68 | A design's cost in five parts (D167), the streamers, crossbar and banks priced from snax-forge-experiments' tables (`pixi run import-costs`), extrapolated and noted off the grid; the crossbar priced by its inputs' kinds since #68 (X71). Energy from what ran moves after the paper; open item 112 stays |
| CSYN1 | snax-forge-experiments #7, #9, #10; snax_cluster #692, #693, #695; #68 | Part A: forge's untiered rows (E7) and DaCe's dot, axpy and rowsum at n2048 against the HLS tools (`results/compare/`). Part B: a design to a synthesized whole cluster in three commands (`pixi run cluster rtl|syn|gls`, X66–X69); dot 3 at W = 16 D123 PASS in 828 cycles, 1 314 306 µm²; HWCOST's five parts within 11 % each, +0.04 % on their sum (X72) |
| CORE1 | #70, snax-gvsoc #49, snax-forge-experiments #11 | One Snitch core drives both the DMA and the accelerator (D170, D131); run tests check what the program computes exactly and its timing within 10 % (D171); the quiet build `forge_quiet.elf` with obs marks on CSR `0x7c5` (D135); the full set on vsim with no FAIL (13 PERFECT, 10 NEAR; open items 95, 129); `dot3_w16` 810 cycles on the RTL and the netlist, 1 261 630 µm², 4.0 % below two cores |
| CLN3 | #73, #74, `ry/cln3-docs`, snax-gvsoc #50 | Housekeeping: no task or contract tags in code (14 `NB (Dn)` notes kept), shorter module docstrings, faster executor, step views and recipe sweeps; the DFG view's edges redrawn until the outlines' margins hold; the cluster view on larger designs; TOOL.md as the command reference, DECISIONS in 43 entries, CONTRACTS renumbered 1–14 |
| PWR1 | snax-forge-experiments #12 | The synthesized one-core cluster's dynamic energy from its netlist run: one VCD of the quiet build, cut into sections by the obs marks on CSR `0x7c5` (D135), PT in averaged mode on the logic and the macros, broken down per component and per section; the run section set beside HWCOST's five parts, an estimate not expected to match |
| PWR2 | snax-forge-experiments #15, #16 | dot_3 at W = 2, 4, 8, 16, 32 through the cluster flow overnight: per W the run section's breakdown beside HWCOST's, as tables and figures; HWCOST follows the synthesized cluster's trend over W, the energy gap left to counting accesses in SNAX-GVSoC (GE) |
| CG1 | snax-forge-experiments #17 | Clock gating in the cluster's synthesis (`CLK_GATE_ENABLE`, `compile_ultra -gate_clock`, the whole cluster): `dot3_w8_cg` measured as the ungated point, the idle cycles' energy most of what gating removes |
| CG2 | snax-forge-experiments #18 | HWCOST's block tables made again from gated blocks (the streamer reader, the crossbar, the char blocks) and dot_3 priced with them in snax-forge-experiments: at W = 8 the five parts 61.60 nJ against the gated cluster's 49.48 nJ run section (1.245), the streamers 0.97, memory 0.99, the datapath and crossbar high on the characterization's random data |
| CG3 | snax-forge-experiments #19 | The gated cluster at W = 2, 4, 16 and 32 against the gated tables, part by part; clock-gated synthesis made the only basis from here on (X87) |
| CG4 | #77, snax-forge-experiments #20 | HWCOST on the clock-gated tables: `import-costs` brings the gated block table too, a unit costs its fires and control $e_{ctrl}\thinspace n_{bits}^{ctrl}\thinspace C_{acc}$ (D154 · D172), dot 3's five parts equal to the experiments' gated pricing; snax-forge-experiments without nangate15 or the ungated data with a gated twin, its docs and code cleaned up (X88). dot 3's five parts fall from about 70 to 61–63 nJ per W, 43 % off the streamers |
| PACK1a | #79, snax-gvsoc #52 | int8, int16 and int32 presets of the tier-done and untiered kernels (`--set types=i8`, `i16`, `i32`; D113 · D173), types following the program's casts (D143 · D175); inputs wire-sliced, a narrow output packed d_out to a word by a `FifoPacker` the flow places (D142 · D174), cycle for cycle in the Chisel and SNAX-GVSoC; L1 37 and L2–L4 110 points on SNAX-GVSoC with 0 data errors, every packed group on its estimate; on vsim the quick and pack sets (31 PERFECT, 5 NEAR) and the full set (15 PERFECT, 10 NEAR, the 23 designs it shares with CORE1 R2 the same to the cycle); open items 95, 132–134 |
| CG6 | snax-forge-experiments #21 | The HLS points (Vitis opt and dace, Bambu, XLS) synthesized and measured clock-gated, their ungated rows removed: inside a busy task window gating moves an HLS block's area −6 % (Vitis), −12 % (Bambu), about 0 (XLS) and its energy 0 to −7 % |
| CG5 | #80, snax-forge-experiments #22 | forge's accelerators alone and the `xbar_mix` grid synthesized gated, every ungated row removed: median area and energy 0.97 of ungated; `char.json` the gated block table, its control constant measured on the gated forge accelerators ($e_{ctrl}$ 7.58 to 5.43 fJ a bit and cycle), `xbar_kinds.json` refitted ($f_{ro}$, $f_{wo}$ 0.3953, 0.4097); the HLS comparison on one gated basis, forge's leads kept; `import-costs` from `char.json`. The 7 W = 32 bundles made before D168 have no gated power (open item 134) |
| CHAR2 | snax-forge-experiments #23 | The char blocks at every width pair (int8 to int64 in, out at least in, the op at out) at one lane count, a narrow fold's lanes priced by its input width, a narrow LinkBuffer by its dtype's, the `FifoPacker` at d_out 2, 4 and 8 (snax-forge-experiments X89, X90); `char`'s lookup at a block's width pair (D176), `DATAPATH_RENAMES` and `TABLE_DEFAULTS` gone; $f_{ctrl}$ 83.20 over the 48 int64 forge points. The model against the forge accelerators: int64 0.997 to 1.035 in energy at W = 8 and 32; int8 and int16 in, 0.92 to 1.40 in energy (int8 high) and 0.99 to 1.12 in datapath area; a packer priced low inside a design (at most 3.7 % of a narrow kernel) |

To do, in four tracks, each task in its own thread, branch and PR (the ROAD1 plan, 16 Nov 2026). Each task runs in a clone set of its own, so tasks in parallel never share `out/`, a GVSoC build or snax_cluster's imported bundles:

- **Track A, tier A and E4** (snax-forge, snax-gvsoc where a task changes the model):
  1. **TIERA1** (`brief`): tier A's forms and E4's sweeps brainstormed together, then tier A's rounds: atax on two engines; the overlap of independent accelerators (open item 109); one unit at several regions (open item 113, against what D165 already does); MVT1's variants 3 and 4; bicg and gesummv (fork, join, broadcast, hold, `augment_init`) as the brainstorm keeps them. It lifts `gen.chain` (a group is a chain). MINI first, SMALL only where a case needs it (open item 54). Every round's vsim check gives E1's points.
     - MVT1, paused at step 1 on 10 Nov and continued inside TIERA1: mvt in the row form on both sides, $x_1 \mathrel{+}= A y_1$ as dots per row and $x_2 \mathrel{+}= A^T y_2$ by the transposed read, at M = N = 40, recipe against recipe, on SNAX-GVSoC and snax_cluster. Part 1 in REUSE1; step 1 on main (`pixi run forge-set mvt`).
     - Built: the DFG view in columns and `mark_transpose` (D164), one instance or linked group shared by both halves (D165), variant 1 as `mvt_1` (both halves on one `vmul`, `vsum`, `vadd`) and `mvt_1_dot2` (one `vmul → link → vsum` group run twice), variant 2 as `mvt_2` (a dot group and a `vadd` per half), each with its `_reuse` twin.
     - Step 1 (done): the six recipes at 40 × 40, W = 4 and 8, `contiguous` and `padded_rows`, on SNAX-GVSoC and vsim (open items 95, 120); mvt_2 at W = 8 dropped (85 crossbar inputs).
     - Left: variant 3, a dot → link → `vadd` at W = 1 per half (no recipe links a fold into a SIMD yet), and variant 4, the seed per half (`fuse_seed`, from `matvec_seed`), both at int64 (mvt's transposed read cannot pack, D142); the column form; `examples/dryrun/4_mvt.md`.
     - The layout stays D86's: W lanes down a column of pitch p words meet $n_\text{banks} / \gcd(p, n_\text{banks})$ banks, so at 40 × 40 on 32 banks W = 4 has no conflict and W = 8 two-way; a pass padding only where W exceeds that would be a new decision.
  2. **E4S1** (kickoff from TIERA1's plan): E4's sweep script over `pixi run flow --set` (no DSE5) and its figure, first on what runs today (W on dot 3, matvec and mvt_1; shared against dedicated ports), then on each tier A form as it lands. The bank-bound sweep at int64 and int32: a dot reads $2 W b$ bytes a cycle against $32 \cdot 8 = 256$, so it crosses at W = 16 (int64) and 32 (int32), and W stops at 32.
- **Track B, costs** (snax-forge-experiments, snax-forge's `char`; Nangate 45 nm only):
  1. **GE1** (`brief`): SNAX-GVSoC tags an energy on every access or transaction of each modelled component and sums it per component and per program section (D135), $E = \sum_c \sum_k n_{c,k}\thinspace e_{c,k}$ with the energies from the same tables as the static estimate (D167): the dynamic estimate beside the graph's static one, checked part by part against PWR2 and CG3's gated cluster at W = 2 to 32. Area stays on the graph. The block energies are CHAR2's, by width pair (D176).
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

## Next Up

The ISPASS 2027 plan (PAP3, PAPER §6), as ROAD1 set it on 16 Nov: TIERA1, CHAR2 and SNB1 start at once, GE1 after CHAR2, E4S1 from TIERA1's plan, track D after track A. Track A's target is 20 Nov. Then, as each lands:

| Step | Waits for | Gives |
|---|---|---|
| E4: the composition sweeps (E4S1) | tier A; GE1's energy and CHAR2's tables for the final sweeps | C3 |
| E7: forge against Vitis opt and dace, Bambu and XLS on tier done and untiered, W ≤ 16 | HSYN1, CSYN1 Part A (both done) | the blocks against HLS |
| E1: mvt and atax (gesummv if time) on vsim, 3–5 compositions each, from tier A's rounds | tier A | C4, C5 |
| E11: the Snitch-only clusters against forge's (SNB1) | SNB1's flow; CSYN1's and PWR1's (done) | cluster PPA, the Snitch baseline |
| E5, E6 and E12: the end-to-end cases (E2E1–E2E5) | track A | where the cycles go per case |

Dates: 20 Nov, track A's target (fine if it slips); 27 Nov, the softmax gate (softmax not running on SNAX-GVSoC by then moves E2E3, E5, to future work); 4 Dec, the abstract's result sentence from E4; 11 Dec, the abstract and the data freeze (figure fixes only after it); 12–18 Dec, writing. PAPER §2 and §3 are written alongside.

## Open Items

Questions not decided yet. Numbers are never reused; a closed item is removed, and where it closed is in git history. Next free number: 136.

1. BRM dataflow notation: `affine` and `stream` (D70), frozen in M6.
2. What automated search reads and how a sweep is configured (DSE1, M8).
4. Stream as a proposer for DNN-shaped workloads, and bank conflicts from the memory passes (D86) in an estimate of our own, on top of the ideal one (D155). Parked.
18. The L1 address map is fixed and word-interleaved, so the design point states no banks; a registered map kind if snax_cluster's differs.
27. Generating a BRM from Chisel, SystemVerilog or HLS sources. Far future.
28. Implementations whose `source` is SystemVerilog or HLS: only `chisel` until one is needed.
32. The DFG viewer's layout (D55), revisited if graphs outgrow it.
33. An MLIR importer that writes `.snaxdfg` directly.
34. A hand-edited `.snaxdfg` runs through the flow (D108) but cannot be replayed or swept, and the design checks' fixes speak of the recipe.
36. Control flow in SNAX-DFG: a sequential `loop` kind and a `branch` kind, registered when a kernel needs them (M9).
37. The expression grammar has `+ - * //`, `max` and `min`, constant casts to the output's or the op's type and value casts (`int32(x)`, a kernel's `astype`, D143 · D175): branches need comparisons.
38. The DMA moves a container as contiguous whole beats (D105); strided or 2-D transfers when a kernel needs them.
40. Switching runs in the viewer clears the selected cycle.
41. A report glossary, a "where the cycles went" ranking and the knobs' `--set` paths: on hold.
44. Reductions over tiles need partial sums and a final combine; until then no tile loop sits around a `reduce` memlet (LOW3b).
50. SNAX-GVSoC's speed on large runs, from the times the runs log (E2 is a sentence or two in PAPER §3); timing-only runs and extrapolation only if it is too slow for C1.
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
82. SNAX-GVSoC dumps the output regions at exit, so the flow could compare them itself; until then the check is the program's own (D136).
86. A custom prefix (`--set accelerator_slot=<x>`) is untested past the generator's file names, and nothing checks that the entry's name starts with `snax_forge`; several entries on one core are open item 108.
88. snax_cluster tests `snax_acc_cfg[0]`'s truthiness for an accelerator and clustergen writes schema defaults into the config, so the schema's `snax_acc_cfg` items carry no defaults.
89. snax_cluster's testharness leaves the cluster's CSR port undriven, so a task-queue CSR (`0x5FE`, `0x5FF`) would hang; its own PR.
90. snax_cluster's generated SV is not reproducible (`ComplexQueue`'s unseeded queue names, the streamer header's timestamp); its own PR.
91. vsum's window over N (64 to 4096) has no per-element gap: SNAX-GVSoC and the RTL differ by at most 16 cycles (2.2%), their slopes' ratio 0.996, with `forge counter` equal. Kept as is (D123).
92. The per-group busy and done pins (`<group>_busy_o`, `<group>_done_o`, D146), which snax_cluster leaves unconnected, into a hardware barrier in the cluster, like `snitch_hw_barrier`, so a core waits on the groups in the barrier instead of polling their STATUS: a snax_cluster template change, then SNAX-GVSoC to match. Parked: snax_cluster leaves them unconnected (Verilator `PINMISSING`, Questa `vopt-2718`).
93. An input of fewer lanes than its word's elements (D142 · D174): the FIFO slicer is built and tested (`hw/chisel/.../stream/`, `snax.forge.stream`) and parked, placed in the flow when a design needs it (the seed port of open item 133 is the first candidate). Outputs are wired or packed by the flow.
94. A masked last beat, the hardware alternative to padding a tail (D144): a lane-valid mask from a CSR, so a tail needs no pads in the data.
95. The cycle gap between SNAX-GVSoC and snax_cluster's RTL, for BASE2 (D123): on the full set, measured on vsim with snax_cluster at #666 and snax-forge at #43, every `forge counter` is equal and every run has `forge data: 0 errors`; the windows are within −4.5% to +8.6%. The quick set measured again in SLOT1 C1 (snax_cluster `ry/slot1` at `5b59191e`, tag `slot1-s3`, snax-forge `slot1-f3`, SNAX-GVSoC `1e988679`, image `d3d8f5572ff4`): every `forge counter` equal and `forge data: 0 errors` again, the windows within −4.8% to +6.8%; dot 1's two streamer sets (`dot_1`) and its encapsulated form (`dot_1_enc`, the earlier `dot_1` row's shape) are its new rows. relu's row is from B2 R2 (snax_cluster main at `ea9c837b`, tag `b2-r2`), axpy's and axpy_baked's from R3b (tag `b2-r3b`), where the quick set measured the same again each time. axpy's K1 write costs SNAX-GVSoC 11 cycles over the baked design and the RTL 4. rowsum_2's two rows are from R4 (tag `b2-r4`): the four lanes of the contiguous layout (rows 64 words apart, all starting in bank 0) collide on their first reads only; each lane is its own channel with its own FIFO, so they then run a cycle apart and never meet again, and padding saves those 3 cycles, the same on the model and the RTL, far below the 4-way conflict a static bank map predicts. R5's rows are from tag `b2-r5` (snax_cluster main at `cad966bf`, SNAX-GVSoC `356c659`, image `d3d8f5572ff4`), where the quick set measured the same again: vsum, now `chisel_adder_tree` (B0), keeps its counter and runs 271 on vsim against 266; matvec_seed's q += A p has matvec_1's counters on SNAX-GVSoC (1027, 1027) and a window of 2842 against its 2845, so the seed's add costs nothing (matvec_1's row needs snax_cluster #688); matvec_seed_serial's sum is 4 · 1024 + 3, one lane a cycle. matvec_seed and rowsum_serial are the first rows whose counters differ, each within its fill (C§10): matvec_seed's 1028 and 1025 on vsim, rowsum_serial's 67 against 65. The windows run from −5.9% (rowsum_serial) to +6.8%. MVT1 and REUSE1 (snax-forge `69c1a51`, snax_cluster main at `82defad8`, SNAX-GVSoC `a95b52bf`, image `d3d8f5572ff4`): the quick set measured the same again on vsim; mvt's rows on SNAX-GVSoC, contiguous and `padded_rows`, each `_reuse` recipe with its hand-written twin's counters (the units' names aside) and window, and padding shortens half 2's transposed read of A (51 → 36 on `mvt_1`'s shared `vmul`, 54 → 39 on the dot group). Their vsim column is the hand-written recipes' (snax-forge `381876d`, the RTL built in image `c5db84ad7bce`): `mvt_1`, `mvt_1_dot2`, `mvt_2` and `mvt_1_dot2` with `padded_rows` each PASS, `forge data: 0 errors`, two counters within their fill (`mvt_1`'s `sum_vsum` 33 against 35, fill 2; the padded `tmp2_mult_mvt` 38 against 39, fill 3), the windows +1.2% (`mvt_1`) to +5.3%; `mvt_1` and `mvt_2` with `padded_rows` are not measured on vsim. The 40 × 40 rows are MVT1-2 step 1's `pixi run forge-set mvt` on SNAX-GVSoC (snax-forge `cd52f3c`, SNAX-GVSoC `a95b52bf`); mvt_2 at `W_x1 = W_x2 = 8` is refused by the design check in both layouts (85 TCDM crossbar inputs, the platform takes 64), so the set stopped there. Half 2's transposed read (the fourth counter of `mvt_1`, the third of the others) against the bank rule (W lanes down a column of pitch p meet 32 / gcd(p, 32) banks): at W = 8 contiguous (p = 40, four banks, two-way) it is 361 and 347 against 232 and 217 padded (p = 41), half 1's own; at W = 4 the rule predicts no conflict, yet contiguous costs 481 against 416 and 452 against 410. MVT1-2 c2's stall traces of `mvt_1` (W = 4 and 8, both layouts) place it between ports: contiguous A stays on one set of banks down a column while y2 sweeps the banks, so A and y2 refuse each other on nearly every other firing all through the task (341 stall cycles at W = 4, 408 at W = 8, beside A's own 910 at W = 8); padded, A moves with y2 and they meet about once per column of A (52 and 152). Outside the tasks, padding leaves the DMA and the gaps between tasks within 16 cycles; `mvt_1` at W = 8 padded starts its first task 114 cycles later, as core0's setup waits twice (142 and 190 cycles) on what looks like instruction fetch behind the DMA's load of A, against once (141) contiguous, a code-placement effect and not the data layout's. The gaps are the core's timing (open items 65, 67; snax-gvsoc NOTES.md, "The comparison with snax_cluster's RTL"). Their vsim column is MVT1-2's (the same bundles, snax_cluster main, image `c5db84ad7bce`): nine PASS, every run `forge data: 0 errors (80 words)`; the two FAILs (NEAR under D123's verdicts since 11 Nov, 1.0 % and 0.5 % beyond the fill) are half 2 at W = 8 with `padded_rows`, where the RTL runs 4 cycles under SNAX-GVSoC on one counter each (`sum_1_vsum` 201 against 205, fill 2; `tmp2_mult_mvt` 213 against 217, fill 3), the case where A, y2 and the output still meet on about 40% of firings (open item 120). The windows are within −4.0% to +4.8%. On the RTL `padded_rows` shortens every window; on SNAX-GVSoC it lengthens three of five (mvt_1 at W = 8, mvt_1_dot2 and mvt_2 at W = 4), so the layout's window ranking flips there, while the counters rank the same on both (open item 120). CTRL2 R4 (snax-forge main `e8bb93e` against `ry/ctrl2` at `50e2228`, SNAX-GVSoC `a95b52bf` against `0a378cdf`, snax_cluster main at `427777e8`, the RTL built in image `c5db84ad7bce`): the quick set, dot 2 and dot 3 measured the same on both, every run PASS with `forge data: 0 errors`, each `forge counter` equal to main's and to SNAX-GVSoC's and each `forge cycles` and window main's, and so did mvt_1_dot2 (1521 on vsim, 1451 on SNAX-GVSoC, on both): its program waits for a group's STATUS before writing that group's CSRs again, so no write meets a busy group and the hold costs it nothing. PORT1 R4's rows are the shared designs (D169; snax-forge `f60d735`, SNAX-GVSoC `831cfec4`, snax_cluster main at `7c14681c`, the RTL built in image `c5db84ad7bce`; the quick set not run again, the unshared SV the same as main's): every run `forge data: 0 errors`, dot_1_shared PERFECT; the three mvt rows NEAR (0.7 %, 2.4 % and 4.9 % beyond the fill; FAIL under the earlier rule), a dot task's counter on the RTL 6 to 19 cycles over SNAX-GVSoC's shared run, past its fill of 3, and within a cycle of the unshared design's on vsim (open item 128). The windows are within +0.4% to +2.8%. Verdicts from 11 Nov (D123 amended, snax_cluster's `snax-forge-check`): PERFECT, NEAR (a counter's gap beyond its fill up to 10 % of its count), WARNING (up to 15 %) and FAIL; an earlier PASS is PERFECT or NEAR, the table's verdicts are the new ones. CORE1 R2, one core (D170; snax-forge `3d64017`, SNAX-GVSoC `ae5b9981`, snax_cluster main at `ac231e42`, the RTL built in image `c5db84ad7bce`): the full set, 13 PERFECT and 10 NEAR, `forge data` equal in all 23; every counter within its fill but rowsum_2's contiguous `sum_acc`, 134 on vsim against 129 on SNAX-GVSoC (3 beyond its fill of 2, NEAR 2.3 %), where SNAX-GVSoC starts the group 10 cycles after its streamer and the RTL does not (open item 129). The windows run from −1.0 % (dot_1_enc) to +23.1 % (vsum_tree_i8), the short designs furthest (vsum +18.5 %, relu +17.2 %, vecadd_i32 and axpy +14.4 %, 28 to 41 cycles each) and the matvec designs within 1.6 % (open item 129). dot_3 at W = 16, N = 2048: 810 on vsim against 778, its counter 132 on both; its quiet build (D135) ran to exit 0 on vsim with `obs_o` taking 1 to 7 in order, the DMA idle at the end of each DMA section and the group idle at the end of the run section (the run section 185 cycles, the group busy 133 of them, the core's waits and counter reads the last 50). PACK1a's close (snax-forge `ry/pack1a`, SNAX-GVSoC with the packer, the same snax_cluster and image): the full set, now 25 with vsum and matvec_3 at i8, 15 PERFECT and 10 NEAR, the 23 designs it shares with CORE1 R2 the same to the cycle on both, counters and windows; the narrow rows (quick and pack sets, R8) are at the end of the table, every packed design PERFECT, three NEAR each within its fill.

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
    | mvt_1 at 40 × 40, W = 8, padded_rows (estimate 812) | 231, 203, 8, 232, 205, 8 (vsim 231, 204, 8, 232, 201, 8: NEAR 1.0 %) | 1585 | 1651 |
    | mvt_1_dot2 at 40 × 40, W = 4 (estimate 824) | 408, 13, 452, 13 | 1521 | 1451 |
    | mvt_1_dot2 at 40 × 40, W = 4, padded_rows (estimate 824) | 410, 13, 410, 13 | 1467 | 1496 |
    | mvt_1_dot2 at 40 × 40, W = 8 (estimate 414) | 216, 8, 347, 8 (vsim 215, 8, 347, 8) | 1209 | 1241 |
    | mvt_1_dot2 at 40 × 40, W = 8, padded_rows (estimate 414) | 217, 8, 217, 8 (vsim 219, 8, 213, 8: NEAR 0.5 %) | 1069 | 1036 |
    | mvt_2, mvt_2_reuse at 40 × 40, W = 4 (estimate 824) | 408, 13, 452, 13 | 1521 | 1451 |
    | mvt_2 at 40 × 40, W = 4, padded_rows (estimate 824) | 410, 13, 410, 13 | 1467 | 1496 |
    | dot_1_shared | 19, 19 | 289 | 287 |
    | mvt_1_dot2_shared at 40 × 40, W = 4 (estimate 824) | 402, 13, 452, 13 (vsim 408, 13, 452, 13: NEAR 0.7 %) | 1466 | 1434 |
    | mvt_1_dot2_shared at 40 × 40, W = 8 (estimate 414) | 208, 8, 337, 8 (vsim 215, 8, 348, 8: NEAR 2.4 %) | 1166 | 1134 |
    | mvt_2_shared at 40 × 40, W = 8 (estimate 414) | 208, 8, 328, 8 (vsim 215, 8, 347, 8: NEAR 4.9 %) | 1200 | 1195 |
    | vsum_tree at i8, W = 8 (snax_cluster #695; packed d_out 2) | 11 | 180 | 158 |
    | rowsum at i8, W = 8 (packed) | 67 | 280 | 246 |
    | matvec_3 at i8, W = 8 (packed) | 1028 | 1345 | 1325 |
    | matvec_3 at i8, W = 32 (packed) | 260 | 577 | 557 |
    | matvec_4 at i8, W = 8 (packed) | 1029 | 1351 | 1325 |
    | matvec_6 at i8, W = 8 (packed) | 1037 | 1357 | 1334 |
    | matvec_1 at i8, W = 8 (packed) | 529, 515 | 1373 | 1349 |
    | matvec_dot1 at i8, W = 8 (packed) | 1057, 1027 | 2461 | 2426 |
    | matvec_dot3 at i8, W = 8 (packed) | 1028 | 1345 | 1325 |
    | matvec_dot4 at i8, W = 8 (packed) | 1029 | 1351 | 1325 |
    | matvec_dot6 at i8, W = 8 (packed) | 1037 | 1357 | 1334 |
    | dot_1 at i8, W = 8 (packed) | 10, 11 | 286 | 245 |
    | dot_1_enc at i8, W = 8 (packed) | 10, 11 (vsim 8, 11: NEAR, 0.0 % beyond the fill) | 329 | 297 |
    | dot_1_shared at i8, W = 8 (packed) | 8, 11 (vsim 10, 9: NEAR, 0.0 % beyond the fill) | 298 | 259 |
    | dot_2 at i8, W = 8 (packed) | 69 | 272 | 251 |
    | dot_3 at i8, W = 8 (packed) | 68 | 272 | 251 |
    | dot_3 at i8, W = 32 (packed) | 20 | 224 | 203 |
    | dot_3 at i16, W = 4 (packed d_out 2) | 132 | 354 | 323 |
    | dot_3 at i32, W = 2 | 260 | 509 | 486 |
    | dot_4 at i8, W = 8 (packed) | 69 | 272 | 251 |
    | dot_5 at i8, W = 8 (packed) | 76 | 278 | 260 |
    | dot_6 at i8, W = 8 (packed) | 77 | 278 | 260 |
    | dot_serial at i32, W = 2 | 32, 67 (vsim 35, 65: NEAR, 0.0 % beyond the fill) | 333 | 299 |
    | vecadd at i8, W = 8 | 10 | 239 | 208 |
    | vecmul at i8, W = 8 | 10 | 239 | 208 |
    | relu at i8, W = 8 | 10 | 197 | 158 |
    | axpy at i8, W = 8 | 11 | 239 | 208 |
    | matadd at i8, W = 8 | 18 | 255 | 236 |

96. The measurements of a linked group: per-unit and per-link events, the converter at beat level for a group of several linked units (`convert.level` refuses it; an accelerator of several one-unit groups, dot 1, is done) and the run viewer's view of it; bank and port statistics, the export CSV, the area × cycles metric. The bank conflicts measured so far (F3, SNAX-GVSoC) stay inside C§10's cycles plus the fill.
97. dot 1's row form at 16 × 512 does not fit: tmp0 needs 132 KiB of the 128 KiB L1, so it runs at 8 × 512.
99. L2 in SNAX-DFG: containers get a level (L2 or L1), and an L2-to-L1 move becomes a DMA copy node placed by a transform, so tiling can choose what moves per tile; today lowering derives the DMAs from the memory plan (D64, D86), and L2V1 only draws them and counts their cycles. It belongs with M15's LOW3b.
100. Unmapped work on a core: a node no accelerator runs is run by a core, its tasklet turned into C; this needs `graph.unbound` (C§6) relaxed and lowering to emit core code. Until then the estimate counts it at one cycle per iteration, an assumption (D155, `UNMAPPED_CYCLES_PER_ITERATION`), which a measured core value could replace.
101. The graph page's edges into a node lower in a map pass behind the nodes above it (`viz/dfg/api.py`'s layout, `static/dfg.js`): a route around them, or the rows inside a map spaced for it.
103. A unit with L ≥ 1 on a wire into a consumer s with T = 1 could borrow s's test as an L-0 unit does, with g_s + popcount(v_k) ≠ R_s (its results in flight counted), dropping `mac_jobs_counter` in dot 4–6 (from CTRL1).
106. A fold reading through a tail a container whose pad words its producer never writes (dot 1 at W_mul = 8, W_sum = 10: tmp0's 2048 words written, 2050 read) needs the program to write the fold's identity there before the fold's task; `connect.pad` refuses it until then (D144, DRY1-2).
108. Several `snax_acc_cfg` entries on one core: the range decoder between entries in place of `snax_acc_mux_demux` and 0x7C6, and `snaxgen.py:620` building every entry; untested since SLOT1 drops the multi-ALU cluster.
109. Overlap of independent accelerators: D158's serial rule dropped for data-only waits; first case PolyBench mvt, run serial as the baseline; `accelerator_cycles` and C_acc change with it.
112. A buffer link's energy against SYN2 (DRY1-3): dot 2's `tmp0` buffer, which loads every beat, is 1.59× (nangate45) and 1.49× (nangate15) high against CHAR1's `link/d1_w8` stream point, not explained. dot 6's `out_partial` was 2.65× and 3.13× high while a buffer loaded once per firing of its producer; it now loads once per output (C§14), once per job behind the Mac.
113. One unit across several DFG regions (atax's A and Aᵀ on one GeMM): a unit bound at several nodes, trading throughput for area; amends D112 (Ryan, 27 Oct 2026, for the paper). tier A's atax is the first case.
115. An ALU SIMD: the `simd` BRM's `op` as an R/W CSR, its op set a design param, so a design pays only for the ops it can select; worth it once one unit runs several ops, a unit bound at several nodes (open item 113).
116. `gen.span` refuses a unit of span > 1 reading an outer input (a serial unit straight from its streamer: vsum or rowsum 1 on `chisel_serial` at W > 1, the beat held for span cycles); lifted when E7 or a kernel needs such a point.
117. Reusing a composite's recipe, done in REUSE1 (#60, D166). Left: a block and a group of one name (`dot`) are selected together; a collapsed matvec in mvt has no outer loop, so both halves read `A[0:M, 0:N]` and the transpose shows one level down or by its mark; `name_block` is left out of the marks (it takes any node).
118. The run viewer (`pixi run view`: banks, interconnect, transactions) is kept working but gets no new features until after the paper (Ryan, 6 Nov 2026); the DFG views (`view-dfg`, `edit-dfg`) stay first.
119. The cost estimate reads the graph only, so a layout pass's padding (`padded_rows`: rowsum 2's 8 × 65 words against 8 × 64) is not in its memory bytes; the design report and the memory plan do count it (B2 R4).
120. MVT1-2's vsim check (open item 95): (1) under heavy bank conflicts (mvt's half 2 at W = 8, `padded_rows`) a counter on the RTL is 4 cycles under SNAX-GVSoC's, past its fill, likely the two crossbars resolving collisions differently; (2) the window differences between layouts are smaller than the model's window error (−4.0% to +4.8%), so three of five layout pairs rank the other way on the RTL, the model's head slower where core0's setup waits on instruction fetch behind the DMA (open item 65). E1 ranks on counters as well as windows, or compares designs further apart than that error; both looked at again if E1's mvt points depend on them.
128. SNAX-GVSoC's shared movers (D169) run a dot task 6 to 19 cycles under the unshared twin, beyond the fill, while the RTL gives the shared and unshared designs the same counters within a cycle (PORT1 R4: mvt_1_dot2_shared at W = 4 and 8, mvt_2_shared at W = 8; open item 95). Suspects: the masked writer and the crossbar component's per-cycle order of events; first check: the beat and stall traces of a dot task, shared against unshared.
122. A Mac expansion: W multipliers into a W → 1 adder tree and one accumulator, as one unit. Splitting both of dot's maps at W (`split_map`, no `split_reduce`) and then `fuse_map` gives one node, W products summed per firing and accumulated over the N / W firings, that nothing binds today: `mac` refuses it (it keeps one partial per lane, so `split_reduce` first, D150) and `accumulate` takes a reduce tasklet only. Its datapath is dot 3's, SimdLanes → wire → TreeAccumulate, which binds it as two units in one group; one unit from one fused node would drop the second unit's counters from the control unit, which its cost would show (Ryan, 10 Nov).
123. Programming the next task while one runs: a design-time option to latch a group's settings at its start, off by default (D168); if built, JOBS counts down so its copy is its counter, and one copy of T per unit whose T comes from a JOB_BEATS.
124. job_width sized by the flow from the largest JOBS and T of the target workload, a design-exploration knob; one job_width of 32 stays the default (D152), like for like with the HLS tools' u32.
125. `<tag>_peripherals` made optional for minimal-area points, the CSR slots kept and reading 0 (D168).
126. `ReaderWriter` (snax_cluster's `readerWriter/ReaderWriter.scala`: a full Reader and Writer behind one set of TCDM channels, the writer first each cycle, no mode bit) for sharing one mover between a read port and a write port (D169 shares a role's movers only); needs a SNAX-GVSoC model (`UNMODELLED_MOVERS` in `snax_cfg.py`).
127. snax_cluster's `streamer/Streamer.scala` lines 188–189 build each `ReaderWriter` from `readerWriterParams(i / 2)` and `(i / 2 + 1)`, where `(2 * i)` and `(2 * i + 1)` are meant: a second pair would take the first pair's parameters (GeMMX has one pair, so nothing breaks today).
129. One core's program timing on SNAX-GVSoC against the RTL (CORE1 R2, D170): the windows of the short designs run 14 to 23 % longer on vsim (28 to 41 cycles; two cores: within +8.6 %), and SNAX-GVSoC starts a group 10 cycles after its streamer at the first start after `snrt_dma_wait_all()` where the RTL does not (rowsum_2 contiguous: 129 against 134, NEAR 2.3 %; vecadd, dot_1's vmul). Likely the core's DMA instructions and waits, now on the same core as the configuration (open item 67's `dmstati` polled every 5 cycles on the RTL against 2); the quiet build's sections (D135), run on both, compare them section by section.
130. A tuned DaCe case for the HLS comparison (E7): beside `dace` (DaCe as it comes), a `dace_tuned` case built only from DaCe's own transforms and library expansions, as DaCe's `tests/npbench/` and `tests/blas/nodes/` use them, with no hand-written C++, pragmas or edits to generated code. Per kernel: dot by `MapReduceFusion` at W = 1 (no 2048-word RAM, one loop) and a `Dot` node on `dace.vector` arrays with `FPGA_Accumulate` at W > 1; axpy by fusion and an `Axpy` node (`fpga` expansion); vsum and rowsum at W = 1 only, since `Reduce`'s FPGA expansion refuses vectors; relu left out unless a NumPy rewrite vectorizes (`max(vec, 0)` does not compile). In a sandbox C-sim at N = 2048 every tuned form computes the right result; `auto_optimize(FPGA)` alone keeps dot's RAM and adds 16 partial sums to every reduction. The lane forms need the kernel restated (`np.dot`, vector-typed arrays), which forge does not; the paper says so. Experiments in hls-exercises `bench/dace_tuned/`.

132. Tier A at int8: the fold type of `@` and `np.dot` (the operands', so a sum that wraps, D143 · D175), mvt's transposed read (a core task writing the transpose into L1 first, against tier A's column form), the outputs the paper needs, mvt's reuse recipes off `matvec_dot2` if matvec_2 is ever removed, and E4's int8 W sweep (dot_3 crosses the 32-bank bound at W = 128 at int8, 16 at int64: W up to 256 needs 67 crossbar inputs).
133. The seed port (D163) at a narrow type: one lane, an input narrower than its word, which the input rule refuses (D174), so `matvec_seed` and `matvec_seed_serial` stay int64. The options: the FIFO slicer on that port alone (open item 93), or one seed per word.
134. Whether `Mac` and `SimdLanes` take a narrow multiplier (w_a + w_b bits sign-extended to `opType`, bit-identical to NumPy's product at `opType`), the experiment ROAD1 left for later; the char table would then gain its pairs.
135. snax-forge-experiments' `cluster-breakdown`, `cluster-energy` and `cluster-sweep` no longer reproduce their files: they read the forge points' ungated `alone` rows, which CG5 removed, so a rerun drops `alone` and `cluster-sweep` stops on it; `breakdown.json` and `energy.json` hold the dot 3 hwcost from before CHAR2 (control 0.4 % low).

## Sync Reminders

- After every change done with Claude, update this file and log any design change in `docs/DECISIONS.md` (rules at its top).
- When a milestone closes, compact `docs/DECISIONS.md` and remove the milestone's done tasks from this file (D117).
