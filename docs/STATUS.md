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
| Generator (hjson, headers, `data.h`, the bundle; `main.c` from SNAX-LOWER's `lower/c.py`) | `snax_forge/gen/` | `tests/gen/` | `platforms/snax_forge_cluster.hjson` |
| SNAX-LOWER | `snax_forge/lower/` | `tests/lower/` | |
| Flow and `pixi run forge` | `snax_forge/flow/` | `tests/flow/` | |
| A flow run's profile and trace (the events converted, D137) | `snax_forge/run/` | `tests/run/` | `tests/run/fixtures/` |
| Viewers | `snax_forge/viz/` | `tests/viz/` | |
| Reports | `snax_forge/report/` | `tests/report/` | |
| Examples | `examples/` | `tests/flow/test_loop1.py` (on SNAX-GVSoC, marker `gvsoc`), `tests/flow/test_dot_example.py` (skipped while dot is parked) | |
| Chisel blocks; a design point's accelerator RTL (`rtl.json`, the shell wrapper) | `hw/chisel/`, `snax_forge/gen/rtl.py` | `hw/chisel/src/test/`, `tests/gen/test_rtl.py` | `hw/chisel/blocks.json`, `hw/chisel/src/test/resources/rtl/` |
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
| M13 | B0 (`done`), B1, B2, B3, B4 | **B0** flow pieces: CLN1's leftovers; the runtime's cluster headers per design point (open item 68); the importer's scalars, views, `max`, `min`, constant casts and maps over several parameters (open item 37); the MatMul and Transpose pass; ND1's slice: 2-D containers, a tag without a split, the streamer from the temporal loops (open items 45, 46, 48, 78); a restart every T < n (open item 81). **B1** one description: `accelerator.json` for Emit and SNAX-GVSoC; each block a datapath and a controller, in three modules; per-unit JOBS, JOB_BEATS, STATUS, COUNTER and pins (open item 83). **B2** blocks, one per round: 1- and 2-stream SIMD, vsum, lane accumulator, MAC, FIFO; `bind` on a group; `split_reduce`; the fold's `init` port. **B3** composition: the composite node, wire and decoupled links on both sides, one start, the link check; dot 2–6 at N = 512, W = 4 and 8. **B4** tier A: fragments applied by output name, `augment_init`, fork and join; bicg, mvt (LOW4, open item 47), gesummv, atax; dot 1 (open item 79). `recipes/dot.json` and `examples/dot/` stay parked until B3. |
| M7 | KER0 (`done`), FE1b, FE1c, PACK0 (`wip`), PACK1, FE1d, FE1e, FE2, FE3 | The tiers of PAPER §6.1 after tier A (D113), which M13 does. KER0 staged every target kernel in `kernels/`, not linked to the flow (`kernels/README.md`, `pixi run kernels`). FE1b is tier B; FE1c tier C; PACK0 packs int32 and int8 into 64-bit words with NumPy's arithmetic (D142–D144; open items 20, 21 and 31); PACK1 keeps tier P: the exact products, requantization and the int8 kernels (the reference executor accumulates as the hardware does); FE1d tier D; FE1e tier E. FE3 is C2's coverage table |
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
reference check. Next is PACK0, the streamer's PR (narrow elements packed in
64-bit words; D142–D144), before B1. Its rounds: R0 docs; R1 design
(platform, streamers, checks, BRM dtypes); R2 lowering and data; R3
generation and the first SNAX-GVSoC runs; R4 Chisel; R5 viewer and close.
PACK0's R0, R1 and R2 are done: the decisions, the design (the platform
keeps only the word, a BRM port's dtypes a rule, the streamers' dtype, k and
channels, `connect.pack` and `connect.channels`), then lowering and data
(layouts by the container's itemsize, stream values in words); generation
refuses a narrow container until R3. snax-gvsoc's R1 is done (PR #32, open
until R2).

## Open Items

Questions not decided yet. Numbers are never reused; a closed item is
removed, and where it closed is in git history. Closed in CLN1: 63 (the ALU
left snax-gvsoc; generated accelerators follow D130). Closed in M13 B0: 45, 46, 48, 81.
Next free number: 93.

1. BRM dataflow notation: `affine` and `stream` (D70), frozen in M6.
2. What automated search reads and how a sweep is configured (DSE1, M8).
4. Stream as a proposer for DNN-shaped workloads, and an analytical
   estimator of our own (the engine alone, open item 49, plus bank conflicts
   from the memory passes, D86). Parked (D23).
18. The L1 address map is fixed and word-interleaved, so the design point
    states no banks; a registered map kind if snax_cluster's differs.
20. Streamer ports are one bank wide; wider when a kernel needs it (PACK0,
    closed in its R5).
21. More than one element per word has never been moved by streamers,
    accelerators or the DMA (PACK0, closed in its R5; R1 checks the ports,
    `connect.pack`). The Accumulator's tree zero-extends
    its lanes (`UInt.pad`): PACK0 makes it sign-extend for int8 into int32
    (D143). relu, axpy and matadd were int32 in KER0 and run as int64 until
    then (M13 B0).
27. Generating a BRM from Chisel, SystemVerilog or HLS sources. Far future.
28. Implementations whose `source` is SystemVerilog or HLS: only `chisel`
    until one is needed.
31. A loop bound that is not a multiple of the spatial bound: `split_map`
    rejects it; PACK0 settles it by padding (D144), closed in its R5.
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
53. A link between two shell accelerators: amends D12 and D51 and the SNAX
    shell. Parked (D112).
54. Which kernels take PolyBench's SMALL in a 128 KiB L1 (M13 B4); the NumPy
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
79. Several accelerators on one core: snax_cluster has `snax_acc_mux_demux`
    and CSR 1990 (`snax_multi_alu_cluster.hjson`); SNAX-GVSoC lacks CSR 1990,
    `snax_hjson` and the generator (`gen.instances`) refuse it, and the
    headers would collide by name. M13's B4 adds it for dot 1.
80. One Snitch core driving both the DMA and the accelerator: the C backend
    takes core roles from the cluster file, `snax_hjson` wants xdma on the
    last core. Never run.
82. SNAX-GVSoC dumps the output regions at exit, so the flow could compare
    them itself; until then the check is the program's own (D136).
83. Control alignment with snax_cluster: a start with a zero count, a done
    CSR (amends D130 and `SnaxShell`), where the counter lives, a Decoupled
    `ctrl` in the blocks. Decided in M13's B1.
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
92. The per-unit busy and done pins (M13 B1) into a hardware barrier in the
    cluster, like `snitch_hw_barrier`, so a core waits on the units in the
    barrier instead of polling their CSRs: a snax_cluster template change,
    then SNAX-GVSoC to match.

## Sync Reminders

- After every change done with Claude, update this file and log any design
  change in `docs/DECISIONS.md` (rules at its top).
- When a milestone closes, compact `docs/DECISIONS.md` and remove the
  milestone's done tasks from this file (D117).
