# SNAX-FORGE Status

Open work: milestones, tasks and open items. What is built:
`docs/ARCHITECTURE.md`; why: `docs/DECISIONS.md`; the artefacts:
`docs/CONTRACTS.md`. Done tasks are removed when their milestone closes
(D117); their tables are in git history.

Status values: `todo`, `brief` (brief written), `wip`, `done`, `deferred`.

## Where the Code Lives

| Component | Code | Tests | Data |
|---|---|---|---|
| SDFG ingest | `snax_forge/sdfg/` | (through `tests/dfg`) | `kernels/` |
| SNAX-DFG, importer, reference executor | `snax_forge/dfg/`, `snax_forge/expr.py` | `tests/dfg/`, `tests/test_expr.py` | `tests/dfg/fixtures/` |
| SNAX-BRM | `snax_forge/brm/` | `tests/brm/` | `snax_forge/brm/library/` |
| SNAX-SANDBOX | `snax_forge/sandbox/` | `tests/sandbox/` | `recipes/` |
| Design step | `snax_forge/design/` | `tests/design/` | `platforms/` |
| Generator (hjson, headers, `data.h`; `main.c` from SNAX-LOWER's `lower/c.py`) | `snax_forge/gen/` | `tests/gen/` | `platforms/*.hjson` |
| SNAX-LOWER | `snax_forge/lower/` | `tests/lower/` | |
| Flow | `snax_forge/flow/` | `tests/flow/` | |
| A flow run's profile and trace (the events converted, D137) | `snax_forge/run/` | `tests/run/` | `tests/run/fixtures/` |
| Viewers | `snax_forge/viz/` | `tests/viz/` | |
| Reports | `snax_forge/report/` | `tests/report/` | |
| Examples | `examples/` | `tests/flow/test_loop1.py` (skipped until FLOW3 round 3), `tests/flow/test_dot_example.py` (skipped while dot is parked) | |
| Chisel blocks | `hw/chisel/` | `hw/chisel/src/test/` | |
| SNAX-GVSoC | the submodule `gvsoc/snax-gvsoc` (its own KULeuven-MICAS repository) | in that repository (`make regress`) | its `forge/` fixtures |
| SNAX-GVSoC from snax-forge | `gvsoc/Makefile` | | |
| Kernels (KER0) | `kernels/` | `tests/kernels/` | |
| Utilities (clean, environment check, kernel list) | `util/` | | |
| Development image (D118) | `container/` | | `container/toolchain/` |

Generated files (`.snaxdfg`, design points, runs) go under `out/`. GVSoC was learnt
in `KULeuven-MICAS/snax-gvsoc-experiment` (X1), which stays a playground.

## Milestones

Done: M1, M4a, M3, M4b, M5, X1, M16. Open, in order (D116):

| Milestone | Content | Status |
|---|---|---|
| M14 | Generation, to SNAX-GVSoC first, then to snax_cluster, then clean-up (D111, D116, D127) | `wip` |
| M7 | Front ends: NPBench's DaCe sources in tiers, integers first (D113) | todo |
| M13 | Composed bind: several unit BRMs as one accelerator (D112) | todo |
| M15 | Tiling: a DMA per tile | todo |
| M9 | `jacobi1d` | todo |
| M11 | Layers: conv2d, softmax and attention, HDC operations (D100) | todo |
| M12 | Blocks and models: ResNet, Transformer, NVSA (D100) | todo |
| P1 | Paper plan, alongside (D100) | `wip` |
| P2 | Paper draft (D100) | todo |
| M6 | Contract freeze, when the contracts call for it | todo |
| M8 | Automated DSE over recipes, when sweeps call for it | todo |
| M10 | Cosim of one accelerator (D52) | todo |

M7's tiers interleave with the rest: tier A after M14, then M13 and LNK3,
tiers B, C and P, M15, tier D with the GeMM (M9 inside it), then tier E.

## M16: SNAX-GVSoC aligned with snax_cluster (D116)

Done: SNAX-GVSoC is the baseline, checked against snax_cluster's RTL (D123);
its tasks SG0 to SG7 and CI1 are in snax-gvsoc's `docs/plans/`, SG8 in its
`docs/notes/NOTES.md`.

## M14: Generation: SNAX-GVSoC, then snax_cluster (D111, D116, D127)

| ID | Scope | Depends | Acceptance | Status |
|---|---|---|---|---|
| LNK1 | A. A platform read from snax_cluster's `cfg/snax_alu_cluster.hjson` (`platforms/snax_alu.json`, `design from-hjson`) with the streamer template fields, checked as any platform is: `temporal_dim` and `fifo_depth` into the default, `spatial_bounds` and `num_channel` derived from the ports (D131) | none | The platform round-trips; `vecadd` at W = 4 designs on it | done |
| GEN3 | A. BRM → GVSoC accelerator (D128–D130): snax-gvsoc as a submodule of snax-forge (`gvsoc/`); `SnaxUnit` and `SnaxShell` in snax-gvsoc; the BRM binding; the generator (the hjson files since LNK2a, D134; the manifest) and the two headers; fixtures in snax-gvsoc `forge/<design>/`, run by hand-written programs until GEN2a (`forge-reduce` stays for T < n); `elementwise_mul`, then `accumulate`. Steps in snax-gvsoc `GEN3.md` | LNK1 for step 3 on | vecadd on the generated accelerator: data correct, counter as predicted, `make regress` passes on the fixture; snax-forge reproduces the fixture; the same for mul and accumulate | `done` |
| LNK2a | A. The cluster hjson from the design point; SNAX-GVSoC's target read from it (D119, D134). Round 1 in snax-gvsoc: the reader, the generic target `snax_hjson`, forge/'s fixtures as hjson by hand; round 2 here: the platform's source hjson and slot, the writer of three hjson files (cluster, streamer, accelerator) and the schemas, forge/'s fixtures from it. Steps and findings in snax-gvsoc `LNK2a.md` | GEN3 | GEN3's runs on the read target | done |
| GEN2a | A. The program for single-accelerator designs: task list → C with a mark per task, the accelerator's and streamer's headers, `data.h` from a datagen (inputs padded to beats, D105; golden from the reference executor), self-checking. Several accelerators on one core are not modelled (open item 79); dot, mul and accumulate driven by one core, is M13's composite (D112). Round 1: the datagen, the C backend, a generic app on snax-gvsoc's fixtures (snax-gvsoc `GEN2a.md`); round 2: the adder tree | LNK2a | `vecadd`, `vecmul`, `vsum`, and `vsum` on the adder tree at W = 4 check themselves on SNAX-GVSoC | `done` |
| FLOW3 | A. The flow ends in a SNAX-GVSoC run (D136, D137). Round 1: **1a** the flow on SNAX-GVSoC: `gvsoc/Makefile` called from pixi, `forge_check()` and the program's window (D135), the stages lower, gen, build, sw, gvsoc and check, `run.json`, CI's job `gvsoc`; **1b** the events (D137): `summary`, `events` and `beats` traces per SNAX component into one `run/events.log`, converted by `snax_forge/run/` into C§7, the level `task`, the reports and the viewer's profile page; **1c** the GVSoC build's output live on the terminal as well as in `build.log` (done). Round 2: **2a** (done) the beat- and stall-level events (D137): SNAX-GVSoC's `beats` and `stalls` lines (grants, DMA beats and bursts, firings, pushes, stalls, queue counts) linked by ids and carrying values (open items 23, 25), the cores' `insn`, the levels `beat`, `stall` and `finegrained`, a source filter, `--vcd` as a gtkwave dump, the converter's beat part with its invariants, L2 regions from the compiled ELF's symbols (`run.json`'s `elf_symbols`, D86 amended); a reader with a repeat (several firings naming one `b`, the nest check's "loop 0 once") is covered only by `tests/run/test_beats.py`'s `agu_nest` case until mat-vec; **2b** (on `ry/flow3-5`) the memory view (D96) and the movement view (D97) rebuilt on those events: `viz/index.py` groups a run's events by their ids and lookups once at load, with the memory tab's marks registered on it; `viz/memory.py` and `viz/movement.py` query it (D97's journey, residency, patterns and conflicts); `run.md`'s movement section from the index. Round 3: the schedule inside the window, with core lanes and folded stretches (D57), and `examples/loop1/` again. Steps in snax-gvsoc `FLOW3.md` | GEN2a | 1a: `pixi run flow` on `vecadd` and `vsum` on `platforms/snax_alu.json` ends in a checked SNAX-GVSoC run (exit 0), in the container and in CI's `gvsoc` job; 1b: the same two flows at `--trace task` pass with the converter's invariants and write `report/design.md` and `report/run.md`, at `--trace off` the profile comes from the summary lines alone, and `pixi run view` opens their profile page; 2a: the same two flows pass at `--trace off`, `task`, `beat`, `stall` and `finegrained` with every invariant, each `forge cycles:` its `off` run's and each `profile.json` at `beat` and above byte-identical to its `task` run's, and one flow with `--vcd` writes the VCD and `view.gtkw` in `run/`; 2b: for those two flows at `off`, `task`, `beat` and `stall`, `viz/api.py` answers the memory tab for both memories with exactly the marks the level allows (none at `off` and `task`; `arrival`, `use`, `wait` at `beat`; `conflicts` too at `stall`) and the reason for each other; vecadd's journey of `A[0]` at `stall` goes burst, DMA beat, `acc_a`'s grant, fire, push, the writer's grant of `C[0]`, the outgoing DMA beat and its burst, the fire's values holding `A[0]` and `B[0]` and the push's `C[0] = A[0] + B[0]`; vsum's one push folds firings 0 to 63 and every input's journey reaches it; the conflicts' total is the profile's SNAX ports' stalls; residency lists every container in each memory; below `stall` the answers that need it say so; round 3: the viewer opens them | `wip` |
| RET1 | A. SNAX-MODEL removed (D116): `snax_model/`, `scenarios/`, their tests and pixi tasks, the command list (`lower_program`, `program.py`), the cluster file (`lower/cluster.py`), `viz/memory.py` and `movement.py`, C§2–C§6 and C§8, and the entries true of SNAX-MODEL alone (to Retired). What the live path uses stays, trimmed: the task list, its values check and its waits (`placed_waits`, `lower/waits.py`), the streamer values and `derive.py` read the design point; `design/` owns the platform's L1, L2 and streamer types, with only the fields something reads; the accelerator kinds the reference executor runs move to `brm/function.py`; a BRM is checked against the generator's blocks. Nothing new is designed | FLOW3 round 1 | `pixi run test` and `pixi run lint` pass; no `snax_model` under `snax_forge/` or `tests/`; `gen-gvsoc` writes byte-identical files for vecadd, vecmul, vsum and vsum_tree before and after, and `tests/gen/test_fixture.py` passes unchanged; in the container `pixi run flow` on `vecadd` (`--trace task`) and `vsum` on `platforms/snax_alu.json` pass with windows 226 and 255 | done |
| GEN1 | B. The Chisel datapath from the binding through `Emit` (D91), the `ctrl` adapter and the generated shell wrapper (D130) | GEN3 | Elaborates for `elementwise_add` and `accumulate` | todo |
| LNK2b | B. The bundle for snax_cluster: the hjson, a `Bender.yml`, the changes snax_cluster needs (open item 51) | GEN1, LNK2a | snax_cluster builds from it with only the listed changes | todo |
| GEN2b | B. GEN2a's ELF on snax_cluster's RTL, compared as D123 does | LNK2b, GEN2a | Deviation explained per phase | todo |
| LNK3 | After M13. The composed accelerator (dot's MAC) generated to both sides and compared as SG5 does | M13, GEN2b | Deviation explained per phase | todo |

After FLOW3 and RET1, a housekeeping pass and the single generator (open item 68). The pass also takes the memory plan's L2 placement and the `memory.align` and `memory.overlap` checks on L2, which work on bases no program uses since D86's amendment (`memory.fit` stays).

## Later Milestones

The tasks are a first cut; each milestone is planned in a round of its own.

| Milestone | Tasks | Content |
|---|---|---|
| P1 | PAP2 (`brief`), BASE1, BASE2, PAP3 | Related work after the reading; the SNAX RTL flow logged step by step (BASE1, after the SG series, once the snax-forge flow runs end to end again); SNAX-GVSoC's speed on large runs (BASE2); the evaluation plan mapped onto the claims (PAP3) |
| M7 | KER0 (`done`), ND1, FE1a, FE1b, FE1c, PACK1, FE1d, FE1e, FE2, FE3 | The tiers of PAPER §6.1 (D113). KER0 stages every target kernel in `kernels/`, not linked to the flow, with `kernels/README.md` and `pixi run kernels`. ND1 first: rank-free unit dataflow, map tagging, N-D fills, the streamer as an input of its own (open items 45, 46, 48, 78). FE1a is tier A, with LOW4 in it (mvt's in-place update, open item 47; gesummv's scalars); FE1b tier B; FE1c tier C; PACK1 tier P (open items 20 and 21; the reference executor accumulates as the hardware does; requantization); FE1d tier D; FE1e tier E. FE3 is C2's coverage table |
| M13 | CMP0–CMP4 | dot is parked until M13: `recipes/dot.json` and `examples/dot/` are kept, out of the tests since FLOW3 (two accelerators, `gen.instances`). One recipe step binds a connected group of tasklets to a chain of unit BRMs as one accelerator, FIFOs inside, streamers outside (D112): the BRM format, the model, the sandbox, the lowering, then composed `dot`, mat-vec and mat-mat against E2E2's chain through L1. That chain on SNAX-GVSoC needs several accelerators on one core (open item 79); M13 decides its baseline |
| M15 | LOW3b | A tile transform and a DMA per tile, an L2 slice per tile, reductions over tiles (open item 44) |
| M9 | J1 | Stencil BRM with reuse and double buffering (see open item 76) |
| M11 | LAY1–LAY3 | conv2d tiled on a MAC-array BRM; softmax and attention with several BRMs; HDC bind, bundle and similarity |
| M12 | BLK2–BLK4 | A ResNet basic block, a Transformer encoder block, an NVSA pipeline; JEPA if it adds a case |
| P2 | PAP4 | The paper, every number from a checked-in command and a test |
| M6 | F1–F3 | Package layout and CI; versioned schemas for every contract (closes open item 1); registries and namespaced attrs |
| M8 | DSE0–DSE5 | The engine-alone number in `design.md` (open item 49); search over recipes; replacement across the BRM library; parameter choices; memory-plan policies (D86); a sweep runner |
| M10 | COS1, COS2 | One accelerator's RTL in place of its model, and the mismatch of its latency and II per BRM (D52) |

## Next Up

M16 is done (D123). In M14, LNK1, GEN3, LNK2a and GEN2a are done: a design
point generates its three hjson files, headers, `data.h` and `main.c`, and
vecadd, vecmul, vsum and vsum_tree check themselves on SNAX-GVSoC; KER0
(M7) staged the target kernels. FLOW3 round 1 is merged: the flow ends in
a checked SNAX-GVSoC run (1a, D136, `docs/RUN.md`), SNAX-GVSoC's `summary`
and `events` lines become the profile and the trace and the flow writes the
reports again (1b, D137), and the GVSoC build's output is on the terminal
(1c). RET1 removed SNAX-MODEL (merged). Round 2a is merged (snax-gvsoc
#23, snax-forge #23): the beat- and stall-level events and the converter's
beat part, vecadd and vsum at every level with every invariant. Round 2b,
the memory and movement views on those events, is under way on
`ry/flow3-5`; then round 3, then GEN1.

## Open Items

Questions not decided yet. Numbers are never reused; a closed item is
removed. Closed with D116: 5, 6, 7, 8 and 10 (SNAX-GVSoC follows
snax_cluster's RTL), 9 (into SG1, D120), 11 (into SG5), 14, 15 and 22 (SNAX-MODEL
details), 52 (the real program and cores do both), 55 (into
FLOW3). Closed in SG3: 59 (the streamer's cycle classes and queue counts)
and 61 (the crossbar's ports named by owner). Closed in SG5: 3 (D123).
Closed in SG6: 62 (D120). Closed in SG7: 58 (the cores, AXI and the DMA in
the SNAX crossbar, D122). Closed in GEN3's plan: 64 (D129). Closed in
LNK2a: 77 (the streamer lowered per design point, D134). Closed in
FLOW3: 56 (D136), 57 and 60 (D137), and in its round 2a 23 and 25 (the
values on beat events and the push event) and 39 (the beat trace's size,
measured on SNAX-GVSoC).
Next free number: 83.

1. BRM dataflow notation: `affine` since D70, frozen in M6. A rank-free
   notation for unit BRMs comes earlier, with ND1 (open item 45).
2. What automated search reads and how a sweep is configured (DSE1, M8); a
   sweep is one recipe with a parameter (D72).
4. Stream as a proposer for DNN-shaped workloads, replayed as a recipe at
   cycle level, and an analytical estimator of our own for the kernels Stream
   cannot describe (the engine alone, open item 49, plus bank conflicts from
   the memory passes' address streams, D86). Parked (D23).
18. The L1 address map is fixed and word-interleaved, which is why the design
    point states no banks. If snax_cluster's map differs, it becomes a
    registered map kind.
20. Streamer ports are one bank wide; a wider port when a kernel needs it
    (PACK1).
21. `elems_per_word > 1` (packed words) has never been moved by streamers,
    accelerators or the DMA. Decided with sub-word packing (PACK1).
    The Accumulator's tree zero-extends its lanes (`UInt.pad`), so PACK1
    makes it sign-extend before int8 lanes feed an int32 accumulator.
27. Generating a BRM from Chisel, SystemVerilog or HLS sources instead of by
    hand. Far future.
28. Implementations whose `source` is SystemVerilog or HLS: only `chisel`
    until one is needed (E8).
31. A loop bound that is not a multiple of the spatial bound: `split_map`
    rejects it; a tail task or padding later.
32. The DFG viewer's layout (rows in Python, SVG edges, no library), revisited
    if graphs outgrow it.
33. An MLIR importer that writes `.snaxdfg` directly.
34. A hand-edited `.snaxdfg` runs through the flow (D108) but cannot be
    replayed or swept, and the design checks' fixes still speak of the
    recipe. Decided when needed.
36. Control flow in SNAX-DFG: a sequential `loop` kind for time steps and a
    `branch` kind with `case` nodes, registered when a kernel needs them (M9).
37. The expression grammar has `+ - * //` only: tail tiles need `min`,
    branches comparisons, jacobi1d a cast. Extended when a kernel needs it.
38. The DMA moves a container as contiguous whole beats (padded, D105);
    strided or 2-D transfers when a kernel needs them, with a check of the
    DMA's loop count.
40. Switching runs in the viewer clears the selected cycle; kept as is until
    a comparison needs it.
41. A report glossary, a "where the cycles went" ranking and a list of the
    knobs with their `--set` paths, on hold; a decision of its own when built.
44. Reductions over tiles need partial sums and a final combine; until then
    no tile loop sits around a `reduce` memlet. Decided with LOW3b.
45. A unit BRM's nest is one-dimensional, so 2-D containers and operands
    reused across an outer loop are rejected; a scratch run showed that with
    the checks relaxed, row and column sums, `A @ x` and `x @ A` run. Decided
    in ND1.
46. No transform tags an outer map `temporal` or `tile` without splitting
    it, and `bind` under an untagged map fails without a named error.
    Decided in ND1.
47. An in-place update (`y += x`) binds but does not lower: the writer waits
    for its own node's reader. Decided in LOW4.
48. The flow sizes inputs from one bound symbol, and fills take 1-D arrays
    only. Decided in ND1.
49. The engine-alone number C3 compares against (firings × II + latency) is
    computed nowhere; it belongs in `design.md` (DSE0).
50. Simulation speed: SNAX-MODEL ran about 4,000 cycles per second, so C1
    rests on turnaround and cause. Measured again on SNAX-GVSoC in BASE2;
    timing-only runs and extrapolation only if that is still too slow.
51. What snax_cluster has to change for a generated accelerator: a Bender
    dependency on the bundle (through `Bender.local`), a target listing the
    files `snaxgen` generates, later a file list `snaxgen` writes itself; the
    wrapper is named `<name>_shell_wrapper`. Also from LNK2a round 1
    (snax-gvsoc `LNK2a.md`): clustergen (`util/clustergen.py`) and snaxgen
    (`util/snaxgen/snaxgen.py`) call `JsonRef.replace_refs` without a
    `base_uri`, so a cross-file `$ref` (`snax_alu_acc.hjson#/datapath`) does
    not resolve; and `docs/schema/snitch_cluster.schema.json` names the
    `snax_acc_cfg` item schema `"items:"`, so no `snax_acc_cfg` entry is
    validated (copied as it is into snax-gvsoc's `sw/docs/schema/`). Since
    LNK2a round 2 the cluster file refers to two whole files beside it
    (`{$ref: "snax_alu_streamer.hjson"}`, `{$ref: "snax_alu_acc.hjson"}`),
    so the `base_uri` fix must resolve a file without a pointer too.
    Settled in LNK2b.
53. A link between two shell accelerators: amends D12 and D51 and the SNAX
    shell. Parked (D112).
54. Which kernels take PolyBench's SMALL in a 128 KiB L1, decided in FE1a;
    the tolerance of the NumPy check for float sums waits for the float pass
    (D113).
63. snax_alu's start and busy rules, and the snax_cluster changes they would
    need (`snax_alu_csr.sv`, the ReqRspManager's `ready`): the ALU never
    holds a start, also while busy, keeps `len_counter` across a start, lets
    the end win over a start in the same cycle, and never ends with length 0;
    the streamer holds a start while busy, which SNAX-GVSoC refuses with a
    warning. Generated accelerators follow D130.
65. Instruction fetch: snax_cluster's L0 (`cluster_icache`'s
    `snitch_icache_l0.sv`) prefetches the next line and the targets of taken
    branches and `jal`; GVSoC's Snitch I-cache does not, so SNAX-GVSoC misses
    more lines, each for fewer cycles: tens of cycles in fixed-cost phases
    (SG5 at 100: C2a 133 against 110, D1 112 against 98). Modelled when a
    kernel's numbers need it. A forge accelerator's counter depends on the
    core's timing between the streamer's and the accelerator's start writes
    (GEN2a round 1b F2: an L0 line starting between them gave 16 instead of
    19), so GEN2b compares that counter with care (D123).
66. Atomics to DRAM: the RTL testbench runs them in `axi_riscv_atomics_wrap`
    (read-modify-write, 2 reads and 2 writes in flight), SNAX-GVSoC as one
    access, so the exit's global barrier and reduction are about 50 cycles
    shorter here (SG5's C7). Outside the cluster; matters only for programs
    with atomics in DRAM.
67. The core's own timing beyond its loads (D125): the RTL polls `dmstati`
    every 5 cycles against GVSoC's 2, so the end of a short DMA is seen up to
    9 cycles later on the RTL (SG5's D2 at 128, accepted in SG5);
    a SNAX CSR poll takes 5 cycles against 6; an L1 load's write-back waits
    2 more cycles in SG5's check loop; the DMA core issues a copy (`dmsrc`,
    `dmdst`, `dmcpyi`) in 9 cycles against 10, and in `dma-tcdm` 9, 5, 6
    against 10, 7, 7 between copies, so its beats land 3 to 7 cycles early
    (snax-gvsoc `SG7.md`, round 4 follow-up). Kept as is (D123) unless a
    kernel's numbers need it.
68. Generation is spread out: snaxgen and Chisel in snax_cluster (the
    wrappers, `streamer_csr_addr_map.h`), clustergen and regtool outputs
    copied into snax-gvsoc's `sw/` (D126), snax-gvsoc's `snax_hjson` and
    its timing constants, `snax_forge/gen` (GEN3, LNK2a), and GEN1, GEN2a,
    GEN2b and LNK2b to come. One
    generator in snax-forge, planned in phase C, after RET1 (D127), which
    also decides whether it makes `sw/generated/`. The L1's size and base
    reach the program only through `sw/generated/` (LNK2a round 1): a design
    point that changes them needs it and the ELF rebuilt; its bank count
    does not.
69. The SNAX crossbar's AXI input (14) is wired on both SNAX targets but no
    program sends through it, on either side (snax-gvsoc `SG7.md`, round 3).
70. On SNAX-GVSoC a store or AMO after a pending L1 load waits for the
    load's data (the LSU keeps one request); the RTL sends it
    (`num_int_outstanding_mem` 4). No measured loop has one (snax-gvsoc
    NOTES, SG7 round 3b step 2).
71. An L1 load after a DRAM load sends its request at the start of D125's
    wait, so it reaches the crossbar up to 13 cycles early; the core's
    timing is right, and no streamer request meets it yet (snax-gvsoc NOTES,
    SG7 round 3b step 2).
72. An L1-to-L1 DMA copy's read and a delayed write go to the crossbar in
    one cycle, where the RTL's `axi_to_mem` takes one a cycle; no program
    makes one (snax-gvsoc `SG7.md`, round 4 follow-up).
73. The DMA reading L1 was not split against the RTL (where the bank read
    falls in D121's read path; the TCDM back-end has no stage per cycle); no
    program has it read L1 against a narrow request (snax-gvsoc `SG7.md`,
    round 4 follow-up).
74. The iDMA through the sparse interconnect as ports, instead of the
    superbank mux that refuses narrow winners: a snax_cluster design change,
    for exploration (snax-gvsoc `SG7.md`, "The sparse interconnect on the
    RTL").
75. Sparse granularity above 1: the crossbar supports it and was checked
    against an elaborated RTL module (snax-gvsoc `SG7.md`, step 5), but no
    target uses it yet.
76. Where reuse lives: placing each loop in the streamer, in the
    accelerator's state or on its lanes, and describing every unit,
    composite and shell to the level above by r, ports, throughput, latency,
    state and area. A candidate research direction (PAPER.md section 10);
    J1 may become jacobi1d in three placements, with jacobi2d.
78. The streamer as an input of its own (ND1 / tier A, amends D84 and
    D134): the knobs and the per-streamer overrides move out of the
    platform into a standalone streamer input, read by the design step
    beside the platform and the BRM, together with a frozen streamer file
    given as input (checked against the ports) and `temporal_dim` lowered
    from the DFG's temporal loops instead of the platform's knob.
79. Several accelerators on one core. snax_cluster has `snax_acc_mux_demux`
    and core CSR 1990 (`multi_acc_mux`), with `snax_multi_alu_cluster.hjson`
    and `snax-multi-alu`. SNAX-GVSoC lacks CSR 1990; `snax_hjson` refuses
    more than one slot per core and the generator refuses with
    `gen.instances`. Every accelerator on a core uses the same CSR
    addresses, so their headers collide by name. Decided with M13's
    baseline.
80. One Snitch core driving both the DMA and the accelerator: snax_cluster
    can, but rarely does. The C backend takes core roles from the cluster
    file; `snax_hjson` wants xdma on the last core, so with one core that is
    core 0. Never run.
81. No kernel gives a reduction with more than one sum per run (T < n).
    `vsum-r16` overrides the task list by hand (`forge-reduce`,
    `RATE_OVERRIDE=16`: one run, four sums of 16-element chunks). The 1-D
    form `out[i // 16] += A[i]` has a non-affine index; making it affine
    splits i into r and k, both temporal, ND1's split-and-tag transform
    (open item 46). `forge-reduce` retires once ND1 or tier A gives a real
    T < n. Revisited at ND1.
82. SNAX-GVSoC dumps the output regions at exit (the manifest names them),
    so the program can drop its check code and the flow compares them with
    the reference itself; until then the check is the program's own
    (D136).

## Sync Reminders

- After every change done with Claude, update this file and log any design
  change in `docs/DECISIONS.md` (rules at its top).
- When a milestone closes, compact `docs/DECISIONS.md` and remove the
  milestone's done tasks from this file (D117).
