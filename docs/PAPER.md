# SNAX-FORGE Paper Plan

What the paper claims, how each claim is measured, and which milestones deliver the evidence (P1, D100). Every number the paper states will come from a checked-in command; the tests that check them, as `examples/loop1`'s do, follow after the submission (PAP4). Items marked **[open]** are listed in section 10.

## 1. Venue

ISPASS 2027 (D100): the abstract on 11 Dec 2026, the paper on 18 Dec. C3 leads, as an analysis of which cluster-level effects decide what a cluster should hold; C4 is why the answer can be trusted, C1 why it is practical, and E7 why the generated blocks are worth using (PAP3). DAC 2027 is dropped; ICCAD and MICRO were the other candidates. Double-blind: SNAX, SNAX-MLIR, OpenGeMM, DataMaestro and Hypercorex are cited in the third person, the repository goes through an anonymous mirror, and the tool is named Atelier in the paper.

## 2. Problem

A SNAX cluster is a shell (shared banked L1, streamers, DMA, a uniform CSR control interface) that holds accelerators. Deciding which accelerators it should hold is slow today:

- **Integration takes about a week** before a first number: the first SNAX GeMM ran an 8x8x8 case after a week of integrating, configuring the streamers and writing the program.
- **Finding the cause takes longer than the number.** It was found in waveforms: bank conflicts in the GeMM convolutions of OpenGeMM and DataMaestro; in Hypercorex, an encoding and a search that ran one after the other where they could have been pipelined.
- **Non-uniform control makes it worse.** Hypercorex needed CSRs plus a custom instruction buffer.
- **The costs that decide are cluster costs:** bank conflicts, streamer back-pressure, control overhead and the hand-off between accelerators. They appear only after RTL integration.

HLS and XLS produce engines, not the shell. Analytical tools (ZigZag, Stream, Timeloop) treat shared memory as a bandwidth budget over DNN-shaped loop nests. Pre-RTL simulators (Aladdin, gem5-SALAM) model one accelerator next to a CPU and caches. None evaluates compositions of engines inside a shared-L1 shell, at cycle level, before RTL, from the workload's dataflow graph.

## 3. Thesis

> An accelerator cluster is a shell holding accelerators that are themselves compositions of unit engines. SNAX-FORGE binds unit engines onto a workload's dataflow graph, evaluates the composed cluster in a cycle-level model of the shell with the real control program, reports the cause of every lost cycle, and generates the chosen composition as RTL that plugs into SNAX, so shell compositions can be compared and explained before RTL.

**Problem and solution, as the paper puts them.** Deciding which accelerators to build into a compute cluster, and how, is slow and unguided: the costs that decide it (bank time, interconnect ports, DMA, control, the hand-offs between accelerators) show only after RTL integration. The solution is one design with three levels of feedback, each checked against the one below: (1) the workload's dataflow graph, where the designer splits loops into streamer and lane work and binds blocks, every step showing its early cycle, DMA, interconnect, area and energy cost (seconds); (2) the accelerators, program and cluster configuration generated from that graph and run on SNAX-GVSoC (minutes); (3) the same design as RTL on vsim and through synthesis (hours), the final form. To our knowledge no other flow puts the cluster's cost on the workload's own graph for a designer who decides, and generates from that graph several accelerators in a shared-L1 cluster checked against RTL; DSAGEN and OverGen cost an architecture graph inside an automated search over a fixed set of fabric primitives (section 8.4).

Composition is one spectrum: unit engines stitched on the graph (a multiplier into an accumulator, `dot`), a unit set that forms a known accelerator (SNAX's GeMM core as a spatial array plus a SIMD unit), or a full accelerator. `bind` in a recipe covers all three (D102, D112).

## 4. Contributions

1. **The engine–shell split.** A unit engine is a BRM (interface, per-port affine dataflow, latency, II, drain, function; D68, D103); the shell is modelled once, in SNAX-GVSoC, and checked against snax_cluster RTL (D116). Shown to capture existing SNAX accelerators and an engine from HLS.
2. **A pre-RTL path from NumPy to a composed cluster.** DaCe SDFG, SNAX-DFG, recipes that bind unit engines, a design point, a generated program, and a cycle-level run with reports that name causes (D99).
3. **Composition as design-space exploration.** Sweeps over compositions with cycles against area, and cases where the cluster-level choice differs from the per-engine choice.
4. **One description, two outputs.** Each BRM gives the model's accelerator and its Chisel RTL, and the hjson describes the cluster to both, so the chosen composition runs in the model and in SNAX from the same design (D111, D116).

Not contributions: the simulation technique (GVSoC's) and the LLM; an agent driving the tool is future work (E10). The BRM interface is open to blocks beyond the basic ones, a design property the paper states but does not demonstrate (E8 is future work).

## 5. Claims

| ID | Claim | Measurement | Baseline | Delivered by |
|---|---|---|---|---|
| C3 | Which accelerators belong in a compute cluster is a cluster-level decision, and it can be made on the workload's dataflow graph: each transform shows its cost in the shell, and the same graph generates the hardware and program that confirm it | Rank of compositions alone against in the cluster; cycles against area and energy; multi-accelerator against homogeneous clusters in area efficiency and energy per task | The engine-alone number (D155's ideal estimate); a cluster of Snitch cores only (E11) | E4, E11, E5 if it lands |
| C4 | Each feedback level ranks design points as the level below does: the graph's estimate as SNAX-GVSoC, SNAX-GVSoC as snax_cluster's RTL, the cost model as synthesis | Kendall tau and top-1 agreement over 3–5 compositions per kernel; mean absolute cycle error; the cost model's error at block and cluster level | snax_cluster RTL on vsim, same ELF (D123); Nangate 45 nm synthesis | E1, HWCOST against the cluster synthesis |
| C1 | Time to insight: the cost and the cause of a design choice in seconds to minutes, not a week of integration and waveform reading | Wall-clock per step from graph to number on each level; simulated cycles per second | The vsim path for the same design, timed step by step; the week of SNAX's first GeMM as the prior anecdote | E2 (BASE1, BASE2) |
| C5 | Generated RTL plugs into SNAX: the chosen composition runs in snax_cluster from the same description | The same ELF on SNAX-GVSoC and snax_cluster's RTL | Hand integration in SNAX | M14, E1 |
| C2 | (supporting) NumPy kernels reach the model with no hand edits: NPBench's own DaCe sources, the dtype a parameter (D113) | The kernels built from basic blocks; an import-level coverage table | — | E3 |

## 6. Evaluation plan

The ISPASS 2027 paper (PAP3, D100) carries E1–E4, E7 and E11, with E5 if it lands; the order of the work is in STATUS's Next Up.

| ID | Experiment | Workload | Output | ISPASS 2027 |
|---|---|---|---|---|
| E1 | Fidelity: SNAX-GVSoC against snax_cluster RTL | Open item 95's 14 designs; mvt and atax (gesummv if time) at 3–5 compositions each that differ in cycles | Rank metrics and error table | in |
| E2 | Turnaround and speed | E1's runs, each step timed on both paths | Steps and times per level; cycles per second | in |
| E3 | The composite kernels | tier done, the untiered kernels and tier A, built from basic blocks; an import-level coverage table over the staged kernels | Cycles per kernel; coverage table | in |
| E4 | Composition sweeps | tier A (mvt serial against overlapped, open item 109; atax on two engines against one shared, open item 113; shared against dedicated crossbar ports, PORT1) and W sweeps across the bank bound, at MINI | Cycles against area and energy per composition, the engine-alone choice marked | in, the headline |
| E5 | End-to-end case: Transformer decode | One attention head (q·Kᵀ, softmax, ·V) at a short context that fits L1 | Where the cycles go; the best composition | if softmax runs by 27 Nov 2026 |
| E6 | Familiar case | A TinyML ResNet slice: conv as GeMM, SIMD requant, maxpool | Where the cycles go per layer | future work (needs tier D's GeMM and M15) |
| E7 | Engine quality | tier done and untiered: forge against Vitis opt and dace, Bambu and XLS at W ≤ 16, 10 ns HLS clock, n2048 for tier done and Vitis's power sizes for untiered | Post-synthesis area and fmax; dynamic energy per task (PrimeTime); control apart where a tool separates it | in |
| E8 | Engine-agnostic shell | An HLS engine as a BRM implementation (open item 28) | Cluster-level cycles against the Chisel engine | future work |
| E9 | Against Stream | The DNN layers of E5 and E6 | Stream's allocation and its rank in SNAX-FORGE | future work |
| E10 | LLM demo | One E4 composition found by an agent | A short trace | future work |
| E11 | Against a homogeneous cluster | E1's kernels on snax_cluster with its Snitch cores only (hand-written C) against the chosen compositions, both synthesized on RAM1's macros at 45 nm | Area efficiency and energy per task, not equal area | in |

### 6.1 Kernel set

NPBench's DaCe sources, their structure kept, with the dtype as one parameter (D113). Integers come first: `int64` with an exact reference check through tier C, then `int8` in and `int32` accumulate from tier P on, as SNAX's GeMM (OpenGeMM) does. Float is a second pass per kernel, as a new BRM implementation. About a dozen kernels, chosen by the tier each exercises, the rest in the coverage table as named errors (FE2).

| Tier | Compute row | Kernels | Width |
|---|---|---|---|
| done | II 1, fold | vecadd, vecmul, vsum, dot | int64 |
| A | fold, no reuse | atax, bicg, mvt, gesummv | int64, exact |
| B | multi-cycle, II > 1 | an integer divider unit; covariance's mean | int64, integer divide |
| C | multi-phase | covariance (mean, center, product), then softmax | int64; softmax quantized: integer max, polynomial or table exp, integer divide (I-BERT style) |
| P | int8 on kernels that already work | vecadd, dot, atax | int8 in, int32 out |
| D | reuse (access axis) | gemver (rank-1 update, stride 0, D70), gemm, k2mm, k3mm, doitgen; jacobi_1d, jacobi_2d | int8 in, int32 accumulate; jacobi in a wider fixed-point type |
| E | DL layers | mlp, conv2d_bias, lenet, resnet | int8 with requantization between layers |

They are built in the tiers of STATUS (M7). Sizes are PolyBench's MINI (`atax` 38 x 42, `gemm` 20 x 25 x 30), and SMALL where containers and transients fit SNAX's 128 KiB L1 (open item 54); NPBench's own presets (hundreds of megabytes) wait for tiling (M15). Left out on purpose: triangular and data-dependent kernels (`cholesky`, `lu`, `trisolv`, `durbin`, `nussinov`). SODA-OPT reported `2mm`, `3mm`, `atax`, `bicg`, `gemm`, `gesummv` and `mvt`, so their numbers can be cited next to ours.

E9 runs Stream on its own inputs for the same layers and accelerators; SNAX-FORGE runs its allocation and the alternatives at cycle level. That the stencil and mixed kernels have no Stream counterpart is part of C3's argument. **[open]** the effort, and whether an export from SNAX-DFG to Stream's format is worth automating.

### 6.2 Kernel regularity: two axes

The kernels are ordered by two kinds of regularity. Access is what the streamers do: affine with no reuse, affine with reuse, then non-rectangular or data-dependent. Compute is what the unit does: one firing per cycle at a fixed latency, a fold with a loop-carried dependency, multi-cycle at a fixed latency or II > 1, multi-phase (a global result gates the next pass), then data-dependent latency or control.

The table, with every kernel placed in it, is in `kernels/README.md`.

Down the table, the unit stops being one firing per cycle, and the question moves from lanes and ports to state, latency, II and phases. To the right, the streamer alone stops being enough: reuse has to live somewhere (open item 76), or the accesses stop being affine. The compute axis comes first (down the first column), since it is easier to implement; the access axis (2-D, reuse) after. Along the compute axis, integers come first (D113).

## 7. Cost model

Area first, energy if characterisation allows; characterised once, then read from tables: post-synthesis area per BRM parameter (lanes, width, depth), fitted linearly where it is; the L1 macros and bank counts on a PDK; TCDM interconnect area against banks and ports; later per-access energy times the counts the model keeps, as Accelergy does. Each fit's error is stated. No place and route. From 5 Nov 2026 characterization is on Nangate 45 nm only (D159): OpenRAM has no 15 nm technology, so the L1 macros (RAM1), the cluster synthesis and E11 exist at 45 nm only; what is at 15 nm already (SYN1, SYN2, CHAR1, HSYN1) stays, and nangate15's L1 and control costs stay D154's placeholders. The PDK question is answered: the Nangate 45 nm and 15 nm Open Cell Libraries, synthesis and gate-level power in KULeuven-MICAS/snax-forge-experiments, with results committed. SYN1 gives forge's area and dynamic energy beside the HLS points at one task of N = 2048, with ctrl and csr apart (`results/forge_vs_hls.md`); CHAR1 characterizes each basic block alone and HWCOST sums them, checked against dot 2–6 synthesized whole.

## 8. Related work and positioning

A draft (PAP2, `brief`) from abstracts and a first pass, before the author's own reading. **[check]** marks a statement the paper has to confirm; section 8.4 is the reading table that turns the draft into the final text.

### 8.1 Comparison

✓ yes, ◐ partly, ✗ no.

| Work | Kernels beyond DNN loop nests (C2) | Engines composed at design time (C3) | Shared banked L1, streamers and control at cycle level (C3) | Reports that name the cause (C1) | Minutes per design point (C1) | Hardware into a cluster (C5) | Checked against RTL or silicon (C4) |
|---|---|---|---|---|---|---|---|
| gem5-SALAM | ✓ | ◐ | ◐ [check: bank arbitration] | ✗ | ◐ | ✗ | ✓ |
| GVSoC (with its Snitch shared-L1 model) | ✓ (as binaries on cores) | ✗ | ✓ for cores, ✗ for accelerators | ✗ | ✓ | ✗ | ✓ |
| Stream | ◐ (DNN, Transformer, SSM) | ◐ (layers onto cores) | ✗ (shared bus) | ◐ | ✓ | ✗ | ✓ |
| Timeloop/Accelergy, AccelForge | ◐ (einsums) | ✗ / ◐ | ✗ | ◐ | ✓ | ✗ | ✓ / ✗ |
| CHARM (Versal), SSR | ✗ (GeMM, Transformer) | ✓ | ✗ | ✗ | ✓ (analytical) | ✓ (FPGA) | ✓ |
| Richie | ✓ | ◐ | ✓ in RTL, after RTL | ◐ [check] | ✗ (FPGA build) | ✓ | ✓ |
| SODA-OPT / SODA Synthesizer | ✓ | ◐ | ✗ | ◐ | ✗ (hours) | ✓ (single engine) | ✓ |
| Allo, HIDA, Stream-HLS | ✓ | ✓ | ✗ | ◐ (HLS reports) | ✗ | ✓ (FPGA) | ✓ |
| **SNAX-FORGE** | ✓ | ✓ | ✓ (on GVSoC) | ✓ | measured in E2 | ✓ (M14) | ✓ (open item 95; E1) |

### 8.2 Works, one line each

Pre-RTL and platform simulators:

- **Aladdin** (ISCA 2014): pre-RTL power and performance of one accelerator, 1 minute per point against 87 in RTL; credited for pre-RTL DSE and the C1 table layout. (C1)
- **gem5-Aladdin** (MICRO 2016): an accelerator designed in isolation is over-provisioned, co-design with the SoC gives up to 7.4x better EDP; the general form of C3. We add composition in a shared-L1 shell. (C3)
- **gem5-SALAM, SALAMv2** (MICRO 2020, JSA 2024): LLVM-IR accelerators with scratchpads, DMA and stream buffers in full-system gem5; the closest simulator. Our engines are block models bound on a graph, not compiled from C; the shell is SNAX's, bank arbitration and streamers included [check SALAM's scratchpad], on a bare-metal core with no OS. (C1, C3)
- **GVSoC** (ICCD 2021) and its **Snitch shared-L1 model** (2026: 1024 cores within 7% of RTL, 115x faster): SNAX-GVSoC is built on it (D116). We do not claim the simulation technique; we add SNAX's accelerators, streamers and register blocks, the path from a NumPy kernel to a design point, generation of the model's accelerator and its RTL from one BRM, and reports that name causes. (C1, C3)
- **gem5-accel, PARADE, MosaicSim**: other accelerator-in-system simulators; PARADE's timing from HLS II and depth is close to the BRM's. (C3)

Analytical DSE:

- **ZigZag** (TC 2021): one accelerator over DNN layers, the "best accelerator alone" view C3 argues against. (C3)
- **Stream** (TC 2025): layer-fused scheduling on heterogeneous multi-cores, 2–5 s per point, shared resources as a bus and a DRAM port; complementary, it proposes and we check at cycle level (E9). (C3)
- **Timeloop/Accelergy, AccelForge**: analytical models over einsums; Accelergy is the precedent for section 7; AccelForge claims heterogeneous architectures without an evaluation yet [check]. (C3)
- **Herald, MAGMA** (HPCA 2021, 2022): choose and schedule heterogeneous sub-accelerators with static partitions; the closest idea-level predecessor to "which mix in one cluster". (C3)

Composition:

- **CHARM, CAMEL** (ISLPED 2012, 2013): accelerators composed from building blocks at runtime; we compose at design time on the graph. (C3)
- **CHARM on Versal, CHARM 2.0** (FPGA 2023, TRETS 2024): several sized matrix-multiply accelerators beat one large one (5.4x BERT, 32.5x ViT), analytically, on FPGA. (C3)
- **SSR** (FPGA 2024): spatial, sequential and hybrid Transformer accelerators; the closest precedent for E5. (C3)

Shells and integration:

- **PULP HWPE**: accelerators on a shared TCDM through streamers with a register file, the shell SNAX uses; credited for it. (C5)
- **Gemmini** (DAC 2021): SoC-level choices change the best configuration. (C3)
- **Richie** (TPDS 2025): generates accelerator-rich multi-cluster SoCs from HWPE-wrapped engines and explores them by FPGA emulation; the same problem, but every point is an FPGA build [check: whether software is generated]. (C1, C5)
- **ESP, Chipyard**: agile SoC flows; ESP's first accelerator on FPGA "in a few hours" is a C1 reference point. (C1)

Compilers and HLS:

- **DaCe/SDFG, NPBench**: our front end and benchmark set; DaCe has no PULP or SNAX target. (C2)
- **SNAX-MLIR, xDSL Snitch backend** (CGO 2025): compile onto a fixed cluster; we decide what it holds. (C2)
- **SODA-OPT, SODA Synthesizer** (ICCAD 2022): Python to GDSII in under 3 hours per design, feedback after synthesis, no engines sharing a banked scratchpad. (C1, C3)
- **Union** (PACT 2021 [check]): MLIR into Timeloop and MAESTRO, analytical. (C2, C3)
- **Allo, HIDA, Stream-HLS, TAPA**: compose kernels into HLS dataflow designs; their engines can be BRM implementations (E8). (C3, C5)
- **Vitis HLS, Bambu, XLS/DSLX**: engine generators, inputs to E7 and E8; Lahti et al. (TCAD 2019) for HLS quality against hand RTL. (C5)

Deployment and LLM-driven DSE:

- **MATCHA** (DAC 2026): deploys DNNs across the accelerators of a fixed PULP SoC; we choose the silicon. (C3)
- **The Configuration Wall** (ASPLOS 2026): configuration overhead caps accelerators below peak; motivates the control part of C3, which the core now measures. (C3, C4)
- **Explainable-DSE** (ASPLOS): bottleneck analysis with hand-written fixes; the precedent for feedback that names causes. (C1)
- **Beacon, AgentDSE, LLM-DSE**: LLM agents driving simulators or HLS; we claim only that the tool is usable by one (E10).

### 8.3 Statements to avoid

| Avoid | Write instead |
|---|---|
| ZigZag, Stream and Timeloop are DNN-only | They model tensor or einsum loop nests with bandwidth-level shared resources |
| DaCe and MLIR do not target PULP or SNAX | DaCe has no PULP target; MLIR flows compile onto a fixed cluster, and none choose what it holds |
| SNAX-MLIR handles only GeMM | Its evaluation centres on GeMM |
| SODA builds hardware by hand, single kernel | SODA generates hardware automatically, with feedback per design after synthesis |
| HLS has 10–30% control overhead | Cite Lahti et al., or measure it (E7) |
| XLS is more efficient than HLS | XLS is one more engine generator |
| SNAX-FORGE's simulator is new | SNAX-GVSoC builds on GVSoC; the contribution is the SNAX components and the path to them |
| SNAX-FORGE is the first to stitch kernels | First to evaluate compositions of fixed engines in a shared-L1 shell at cycle level before RTL (to our knowledge) |

### 8.4 Reading table

The draft's line, then the author's after reading, which wins. A row whose second line cannot be written is a threat (Q11).

| Work | Draft: what it cannot do | After reading |
|---|---|---|
| gem5-SALAM, SALAMv2 | Compose engines from a dataflow graph; model bank arbitration and streamers | |
| GVSoC shared-L1 model | Model accelerator streamers; go from a kernel to a design point; report causes | |
| Stream | See bank conflicts, chaining waits and control; take kernels that are not DNN-shaped | |
| CHARM (Versal), CHARM 2.0 | Leave GeMM; model contention in a shared L1 at cycle level | |
| SSR | The same, for spatial against sequential | |
| Richie | Give a result without an FPGA build; name bank conflicts or layout as the cause | |
| SODA-OPT, SODA Synthesizer | Model engines sharing a banked scratchpad; answer in minutes | |
| MATCHA | Choose the silicon | |
| AccelForge | Model cycle-level contention | |
| DSAGEN (ISCA 2020), OverGen (MICRO 2022) | [first pass, PAP3] C with pragmas in, the workload graph internal to the compiler; one architecture graph of a fixed set of primitives (PEs, switches, delay and sync elements, memories, a control unit) drives compiler, simulator and RTL; a regression area and power model (DSAGEN, 4–7 % of synthesis) or a learned resource model (OverGen) inside an automated search (perf²/mm², simulated annealing); one programmable fabric, OverGen an FPGA overlay as an alternative to HLS. Where we differ: the cost on the workload's own graph for a designer who decides, fixed-function accelerators composed in a shared-L1 cluster with a control core, blocks open through the BRM | |
| APEX (ASPLOS 2023) | [unread] Merges frequent subgraphs of primitive operations into CGRA elements | |
| Aetherling (PLDI 2020) | [unread] Image pipelines on FPGA; its space-time types are close to `split_map`'s temporal and spatial loops | |
| Softbrain, stream-dataflow (ISCA 2017) | [unread] The reference for the streamer-shell pattern | |

## 9. Features the evidence needs

| Feature | Needed by | Where |
|---|---|---|
| SNAX-GVSoC matching snax_cluster RTL | E1, every cycle count | M16 (done); the local RTL check, open item 95 |
| Generated RTL, GVSoC accelerator, hjson and program | C5, E1 on the same design | M14 (done) |
| The composite kernels | E1, E3, E4, E11 | M13 track A: B2, tier A (open items 47, 54, 109, 113) |
| SIMD and the CSR scalar | E3, E7's untiered rows | M13 B2 |
| Overlap of independent accelerators | E4 (mvt) | open item 109, in tier A |
| One unit across DFG regions | E4 (atax) | open item 113, in tier A |
| Shared crossbar ports across streamer sets | E4 | M13 PORT1, model level first (open item 114) |
| DMA cost on the graph | Level 1's feedback, E4 | M13 L2V1 (open item 99 for the graph-level node) |
| Interconnect markers | Level 1's feedback, E4 | M13 XBAR1 |
| The engine-alone number | C3's baseline in E4 | M13 VIEW2, the ideal estimate (D155, C§19) |
| Cost model: blocks, streamers, memory, interconnect | E4, E7, C4's cost check | M13 CHAR1, CHAR1b (done), RAM1, HWCOST (section 7, D159) |
| A sweep script | E4 | over `pixi run flow --set`; DSE5 (M8) is not needed |
| Cluster synthesis on the same macros | C4's cost check at cluster level, E11 | RAM1, then 2–3 composites and the homogeneous cluster at 45 nm |
| `exp`, `max`, division | softmax in E5 | M7 tiers B and C (FE1b, FE1c; open item 37) |
| Speed on long runs | E2 | open item 50 |
| Packing (int8), tiling, the GeMM | E6, E9 | PACK1, M15, tier D: after the paper |
| HLS engine as a BRM implementation | E8 | open item 28: after the paper |

## 10. Open questions

1. BASE1 is E2's timed vsim path; the GeMM bring-up stays an anecdote, not a logged baseline (PAP3).
2. Floating point, the second pass per kernel after the integer tiers (D113): Berkeley HardFloat ops inside the existing blocks' lanes is one option. Either way the multi-cycle rows of section 6.2 stay.
3. E5's head: size, context length and precision (one head that fits L1, e.g. d = 64 and a context of 128); E6 is future work.
4. The snax_cluster version to pin for SNAX-GVSoC and the bundle.
5. The newer DIMC: when it joins, and whether as a second anchor or as the heterogeneous case.
6. E9's effort and who to involve.
7. Limitations to state: compositions are hand-written recipes (search is M8); engines whose dataflow changes per kernel (Hypercorex, CGRAs) do not fit a fixed BRM yet; fidelity holds as far as E1 shows.
8. The story: composition first (kernel-sized, designer-written units stitched from the SNAX-DFG into a datapath in a shell), with the simulator as the evaluation engine; the flow needs a name of its own, since "mid-level synthesis" is XLS's term.
9. Not settled: in a shared-L1 cluster, how should a workload be split into accelerators, and each accelerator into units and loop placements, given limits on any of bandwidth, area and throughput, when every level is described to the one above by the same small set of numbers (open item 76)?

## 11. Questions that shape the paper

A guide, not a verdict: what has to be settled before the outline (section 12). Answers are written under the question as they come.

**The claim.**

- **Q1. Tool or finding?** The thesis reads as a finding, the contributions as a tool. Settled once E4–E6 show whether a result stands without the tool's name.
- **Q2. C1 or C3 first?** Leaning to the pair. Settled by which evidence is stronger: a turnaround table (E2) or a ranking that flips (E4–E6). PAP3: C3 leads, C4 and C1 behind it (section 1).
- **Q3. Is the engine–shell split a contribution?** Claim one model of the shell any engine plugs into, and composition at design time on the graph; not the shell (HWPE, SNAX) nor composition as such (CHARM). Settled by reading HWPE and CHARM.

**The reader.**

- **Q4. Beyond SNAX?** A PULP HWPE cluster has the same shell; one paragraph, or a second platform file as evidence.
- **Q5. How much SNAX to explain?** One figure that shows the flow and the shell together.

**The evidence.**

- **Q6. Least fidelity evidence?** Ranking agreement on 3–5 GeMM sizes with the absolute error beside it, plus one DMA-heavy case; it started on SNAX's ALU in M16 and stands now on the generated vecadd and vsum (GEN2b: counters equal, windows within 10%).
- **Q7. Which C3 case, and what if no ranking flips?** E4 is cheap and run first; a scratch run suggests flips exist (`x @ A` with as many columns as banks puts a column's lanes in one bank, and one column of padding restores the gain). Fallback: the cluster changes how much an engine is worth, with the cause named.
- **Q8. What does C1 measure?** The time to a first number and the time to the cause, with simulation speed apart (BASE2).
- **Q9. Area only, or energy too?** Area first.
- **Q10. Generation in the first paper?** Yes (D111): E1 compares the same design in model and RTL.

**Related work.**

- **Q11. Does each separating sentence hold after the reading?** Tested in section 8.4; one that fails changes the claim.
- **Q12. Head-to-head run or a feature table?** Stream as E9, exploratory and parked (D23).
- **Q13. HLS and XLS: input or competitor?** Input (E7, E8). PAP3: E7 only; E8 is future work.
- **Q14. The name clash with AccelForge?** At least one sentence of contrast.

**Scope and story.**

- **Q15. One paper or two?** Decided when the evidence exists, now that there is no deadline.
- **Q16. The LLM paragraph?** One short demonstration (E10), if clean. PAP3: future work, one sentence.
- **Q17. Limitations up front?** Section 10, item 7.
- **Q18. Open with the experience or the question?** The question first, the experience as its evidence in two or three sentences.
- **Q19. Figure 1?** The flow with the shell for a tool paper; the composition spectrum on one graph for a finding paper.

## 12. Paper outline

After the questions of section 11, starting from the author's own outline.
