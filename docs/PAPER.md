# SNAX-FORGE Paper Plan

What the paper claims, how each claim is measured, and which milestones deliver
the evidence. Written in P1 (D100); the order of the milestones it implies is
logged as a decision (PAP3). Every number the paper states will come from a
checked-in command and be checked by a test, as `examples/loop1` is (PAP4).

Written by PAP1 (claims, evidence, evaluation plan). PAP2 added section 8 as
a draft and section 13, the questions that shape the paper; section 10 is
decided by PAP3, and section 14 follows the answers to section 13. Items marked
**[open]** are listed in section 11; they are about the paper, so they live
here, and those that change the system become open items in `docs/STATUS.md`
when PAP3 decides them.

## 1. Venue and timeline

- **Quality first.** The full system does not fit in six pages; the target is
  the paper the work deserves, not the nearest deadline.
- **DAC 2027 as a checkpoint.** Abstract 11 Nov 2026, paper 18 Nov 2026
  (17:00 PST), 6 pages + 1 page of references, double-blind. Around mid-November
  we check whether E1, E4 and one case of E5 or E6 (section 6) exist and tell a
  clean story in six pages. If yes, submit a focused DAC paper; if not, carry on.
- **Fallback:** ICCAD 2027 or MICRO 2027 (deadlines usually in spring; check
  once their calls are out). ISCA 2027 falls in the same week as DAC, so it is
  not a later option. A focused DAC paper followed by a full paper needs
  substantially new results in the second (E5 and E6 at scale).
- **Double-blind:** SNAX, SNAX-MLIR, OpenGeMM, DataMaestro and Hypercorex are
  cited in the third person; the repository goes through an anonymous mirror.

## 2. Problem

A SNAX cluster is a shell (shared banked L1, streamers, DMA, a uniform CSR
control interface) that holds accelerators. Deciding which accelerators it
should hold is slow today:

- **Integration takes about a week** before a first number: the first SNAX
  GeMM ran an 8x8x8 case after a week of integrating the accelerator,
  configuring the streamers and writing the program.
- **Finding the cause takes longer than the number.** When performance varied,
  the reason was found in waveforms: poor bank-conflict management in the GeMM
  convolutions of the OpenGeMM and DataMaestro work, and in Hypercorex an
  encoding and an associative-memory search that ran one after the other where
  they could have been pipelined.
- **Non-uniform control makes it worse.** Hypercorex needed CSRs plus a custom
  instruction buffer, which made programming it inside a multi-accelerator SoC
  hard.
- **The costs that decide are cluster costs:** bank conflicts, streamer
  back-pressure, control overhead and the hand-off between accelerators. They
  appear only after RTL integration.

HLS and XLS produce engines, not the shell around them. Analytical tools
(ZigZag, Stream, Timeloop) treat shared memory as a bandwidth budget and cover
DNN-shaped loop nests. Pre-RTL simulators (Aladdin, gem5-SALAM) model one
accelerator next to a CPU and caches. None of them evaluates compositions of
engines inside a shared-L1 shell, at cycle level, before RTL, from the
workload's dataflow graph.

## 3. Thesis

> An accelerator cluster is a shell holding accelerators that are themselves
> compositions of unit engines. SNAX-FORGE binds unit engines onto a workload's
> dataflow graph, evaluates the composed cluster in a cycle-level model of the
> shell in seconds, and reports the cause of every lost cycle, so that shell
> compositions can be compared and explained before RTL, and the chosen one
> generated as RTL that plugs into SNAX.

Composition is one spectrum: unit engines stitched on the graph (a multiplier
into an accumulator, `dot`), a unit set that forms a known accelerator (SNAX's
GeMM core as a spatial array plus a SIMD unit), or a full accelerator. The same
mechanism, `bind` in a recipe (D79, D104), covers all three.

## 4. Contributions

1. **The engine–shell split.** A unit engine is a BRM (interface, per-port
   affine dataflow, latency, II, drain, function, D68, D103); the shell is
   modelled once (TCDM banks and arbitration, streamers, DMA, uniform register
   interface, D36). Shown to capture existing SNAX accelerators and an engine
   from HLS.
2. **A pre-RTL path from NumPy to a composed cluster.** DaCe SDFG, SNAX-DFG,
   recipes that bind unit engines, a design point, a lowered control program,
   and a cycle-level model with reports that name causes (D99, D106). Checked
   against SNAX RTL.
3. **Composition as design-space exploration.** Sweeps over compositions with
   cycles against area, and case studies where the cluster-level choice differs
   from the per-engine choice.
4. **Generation into SNAX.** The chosen composition is emitted as Chisel RTL
   that plugs into the SNAX cluster, closing the loop from model to hardware.

Not a contribution: the LLM. The tool works as a black box an LLM agent can
drive, because every input and output is text; the paper shows this in one
short demonstration (E10).

## 5. Claims

The claim IDs replace PAP1's first cut: C1 and C3 keep their meaning, C2 becomes
supporting, C4 becomes fidelity (the old C4, pre-RTL seconds against SODA and
Richie, moves into C1 and related work), C5 is new.

| ID | Claim | Measurement | Baseline | Delivered by |
|---|---|---|---|---|
| C1 | Time to insight: the cost and the cause of a design choice in seconds, not a week of integration and waveform reading | Time to a first cycle count and time to the cause, per step; simulated cycles per second | The SNAX RTL flow with each step logged (BASE1); RTL simulation speed; Aladdin's and GVSoC's reported speedups for context | BASE1, BASE2, `flow.log` (D107) |
| C3 | Composition must be decided at cluster level: the best engine or mix in isolation is not the best in the shell | Rank of compositions alone against in the cluster; cycles against area | Per-engine evaluation (the BRM alone, roofline); Stream's analytical ranking on the DNN overlap (E9) | E4, E5, E6 |
| C4 | The model ranks design points as SNAX RTL does | Kendall tau and top-1 agreement over 3–5 configurations; mean absolute cycle error stated | SNAX RTL simulation of the default 8x8x8 GeMM, plus one DMA-heavy case | E1 (open item 42, amends D51) |
| C5 | The shell is engine-agnostic: existing SNAX accelerators and HLS engines fit the unit-engine description, and generated RTL plugs into SNAX | Number of SNAX accelerators expressed as BRMs; an HLS engine run as a BRM implementation; generated RTL running in SNAX | Hand integration in SNAX | E7, E8, GEN |
| C2 | (supporting) NumPy kernels reach the model with no hand edits | Coverage table: imported and checked, or a named error | — | FE1–FE3 |

## 6. Evaluation plan

| ID | Experiment | Workload | Output | Paper |
|---|---|---|---|---|
| E1 | Fidelity: model against SNAX RTL | Default 8x8x8 GeMM at 3–5 sizes from 8x8x8 to 128x128x128; one DMA-heavy case | Rank metrics and error table | DAC and full |
| E2 | Turnaround and speed | GeMM bring-up (logged), `vecadd`, `dot`, one PolyBench chain | Table: steps and times, model against RTL; cycles per second | DAC and full |
| E3 | Breadth | The kernel set in section 6.1 | Coverage table and one cycles-per-kernel figure | Full (one line in DAC) |
| E4 | Composition sweep | PolyBench chains (mat-vec and mat-mat) at sizes that fit in L1 | Cycles against area, one point per composition; the per-engine choice marked | DAC and full |
| E5 | Headline case: prefill against decode | One attention block of a small Transformer, 2–3 sequence lengths | The best shell for each phase and the best single shell for both, at equal area | Full (DAC if ready) |
| E6 | Familiar case | A TinyML ResNet slice: conv as GeMM, SIMD requant, maxpool | Where the cycles go per layer; one composition change and its effect | Full (DAC if ready) |
| E7 | Engine quality | One unit (mat-vec or 8x8x8 GeMM) in Chisel, Vitis HLS or Bambu, and XLS, at equal II and clock | Post-synthesis area and timing | Full |
| E8 | Engine-agnostic shell | The HLS engine from E7 plugged in as a BRM implementation (latency and II from its report) | Cluster-level cycles against the Chisel engine | Full |
| E9 | Against Stream (exploratory) | DNN layers of E5 and E6 only | Stream's chosen allocation and its rank against SNAX-FORGE's | Full, if feasible |
| E10 | LLM demo | One E4 composition found by an agent reading the reports | A short trace of the agent's steps | One paragraph |

### 6.1 Kernel set

Chosen by the unit pattern each exercises; about a dozen, with the rest in the
coverage table as named errors (FE2).

| Pattern | Kernels |
|---|---|
| Elementwise, reduction | `vecadd`, `dot` (done) |
| Mat-vec chains | `atax`, `bicg`, `mvt`, `gesummv`, `gemver` |
| Mat-mat chains | `gemm`, `2mm`, `3mm` |
| Batched contraction | `doitgen` |
| Stencil | `jacobi1d`, `jacobi2d` |
| Mixed | `covariance` |
| NPBench DL kernels | `softmax`, `mlp`, `conv2d` |

Left out on purpose (named errors): triangular and data-dependent kernels
(`cholesky`, `lu`, `trisolv`, `durbin`, `nussinov`), outside `split_map`'s
rectangular maps. SODA-OPT reported `2mm`, `3mm`, `atax`, `bicg`, `gemm`,
`gesummv` and `mvt`, so their numbers can be cited next to ours.

### 6.2 Comparison with Stream (E9)

Stream is not plugged into SNAX-FORGE. The same DNN layers and the same
accelerators are described once in Stream's inputs (workload graph, cores) and
once in SNAX-FORGE; Stream proposes an allocation, SNAX-FORGE runs it and its
alternatives at cycle level, and the paper reports where the two rankings
disagree and why (bank conflicts, chaining waits, control). This only covers
the DNN-shaped parts; that the stencil and mixed kernels of E3 have no Stream
counterpart is itself part of C3's argument. **[open]** effort and whether an
export from SNAX-DFG to Stream's workload format is worth automating.

## 7. Cost model

Area first, energy if characterisation allows. Pre-characterise once, then the
DSE reads tables:

- Unit engines: post-synthesis area per BRM parameter (lanes, width, depth),
  fitted as linear in the parameters where it is.
- Memories: the L1 macros and bank counts from post-synthesis on a PDK.
- TCDM interconnect: area against the number of banks and ports.
- Energy (second): per-access energy per component, multiplied by the counts
  the model already keeps (bank grants, beats, firings), as Accelergy does.

The paper states each fit's error. No place and route. This moves the HW cost
estimator out of "Later" (ARCHITECTURE section 8). **[open]** the PDK and what
can be published under its NDA.

## 8. Related work and positioning

A draft (PAP2, `brief`): written from abstracts, documentation and a first
pass over the papers, before the author's own reading. **[check]** marks a
statement that the full paper has to confirm. Section 8.10 is the reading
table that turns the draft into the final text; until it is filled,
ARCHITECTURE section 2 and D23 stand as they are. The citations are kept
outside the repository until they are rebuilt from DBLP.

SNAX-FORGE asks which engines a cluster shell should hold, and how they should
be composed for a workload, before RTL. No single group of prior work answers
this. Analytical DSE tools are fast but treat shared memory as a bandwidth
budget. Pre-RTL simulators model accelerators next to a CPU and caches rather
than inside a shared-L1 shell. Generators and system flows produce RTL for one
engine or assemble a platform, and give feedback only after synthesis or on
FPGA. Each line below says what a work does, what SNAX-FORGE adds, and which
claim it bears on.

### 8.1 Comparison

✓ yes, ◐ partly, ✗ no. The last two columns are where SNAX-FORGE is still
behind: they depend on E1 and on generation into SNAX.

| Work | Kernels beyond DNN loop nests (C2) | Engines composed at design time (C3) | Shared banked L1, streamers and control at cycle level (C3) | Reports that name the cause (C1) | Seconds per design point (C1) | Hardware into a cluster (C5) | Checked against RTL or silicon (C4) |
|---|---|---|---|---|---|---|---|
| gem5-SALAM | ✓ | ◐ | ◐ [check: no explicit bank arbitration] | ✗ | ◐ | ✗ | ✓ |
| GVSoC (with its Snitch shared-L1 model) | ✓ (as binaries on cores) | ✗ | ✓ for cores, ✗ for accelerators | ✗ | ✓ | ✗ | ✓ |
| Stream | ◐ (DNN, Transformer, SSM) | ◐ (layers onto heterogeneous cores) | ✗ (shared bus) | ◐ | ✓ | ✗ | ✓ |
| Timeloop/Accelergy, AccelForge | ◐ (einsums) | ✗ / ◐ | ✗ | ◐ | ✓ | ✗ | ✓ / ✗ |
| CHARM (Versal), SSR | ✗ (GeMM, Transformer) | ✓ | ✗ | ✗ | ✓ (analytical) | ✓ (FPGA) | ✓ |
| Richie | ✓ | ◐ | ✓ in RTL, after RTL | ◐ [check] | ✗ (FPGA build) | ✓ | ✓ |
| SODA-OPT / SODA Synthesizer | ✓ | ◐ | ✗ | ◐ | ✗ (hours) | ✓ (single engine) | ✓ |
| Allo, HIDA, Stream-HLS | ✓ | ✓ | ✗ | ◐ (HLS reports) | ✗ | ✓ (FPGA) | ✓ |
| **SNAX-FORGE** | ✓ | ✓ | ✓ | ✓ | ✓ | planned (C5) | planned (E1) |

### 8.2 Pre-RTL and platform simulators

- **Aladdin** (Shao et al., ISCA 2014). Pre-RTL power and performance of one
  accelerator, from a C kernel's dependence graph, about 1 minute per point
  against 87 minutes in the RTL flow. Credited for pre-RTL DSE and for the C1
  table layout. (C1)
- **gem5-Aladdin** (Shao et al., MICRO 2016). Aladdin inside gem5, showing that
  an accelerator designed in isolation is over-provisioned and that co-design
  with the SoC interface gives up to 7.4x better EDP. Credited for the general
  form of C3. SNAX-FORGE adds composition of several engines in a shared-L1
  shell. (C3)
- **gem5-SALAM** (Rogers et al., MICRO 2020) and **SALAMv2** (Spencer et al.,
  JSA 2024), now in gem5 mainline. Several LLVM-IR accelerators with
  scratchpads, DMA and stream buffers inside full-system gem5. This is the
  closest simulator. SNAX-FORGE differs in three ways. Its engines are
  hand-written block models bound on a dataflow graph, not compiled from C. It
  models bank arbitration, streamers and a uniform register interface
  explicitly [check SALAM's scratchpad model]. It needs no CPU or OS, and a
  design point takes seconds. (C1, C3)
- **GVSoC** (Bruschi et al., ICCD 2021) and its **Snitch shared-L1 model** (Li
  et al., 2026). An event-driven PULP platform simulator; the 2026 model
  reproduces shared-L1 contention of 1024 cores within 7% of RTL, 115x faster.
  SNAX-FORGE does not claim the simulation technique. It adds accelerator
  streamers rather than cores, composition from a NumPy kernel, and reports
  that name causes. (C1, C3)
- **gem5-accel** (Vieira et al., CAL 2024), **PARADE** (Cong et al., ICCAD
  2015), **MosaicSim** (ISPASS 2020). Other accelerator-in-system simulators.
  PARADE's timing from HLS II and depth is close to the BRM's latency and II.
  (C3)

### 8.3 Analytical design-space exploration

- **ZigZag** (Mei et al., TC 2021). Analytical cost model and mapping search
  for one accelerator, over DNN layers. This is the "best accelerator alone"
  view that C3 argues against. (C3)
- **Stream** (Symons et al., TC 2025), with its Transformer and SSM extensions.
  Layer-fused scheduling on heterogeneous multi-core accelerators, 2–5 s per
  point, 91–99% latency accuracy on three chips. Shared resources are a
  first-come-first-serve bus and a DRAM port. SNAX-FORGE is complementary:
  Stream proposes an allocation, and SNAX-FORGE checks it at cycle level and
  shows where bank conflicts, chaining and control change the ranking (E9).
  (C3)
- **Timeloop/Accelergy** (Parashar et al., ISPASS 2019; Wu et al., ICCAD 2019)
  and **AccelForge** (Emer et al., 2026). Analytical models over einsums, with
  energy and area from per-component tables. Accelergy is the precedent for the
  cost model in section 7. AccelForge claims heterogeneous architectures,
  analytically, without an experimental evaluation yet [check]. (C3, cost
  model)
- **Herald** (Kwon et al., HPCA 2021) and **MAGMA** (Kao and Krishna, HPCA
  2022). Choose and schedule heterogeneous sub-accelerators with shared
  resources as static partitions. Herald is the closest idea-level predecessor
  to "which mix in one cluster". (C3)

### 8.4 Composition

- **CHARM** (Cong et al., ISLPED 2012), with **CAMEL** (ISLPED 2013). Small
  building blocks composed into accelerators at runtime by a hardware composer.
  Credited for composing accelerators from units. SNAX-FORGE composes at design
  time, on the dataflow graph, and evaluates the shell. (C3)
- **CHARM on Versal** (Zhuang et al., FPGA 2023) and **CHARM 2.0** (TRETS
  2024). Several differently sized matrix-multiply accelerators beat one large
  one: 5.4x on BERT, 32.5x on ViT. This is the published "one big accelerator
  is not the best mix" result, for GeMM on FPGA with an analytical partitioner.
  (C3)
- **SSR** (Zhuang et al., FPGA 2024). Spatial, sequential and hybrid
  Transformer accelerators trading latency against throughput. The closest
  precedent for E5 (prefill against decode). SNAX-FORGE asks the same question
  for a shared-L1 cluster, at cycle level, beyond GeMM. (C3)

### 8.5 Shells and system integration

- **PULP HWPE** (Conti et al., CODES+ISSS 2013; Dehyadegari et al., TC 2015).
  Accelerators on a shared TCDM through streamers, with a register-file control
  interface: the same shell SNAX uses. Credited for the shell. SNAX-FORGE adds
  one executable model of it that any engine plugs into. (C5)
- **Gemmini** (Genc et al., DAC 2021). Full-stack evaluation showing that
  SoC-level choices change the best configuration, and that accelerators are
  often evaluated in isolation. (C3)
- **Richie** (Bellocchi et al., TPDS 2025). Generates accelerator-rich
  multi-cluster SoCs from HLS or HDL engines wrapped as HWPEs, and explores
  them by FPGA emulation. It has the same problem statement as SNAX-FORGE, but
  every design point is an FPGA build and the feedback is execution-time
  breakdowns and counters [check: whether the software is generated]. (C1, C5)
- **ESP** (Mantovani et al., ICCAD 2020) and **Chipyard** (Amid et al., IEEE
  Micro 2020). Agile SoC flows with standard accelerator sockets. ESP reports a
  first accelerator on FPGA "in a few hours". These are C1 reference points for
  mature flows. (C1)

### 8.6 Compilers and HLS

- **DaCe/SDFG** (Ben-Nun et al., SC 2019) and **NPBench** (Ziogas et al., ICS
  2021). The front end and the benchmark set SNAX-FORGE builds on. DaCe
  offloads to CPU, GPU and FPGA, but has no PULP or SNAX target. (C2)
- **SNAX-MLIR** (in the SNAX work) and the **xDSL Snitch backend** (Lopoukhine
  et al., CGO 2025). Compile onto a fixed cluster. SNAX-FORGE decides what the
  cluster holds before that. (C2)
- **SODA-OPT and the SODA Synthesizer** (Agostini et al., ICCAD 2022; IEEE
  Micro 2022). MLIR outlining into Bambu HLS, a DSE over compiler passes,
  Python to GDSII in under 3 hours per design. Hardware is generated
  automatically. Feedback comes per design after synthesis, and nothing models
  engines sharing a banked scratchpad. Their PolyBench kernels overlap section
  6.1. (C1, C3)
- **Union** (Jeong et al., PACT 2021 [check venue]). An MLIR front end feeding
  Timeloop and MAESTRO for one accelerator. Same intent, analytical. (C2, C3)
- **Allo** (Chen et al., PLDI 2024), **HIDA** (ASPLOS 2024), **Stream-HLS**
  (FPGA 2025), **TAPA** (TRETS 2023). Compose kernels into HLS dataflow
  designs. Composing kernels is solved on the HLS side. SNAX-FORGE composes
  fixed engines that share a banked L1 and evaluates the cluster before RTL.
  Their engines can be BRM implementations (E8). (C3, C5)
- **Vitis HLS, Bambu, XLS/DSLX.** Engine generators, used as inputs in E7 and
  E8. Lahti et al. (TCAD 2019) is the safe citation for HLS quality against
  hand RTL. (C5)

### 8.7 Deployment on fixed silicon

- **MATCHA** (Russo et al., DAC 2026). Deploys DNNs across several accelerators
  of a PULP SoC with a shared scratchpad. It is a compiler for silicon that
  exists. SNAX-FORGE chooses the silicon. (C3)
- **The Configuration Wall** (Van Delm et al., ASPLOS 2026). Configuration
  overhead caps accelerators well below peak. Motivates the control part of C3;
  SNAX-FORGE's control costs are declared defaults until E1 (D37). (C3, C4)

### 8.8 LLM and explainable DSE

- **Explainable-DSE** (Dave et al., ASPLOS). Bottleneck analysis with
  hand-written fixes, 47x fewer iterations than black-box search. The non-LLM
  precedent for feedback that names causes. (C1)
- **Beacon** (Li et al., 2026), **AgentDSE** (Wang et al., 2026), **LLM-DSE**
  (2025). LLM agents driving simulators or HLS from reports. SNAX-FORGE does
  not claim the agent. It claims that its text artefacts and reports make it
  usable by one (E10). (E10)

### 8.9 Statements to avoid

| Avoid | Write instead |
|---|---|
| ZigZag, Stream and Timeloop are DNN-only | They model tensor or einsum loop nests with bandwidth-level shared resources |
| DaCe and MLIR do not target PULP or SNAX | DaCe has no PULP target; MLIR flows compile onto a fixed cluster, and none choose what it holds |
| SNAX-MLIR handles only GeMM | Its evaluation centres on GeMM |
| SODA builds hardware by hand, single kernel | SODA generates hardware automatically, with feedback per design after synthesis |
| HLS has 10–30% control overhead | Cite Lahti et al., or measure it (E7) |
| XLS is more efficient than HLS | XLS is one more engine generator |
| GVSoC is hard to use | GVSoC runs binaries on modelled cores and has no path from a kernel to a cluster design point |
| SNAX-FORGE is the first to stitch kernels | First to evaluate compositions of fixed engines in a shared-L1 shell at cycle level before RTL (to our knowledge) |

### 8.10 Reading table

One row per work a reviewer will put next to SNAX-FORGE. The first line is the
draft's; the second is the author's, after reading, and wins where they differ.
A row whose second line cannot be written is a threat to deal with in
section 13 (Q11).

| Work | Draft: what it cannot do | After reading |
|---|---|---|
| gem5-SALAM, SALAMv2 | Compose engines from a dataflow graph; model bank arbitration and streamers; run without a CPU model, in seconds | |
| GVSoC shared-L1 model | Model accelerator streamers rather than cores; go from a kernel to a cluster design point; report causes | |
| Stream | See bank conflicts, chaining waits and control; take kernels that are not DNN-shaped | |
| CHARM (Versal), CHARM 2.0 | Leave GeMM; model contention in a shared L1 at cycle level | |
| SSR | The same, for the spatial against sequential trade-off | |
| Richie | Give a result without an FPGA build; name bank conflicts or layout as the cause | |
| SODA-OPT, SODA Synthesizer | Model engines sharing a banked scratchpad; answer in seconds | |
| MATCHA | Choose the silicon; it deploys on a fixed SoC | |
| AccelForge | Model cycle-level contention; it is analytical | |

## 9. Features the evidence needs

| Feature | Needed by | Where it lives today |
|---|---|---|
| Import and run the kernel set | E3, E4 | M7 (FE1, FE3) |
| Mat-vec, spatial-array and SIMD unit BRMs | E1, E4–E6 | new BRM tasks |
| Cost model and characterisation | E4–E7 | "Later" in ARCHITECTURE section 8 |
| Sweep runner and a sweep view (cycles against area) | E4, E5 | DSE5 in M8; a sweep view is new (VIS6 was dropped, D98) |
| Light anchor against SNAX RTL | E1 | open item 42, M2 deferred (D51) |
| Tiling, per-tile DMA, reductions over tiles, 2D DMA | E1 at large sizes, E5, E6 | LOW3b (M9), open items 38, 44 |
| `exp` and casts in the expression grammar | softmax in E5 | open item 37 |
| Generated RTL that plugs into SNAX | C5, and an easier E1 | GEN1 (M10); kept outside the plan by D52, open item 43 |
| Long runs at trace off; extrapolation if needed | E5, E6 | BASE2, BLK1 |
| HLS engine as a BRM implementation | E8 | open item 28 (only `chisel` sources accepted) |

## 10. Proposed order

A proposal, decided in PAP3. Until then the order of D100 and D101 stands,
except that steps 1 and 2 go ahead alongside P1 and M13 waits (D109).

1. Kernel set at sizes that fit in L1: import and run (slice of M7).
2. Unit BRMs: mat-vec, spatial array, SIMD.
3. Cost model and characterisation; sweep runner and sweep view.
4. Light anchor on the default 8x8x8 SNAX GeMM (E1).
5. Generation into SNAX for the anchored design (amends D52).
6. Tiling and per-tile DMA (LOW3b, open items 38, 44).
7. TinyML ResNet slice, then the attention block (E6, E5).
8. HLS and XLS engines (E7, E8); Stream comparison if feasible (E9).
9. Later: M13 (A2A), M6 (contract freeze), the rest of M7–M9, M11, M12.

Mid-November checkpoint after step 4 or 5 (section 1). Amends D100 and D101
(order), D51 (light anchor), D52 (generated accelerators in SNAX), and moves
the HW cost estimator forward.

## 11. Open questions

1. The SNAX RTL flow baseline (BASE1): who logs the steps and times, on which
   accelerator (the GeMM bring-up is the natural one)?
2. The PDK for characterisation, the synthesis tool, and what may be published.
3. Data types: SNAX GeMM is int8 with int32 accumulation; the model's kernels
   are int64 today (D102). Precision affects area and the SIMD requant unit.
4. The exact Transformer (model, head size, sequence lengths, precision) and
   ResNet slice for E5 and E6.
5. Generation into SNAX: which SNAX version and which of its configuration
   files the generator should emit, so SNAX's own build consumes them.
6. The newer DIMC: when it joins (planned with a colleague), and whether as a
   second anchor or as the heterogeneous case.
7. E9: effort of describing the same layers in Stream, and who to involve.
8. Limitations to state: platform values are declared defaults until E1;
   compositions are hand-written recipes (search is M8); accelerators whose
   dataflow changes per kernel (Hypercorex, CGRAs) do not fit a fixed BRM yet;
   the model is Python, so speed must be shown (BASE2).

## 12. Figure budget (DAC version)

1. The flow and the engine–shell split (one diagram).
2. A report excerpt naming a cause (bank conflicts or a chaining wait).
3. E1: model against RTL.
4. E2: turnaround table.
5. E4: cycles against area for the PolyBench compositions.
6. E5 or E6: one case where the cluster-level choice differs.

## 13. Questions that shape the paper

A guide, not a verdict: what has to be settled before the outline (section 14)
can be written, with what has been thought so far. Each question says why it
matters, where the thinking stands, and what settles it: a paper to read, an
experiment, or a choice only the author can make. Answers are written here,
under the question, as they come.

### The claim

**Q1. A tool paper or a finding paper?**

- Why: a tool paper says "here is SNAX-FORGE and what it can do"; a finding
  paper says "which engines a shell holds must be decided at cluster level, and
  can be, before RTL", with the tool as the means. Reviewers read "a
  combination of many things" as integration work, and the venues weigh the two
  differently (DAC and ICCAD take tools, MICRO wants the finding).
- So far: the thesis (section 3) is written as a finding, the contributions
  (section 4) as a tool.
- Settled by: the author, after seeing whether E4–E6 give a result that stands
  without the tool's name on it.

**Q2. Which claim leads, C1 or C3?**

- Why: the first figure and the title follow the lead claim; six pages carry
  one.
- So far: both, leaning to the pair. C1 is the pain every SNAX user knows; C3
  is the result a reader outside SNAX can use.
- Settled by: which evidence is stronger by the checkpoint: a turnaround table
  (E2) or a ranking that flips (E4–E6).

**Q3. Is the engine–shell split a contribution or the background?**

- Why: PULP HWPE and SNAX own the shell in hardware, and CHARM (2012) owns
  composing accelerators from building blocks. Claiming either as new invites a
  rejection.
- So far: claim one executable model of the shell that any engine plugs into,
  and composition at design time on the dataflow graph. Not the shell, and not
  composition as such.
- Settled by: reading the HWPE and CHARM papers (section 8.10), then one
  sentence that survives both.

### The reader

**Q4. What supports "cluster designers in general" beyond SNAX?**

- Why: every experiment runs on SNAX; a reviewer can call it a SNAX tool.
- So far: a PULP cluster with HWPEs has the same shell (streamers, a register
  file, a shared TCDM), and Richie builds on it. One paragraph arguing that the
  platform file would describe it.
- Settled by: whether a second platform file (an HWPE cluster's parameters) is
  worth writing as evidence, or the paragraph is enough.

**Q5. How much of SNAX does the paper explain?**

- Why: space. Streamers, the TCDM and the CSR interface need a figure before
  any result makes sense, and the SNAX work must be cited in the third person.
- So far: one figure that shows the flow and the shell together (section 12,
  figure 1).
- Settled by: the outline; a first draft of that figure.

### The evidence

**Q6. What is the least fidelity evidence that makes a cycle count
believable?**

- Why: every comparable tool reports model against RTL. Platform values are
  declared defaults today (D51), and "the trends are right even if the numbers
  are not" is not accepted unless shown.
- So far: ranking agreement on 3–5 sizes of the default 8x8x8 GeMM (Kendall
  tau, the same best point), absolute error stated beside it, and one DMA-heavy
  case because E5's decode phase rests on the DMA model (open item 7).
- Settled by: E1, and which mechanisms the C3 cases depend on. Each of those
  mechanisms needs its own anchor point.

**Q7. Which C3 case carries the paper, and what if no ranking flips?**

- Why: C3 is only persuasive with a case where the per-engine choice loses in
  the cluster. `dot` does not show one: the adder tree wins at every W.
- So far: three candidates, by cost: PolyBench chains at sizes that fit in L1
  (E4), a ResNet slice (E6), prefill against decode (E5). Prefill against
  decode is the best story and the most expensive.
- Fallback if nothing flips: the weaker claim that the cluster changes how much
  an engine is worth (the gain shrinks), with the cause named. That is still a
  result, and gem5-Aladdin's own was of this kind.
- Settled by: running E4 early; it is cheap and tells whether flips exist at
  all.

**Q8. What exactly does C1 measure?**

- Why: the SNAX work reports a Verilator build of about three minutes, so
  "weeks against seconds" will be challenged.
- So far: two times, not one: the time to a first number (integration, streamer
  configuration, program; about a week for the first GeMM) and the time to the
  cause (reading waveforms to find bank conflicts). Simulation speed is
  reported separately (BASE2).
- Settled by: BASE1, with every step of the SNAX flow logged, on the GeMM
  bring-up.

**Q9. Area only, or energy too?**

- Why: without a cost, more lanes always wins. Each cost axis is one more thing
  to validate.
- So far: area first, from pre-characterised tables (section 7); energy if
  characterisation is cheap.
- Settled by: the PDK and what may be published (section 11, item 2).

**Q10. Does generation into SNAX belong in the first paper?**

- Why: it closes the loop and makes E1 a comparison of the same design in model
  and RTL, but it is the largest feature still to build, and D52 keeps it
  outside the plan today.
- So far: wanted as a feature; in the proposed order it comes right after the
  anchor (section 10, step 5).
- Settled by: PAP3, and whether the first paper is the focused one or the full
  one (Q15).

### Against related work

**Q11. After the reading, does each separating sentence still hold?**

- Why: four works can say "already done" for part of a claim: gem5-Aladdin (C3
  in general), CHARM on Versal and SSR (composition), GVSoC's shared-L1 model
  (the TCDM model), AccelForge (heterogeneous, and the name).
- So far: one sentence per work in section 8; the table in section 8.10 is
  where they are tested.
- Settled by: the reading. A sentence that does not survive changes the claim,
  not only the wording.

**Q12. Is a head-to-head run needed, or is a feature table enough?**

- Why: "nobody does the same thing" does not excuse a paper from comparison.
  The closest overlaps are Stream (same lab, DNN-shaped parts only) and
  gem5-SALAM (heavy to set up).
- So far: Stream as E9, exploratory: the same layers and engines described in
  both, and where the rankings disagree. gem5-SALAM only for speed and set-up
  effort, if at all.
- Settled by: the venue (a feature table carries a 6-page paper; a full paper
  is expected to run something) and the effort E9 turns out to need.

**Q13. Are HLS and XLS an input or a competitor?**

- Why: comparing "our RTL" with HLS RTL measures the unit library and its
  author, not the method, and the engines are hand-written Chisel.
- So far: an input. E7 compares one unit across Chisel, HLS and XLS after
  synthesis; E8 plugs the HLS engine in as a BRM implementation and shows the
  shell matters as much as the engine.
- Settled by: whether E7 and E8 fit the first paper (Q15).

**Q14. Does the name clash with AccelForge matter?**

- Why: AccelForge (September 2026) is an analytical co-design framework for
  accelerators, from a well-known group, and appeared weeks before the
  checkpoint.
- So far: noted, not decided.
- Settled by: the author; at the least, one sentence of contrast in the related
  work.

### Scope and venue

**Q15. One paper or two, and what is cut from six pages?**

- Why: a focused paper followed by a full one needs substantially new results
  in the second.
- So far: the focused version holds E1, E2, E4 and one of E5 or E6; the full
  version adds scale (E5, E6), E7–E9 and generation.
- Settled by: the mid-November checkpoint (section 1).

**Q16. Is the LLM paragraph in or out?**

- Why: it is not a claim, and agent-driven DSE is crowded in 2026 (Beacon,
  AgentDSE); a weak paragraph draws questions it cannot answer.
- So far: one short demonstration (E10): an agent drives the tool as a black
  box because every input and output is text.
- Settled by: space, and whether the demonstration is clean.

**Q17. Which limitations are stated up front?**

- Why: stated limitations are forgiven; discovered ones are not.
- So far: declared default platform values until E1; compositions are
  hand-written recipes; engines whose dataflow changes per kernel (Hypercorex,
  CGRAs) do not fit a fixed BRM yet; the model is Python.
- Settled by: the outline; which of these the evidence has removed by then.

### Story order

**Q18. Does the paper open with the experience or with the question?**

- Why: the week of integration and the waveforms are concrete and true, but
  they are one lab's experience, cited in the third person; the question
  ("which engines should a shell hold?") is general.
- So far: the question first, the experience as its evidence, in two or three
  sentences.
- Settled by: the author's own outline.

**Q19. What is figure 1?**

- Why: it is what a reviewer remembers. Candidates: the flow with the shell;
  one composition spectrum (units, a unit set, a full accelerator) on the same
  graph; a report excerpt that names a cause.
- So far: the flow with the engine–shell split (section 12).
- Settled by: Q1 and Q2: a finding paper leads with the composition spectrum, a
  tool paper with the flow.

## 14. Paper outline

After the questions of section 13: the sections of the paper, what each holds
and which figure or table carries it, starting from the author's own outline.
