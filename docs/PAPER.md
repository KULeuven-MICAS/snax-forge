# SNAX-FORGE Paper Plan

What the paper claims, how each claim is measured, and which milestones deliver
the evidence. Written in P1 (D100); the order of the milestones it implies is
logged as a decision (PAP3). Every number the paper states will come from a
checked-in command and be checked by a test, as `examples/loop1` is (PAP4).

Written by PAP1 (claims, evidence, evaluation plan). Section 8 is filled by
PAP2, section 10 is decided by PAP3, and section 13 follows PAP2. Items marked
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

Filled by PAP2: one line per work on what it does and what SNAX-FORGE adds,
each pointing to the claim it bears on, grouped as pre-RTL simulators,
PULP-world simulators, analytical DSE, composition, shells and integration,
compilers and HLS, deployment on fixed silicon, and LLM and explainable DSE.

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

A proposal, decided in PAP3; until then the order of D100 and D101 stands.

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

## 13. Paper outline

After PAP2: the sections of the paper, what each holds and which figure or
table carries it, starting from the author's own outline.
