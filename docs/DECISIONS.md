# SNAX-FORGE Decisions

Why SNAX-FORGE is built the way it is: one entry per live decision, in its current form. What the
system is: `docs/ARCHITECTURE.md`; the artefacts' fields: `docs/CONTRACTS.md` (C§n); open work:
`docs/STATUS.md`.

Rules (D117):

- One fact in one place. This file holds the why; an entry that only repeats ARCHITECTURE,
  CONTRACTS or a module's docstring does not belong here.
- A decision is logged only for a choice with a real alternative that someone may want to
  reverse: direction, ownership, the interfaces between components, the milestone order.
  Implementation details live in the module's docstring and its tests.
- An entry says what was decided and why in at most three lines, then what it amends and where the
  detail lives. An amendment is folded into the entry it changes, not chained after it.
- A replaced or merged entry is removed, and its number goes under Retired with the entry that
  holds it now. A number is never changed or reused, since code, tests and docs cite them; the
  text of every retired entry is in git history (tag `pre-gvsoc`; RET1's at `cad6619`).
- Closing a milestone compacts this file and removes the done tasks from STATUS.
- A new decision gets the next number at the end of this file, with its area tag. Next free
  number: **D141**.

Area tags: `scope` (plan, ownership, order), `lower` (SNAX-LOWER), `gvsoc` (SNAX-GVSoC), `brm`
(SNAX-BRM), `dfg` (SNAX-DFG and importers), `ref` (reference executor), `sandbox`, `design`
(platform, checks, memory plan, design point), `flow`, `viz`, `docs`.

| Area | Decisions |
|---|---|
| scope | D8 D23 D51 D52 D91 D98 D100 D111–D113 D116 D118 D123 D126 D127 D130 D138 D139 D140 |
| lower | D64 D75 D89 D105 D135 |
| gvsoc | D120 D121 D122 D125 D128 D134 D137 |
| brm | D68 D69 D70 D82 D103 D129 D133 |
| dfg | D19 D71 D77 D78 D102 |
| ref | D28 D79 |
| sandbox | D72 D73 D104 |
| design | D12 D13 D84–D87 D131 |
| flow | D50 D90 D108 D136 |
| viz | D55 D57 D81 D96 D97 D99 |
| docs | D26 D46 D117 |

Retired, and where each now lives:

| Now in | Retired |
|---|---|
| D19 | D43 |
| D51 | D22 |
| D55 | D54 |
| D57 | D56 D58 D60 D61 |
| D64 | D63 D66 |
| D68 | D3 D5 D25 |
| D70 | D15 |
| D71 | D1 |
| D72 | D14 D27 D80 |
| D77 | D2 |
| D79 | D17 D20 |
| D81 | D16 D76 |
| D84 | D7 |
| D87 | D74 |
| D90 | D94 D107 |
| D99 | D106 |
| D100 | D109 |
| D105 | D101 |
| D111 | D18 |
| D113 | D9 |
| D116 | D4 D6 D10 D11 D21 D24 D29 D30 D31 D32 D33 D34 D36 D37 D40 D41 D42 D44 D47 D48 D49 D53 D62 D65 D67 D83 D88 D92 D95 D110 D114 D115 D119 |
| D117 | D93 |
| D123 | D124 |
| D129 | D35 |
| D134 | D132 |
| D135 | D45 |
| D137 | D38 D39 D59 |

---

**D8** · scope — The AI thinker is a commercial LLM, so every artefact is text that humans and LLMs
read alike.

**D12** · design — One streamer per accelerator port, named after it (D75), its width (`n_ports`,
the hjson's `num_channel`) the port's lanes. → `design/streamers.py`, C§13.

**D13** · design — The unit of transfer is a bank word, one element per word for now; element type
and elements per word are in the platform so packing can come later (open item 21, tier P). → C§1,
C§13.

**D19** · dfg — Everything is extended by registration (node kinds, transforms, checks, passes, the
accelerator kinds and ops of `brm/function.py`, the generator's blocks) and namespaced attrs, never
by editing core classes. Merges D43. → principle 6.

**D23** · scope — SNAX-FORGE complements ZigZag and Stream: cycle-level and RTL-backed rather than
analytical. Parked: Stream as a proposer for DNN-shaped workloads, and an analytical estimator of
our own for the kernels Stream cannot describe (open item 4).

**D26** · docs — Contracts are Python dataclasses with `to_dict` / `from_dict` and plain JSON until
the contract freeze (M6), versioned schemas after. → C§1.

**D28** · ref — Reductions are exact: integers first, and a float reduction follows the
accumulation order its BRM defines, so the check against the reference executor stays exact and
the check against NumPy is a tolerance (D113).

**D46** · docs — The contracts are written in `docs/CONTRACTS.md`, and every snippet in it is
checked against its source (`tests/test_contracts.py`).

**D50** · flow — A run's `run.json` records what ran (the design point, the hjson files, the ELF,
the snax-gvsoc commit), so a run directory says on its own which hardware produced it. →
`flow/run.py`, C§7.

**D51** · scope — The user owns the accelerator, through its BRM (lanes, rates, latency, II, op);
everything around it is the model of the SNAX platform. Model cycles are trusted only as far as
the comparison with SNAX RTL shows (M16, D116).

**D52** · scope — Cosim (M10) swaps one accelerator for its RTL and checks its output, latency and
II; the whole cluster is checked by running the same program on SNAX-GVSoC and on snax_cluster
RTL (D116). Neither blocks the inner loop.

**D55** · viz — The viewers are a local server bound to 127.0.0.1 plus a static page: stdlib only,
no build step, offline, for humans; LLMs read the reports (D99). Views read the run's and the
design's dataclasses, not raw JSON. → `viz/`.

**D57** · viz — The run views are a schedule with the cluster view under it at the selected cycle,
kept in the URL hash. Python answers every fold, pairing, lookup and step and `static/` only draws
the answers, so a view never disagrees with the reports; one colour means one thing in every view.
The schedule covers the program's window with core lanes; quiet stretches of at least FOLD_MIN
cycles fold, open on a click and stay open in the hash. The cluster view shows the data side at the
cycle, stepped by event or task boundary. → `viz/schedule.py`, `viz/cluster.py`, `viz/static/`.

**D64** · lower — The task list has `configure`, `start`, `sync` and `read` steps, placed where the
list chooses. Lowering adds only the waits correctness needs, one per component, and leaves out a
wait another one already covers. → C§9, `lower/waits.py`.

**D68** · brm — A BRM is a hand-written JSON file: a shared part (an interface with a rate per port,
so reductions and elementwise blocks share one; function, dataflow, pattern) and implementations
(source, supports, timing, binding D129), its timing the user's numbers (D51). `resolve` checks an
instance against the generator's `BLOCKS`. Merges D25. → C§10, `brm/`.

**D69** · brm — A reader whose innermost temporal stride is 0 reads a beat once and hands it out
`bound[0]` times, as SNAX's reader does, so a BRM nest may repeat a beat with a stride of 0. →
`brm/affine.py`, C§10, snax-gvsoc `snax/snax_streamer.cpp`.

**D70** · brm — A port's dataflow is a nest in a registered notation. The first, `affine`, gives
the operand's shape, an offset and loops (bound, strides, spatial flag), spatial loops last; it
checks, by address, the order in which the memlets deliver the operand. → C§10.

**D71** · dfg — SNAX-DFG is its own JSON format, `.snaxdfg`: SDFG's concepts without SDFG's JSON.
Importers derive it and decide nothing; streamers are not nodes, a memlet on an accelerated node's
connector becomes one at lowering.

**D72** · sandbox — Decisions are made by hand in SNAX-SANDBOX: registered transforms on a
`.snaxdfg`, applied in order from a recipe (kernel, params, symbols, steps), each step checked by
the reference executor. A sweep is one recipe with a parameter; automated search writes recipes.

**D73** · sandbox — Loops are tagged `tile`, `temporal` or `spatial`, and `bind` reads a BRM's
design params off the graph (W from the spatial bound), so each is decided once. Streamer values
come from memlets through the containers' layouts.

**D75** · lower — Names are derived, never stated: streamer `<instance>_<port>`, task
`<node>_<component>`, DMA task `load_<container>` or `store_<container>`.

**D77** · dfg — A `.snaxdfg` holds symbols, containers and an ordered body of registered node kinds
(`map`, `tasklet`, `accelerated`), memlets on connectors, and one expression grammar (`expr.py`)
shared with the BRM. → C§11.

**D78** · dfg — The SDFG importer maps a simplified SDFG onto `.snaxdfg` with readable names and
raises a named error for anything it does not support. → `dfg/import_sdfg.py`.

**D79** · ref — The reference executor runs any `.snaxdfg` in NumPy, accelerated nodes through
their BRM's function; every sandbox step and every run is compared with it, so DaCe's CPU
reference is not needed. → `dfg/execute.py`.

**D81** · viz — The DFG viewer is a second mode of the same server: graphs side by side, top to
bottom in execution order, rows computed in Python, edges in SVG, no library. → `viz/dfg/`.

**D82** · brm — A BRM says what one lane computes (`function.code`), and `bind` checks the tasklet
against it. A bound graph records what it replaced, so `unbind` and `join_map` take it back.

**D84** · design — The platform is a file of its own in `platforms/`, paired with the recipe's bound
graph by the design step, so either changes without the other; cluster parameters are design knobs.
It holds the streamer knobs, never a template (D134); a streamer input of its own is open item 78. →
C§13.

**D85** · design — Design checks report every problem before anything is written, each with its
fix. → C§14.

**D86** · design — The memory plan comes from registered residency, layout and placement passes,
contiguous by default, with pins, seeing each port's element-index stream to avoid bank conflicts
without a run; it fixes residency and shape in both memories and the L1 base, the L2 base being the
program's link (its ELF's symbols, in `run.json`). → C§15, C§7.

**D87** · design — `design_point.json` holds the bound graph, the platform, the resolved streamers
and the memory plan; it is checked again on load and is SNAX-LOWER's and the generator's only
input, the streamer file lowered from its streamers (D134). → C§15.

**D89** · lower — The task list is derived from a design point: a group per accelerated node,
loads and stores from the memory plan, streamer values from memlets checked against the BRM's
nest, and `after` holding data dependences only. → `lower/derive.py`.

**D90** · flow — `pixi run flow` runs the inner loop once, from a recipe to a checked run, and
decides nothing. Everything goes to a folder named after every `--set`: the run (on SNAX-GVSoC,
D136), the reports and `flow.log` with each stage's time. → `flow/run.py`.

**D91** · scope — The old SDFG → descriptor → RTL path is gone; the Chisel blocks and `Emit` are a
project of their own in `hw/chisel/`.

**D96** · viz — The memory tab shows L1 as banks × rows through the address map and L2 as rows of
one DMA beat, folded so the answer grows with the regions, not with the depth. → `viz/memory.py`
on `viz/index.py`.

**D97** · viz — Data movement is read off the beat trace and the regions, with no timing model: an
element's journey, residency, per-port address nests and the L1 conflicts on the layout. →
`viz/movement.py` on `viz/index.py`.

**D98** · scope — No run diff view: runs are compared by switching in the viewer and reading their
reports, until sweeps need one. Walk-throughs live in `examples/`, with a test checking every
number they state.

**D99** · viz — Design and run reports in the layout of HLS reports, Markdown from dataclasses, for
humans and LLMs alike. Every number comes from the design point, profile or trace, nothing grows
with cycles, and a wait is named by what it was for. → `report/`.

**D100** · scope — SNAX-FORGE is the subject of a paper, with no fixed deadline: a plan (P1) runs
alongside development, a draft (P2) follows the evidence. The evaluation climbs NPBench from
kernels to layers to blocks and models (ResNet, Transformer, NVSA). → `docs/PAPER.md`.

**D102** · dfg — A reduction is a map whose tasklet writes through a memlet with a `reduce` (a
registered associative op and its identity), so split and bind treat it like any map. The importer
turns DaCe's `Reduce` into one; the field is named for what it means. → C§11, `dfg/reduce.py`.

**D103** · brm — Timing has a third number, `drain`: the cycles from a push in which no input is
taken (1 for the Chisel Accumulator). Port names are not Python keywords and do not start with
`_`, since they appear in `code` and in streamer names. Amended by D133. → C§10.

**D104** · sandbox — `bind` binds a reduction through a `reduce` matcher; a folding output without
the spatial variable folds every lane into one. Named rates come from one function, `named_rates`,
shared by bind, the reference executor and SNAX-LOWER. → `sandbox/patterns.py`.

**D105** · lower — Accelerators chain through L1: the reader waits on the writer that put its
container there, the only wait between them. A container the DMA moves is padded to whole beats,
and the padding is kept free in both memories. → `lower/derive.py`, `lower/layout.py`.

**D108** · flow — The flow also starts from a bound `.snaxdfg`, the suffix telling the two forms
apart. That is how a hand-edited graph is run and checked, though it cannot be replayed or swept
(open item 34).

**D111** · scope — SNAX-FORGE generates a bundle per design point for snax_cluster (the accelerator
RTL with a shell wrapper carrying its own CSR manager (D130) in `rtl/` with a file list the cluster
hjson names (`snax_forge_rtl`), the hjson of D134, the program as direct CSR writes) and for
SNAX-GVSoC the same hjson and headers (D128); SNAX's own generators build the rest, so the hjson
describes the cluster once. A file list, not a `Bender.yml`: snax_cluster depends on no bundle
(D139). → ARCHITECTURE §5.8, C§18.

**D112** · scope — A composition is one accelerator: one recipe step binds a connected group of
tasklets to a chain of unit BRMs linked by FIFOs, with streamers on the outer ports only, so one
register window and one SNAX shell wrapper (M13). Links between shells are parked (open item 53).

**D113** · scope — The kernels keep NPBench's DaCe structure with the dtype as one parameter, at
PolyBench's MINI and SMALL sizes. Integers first, for an exact reference check: `int64` through tier
C, `int8` in and `int32` accumulate from tier P (as OpenGeMM); float later, as a new BRM
implementation. Reverses NPBench's own dtypes. → PAPER §6.1, STATUS (M7).

**D116** · scope — SNAX-GVSoC replaced SNAX-MODEL (GVS4, closing X1): a GVSoC model of the SNAX
cluster in which a Snitch core runs the real program, so control cost is measured, not declared,
checked against snax_cluster's RTL (M16, D123); RET1 removed SNAX-MODEL and its entries. Amends D51,
D52, D68, principles 1 and 3, the RISC-V non-goal. → STATUS (M16, M14).

**D117** · docs — The documents stay short: one fact in one place, a decision only for a choice
someone may want to reverse, each entry in its current form in at most three lines, and the done
tasks out of STATUS at every milestone close. The rules are at the top of this file. Replaces D93.

**D118** · scope — One development image, `ghcr.io/kuleuven-micas/snax-forge:main` from
`container/`, for snax-forge, snax-gvsoc and the programs built from snax_cluster: the image holds
the tools, each repo keeps its own environment. → `container/Dockerfile`, README (In the container).

**D120** · gvsoc — The Snitch core sends CSRs 0x3C0 to 0x5FF as 32-bit IO requests on its `snax_csr`
port to an accelerator component per core, so a SNAX block is a plain IO slave; both sides follow
Zicsr since SG6. → ARCHITECTURE §5.6, snax-gvsoc `SG1.md`, `SG6.md`.

**D121** · gvsoc — SNAX-GVSoC keeps GVSoC's iDMA model and fits it to snax_cluster's iDMA RTL by
measurement, not a new model: the target sets the DMA and DRAM values, the model keeps the DMA's own
read path. → snax-gvsoc `SG2.md`, `SG7.md`, `snax/snax_hjson.py`.

**D122** · gvsoc — SNAX-GVSoC's crossbar takes the SNAX ports, cores and AXI, numbered as on the
RTL, and copies its arbiter (plain and sparse), checked against the RTL module in Verilator; the DMA
takes whole superbanks and refuses narrow winners. Closes open item 58. → snax-gvsoc `SG3.md`,
`SG7.md`.

**D123** · scope — SNAX-GVSoC is checked against snax_cluster's RTL by one ELF on both, comparing
the counters free of the core's timing, by hand on the forge fixtures against GEN2b's tiers, the
cluster tests' references on the forge cluster (SG5 is history). Merges D124. → `GEN2b.md`.

**D125** · gvsoc — The Snitch core's loads follow the hjson and the RTL: one integer load in flight,
the RTL's load-to-load cycles, and the core going on after a load answered later until the next load
or a use of its register; fetch keeps GVSoC's cache (open item 65). → snax-gvsoc
`snax/snax_hjson.py`, `SG5.md`, `SG7.md`.

**D126** · scope — snax-gvsoc builds its programs from a pinned copy of snax_cluster's software in
`sw/`, its synced config the forge vecadd fixture's, refreshed by `make sync-snax-cluster`, whose
diff is the check; only the RTL needs a snax_cluster checkout. → snax-gvsoc `sw/README.md`, CI1.

**D127** · scope — M14 goes to SNAX-GVSoC first, then to snax_cluster, then clean-up, so the model
side is tried first: LNK1, GEN3, LNK2a, GEN2a, FLOW3, and RET1 as its own task after FLOW3 round 1;
GEN1, LNK2b, GEN2b; a housekeeping pass and one generator (open item 68). → STATUS (M14).

**D128** · gvsoc — Generated accelerators run on two hand-written generic components in snax-gvsoc,
`SnaxUnit` and `SnaxShell`, every number a property, so snax-forge generates only hjson and headers
(D134); each design point snax-gvsoc regresses is a fixture in its `forge/`, so its CI never needs
snax-forge. → snax-gvsoc `GEN3.md`, `forge/README.md`.

**D129** · brm — An implementation's `binding` names its Chisel block and how the BRM fills it
(`block`, `args`, `control`, `csrs`, `wait`, and optionally `ports`: BRM port → block port, by default
the same name), each kind registered (D19), so one description drives both generators: GEN3 models
it, GEN1 elaborates it. Closes open item 64. → C§10, `brm/binding.py`.

**D130** · scope — Generated accelerators meet snax_cluster at the CSR request/response port with its
manager off (`snax_disable_csr_manager`, as Hypercorex and DIMC) and carry their own manager on
ReqRspManager's protocol as it is (the fields, then start, held while busy; busy and a counter
read-only), a `ctrl` with valid and ready to the block inside, and GeMMX's data ports, so one
contract serves every block and `CsrLayout`, the headers and `SnaxShell` stay; snax_alu keeps its
own rules as the baseline. Narrows open item 63. → C§17, snax-gvsoc `GEN3.md`.

**D131** · design — The base cluster is snax-forge's own file, `platforms/snax_forge_cluster.hjson`
(from snax_cluster's), which `design from-hjson` reads: one accelerator, its movers agreeing (D84),
the rest small16's; `accelerator_slot` names its slot (D134). → `design/hjson_platform.py`, C§13.

**D133** · brm — The Chisel Accumulator takes `ElementwiseTiledSpatial`'s control plus `len`, so
both run under one shell contract (D130); one register loads the next input as the result leaves
(drain 0), and a combinational W → 1 adder tree makes `chisel_accumulator` and `chisel_adder_tree`
one block. Amends D103. → `samples/Accumulator.scala`.

**D134** · gvsoc — The hjson is the contract between snax-forge and SNAX-GVSoC, so a sweep compiles
nothing: per design point three files, each changing for its own reason (cluster, streamer,
accelerator), which `snax_hjson` reads at run time. Retires D132. → `gen/gvsoc.py`, C§16, snax-gvsoc
`LNK2a.md`.

**D135** · lower — The task list becomes C on snRuntime: each step on the core that owns its
component, D64's waits as polls with a hardware barrier where another core goes on, the DMA waited
for on dependences only, and two `mcycle` reads (main, `forge_check()`) bounding the self-checking
program's window. One accelerator per program. Merges D45. → `lower/c.py`, `gen/data.py`, C§16.

**D136** · flow — The flow runs SNAX-GVSoC only, with no backend flag: lower, gen, build when
needed, sw, gvsoc, check; the check is the program's own exit code and `total_cycles` its window
(D135), and it calls `gvsoc/Makefile` from pixi. Amends D90; closes open item 56. → `flow/run.py`,
C§7.

**D137** · gvsoc — Events in two stages, so the model owns what it counts and snax-forge only names
it: SNAX-GVSoC's traces go to one `events.log`, a level is the trace names on (`beat`: `beats`,
`stall`: `stalls`, `finegrained`: `insn`), lines linked by ids, filtered by source at the crossbar;
the converter renames, never recounts, cuts the window; `--vcd` is a gtkwave dump only. Merges
D38, D39, D59. → `run/`, C§7, snax-gvsoc `docs/EVENTS.md`.

**D138** · scope — `rtl.json` per design point and the checked-in block catalogue
`hw/chisel/blocks.json` are the boundary between `snax_forge/gen` and `hw/chisel/`: Python resolves
the binding and checks it against the catalogue, Scala only constructs (`Emit --spec`), and the
chisel job checks `Emit --describe` equal to the catalogue, so the Python side never needs a JVM.
→ C§17, `gen/rtl.py`, `hw/chisel/README.md`.

**D139** · scope — snax-forge and snax_cluster are separate checkouts joined only by the bundle's
files, so snax_cluster never depends on snax-forge and a bundle is made by the snax-forge checked
out; snax-gvsoc stays a submodule, pinned for snax-forge's flow and tests. → C§18.

**D140** · scope — The ELF compared on SNAX-GVSoC and snax_cluster's RTL is snax-forge's, built
from the bundle, and snax-forge never runs RTL; snax_cluster builds its own ELF from a fixture's
`main.c` (`sw/apps/snax-forge`) only for its CI's pass/fail. Narrows D123, follows D139. → C§18,
RUN.md §8.
