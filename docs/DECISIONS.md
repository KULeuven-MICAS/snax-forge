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
  text of every retired entry is in git history (tag `pre-gvsoc`).
- Closing a milestone compacts this file and removes the done tasks from STATUS.
- A new decision gets the next number at the end of this file, with its area tag. Next free
  number: **D127**.

The `model` and `scenario` entries, and D45 and D53, which lower into SNAX-MODEL's inputs, are kept
as first written until SNAX-MODEL is removed (RET1, D116).

Area tags: `scope` (plan, ownership, order), `model` (SNAX-MODEL), `scenario` (scenario files),
`lower` (SNAX-LOWER), `gvsoc` (SNAX-GVSoC), `brm` (SNAX-BRM), `dfg` (SNAX-DFG and importers), `ref` (reference
executor), `sandbox`, `design` (platform, checks, memory plan, design point), `flow`, `viz`,
`docs`.

| Area | Decisions |
|---|---|
| scope | D8 D23 D51 D52 D91 D98 D100 D111–D113 D116 D118 D119 D123 D124 D126 |
| model | D6 D10–D13 D21 D25 D29–D40 D43 D44 D47–D50 D59 D62 D69 |
| scenario | D41 D42 D65 D67 D83 D95 |
| lower | D45 D53 D64 D75 D88 D89 D105 |
| gvsoc | D120 D121 D122 D125 |
| brm | D68 D70 D82 D103 |
| dfg | D19 D71 D77 D78 D102 |
| ref | D28 D79 |
| sandbox | D72 D73 D104 |
| design | D84–D87 |
| flow | D90 D108 |
| viz | D55 D57 D81 D96 D97 D99 |
| docs | D26 D46 D117 |

Retired, and where each now lives:

| Now in | Retired |
|---|---|
| D51 | D22 |
| D55 | D54 |
| D57 | D56 D58 D60 D61 |
| D64 | D63 D66 |
| D65 | D92 |
| D68 | D3 D5 |
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
| D116 | D4 D24 D110 D114 D115 |
| D117 | D93 |

---

**D6** · model — SNAX-MODEL is pure Python: banks, TCDM interconnect, streamers, DMA and L2, a
register interface and a controller. It has no CPU.

**D8** · scope — The AI thinker is a commercial LLM, so every artefact is text that humans and LLMs
read alike.

**D10** · model — Cycle-level and event-driven: only components with pending work are ticked, and
idle cycle ranges are skipped. → `sched.py`.

**D11** · model — The control program is JSON; a wait polls a block's `busy` register or blocks on
its completion signal. Commands: D36, D42.

**D12** · model — One streamer per accelerator port; how many interconnect ports a streamer has is
configurable.

**D13** · model — The unit of transfer is a bank word, one element per word for now; element type
and elements per word are in the data model so packing can come later (open item 21).

**D19** · dfg — Everything is extended by registration (node kinds, transforms, checks, passes,
accelerator kinds) and namespaced attrs, never by editing core classes. → principle 6.

**D21** · model — The exact sequential simulator is the source of truth; its state is kept
array-friendly so it can be accelerated later without changing results. Amended by D48.

**D23** · scope — SNAX-FORGE complements ZigZag and Stream: cycle-level and RTL-backed rather than
analytical. Parked: Stream as a proposer for DNN-shaped workloads, and an analytical estimator of
our own for the kernels Stream cannot describe (open item 4).

**D25** · model — The accelerator interface has a per-port element rate, so reductions and
elementwise blocks share one interface. → C§4.

**D26** · docs — Contracts are Python dataclasses with `to_dict` / `from_dict` and plain JSON until
the contract freeze (M6), versioned schemas after. → C§1.

**D28** · ref — Reductions are exact: integers first, and a float reduction follows the
accumulation order its BRM defines, so the check against the reference executor stays exact and
the check against NumPy is a tolerance (D113).

**D29** · model — RTL-like cycle semantics: fixed phases per cycle, reads see the previous cycle's
committed state, and everything commits at the end of the cycle. → `sched.py`, rules in C§8
(D47).

**D30** · model — L1 banks are shared state touched by whoever requests them, not a ticked
component. Two accesses to one bank in a cycle is an error, because arbitration is the
interconnect's job. → `mem.py`.

**D31** · model — The interconnect copies the SNAX SparseInterconnect arbiter: per bank a priority
mask, then round-robin from the last selection, and a lock on a refused request; no added latency.
Extended by D33. → `xbar.py`.

**D32** · model — The streamer copies SNAX's reader/writer timing: each port advances on its own
with an address queue and credit, and the FIFO depth bounds reads in flight. Dynamic TCDM priority
is not copied (open item 5). → `streamer.py`.

**D33** · model — Interconnect ports have a width: a w-bit port covers an aligned group of banks
(512 bits = 8 banks), the wider port wins per cycle and group, and stalls caused by a wider grant
are counted apart (open item 6). → `xbar.py`.

**D34** · model — The DMA sits on one wide port between a flat L2 and L1 and moves affine beat
patterns, one beat per `beat_interval`, with declared startup and latencies. iDMA features not
copied: open item 7. → `dma.py`, `l2.py`.

**D35** · model — The accelerator sits between FIFOs: it joins its inputs, runs an L-stage pipeline
that stalls as a whole on a full output, and fires at most once per II (open item 8). Cycle
classes amended by D59. → `accel.py`.

**D36** · model — Every block (streamer, accelerator, DMA) has one register window: `start`,
`busy`, `busy_cycles`, then buffered configuration registers listed by a per-kind adapter; writing
1 to `start` launches it. Mapping onto SNAX's real registers: open item 9. → C§3, C§5.

**D37** · model — The controller runs the program in order, one command at a time, with a cost per
command kind, and keeps control overhead apart from waiting. Costs are declared defaults (open
item 10). → `ctrl.py`.

**D38** · model — The profile is built only from counters the components already keep; the trace
is an independent log that tests check against them. → C§7.

**D39** · model — Trace levels `off`, `task` and `beat`; an event is a flat record with its cycle,
kind and source, and kinds are registered. Amended by D49, D62. → C§7.

**D40** · model — Cycle-class intervals and FIFO occupancy are recorded so they come out identical
with skipping on and off. → C§7.

**D41** · scenario — A run's inputs are two JSON files: the cluster file (the hardware) and the
scenario (memory, program, cycle limit), so one cluster serves many programs. Expected results
live in the tests (open item 15). → C§2, C§6.

**D42** · scenario — A scenario's program is a plain command list; nothing expands inside the
model. Expanding blocks into register writes is SNAX-LOWER's job.

**D43** · model — Components, accelerator kinds and operations are registered, so new ones need no
change to the core.

**D44** · model — `pixi run model-run` writes the run record, profile, trace and memory dumps,
byte-identical on every run and with skipping on or off. Amended by D50, D95. → C§6, C§7.

**D45** · lower — SNAX-LOWER works in two steps: design point → task list → command list, the
second through the model's own register adapters. Refines D18.

**D46** · docs — The model-side contracts are written in `docs/CONTRACTS.md`, and every snippet in
it is checked against its source (`test_contracts.py`).

**D47** · model — Rules for skipping idle cycles, so a sleeping component cannot report wrong
statistics. → C§8, `test_gaps.py`.

**D48** · model — Everything touched every cycle is plain Python, not NumPy; arrays only hold data
and address streams. Faster at cluster sizes, with identical results. Amends D21.

**D49** · model — The beat-level trace can be filtered by source and cycle window; task events and
the profile never are. Closes open item 12. → C§7.

**D50** · model — `run.json` records the cluster configuration, so an output directory says on its
own which hardware produced it.

**D51** · scope — The user owns the accelerator, through its BRM (lanes, rates, latency, II, op);
everything around it is the model of the SNAX platform. Model cycles are trusted only as far as
the comparison with SNAX RTL shows (M16, D116).

**D52** · scope — Cosim (M10) swaps one accelerator for its RTL and checks its output, latency and
II; the whole cluster is checked by running the same program on SNAX-GVSoC and on snax_cluster
RTL (D116). Neither blocks the inner loop.

**D53** · lower — SNAX-LOWER produces both inputs of a model run: the cluster file and the control
program. Refines D18; built by D88 and D89.

**D55** · viz — The viewers are a local server bound to 127.0.0.1 plus a static page: stdlib only,
no build step, offline, for humans; LLMs read the reports (D99). Views read the model's own
dataclasses, not raw JSON. → `viz/`.

**D57** · viz — The run views: a schedule (per component its classes, tasks and commands over a
cycle window) with the cluster view under it at the selected cycle, kept in the URL hash; tasks
are paired once in Python, and one colour means one thing in every view. → `viz/api.py`.

**D59** · model — An accelerator's `busy` class includes its II gap, so utilisation is the share of
cycles the datapath is occupied; the profile also counts firings. Amends D35, D38.

**D62** · model — Read responses are traced as `resp` events and drawn apart from requests.
Amends D39, D61.

**D64** · lower — The task list has `configure`, `start`, `sync` and `read` steps, placed where the
list chooses. Lowering adds only the waits correctness needs, one per component, and leaves out a
wait another one already covers. → C§9, `lower/commands.py`.

**D65** · scenario — One folder per scenario, holding its `scenario.py` and a hand-written
`tasks.json` that it lowers into the program. Amended by D67, D92, D95.

**D67** · scenario — Generated scenario files (`scenario.json`, `.npy` inputs, cluster files) are
not in git; `pixi run scenarios`, `model-run` and every test session write them. Amended by D92.

**D68** · brm — A BRM is a hand-written JSON file: a shared part (interface, function, dataflow,
pattern) and implementations (source, supported values, timing, binding); timing is always the
user's numbers. `resolve` makes an instance and checks it against the model's kind. → C§10.

**D69** · model — A reader whose loop-0 temporal stride is 0 reads each group once and hands it
out `bound[0]` times, as SNAX's reader does. Closes that half of open item 5.

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

**D83** · scenario — The scenarios use the derived names (`acc_a`, `add_acc_a`, `load_A`), and
vecadd's L2 is packed as the default memory plan packs it; every cycle count is kept.
Amends D75.

**D84** · design — The platform is a file of its own in `platforms/`, streamer shell included; the
design step pairs it with the recipe's bound graph, so either changes without the other. Cluster
parameters are design knobs. → C§13.

**D85** · design — Design checks report every problem before anything is written, each with its
fix. → C§14.

**D86** · design — The memory plan comes from registered residency, layout and placement passes,
contiguous by default, with pins. Passes see each port's element-index stream, so a policy can
avoid bank conflicts without a run. → C§15.

**D87** · design — `design_point.json` holds the bound graph, the platform, the resolved streamers
and the memory plan; it is checked again on load and is SNAX-LOWER's only input. → C§15.

**D88** · lower — The cluster file is derived from a design point, and the checked-in clusters are
built by the same code. → `lower/cluster.py`.

**D89** · lower — The task list is derived from a design point: a group per accelerated node,
loads and stores from the memory plan, streamer values from memlets checked against the BRM's
nest, and `after` holding data dependences only. → `lower/derive.py`.

**D90** · flow — `pixi run flow` runs the inner loop once, from a recipe to a checked run, and
decides nothing. Everything goes to a folder named after every `--set`: the run (traced at
`task`), the reports and `flow.log` with each stage's time. → `flow/run.py`.

**D91** · scope — The old SDFG → descriptor → RTL path is gone; the Chisel blocks and `Emit` are a
project of their own in `hw/chisel/`.

**D95** · scenario — A scenario may name its data: `regions` (name, memory, base, shape, strides),
which the model ignores and `run.json` records, so a run directory says where its data lives as it
says which hardware ran it (D50). The flow fills them from the memory plan; vecadd and
vecadd_conflict declare theirs, and a region must fit its memory when the scenario is made.
Amends D44, D65. → C§6.

**D96** · viz — The memory tab shows L1 as banks × rows through the address map and L2 as rows of
one DMA beat, folded in Python so the answer grows with the regions, not with the depth.
→ `viz/memory.py`.

**D97** · viz — Data movement is read off the beat trace and the regions, with no timing model: an
element's journey, residency, per-port address nests and the L1 conflicts on the layout.
→ `viz/movement.py`.

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
`_`, since they appear in `code` and in streamer names. → C§10.

**D104** · sandbox — `bind` binds a reduction through a `reduce` matcher; a folding output without
the spatial variable folds every lane into one. Named rates come from one function, `named_rates`,
shared by bind, the reference executor and SNAX-LOWER. → `sandbox/patterns.py`.

**D105** · lower — Accelerators chain through L1: the reader waits on the writer that put its
container there, the only wait between them. A container the DMA moves is padded to whole beats,
and the padding is kept free in both memories. → `lower/derive.py`, `lower/layout.py`.

**D108** · flow — The flow also starts from a bound `.snaxdfg`, the suffix telling the two forms
apart. That is how a hand-edited graph is run and checked, though it cannot be replayed or swept
(open item 34).

**D111** · scope — SNAX-FORGE generates a bundle per design point for SNAX: the accelerator RTL
with a shell wrapper, the cluster hjson, a `Bender.yml` and the program from the task list as
direct CSR writes. SNAX's own generators build the rest, so the hjson describes the cluster once.

**D112** · scope — A composition is one accelerator: one recipe step binds a connected group of
tasklets to a chain of unit BRMs linked by FIFOs, with streamers on the outer ports only, so one
register window and one SNAX shell wrapper (M13). Links between shells are parked (open item 53).

**D113** · scope — The kernels are NPBench's own DaCe sources with their data types, at PolyBench's
MINI and SMALL sizes where they fit SNAX's 128 KiB L1, on SNAX's own cluster first (32 banks, from
its hjson). `vecadd` and `dot` stay `int64`. → STATUS (M7).

**D116** · scope — SNAX-GVSoC replaces SNAX-MODEL (GVS4, closing X1): a GVSoC model of the SNAX
cluster in which a Snitch core runs the real program, so control cost is measured, not declared.
It is first aligned with snax_cluster by hand and checked against its RTL (M16); then each BRM
generates a GVSoC accelerator and its Chisel RTL, accepted when the pair reproduces that baseline
(M14), and SNAX-MODEL goes with its entries here (RET1). Amends D6, D36–D40 (the CSR transport
stays, the layout becomes SNAX's), D45, D51–D53, D68, principles 1 and 3, the RISC-V non-goal.
→ STATUS (M16).

**D117** · docs — The documents stay short: one fact in one place, a decision only for a choice
someone may want to reverse, each entry in its current form in at most three lines, and the done
tasks out of STATUS at every milestone close. The rules are at the top of this file. Replaces D93.

**D118** · scope — One development image, `ghcr.io/kuleuven-micas/snax-forge:main` from
`container/`, for snax-forge, snax-gvsoc and the programs built from snax_cluster: the hub owns
the tools (GVSoC's build tools, snax_cluster's RISC-V toolchain, pixi); each repo keeps its own
environment. Replaces SG0's two images (one in snax-gvsoc). → README (Setup).

**D119** · scope — M16 follows what `snax-alu` needs on the snax_alu cluster, plain interconnect as
baseline and sparse: target and CSR port (SG1), DMA (SG2), streamer (SG3), ALU (SG4), cycles against
RTL (SG5), Zicsr (SG6), the sparse interconnect (SG7), LR/SC on the sparse RTL (SG8). Reading the
hjson moves to LNK2, the generic accelerator to GEN3. → STATUS (M16).

**D120** · gvsoc — The Snitch core sends CSRs 0x3C0 to 0x5FF as 32-bit IO requests at 4 × CSR on
its `snax_csr` port to an accelerator component per core, so a SNAX block is a plain IO slave. Both
sides follow Zicsr since SG6: snax_cluster #657 (decoder, translator), GVSoC's `priv.hpp`, all cores.
→ ARCHITECTURE §5.6, snax-gvsoc `SG6.md`.

**D121** · gvsoc — SNAX-GVSoC keeps GVSoC's iDMA model and fits it to snax_cluster's iDMA RTL by
measurement, not a new model: the target sets the DMA and DRAM values (with SG7, the TCDM write
latency and one stage per cycle, from splitting beat cycles against the RTL), the model the DMA's
own read path; `dma-timing` checks `busy`, `r` and `w` against the RTL's on every run. → snax-gvsoc
`SG2.md`, `SG7.md`.

**D122** · gvsoc — SNAX-GVSoC's crossbar takes the SNAX ports, cores and AXI, numbered as on the
RTL, and copies its arbiter, checked against the RTL module in Verilator: `rr_arb_tree` with
`LockIn` (plain); priority from the streamer's bit, round robin from the last pick, idle restart,
lock (sparse). The DMA takes whole superbanks (`dma_data_width / data_width` banks), refusing narrow
winners. Closes open item 58 (SG7). → snax-gvsoc `SG3.md`, `SG7.md`.

**D123** · scope — A SNAX-GVSoC model is checked against snax_cluster's RTL by building one ELF
and running it on both, comparing the counters that do not depend on the core's timing (each
program names them), not by predicted values. Build once, run both, compare becomes a feature of
the whole system: D124's comparison, FLOW3 and GEN2 build on it. → snax-gvsoc `SG4.md`,
STATUS (M16).

**D124** · scope — SNAX-GVSoC is held to snax_cluster's RTL per phase of one ELF: counters and
errors equal, growing phases within 10%, the rest explained (open items 65–67), checked by hand from
fresh runs; `make regress` (CI, each push to main and PR) checks every program exits 0, and
`dma-timing` that `busy`, `r`, `w` stay the RTL's (D123, SG7). → snax-gvsoc `SG5.md`–`SG7.md`,
STATUS (M16, CI1).

**D125** · gvsoc — The Snitch core's loads follow the hjson and the RTL: one integer load in flight
(`num_int_outstanding_loads`, a core property in GVSoC's LSU), the RTL's load-to-load cycles (4 from
L1, 14 from DRAM; `tcdm` 2), and, since SG7, the core going on after a load answered later until the
next load or a use of its register. Fetch keeps GVSoC's cache (open item 65). → snax-gvsoc `SG5.md`,
`SG7.md`, `snax/snax_alu.py`.

**D126** · scope — snax-gvsoc builds its programs from a pinned copy of snax_cluster's software and
generated headers in `sw/`, refreshed by hand with `make sync-snax-cluster`, whose diff is the check
that the two still match; the sparse hjson is copied too, its headers checked equal to the old one's
(SG7). A snax_cluster checkout is needed only for the RTL and sync. Pin: snax_cluster `05615e4`
(#658, SG8). → snax-gvsoc `sw/README.md`, STATUS (M16, CI1).
