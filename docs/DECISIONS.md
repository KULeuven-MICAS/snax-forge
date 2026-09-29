# SNAX-FORGE Decisions

What was decided, and why, in the order it was decided. `docs/ARCHITECTURE.md`
describes the system as it is now; this file keeps the decisions behind it.
Field-level detail lives in `docs/CONTRACTS.md` (cited as C§n) or in the
docstring of the module named; plans and open items live in `docs/STATUS.md`.
The full wording of D1–D92 as first logged is in git history: section 10 of
`docs/ARCHITECTURE.md` before D93.

Rules (D93):

- A number is never changed or reused: code, tests and docs cite them.
- An entry says what was decided and why in one to three sentences, then
  what it amends and where the detail lives. Test names, cycle counts and
  file listings belong in the tests, STATUS or CONTRACTS, not here.
- A decision that a later one replaces keeps its entry, shortened to what it
  said and which decision replaced it.
- A new decision gets the next number at the end of this file, with its
  area tag. Next free number: **D97**.

Area tags: `scope` (plan, ownership, order), `model` (SNAX-MODEL),
`scenario` (scenario files), `lower` (SNAX-LOWER), `brm` (SNAX-BRM),
`dfg` (SNAX-DFG and importers), `ref` (reference executor), `sandbox`,
`design` (platform, checks, memory plan, design point), `flow`, `viz`,
`docs`, `test`.

By area:

| Area | Decisions |
|---|---|
| scope | D4 D8 D9 D14 D22 D23 D24 D27 D51 D52 D63 D91 |
| model | D6 D10–D13 D21 D25 D29–D40 D43 D44 D47–D50 D59 D62 D69 |
| scenario | D41 D42 D65 D67 D83 D95 |
| lower | D18 D45 D53 D64 D66 D75 D88 D89 |
| brm | D3 D5 D15 D68 D70 D82 |
| dfg | D1 D2 D19 D71 D77 D78 |
| ref | D17 D20 D28 D79 |
| sandbox | D72 D73 D80 |
| design | D7 D74 D84–D87 |
| flow | D90 D94 |
| viz | D16 D54–D58 D60 D61 D76 D81 D96 |
| docs, test | D26 D46 D92 D93 |

---

**D1** · dfg — SNAX-DFG is SNAX-FORGE's own serialisable, SDFG-inspired, extensible format.
Refined by D71 and D77.

**D2** · dfg — Nodes start at SDFG granularity, with coarse accelerated nodes on top that replace a
subgraph. Refined by D77.

**D3** · brm — A block runtime model (BRM) has an interface, a dataflow, timing, a function, a
pattern and an optional hardware binding. Format: D68.

**D4** · scope — Model first: the Python models are the primary artefact, and the RTL library
becomes the BRMs' hardware bindings.

**D5** · brm — Latency and II are the user's numbers, analytic or measured from RTL.

**D6** · model — SNAX-MODEL is pure Python: banks, TCDM interconnect, streamers, DMA and L2, a
register interface and a controller. It has no CPU.

**D7** · design — Cluster parameters are part of the design space. They live in the platform
(D84).

**D8** · scope — The AI thinker is a commercial LLM, so every artefact is text.

**D9** · scope — First targets: vecadd, dot, then jacobi1d.

**D10** · model — Cycle-level and event-driven: only components with pending work are ticked, and
idle cycle ranges are skipped. → `sched.py`.

**D11** · model — The control program is JSON; a wait polls a block's `busy` register or blocks on
its completion signal. Commands: D36, D42.

**D12** · model — One streamer per accelerator port; how many interconnect ports a streamer has is
configurable.

**D13** · model — The unit of transfer is a bank word, one element per word for now; element type
and elements per word are in the data model so packing can come later (open item 21).

**D14** · scope — *Replaced by D72.* Thinkers were to act through declarative config files, with a
GUI showing design-point diffs.

**D15** · brm — Each BRM port gives an exact affine loop nest. Notation: D70.

**D16** · viz — SNAX-DFG can be viewed. Viewer: D76, D81.

**D17** · ref — DaCe's CPU reference is optional; correctness comes from the reference executor
(D20) and later cosim.

**D18** · lower — SNAX-LOWER produces the control program, and later the SW kernel from the same
logic, so the model and the chip run the same task sequence. Refined by D45, D53.

**D19** · dfg — Extension by a small core schema, registered node kinds and namespaced attrs, never
by editing core classes. Amended by D77.

**D20** · ref — A NumPy executor of the SNAX-DFG gives the golden output. Built: D79.

**D21** · model — The exact sequential simulator is the source of truth; its state is kept
array-friendly so it can be accelerated later without changing results. Amended by D48.

**D22** · scope — *Replaced by D24 and D51.* A vertical vecadd slice first (still the rule), with
the model validated against RTL early (now deferred to M2).

**D23** · scope — SNAX-FORGE complements ZigZag and Stream: cycle-level and RTL-backed rather than
analytical (open item 4).

**D24** · scope — Build order: the kernel-agnostic SNAX-MODEL first, then vecadd end to end, the
visualiser before dot, and the contract freeze after dot. The order is now M1, M4a, M3, M4b,
M5–M10, M2 (D51, D54, D76).

**D25** · model — The accelerator interface has a per-port element rate, so reductions and
elementwise blocks share one interface. → C§4.

**D26** · docs — The model-side contracts follow from what SNAX-MODEL needs: dataclasses and plain
JSON until the M6 freeze, versioned schemas after. Written down by D46.

**D27** · scope — *Replaced by D72.* Until SNAX-DSE existed, thinkers were to edit the design point
directly.

**D28** · ref — Reductions use integer types first; a float reduction follows the accumulation
order its BRM defines, so results match exactly.

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
byte-identical on every run and with skipping on or off. Amended by D50. → C§6, C§7.

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

**D51** · scope — The user owns the accelerator entry (lanes, rates, latency, II, op); everything
else is SNAX-MODEL's model of the SNAX platform, with declared defaults. Model cycles compare
design points and do not predict SNAX's timing; the RTL anchor (M2) comes after M10.

**D52** · scope — Cosim does not block the inner loop: it swaps only the accelerator for its RTL
and checks its output, latency and II, not the platform.

**D53** · lower — SNAX-LOWER produces both inputs of a model run: the cluster file and the control
program. Refines D18; built by D88 and D89.

**D54** · viz — The visualiser is split: the run views (M4a) before M3, the design-point views and
the first manual loop (M4b) after it.

**D55** · viz — The visualiser is a local server bound to 127.0.0.1 plus a static viewer: stdlib
only, no build step, works offline. It is for humans; LLMs read the profile (VIS7).
→ `viz/server.py`, `viz/api.py`.

**D56** · viz — A FIFO's occupancy is also shown over its busy window, taken from task events.
Closes that half of open item 11.

**D57** · viz — The schedule view shows per component its cycle classes, tasks and commands over a
cycle window, with one selected cycle; view state lives in the URL hash. Amended by D58, D60, D61.

**D58** · viz — Schedule details: port rows named by what they show, each DMA task's direction,
wheel zoom. Its `dma_tasks` is replaced by D60.

**D59** · model — An accelerator's `busy` class includes its II gap, so utilisation is the share of
cycles the datapath is occupied; the profile also counts firings. Amends D35, D38.

**D60** · viz — The run detail carries each block's tasks, paired once in Python and reused by the
views. Replaces D58's `dma_tasks`.

**D61** · viz — The cluster view shows banks, interconnect, streamers, accelerators, DMA and
controller at the selected cycle, under the schedule; one colour means one thing across all views.
Data values: open item 23.

**D62** · model — Read responses are traced as `resp` events and drawn apart from requests.
Amends D39, D61.

**D63** · scope — Inside M3, task list → commands (LOW1b) was built before design point → task
list (LOW1a), on hand-written task lists. Done.

**D64** · lower — The task list: `configure`, `start`, `sync` and `read` steps. The lowering adds
only the waits correctness needs, one per component, and a wait on a writer streamer also covers
what started with it. Amended by D66. → C§9, `lower/commands.py`.

**D65** · scenario — One folder per scenario, holding its `scenario.py` and a hand-written
`tasks.json` that it lowers into the program. Amended by D67, D92.

**D66** · lower — A wait that another wait of the same start already covers is left out; with it,
fmul becomes a task list with its hand-scheduled program.

**D67** · scenario — Generated scenario files (`scenario.json`, `.npy` inputs, cluster files) are
not in git; `pixi run scenarios`, `model-run` and every test session write them. Amended by D92.

**D68** · brm — A BRM is a hand-written JSON file with a shared part (interface, function,
dataflow, pattern) and a map of implementations (source, supported values, timing, binding).
`resolve` checks an instance against the model's registered kind, so an instance that exists fits
the model. → C§10.

**D69** · model — A reader whose loop-0 temporal stride is 0 reads each group once and hands it
out `bound[0]` times, as SNAX's reader does. Closes that half of open item 5.

**D70** · brm — The first dataflow notation, `affine`: per port the operand's shape, an offset and
loops (bound, strides, spatial flag), spatial loops last. SNAX-LOWER maps it through a buffer
layout to check streamer values. Amended by D73. → C§3, C§10.

**D71** · dfg — SNAX-DFG is its own JSON format, `.snaxdfg`: SDFG's concepts without SDFG's JSON.
Importers derive it and decide nothing, and streamers are not nodes. Format: D77, importer: D78.

**D72** · sandbox — SNAX-SANDBOX is where decisions are made by hand: registered transforms on a
`.snaxdfg`, applied in order from a recipe, each step checked by the reference executor. A sweep is
one recipe with a parameter; automated search (M8) writes recipes later. Replaces D14, D27;
amended by D80, D84.

**D73** · sandbox — Loops are tagged `tile`, `temporal` or `spatial`. `bind` reads a BRM's design
params off the graph (W from the spatial bound), so each is decided once. Streamer values come from
memlets through layouts, and the BRM's nest checks their order.

**D74** · design — The design point holds the mapped graph, the memory plan, the accelerator
instances and the platform. How it is made is replaced by D84 (design step), D86 (memory plan) and
D87 (the file).

**D75** · lower — Derived names: streamer `<instance>_<port>`, task `<node>_<component>`, DMA tasks
`load_<container>` and `store_<container>`. SNAX-LOWER derives them; the design point names none.
Amended by D83.

**D76** · viz — The DFG viewer is a second mode of the viewer (VIS5), and M3 closes vecadd from the
kernel forwards. Viewer amended by D81.

**D77** · dfg — The `.snaxdfg` format: symbols, containers and an ordered body of registered node
kinds (`map`, `tasklet`, `accelerated`), memlets on connectors, and one expression grammar
(`expr.py`) shared with the BRM. Amends D19, D71, D73, D74. → C§11.

**D78** · dfg — The SDFG importer maps a simplified SDFG onto `.snaxdfg` with readable names, and
raises a named error for anything it does not support. The vecadd kernel became `int64`
(open item 30). → `dfg/import_sdfg.py`.

**D79** · ref — The reference executor runs a `.snaxdfg` in NumPy, accelerated nodes through their
BRM's function, and is what every sandbox step and model run is compared with.
→ `dfg/execute.py`.

**D80** · sandbox — Recipes (kernel, params, symbols, steps) and the first transforms, `split_map`
and `bind` with registered pattern matchers; `--set` runs one point of a sweep. → C§12,
`sandbox/transforms.py`.

**D81** · viz — The DFG viewer draws graphs top to bottom in execution order, with rows computed in
Python, SVG edges between neighbouring rows, and highlighting across panels.
→ `viz/dfg/api.py`.

**D82** · brm — A BRM states what one lane computes (`function.code`), and `bind` checks the
tasklet against it. A bound graph records what it replaced, so `unbind` and `join_map` take it
back. Amends D68, D77, D80, D81.

**D83** · scenario — The scenarios use the derived names (`acc_a`, `add_acc_a`, `load_A`), and
vecadd's L2 is packed as the default memory plan packs it; every cycle count is kept.
Amends D75.

**D84** · design — The platform is a file of its own in `platforms/`, including the streamer shell;
the design step pairs it with the recipe's bound graph, so either can change without the other.
`--set platform.PATH=VALUE` writes a working copy, `design save` keeps one. Closes open item 35.
→ C§13.

**D85** · design — Design checks find every problem before anything is written, each named with
its fix as a `--set` or recipe change. → C§14.

**D86** · design — The memory plan comes from registered residency, layout and placement passes,
contiguous by default, with pins (`--set memory.<c>.<mem>.base=N`) and a working copy. Passes see
each port's element-index stream, so later policies can avoid bank conflicts without running the
model (DSE4). Replaces D74's place step; closes open item 29. → C§15.

**D87** · design — `design_point.json` holds the bound graph, the platform, the resolved streamers
and the memory plan. It is checked again on load and is SNAX-LOWER's only input. → C§15.

**D88** · lower — The cluster file is derived from a design point; the checked-in clusters are
built by the same code, so they cannot drift apart. → `lower/cluster.py`.

**D89** · lower — The task list is derived from a design point: one group per accelerated node,
loads and stores from the memory plan, and streamer values from memlets, checked against the BRM's
nest by address. → `lower/derive.py`.

**D90** · flow — `pixi run flow` runs recipe → design point → cluster file and task list → scenario
→ model run, and checks the output against the kernel's reference and the reference executor. It
closes M3 and open item 13. → `flow/run.py`.

**D91** · scope — The old SDFG → descriptor → RTL path is removed: M3's flow never used it. The
Chisel blocks and `Emit` stay as a project of their own in `hw/chisel/`, the `hw` environment is
JVM-only, and unused dependencies are dropped. Amends D72, D78 and GEN1.

**D92** · test — Scenario helpers that SNAX-LOWER replaced are removed, the two hand-written MOD7
vecadds become one test helper, and tests that only repeated others are removed.
`scenarios/vecadd/tasks.json` stays hand-written as LOW1a's reference. Amends D65, D67, D83.

**D93** · docs — The decision log moves to this file and is condensed by the rules at its top;
`docs/ARCHITECTURE.md` describes the current state only.

**D94** · flow — The flow traces at `task` by default, so its run opens in the viewer with a
schedule, and its default folder name carries every `--set` (recipe params, then platform and
memory paths without their prefix), so runs that differ only in the platform or the memory plan
no longer overwrite each other. The design point and its task list keep the name of the recipe
and its params, so pinning B and C still gives `scenarios/vecadd/tasks.json`, and the viewer names
a flow's `run/` after its folder. Amends D55 (run names) and D90. → `flow/run.py`, `viz/api.py`.

**D95** · scenario — A scenario may name its data: `regions` (name, memory, base, shape, strides),
which the model ignores and `run.json` records, so a run directory says where its data lives as it
says which hardware ran it (D50). The flow fills them from the memory plan; vecadd and
vecadd_conflict declare theirs, and a region must fit its memory when the scenario is made.
Amends D44, D65. → C§6.

**D96** · viz — The memory layout is a tab of the run viewer, not a design point view: L1 first, as
banks × rows through the model's address map, L2 as rows of one DMA beat. Rows are computed and
folded in Python: neighbouring rows fold when every column holds the same regions and each region
moves by one constant step, so the answer grows with the regions and not with the depth, and
folded rows are served on request, at most 256 at a time. Region colours are a new meaning, used
only in that tab. Amends D54, D61. → `viz/memory.py`.
