# SNAX-FORGE Decisions

Why SNAX-FORGE is built the way it is: one entry per live decision, in its current form. What the system is lives in `docs/ARCHITECTURE.md`, the artefacts' fields in `docs/CONTRACTS.md`, open work in `docs/STATUS.md`.

Rules (D117):

- An entry is a choice with a real alternative that someone may want to reverse: direction, ownership, the interfaces between components, the order of work. Implementation details live in the module's docstring and its tests.
- An entry says what was decided, why, and where the detail lives, in a few short bullets; an edge case only when there is one. A change is written into the entry, not chained after it.
- Entries merged into one keep all their numbers in its heading, so a cite of any of them still finds it. A number is never changed or reused; the text of a removed entry is in git history.
- A new decision gets the next free number, under the area it belongs to. Next free number: **D172**.

Numbers retired, never reused: D1–D7, D9–D11, D13–D18, D20–D25, D27, D29–D45, D47–D50, D53, D54, D56, D58–D63, D65–D67, D69, D74, D76–D78, D80, D81, D83, D88–D97, D100, D101, D103, D104, D106, D107, D109, D110, D114, D115, D119, D124, D127, D128, D132, D133, D138–D141, D151, D156, D159, D160.

## Scope

### D51 · D52 — The user owns the accelerator; everything around it is the model

- **Decided:** the user describes the accelerator through its BRM (lanes, rates, latency, II, op). Everything around it (cores, DMA, streamers, crossbar, memory) is the model of the SNAX platform. One accelerator can be swapped for its RTL in a cosim (M10); the whole cluster is checked by running the same program on SNAX-GVSoC and on snax_cluster's RTL (D123). Neither check blocks the inner loop.
- **Why:** the user knows their block; the platform is shared and must be right for everyone. Model cycles are trusted only as far as the comparison with the RTL shows.
- **Where:** ARCHITECTURE section 4.

### D116 — SNAX-GVSoC is the model

- **Decided:** the model is SNAX-GVSoC, a GVSoC model of the SNAX cluster in which a Snitch core runs the real program.
- **Why:** control cost (the core configuring, starting and polling) is measured, not declared, and the model is checked against snax_cluster's RTL by one ELF on both (D123).
- **Where:** ARCHITECTURE, "SNAX-GVSoC"; the snax-gvsoc repository.

### D19 · D8 — Text artefacts, extended by registration

- **Decided:** every artefact is text that humans and LLMs read alike (the AI thinker is a commercial LLM). Everything is extended by registration (node kinds, transforms, checks, passes, the accelerator kinds and ops of `brm/function.py`, the generator's blocks) and namespaced attrs, never by editing core classes.
- **Why:** an LLM can read and write every step, and a new kind of block or transform touches no core code.
- **Where:** the registries in each package; CLAUDE.md's principles.

### D113 — Kernels: NPBench's structure, integers first

- **Decided:** the kernels keep NPBench's DaCe structure with a dtype per container, so one kernel may mix them (vsum: `int32` in, `int64` out), at PolyBench's MINI and SMALL sizes. Integers first: `int64` through tier C, `int8` in and `int32` accumulate from tier P, exact products by an explicit cast (D143); floats later.
- **Why:** an integer check against the reference is exact.
- **Where:** `kernels/README.md`, PAPER section 6.1.

### D111 · D130 — A bundle for snax_cluster, one wrapper contract

- **Decided:** per design point snax-forge writes a bundle for snax_cluster (the accelerator's RTL with its shell wrapper and file list, the cluster and streamer hjson, the program) and the same hjson and headers for SNAX-GVSoC. A generated accelerator carries its own CSR manager (ReqRspManager's protocol, snax_cluster's own off) and its CSRs per group. Python checks `accelerator.json` against the datapath catalogue; Scala only builds.
- **Why:** SNAX's own generators build the rest of the cluster from one hjson, so the cluster is described once, and snax_cluster sees only the bundle.
- **Where:** CONTRACTS, "Bundle" and "Accelerator RTL spec".

### D118 · D126 — One image, a pinned copy of snax_cluster's software

- **Decided:** one development image, `ghcr.io/kuleuven-micas/snax-forge:main` from `container/`, for snax-forge, snax-gvsoc and the programs; the image holds the tools, each repository keeps its own environment. snax-gvsoc builds its programs from a pinned copy of snax_cluster's software in `sw/`, refreshed by `make sync-snax-cluster` (its diff is the check); a design point's program takes the cluster headers snax-forge writes for it.
- **Why:** the same tools everywhere, and only the RTL needs a snax_cluster checkout.
- **Where:** `container/Dockerfile`; snax-gvsoc's `sw/README.md`.

### D98 · D171 — Tests check runs loosely; walk-throughs are of two kinds

- **Decided:** a test on a SNAX-GVSoC run checks what the program computes exactly (exit 0, no data error, every counter at least its beats, the counts timing does not set: firings, beats, words, bytes) and its timing loosely: each counter within 10 % of the estimate's expected BUSY_COUNTER beyond its fill (D155), no window pinned. Walk-throughs under `examples/dryrun/` are tutorials with no test, their numbers as printed when written; the others keep a test checking every number they state, cycles the same loose way. Runs are compared by switching in the viewer and reading their reports; there is no diff view.
- **Why:** a program or platform change moves cycles without breaking anything; the exact numbers go to run notes and open item 95, and a counter beyond 10 % is investigated before any number changes.
- **Where:** `tests/flow/helpers.py` (`sound`, `near`); `examples/`.

## Graph and reference

### D71 — `.snaxdfg`, SNAX-DFG's own format

- **Decided:** a JSON format with SDFG's concepts and one expression grammar (`expr.py`) shared with the BRM. MatMul and Transpose are first expanded into dot's shape (`sdfg/expand.py`). Importers derive, decide nothing and name what they lack; streamers appear only at lowering.
- **Why:** one small format the sandbox, the design step and an LLM all read, without DaCe's library nodes.
- **Where:** CONTRACTS, "SNAX-DFG".

### D102 — A reduction is a map

- **Decided:** a reduction is a map whose tasklet writes through a memlet with a `reduce` (an associative op, and its identity or a seed, D163). `bind` folds every lane of an output without the spatial variable into one.
- **Why:** import, split and bind treat it as any map.
- **Where:** CONTRACTS, "SNAX-DFG".

### D162 — Loop variables are named by dimension, piece and tag

- **Decided:** a map over `0:X` (X one symbol) is named `lower(X)` at import (rowsum's `m` and `n`), else it keeps DaCe's name. A split dimension's pieces are numbered outer to inner (`n_1`, `n_2`, `n_3`), and a tag adds `_t`, `_s` or `_tile`. The name is derived from `loop.dim` and `loop.piece` after every step, never typed; map ids do not change.
- **Why:** a variable says which dimension it runs over and how it is used, so `A[m, n]` reads as it should, and recipes keep working.
- **Where:** CONTRACTS, "SNAX-DFG" and "Recipe".

### D164 — A transposed read is a mark, not a view

- **Decided:** a read across the loop order (mvt's `A[m, n]` under the loops (n, m)) is marked by a step, `mark_transpose`, as an annotation on the reader, not as a view container as DaCe's Transpose makes. The memlets keep the read order and LOWER derives the strides from them.
- **Why:** the mark is what a recipe's match and a later DMA-transpose variant read; the viewer draws the container once with `Aᵀ` on the edge.
- **Where:** CONTRACTS, "SNAX-DFG" and "Recipe".

### D28 · D79 — The reference executor is exact

- **Decided:** the reference executor runs any `.snaxdfg` in NumPy, accelerated nodes through their BRM's function, a slot's value from its params or its slot connector. Every sandbox step and every run is compared with it. Reductions are exact: integers first, a float reduction in the accumulation order its BRM defines.
- **Why:** the check against the executor stays exact and DaCe's CPU reference is not needed; the check against NumPy is a tolerance.
- **Where:** `dfg/execute.py`.

## BRMs and blocks

### D68 · D70 · D129 — A BRM is a hand-written JSON file

- **Decided:** a BRM has a shared part (an interface with a rate per port and a `when` naming the design params a port exists at; function, dataflow, pattern) and implementations (source, supports, timing, binding). Its timing is the user's numbers (D51). A port's dataflow is a nest in a registered notation: `affine` gives the operand's shape, an offset and loops; `stream` gives only lanes and a rate. A `binding` names its datapath: one Chisel module, one `blocks.json` entry and one SNAX-GVSoC class by name.
- **Why:** reductions and SIMD blocks share one interface; a new block is one of each of the three, and the user's numbers stay theirs.
- **Where:** CONTRACTS, "Block runtime model" and "Accelerator RTL spec".

### D82 — A BRM says what one lane computes

- **Decided:** a BRM gives one lane's `function.code`, or a table of codes each fixing design params and naming scalar slots (`simd`'s per `op` and `streams`, with `k`, `k1`, `k2`), matched with `+`, `*`, `min` and `max` commuting. `bind` checks the tasklet, or a fused map's tasklets joined by a local value, against it; the first entry that matches gives the instance its params and slots. A bound graph records what it replaced, so `unbind` and `join_map` take it back.
- **Why:** one block serves several ops, and binding is checked, not trusted.
- **Where:** CONTRACTS, "Block runtime model".

### D143 — Arithmetic follows NumPy

- **Decided:** each op runs at its operands' result type, with signedness from each port's container, then the result is extended or wrapped into its target. A fold's lanes are cast to its output's type before its tree. A literal operand is weak (NumPy's NEP 50): the op runs at the other operands' type, which the literal must fit (relu's `k`).
- **Why:** the model, the RTL and the reference compute the same bits.
- **Where:** CONTRACTS, "Generated hjson files"; snax-gvsoc's `snax/snax_types.hpp`.
- **Edge:** exact products (tier P) need an explicit cast in the kernel (open item 37).

### D149 — Blocks with variants, not chains

- **Decided:** the MAC is one block with design-time variants (`accumulate`, or a·b + c with `regOut` for a later systolic PE), not vmul into a lane accumulator. The lane accumulator without an op is its own block, `LaneAccumulate`; the one-lane vsum is `SerialAccumulate` (`span` = lanes); a buffer is `LinkBuffer`, not a FIFO.
- **Why:** each block is the one the RTL has, so the cost of each is measured alone.
- **Where:** CONTRACTS, "Accelerator RTL spec"; `hw/chisel/`.

### D163 — A fold's seed is a port

- **Decided:** a fold's result read only by an elementwise nest of the same op (mvt's x₁ + A y₁) is folded in as the fold's seed, a per-job port read on a job's first firing, by a step of its own, `fuse_seed` (inverse `unfuse_seed`). Only `TreeAccumulate` and `SerialAccumulate` take one; their unseeded RTL is unchanged.
- **Why:** no vecadd after the fold and no "init" port on every block.
- **Where:** CONTRACTS, "Block runtime model" and "Recipe".

## Sandbox and recipes

### D72 — Decisions are made by hand, as recipes

- **Decided:** decisions are made in SNAX-SANDBOX: registered transforms applied in order from a recipe, each step checked by the reference executor. A sweep is one recipe with a parameter; automated search writes recipes. A transform may give `site(g, selected)`, the graph-shape test of where it applies, which it calls first and the editor marks and lists from. A site gets the editor's selection, one item or several; a transform that takes several (`encapsulate`, `fuse_map`) has a site over the list, any other refuses several.
- **Why:** DSE decides, LOWER derives: every decision is a readable, replayable step.
- **Where:** CONTRACTS, "Recipe"; `sandbox/`.

### D73 · D150 — The mapping is spelled step by step

- **Decided:** `split` only splits, with neutral names; `tag_map` tags `tile`, `temporal` or `spatial`; `split_reduce`, `fuse_map`, `fuse_seed`, `interchange_map` and `link` each do one thing, each with its inverse, one recipe per variant. `bind` reads design params off the graph (W from the spatial bound) and refuses with the fix named ("fuse_map first", "split_reduce first").
- **Why:** each choice is made once, in its own step, and undone the same way.
- **Where:** CONTRACTS, "Recipe".
- **Edge:** `split_map` splits and tags in one step and keeps the ids `<id>` and `<id>_s`, since old recipes (dot 4–6, matvec 4–6) bind `mult_map_s`.

### D161 — A scalar argument is read from a CSR, or baked

- **Decided:** a kernel's scalar argument binds to a slot read at run time from an R/W CSR, so one ELF serves any value (axpy's alpha as K1). How it is read is a step of its own, `bake_scalar` with a value (baked into the hardware) or `csr`; `bind` still reads an unmarked scalar from a CSR.
- **Why:** both variants are recipes with the same steps, and the editor shows the choice.
- **Where:** CONTRACTS, "Block runtime model", "Recipe" and "Accelerator RTL spec".

### D165 — Two nests on one set of hardware

- **Decided:** two nests run on one set of hardware by binding them to one instance (mvt's variant 1: both halves on one `vmul`, `vsum` and `vadd`), chosen in `bind`'s form ("share with") or by `share` for a linked group. LOWER runs the group once per nest on the same streamers, and the estimate counts the area once.
- **Why:** reuse of hardware is a mapping choice, not a new block.
- **Where:** CONTRACTS, "SNAX-DFG", "Recipe" and "Estimate".

### D166 — A recipe is the unit of reuse

- **Decided:** a recipe applies another on a region of its graph that matches the other's kernel (`apply_recipe`), so matvec is dot's variant on each row and mvt is matvec's on each half. The user lifts a kernel's regions as blocks by hand (`lift`), rather than a search applying recipes unasked. The match leaves the outer loops free; the editor draws a block as one box in its firing loops.
- **Why:** a larger kernel is built from the smaller ones' recipes, and the user stays in charge of what is reused.
- **Where:** CONTRACTS, "Recipe"; `sandbox/blocks.py`.

## Accelerators

### D112 · D145 · D146 · D148 — An accelerator is groups of linked units, described once

- **Decided:** a bound graph nests accelerator ⊃ groups ⊃ units and links. A group is a stand-alone accelerator with its own START, JOBS, STATUS, BUSY_COUNTER and control unit; its units are joined by links of a depth (0 a wire, ≥ 1 a buffer of that many beats), with streamers on the outer ports only. A unit fires JOBS · ∏ T over itself and the folds downstream. `link` merges two bound units' groups; `encapsulate` puts several groups in one accelerator. One description per design point, `accelerator.json`, nested the same way, is what Emit, the catalogue check and SNAX-GVSoC all read.
- **Why:** a composition is one accelerator with one register window and one wrapper, groups run apart, and the model and the RTL cannot be configured apart.
- **Where:** CONTRACTS, "SNAX-DFG" and "Accelerator RTL spec".
- **Edge:** an empty run still ends in a done, so there is no ERROR CSR. Links between accelerators stay parked (open item 53).

### D147 · D152 · D153 · D168 — Datapath, control and CSRs each hold their own

- **Decided:** every block is a datapath of data and enables only, under one control unit per group generated from the description. `<tag>_csr` is the accelerator's interface: what the core writes (settings, START) and reads (STATUS), every CSR at its platform width; it holds a write to a busy group's settings until the group is idle. `<tag>_ctrl` holds what a firing changes (busy, counters, the II gap, the buffers' occupancy) and every test that decides a firing, reading the settings live with no copy; it counts per segment in mixed radix, with no product. `<tag>_peripherals` holds what the run never needs (BUSY_COUNTER) and is left out of every reported figure.
- **Why:** control is written once for model and RTL, never per variant, and `<tag>_ctrl` plus `<tag>_datapath` compare with a whole HLS block. SnaxGroupRule's flat count of firings stays the spec.
- **Where:** CONTRACTS, "Accelerator RTL spec"; `GroupRule.scala`.
- **Edge:** programming the next task during a run is open item 123.

### D157 — A design's accelerators are streamer sets of one entry

- **Decided:** a design's accelerators are streamer sets in one snax_cluster entry, `snax_forge_<design>`, each with its own streamer, START and CSRs behind `snax_csr_range_demux`.
- **Why:** not entries behind `snax_acc_mux_demux`, whose 0x7C6 select can misroute a delayed write, nor address windows (open item 108).
- **Where:** CONTRACTS, "Generated hjson files" and "Accelerator RTL spec".

## Design

### D84 · D131 · D170 — The platform is its own file

- **Decided:** the platform is a file in `platforms/`, paired with the bound graph by the design step; its cluster parameters and the streamers' FIFO depths are design knobs, the temporal loops come from the graph. The base cluster is snax-forge's own `platforms/snax_forge_cluster.hjson`, read by `design from-hjson`: one accelerator entry, named `<accelerator_slot>_<design>`, and one Snitch core, which drives both the DMA and the accelerator.
- **Why:** either the graph or the platform changes alone. A second core only idled at barriers and was still clocked; with one, only core 0's `obs_o` matters.
- **Where:** CONTRACTS, "Platform".
- **Edge:** a cluster file with more cores still lowers (`lower/c.py`).

### D85 — Design checks name every problem and its fix

- **Decided:** the design checks report every problem before anything is written, each with its fix. A limit of the model's cluster is a platform field the checks hold the design to, such as the TCDM crossbar's inputs (`l1.max_xbar_inputs`, SNAX-GVSoC's 128), with a note above `l1.warn_xbar_inputs` (64).
- **Why:** a design is refused before a run would assert, and the note marks costs extrapolated beyond the tables.
- **Where:** CONTRACTS, "Platform" and "Design checks".

### D12 · D142 · D169 — Streamers: one per port, packed words, shared movers

- **Decided:** one streamer per accelerator port, named after it; its width is the port's lanes W, and the hjson's `num_channel` is num_ch = W / num_slc, where a bank word packs num_slc = 64 / b elements of the container's dtype, little-endian as NumPy. `share_ports` on an L1 container one accelerator writes and another reads puts both on one streamer set whose movers they share; a data-side mux picks the group by `PORT_MUX_SEL`, which the core writes before each `STREAMER_START`.
- **Why:** the streamer matches snax_cluster's, the platform keeps only the word, and sharing movers saves crossbar inputs without new hardware between accelerators.
- **Where:** `design/streamers.py`; CONTRACTS, "Platform", "Design checks" and "Accelerator RTL spec".
- **Edge:** a W that is not a multiple of num_slc needs the FIFO slicer (parked, open item 93), so `connect.num_ch` refuses it. `share_ports` is off by default and the sets still run in series (D158).

### D86 · D105 · D144 — Memory: planned by passes, chained through L1, padded in the data

- **Decided:** the memory plan comes from registered residency, layout and placement passes, contiguous by default, with pins; `padded_rows` avoids bank conflicts without a run and is chosen with a `--set`, never in a recipe. Accelerators chain through L1: a reader waits on the writer that put its container there. Padding is in the data and the hardware masks nothing: a split dimension pads to a multiple of W, the container to whole DMA beats, with zero or a fold's identity.
- **Why:** each pass is one choice; the only wait between accelerators is a real dependence; masks would cost hardware in every block.
- **Where:** CONTRACTS, "Memory plan and design point"; `lower/layout.py`, `dfg/pad.py`, `gen/data.py`.
- **Edge:** a fold may read through a tail a container another unit wrote only when the producer's tasklet on its pad values gives the fold's identity (dot 1 at W_mul = W_sum = 10); otherwise `connect.pad` refuses (open item 106).

### D87 — The design point is LOWER's only input

- **Decided:** `design_point.json` holds the bound graph, the platform, the resolved streamers and the memory plan; it is checked again on load and is SNAX-LOWER's and the generator's only input.
- **Why:** what was decided is one file, and LOWER derives without deciding.
- **Where:** CONTRACTS, "Memory plan and design point".

## Lower and program

### D64 · D75 · D158 — The task list is derived, accelerators run in series

- **Decided:** the task list is derived from a design point: per accelerated node `configure`, `start` and `sync` steps, loads and stores from the memory plan, `after` for data dependences only, one wait per component. Names are derived, never stated: streamer `<instance>_<port>`, task `<node>_<component>`, DMA task `load_<container>` or `store_<container>`. Accelerators run in series: before each one's start, a `sync` on the previous one's task, its configures before that sync.
- **Why:** lowering adds only the waits correctness needs, and the next accelerator is set up while the current one runs.
- **Where:** CONTRACTS, "Task list".
- **Edge:** overlap of independent accelerators is open item 109.

### D135 — The task list becomes C on snRuntime

- **Decided:** one program per design point: each step on the core that owns its component, D64's waits as polls, the DMA waited on for dependences only, two `mcycle` reads bounding the window. The quiet build (`-DFORGE_QUIET`, `forge_quiet.elf`) waits on every start right after it, marks each section in CSR `0x7c5` (core 0's `obs_o`) and prints nothing; its parts are preprocessor blocks, so the default build is unchanged.
- **Why:** the program the RTL runs is the program the model runs; the quiet build cuts a netlist run into sections for power.
- **Where:** CONTRACTS, "Generated hjson files"; `lower/c.py`.

## SNAX-GVSoC

### D120 · D121 · D122 · D125 — SNAX-GVSoC follows the RTL, by measurement

- **Decided:** the Snitch core sends CSRs 0x3C0 to 0x5FF as 32-bit IO requests to an accelerator component per core, so a SNAX block is a plain IO slave. GVSoC's iDMA model is kept and fitted to snax_cluster's iDMA RTL by measurement. The crossbar takes the SNAX ports, cores and AXI, numbered as on the RTL, and copies its sparse arbiter, checked against the RTL module in Verilator. The core's loads follow the RTL: one integer load in flight, its load-to-load cycles.
- **Why:** reuse GVSoC's models where they fit and copy the RTL where cycles depend on it, each checked.
- **Where:** ARCHITECTURE, "SNAX-GVSoC"; snax-gvsoc's `docs/notes/NOTES.md`.
- **Edge:** fetch keeps GVSoC's cache (open item 65).

### D134 — Three files are the contract with SNAX-GVSoC

- **Decided:** per design point, the cluster hjson, the streamer hjson and `accelerator.json` are read by SNAX-GVSoC at run time (`snax_cfg.py`). snax-gvsoc's `forge/` fixtures are those files, checked in.
- **Why:** a sweep compiles nothing, and snax-gvsoc's tests need no snax-forge.
- **Where:** CONTRACTS, "Generated hjson files" and "Accelerator RTL spec".

### D137 — Events in two stages

- **Decided:** SNAX-GVSoC's traces go to one `events.log`, a level turns trace names on, lines are linked by ids. snax-forge's converter renames, never recounts, and cuts the window.
- **Why:** the model owns what it counts and snax-forge only names it.
- **Where:** CONTRACTS, "Profile and trace"; `run/`; snax-gvsoc's `docs/EVENTS.md`.

### D123 — SNAX-GVSoC is checked against the RTL by one ELF

- **Decided:** snax-forge's ELF runs on SNAX-GVSoC and on snax_cluster's RTL. The data must match exactly; each group's BUSY_COUNTER is compared by its gap beyond the group's streamer fill, $e = \max(0, |c_\text{RTL} - c_\text{GVSoC}| - \text{fill}) / c_\text{GVSoC}$. The worst counter gives the verdict: PERFECT when all are equal, NEAR up to 10 %, WARNING up to 15 % (each with an open item, neither blocks a merge), FAIL beyond or on any data error.
- **Why:** the BUSY_COUNTER counts the streamer's head start, which the core's timing sets and SNAX-GVSoC does not model (open items 65, 67). snax-forge never runs RTL; snax_cluster's `snax-forge-check` does.
- **Where:** CONTRACTS, "Bundle"; docs/TOOL.md section 6.

## Flow and estimate

### D136 · D108 — The flow runs the loop once and decides nothing

- **Decided:** `pixi run flow` runs the inner loop once on SNAX-GVSoC: lower, gen, build when needed, sw, gvsoc, check (the program's exit code, `total_cycles` its window), into a folder named after every `--set`. It has no backend flag. It also starts from a bound `.snaxdfg`, the suffix telling the two forms apart.
- **Why:** one command from recipe to checked run; a hand-edited graph can be run and checked, though not replayed or swept (open item 34).
- **Where:** `flow/run.py`; CONTRACTS, "Profile and trace".

### D155 — An ideal estimate, and what it is not

- **Decided:** per group, snax-gvsoc's group rule on ideal streams, groups adding through L1; unmapped work at one cycle per iteration, a labelled assumption in one constant; the total adds both. The expected BUSY_COUNTER (cycles + fill) is a guide, not a check. `bank_bound`, ⌈Σ beats × num_ch / n_banks⌉, bounds the streams' L1 time, not the BUSY_COUNTER.
- **Why:** a quick number before any run. A run's BUSY_COUNTER counts from START until busy drops, so it includes bank conflicts and is shortened by the streamers' head start.
- **Where:** CONTRACTS, "Estimate".

### D154 — A first-order cost model before synthesis

- **Decided:** the cost of a design's groups comes from the description: registers are bits (e_load when loading, e_hold every cycle), a block its comb energy per fire, control its bits × e_hold × a factor; the table is `snax_forge/char/<tech>.json`. A control unit's area is its flops' area plus an equal comb area.
- **Why:** an early estimate (about 30 % on the large costs), shown as such, with no synthesis in the loop.
- **Where:** CONTRACTS, "Cost model"; `snax_forge/char/README.md`.

### D167 — The cost in five parts

- **Decided:** a design's cost is reported in five parts (datapath, control, streamers, crossbar, memory), each with its share, beside the total and the total without the banks. Streamers, crossbar and banks are priced from snax-forge-experiments' tables, copied as they are (`pixi run import-costs`) and interpolated, extrapolated with a note beyond them. A streamer is priced per beat and a crossbar access at one price per bank count, so the energy follows the accesses. The crossbar's area prices a reader's channels as read-only inputs and a writer's as write-only. The cores, I-cache, DMA and csr are not costed. nangate45 only.
- **Why:** the parts a user can change are the parts the estimate shows.
- **Where:** CONTRACTS, "Cost model"; `snax_forge/char/tables.py`, `estimate/cost.py`.

## Viewers, reports and docs

### D55 · D57 · D99 — Local viewers that only draw; reports for people and LLMs

- **Decided:** the viewers are a local server on 127.0.0.1 plus a static page: stdlib only, no build step or library, offline, every POST `application/json`. `view` shows runs, `view-dfg` given `.snaxdfg` files, `edit-dfg` a session the server holds, saved as a new recipe in `recipes/custom/`, never overwriting. Python answers every fold, lookup and step and the page only draws; the view's state is in the URL hash. Design and run reports are Markdown from dataclasses in the layout of HLS reports, every number from the design point, profile or trace.
- **Why:** nothing to install, a view never disagrees with a report, and LLMs read the reports.
- **Where:** `viz/`, `report/`; CONTRACTS, "Recipe".

### D26 · D46 — Contracts are dataclasses, written down and checked

- **Decided:** contracts are Python dataclasses with `to_dict` / `from_dict` and plain JSON until the contract freeze (M6), versioned schemas after. They are written in `docs/CONTRACTS.md`, and every snippet in it is checked against its source.
- **Why:** one source for each artefact's form, and a document that cannot drift from the code.
- **Where:** CONTRACTS, "Conventions"; `tests/test_contracts.py`.

### D117 — The documents stay short

- **Decided:** one fact in one place; a decision only for a choice someone may want to reverse; done tasks out of STATUS at each milestone close; prose one line per paragraph, so a diff shows only what changed.
- **Why:** people and LLMs read the documents before every change; every extra line is read again.
- **Where:** the rules at the top of this file; `util/md_reflow.py`.
