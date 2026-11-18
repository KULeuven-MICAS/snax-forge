# SNAX-FORGE Decisions

Why SNAX-FORGE is built the way it is: one entry per live decision, in its current form. What the system is lives in `docs/ARCHITECTURE.md`, the artefacts' fields in `docs/CONTRACTS.md`, open work in `docs/STATUS.md`.

Rules (D117):

- An entry is a choice with a real alternative that someone may want to reverse: direction, ownership, the interfaces between components, the order of work. Implementation details live in the module's docstring and its tests.
- An entry says what was decided, why, and where the detail lives, in a few short bullets; an edge case only when there is one. A change is written into the entry, not chained after it.
- Entries merged into one keep all their numbers in its heading, so a cite of any of them still finds it. A number is never changed or reused; the text of a removed entry is in git history.
- A new decision gets the next free number, under the area it belongs to. Next free number: **D185**.

Numbers retired, never reused: D1–D7, D9–D11, D13–D18, D20–D25, D27, D29–D45, D47–D50, D53, D54, D56, D58–D63, D65–D67, D69, D74, D76–D78, D80, D81, D83, D88–D97, D100, D101, D103, D104, D106, D107, D109, D110, D114, D115, D119, D124, D127, D128, D132, D133, D138–D141, D151, D156, D159, D160, D181.

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

### D113 · D173 — Kernels: NPBench's structure, integers first

- **Decided:** the kernels keep NPBench's DaCe structure with a dtype per container, so one kernel may mix them (vsum: `int32` in, `int64` out), at PolyBench's MINI and SMALL sizes. Integers first: `int64` through tier C; before the paper, every kernel whose streams pack (tier done and untiered) also runs `int8` in and `int32` accumulate, as OpenGeMM does, with `int16` and `int32` in alongside, as type presets of one kernel source (`--set types=i8|i16|i32`); exact products by the kernel's own cast (D143 · D175). Tier A's kernels and E7's points stay `int64` for now; floats later.
- **Why:** an integer check against the reference is exact; narrow data makes the streams, crossbar and banks cost what a real design's would.
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
- **Edge:** the mark is a strided read, so it serves only an element of a bank word or wider; below it, the read is a copy (D182).

### D182 — A read across the loop order: a mark or a copy

- **Decided:** a read across the loop order is served by `mark_transpose` (D164: a strided read, no copy) or by `data_transpose`: a `transpose` node, a kind of its own, copies the container transposed into a new L1 container, which the nest then reads in loop order. The copy is made `by` the DMA (later the core); `mark_transpose` refuses an element narrower than the bank word, whose packed lanes are contiguous elements of one word (D142 · D174 · D180). LOWER makes the DMA's copy `transpose_<dst>` (`l1_to_l1`), one burst of an element per step, the innermost loop one 2-D copy's repeat, where a load of the destination would be. The estimate prices the copy at the DMA's rate measured on the RTL, 8/3 cycles a burst, one burst per element (D155), and its energy by its L1 accesses (D184).
- **Why:** int8 mvt's half 2 cannot be read by strides; an explicit node keeps the reference check exact (D28 · D79), draws as a block, and gives LOWER one place to emit the DMA's L1-to-L1 task.
- **Where:** CONTRACTS, "SNAX-DFG", "Recipe", "Task list" and "Estimate"; `sandbox/transpose.py`, `lower/derive.py`, `estimate/copies.py`.
- **Edge:** an element a burst is the DMA's slowest form; a copy whose rows are whole beats in both layouts could move beats, which no recipe needs yet. Rank 2 only.

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

### D143 · D175 — Arithmetic follows NumPy; types follow the program

- **Decided:** each op runs at its operands' result type, with signedness from each port's container, then the result is extended or wrapped into its target. A cast in the kernel (`np.sum(A.astype(OP_TYPE) * B.astype(OP_TYPE))`) is a type, not code: the importer folds DaCe's convert map into its consumer as a value cast, which sets the op's `opType` and is stripped at `bind`, the BRMs unchanged. A fold runs at NumPy's type for its reduction (`np.sum` of a narrower int at `int64`, `@` and `np.dot` at the operands'), then its result is extended or wrapped into the output; for `np.sum` that is the same bits as casting the lanes to the output's type before the tree, which the hardware does. A literal operand is weak (NumPy's NEP 50): the op runs at the other operands' type, which the literal must fit (relu's `k`).
- **Why:** the model, the RTL and the reference compute the same bits, and the program stays the only description of the types: NumPy's wrapped products and exact products by a cast are both built.
- **Where:** CONTRACTS, "SNAX-DFG" (import) and "Block runtime model"; `brm/dtypes.py`; snax-gvsoc's `snax/snax_types.hpp`.
- **Edge:** the width rule at `bind`, an output at least as wide as every input, is a hardware rule on top of NumPy, which allows a narrower output.

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

### D12 · D142 · D169 · D174 · D180 — Streamers: one per port, packed and sliced words, shared movers

- **Decided:** one streamer per accelerator port, named after it; a bank word packs num_slc = 64 / b elements of the container's dtype, little-endian as NumPy. An input is wire-sliced: num_slc divides its lanes W, and the hjson's `num_channel` is num_ch = W / num_slc; or sliced when W divides num_slc: a slicer between its reader and the unit takes one word on num_ch = 1 and gives d_in = num_slc / W pops of W lanes, low slice first; it takes words only while its group is busy, and the group's start drops what it holds. An output is wired the same way, or packed when W divides num_slc: a packer between the unit and its writer collects d_out = num_slc / W results per word on num_ch = 1, cleared at START, the word sent when full or on the run's last push, the slices above the last one zero. The flow derives d_in and d_out from the port's dtype, lanes and direction and places the slicer and the packer; a recipe never names them. `share_ports` on an L1 container one accelerator writes and another reads puts both on one streamer set whose movers they share; a data-side mux picks the group by `PORT_MUX_SEL`, which the core writes before each `STREAMER_START`.
- **Why:** the streamer matches snax_cluster's, the platform keeps only the word, a one-lane output keeps that word without being widened or byte-masked, the slicer's gate and drop make it safe whatever the order of `STREAMER_START` and `START`, its registered output costs one cycle a run and keeps the reader's FIFO off the unit's path, and sharing movers saves crossbar inputs without new hardware between accelerators.
- **Where:** `design/streamers.py`; `hw/chisel/`'s `FifoSlicer` and `FifoPacker`; snax-gvsoc's `SnaxSlicer` and `SnaxPacker`; CONTRACTS, "Platform", "Design checks" and "Accelerator RTL spec".
- **Edge:** a port whose W neither divides nor is a multiple of num_slc is refused with the fix, on either side. With an outer loop, d_in or d_out divides the innermost count, since only the run's last word may be short. The program starts a set's streamer only after its group is idle (D179's polls); a word offered earlier is dropped by the start. `share_ports` is off by default and the sets still run in series (D158).

### D86 · D105 · D144 — Memory: planned by passes, chained through L1, padded in the data

- **Decided:** the memory plan comes from registered residency, layout and placement passes, contiguous by default, with pins; `padded_rows` avoids bank conflicts without a run and is chosen with a `--set`, never in a recipe. Accelerators chain through L1: a reader waits on the writer that put its container there. Padding is in the data and the hardware masks nothing: a split dimension pads to a multiple of W, the container to whole DMA beats, with zero or a fold's identity.
- **Why:** each pass is one choice; the only wait between accelerators is a real dependence; masks would cost hardware in every block.
- **Where:** CONTRACTS, "Memory plan and design point"; `lower/layout.py`, `dfg/pad.py`, `gen/data.py`.
- **Edge:** a fold may read through a tail a container another unit wrote only when the producer's tasklet on its pad values gives the fold's identity (dot 1 at W_mul = W_sum = 10); otherwise `connect.pad` refuses (open item 106).
- **Edge:** the DMA moves whole beats, but a `transpose` node's copy moves one element a burst (D182).

### D87 — The design point is LOWER's only input

- **Decided:** `design_point.json` holds the bound graph, the platform, the resolved streamers and the memory plan; it is checked again on load and is SNAX-LOWER's and the generator's only input.
- **Why:** what was decided is one file, and LOWER derives without deciding.
- **Where:** CONTRACTS, "Memory plan and design point".

## Lower and program

### D64 · D75 · D158 — The task list is derived, accelerators run in series

- **Decided:** the task list is derived from a design point: per accelerated node `configure`, `start` and `sync` steps, loads and stores from the memory plan, `after` for data dependences only, one wait per component. Names are derived, never stated: streamer `<instance>_<port>`, task `<node>_<component>`, DMA task `load_<container>` or `store_<container>`. Under the `serial` schedule (the default, D179) accelerators run in series: before each one's start, a `sync` on the previous one's task, its configures before that sync.
- **Why:** lowering adds only the waits correctness needs, and the next accelerator is set up while the current one runs.
- **Where:** CONTRACTS, "Task list".
- **Edge:** overlap of independent accelerators is D179's schedules.

### D179 — The schedule: independent accelerators may overlap

- **Decided:** a design point holds a `schedule` (`--set design.schedule=`), amending D158. `serial`: D158's. `sequential`: D183's. `pipeline`, from configuration: no serial sync, so a start waits only for its tasks' `after`; the next accelerator is configured while the previous one runs and started at once. `pipeline_compute`, from compute: consecutive starts with only loads to wait for, sharing no component, are gathered: their loads, then their configures, then the starts back to back. Under both, a store moves past the later starts of other accelerators, to before its own accelerator's next block or to the end, since its wait on its writer would otherwise hold the next start. Whatever the schedule, a group never runs two tasks at once (the program polls a busy component before starting it again), so a shared form's task list is the same under all three; the DMA's waits stay D135's.
- **Why:** mvt's halves share no data, yet D158 runs them one after the other. With c the core's configuration and r a run: serial $c_1 + r_1 + r_2$, pipeline $c_1 + \max(r_1, c_2 + r_2)$, pipeline_compute $c_1 + c_2 + \max(r_1, r_2)$, the last with both chains on the banks at once. The store's move keeps those true: left in place, the store's wait on the first half would come before the second half's start.
- **Where:** CONTRACTS, "Task list" (its "Derived" part) and "Estimate"; `lower/derive.py`, `estimate/schedule.py`.
- **Edge:** the estimate's `accelerator_cycles` becomes the longest path through the runs (their waits on data, their group and their accelerator), and C_acc with it (the static cost's control and D177's graph counts), still ideal (D155), with a note when overlapping accelerators move more words a cycle than there are banks. It holds no core configuration, so it gives both pipelined schedules the same cycles. The quiet build still waits on every start at once, so it runs a pipelined program serially (open item 140).

### D183 — The sequential schedule: one step at a time

- **Decided:** a fourth schedule, `sequential`, amending D179: every start is followed at once by its sync (an accelerator wait or a DMA wait), so each step, configures included, begins after the previous one has ended. The estimate adds up as under `serial`, the copies' cycles included.
- **Why:** a copy before its reader (D182) and an accelerator run without the copy are compared step by step only when nothing overlaps: each step's cycles are then its own, on SNAX-GVSoC and the RTL alike.
- **Where:** CONTRACTS, "Task list" (its "Derived" part); `lower/derive.py`'s `sequential`.
- **Edge:** the configures no longer hide under the previous run, so the window is longer than `serial`'s.

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

- **Decided:** per group, snax-gvsoc's group rule on ideal streams, groups adding through L1; unmapped work at one cycle per iteration, a labelled assumption in one constant; a copy (D182) at the DMA's measured rate, another constant; the total adds all three. The expected BUSY_COUNTER (cycles + fill) is a guide, not a check. `bank_bound`, ⌈Σ beats × num_ch / n_banks⌉, bounds the streams' L1 time, not the BUSY_COUNTER.
- **Why:** a quick number before any run. A run's BUSY_COUNTER counts from START until busy drops, so it includes bank conflicts and is shortened by the streamers' head start.
- **Where:** CONTRACTS, "Estimate".

### D154 · D172 — A first-order cost model on clock-gated blocks, before synthesis

- **Decided:** the cost of a design's groups comes from the description: a block costs its fires, each its comb energy and its register bits loading (e_load); a flop that holds costs nothing, as clock gating makes it; control is its bits × e_ctrl every cycle, e_ctrl measured on the clock-gated forge accelerators alone (5.45 fJ a bit and cycle on nangate45). The table is `snax_forge/char/<tech>.json`, snax-forge-experiments' clock-gated block table (its `char.json`, X37). A control unit's area is its flops' area plus an equal comb area.
- **Why:** an early estimate (about 30 % on the large costs), shown as such, with no synthesis in the loop. Clock-gated synthesis is the basis (snax-forge-experiments X79 · X87 · X88): a held bit costs about 0.07 fJ a cycle there against 7.1 fJ ungated, so the hold term moved dot 3's datapath by 0.004 pJ and is dropped. Control keeps one constant, so a group's control unit stays visible beside its datapath, the HLS comparison's basis.
- **Where:** CONTRACTS, "Cost model"; `snax_forge/char/README.md`.

### D176 — A block priced at its width pair

- **Decided:** the block table holds each block at its width pairs (snax-forge-experiments X89), and a unit is looked up at its pair: its widest input and its output, a SimdLanes' output at most its op's width (the low bits of a sum or product need only its operands' low bits, a wider output only copies the sign), a Mac's at its output, an unsigned type as the signed one of its width. Over lanes: `copy` from the pair's point, a narrow fold's `slope` (the int64 cost per lane × w_in / 64), a narrow LinkBuffer the int64 one × w / 64. A packed output's FifoPacker is looked up by its lane type and d_out, a sliced input's FifoSlicer by its lane type, lanes and d_in (D180), each point naming the bits that load per push or pop. A seeded fold (D163) is priced as its unseeded entry (`unseeded`): the same adder and register, its first beat's mux choosing the seed for the register instead of 0 for the sum, until a seeded point is measured. The table is keyed by the catalogue's names and full params.
- **Why:** synthesis removes the logic on copied sign bits, so a narrow block costs what it computes; one measured point per pair and one rule over lanes keep the table small and every block priced the same way.
- **Where:** CONTRACTS, "Cost model"; `snax_forge/char/README.md`.

### D167 — The cost in five parts

- **Decided:** a design's cost is reported in five parts (datapath, control, streamers, crossbar, memory), each with its share, beside the total and the total without the banks. Streamers, crossbar and banks are priced from snax-forge-experiments' clock-gated tables, copied as they are with the block table (`pixi run import-costs`) and interpolated, extrapolated with a note beyond them. A streamer is priced per beat and a crossbar access at one price per bank count, so the energy follows the accesses. The crossbar's area prices a reader's channels as read-only inputs and a writer's as write-only. The cores, I-cache, DMA engine and csr are not costed; a copy's L1 accesses are (D184). nangate45 only.
- **Why:** the parts a user can change are the parts the estimate shows.
- **Where:** CONTRACTS, "Cost model"; `snax_forge/char/tables.py`, `estimate/cost.py`.

### D177 — The dynamic energy: the model's counts at the static prices

- **Decided:** a run's dynamic energy is D167's five parts with SNAX-GVSoC's counts in place of the graph's, each priced at the instance's existing price (D154 · D172, D176), per window between two snapshots of the counters (core 0's writes to CSR `0x7C5`, the quiet build's marks, D135; the end of the simulation). A unit costs its firings × span, a buffer link and a packer their producer's pushes, a slicer its pops, the control units every group's busy-flag cycle (the static estimate's $C_{acc}$), a mover its crossbar requests over its channels, the crossbar every request, the banks the streamers' grants, and a copy's L1 reads and writes the banks and the crossbar (D184). A refused request costs a granted one's price; held and idle cycles cost nothing. A shared mover is priced at its own channels; each request at its mover's kind (D178).
- **Why:** the prices were measured at full rate, so a transaction is an active cycle there, and only granted ones were measured; with the counts the only change, the dynamic and static numbers differ only where the run differs from the graph (with the graph's counts they are equal). Aimed at trends over a sweep, not at an accurate number: the tables' random data stays.
- **Where:** CONTRACTS, "Profile and trace" (the energy run) and "Cost model" (the dynamic energy); `estimate/dynamic.py`, `run/snapshots.py`, `flow/energy.py`.

### D178 — The crossbar's energy by input kind

- **Decided:** a crossbar access is priced at a full crossbar's $e_{access}(b, b)$ less its input kind's saving at the design's bank count, $e_{access}\thinspace(1 - g_k(b))$: a streamer's read as a read-only input's, its write as a write-only input's (`xbar_kinds.json`'s `energy`, snax-forge-experiments X93). The static estimate prices the words so, the dynamic energy (D177) each mover's requests. Amends D167's one price per access.
- **Why:** the table's $e_{access}$ is a read-write input's (each input alternating reads and writes with random write data); a cluster's inputs are a reader's or a writer's channels, as the area already prices them (X71). The dot 3 cluster paid 0.67 to 0.81 of the read-write price; by kind its crossbar is 0.94 to 0.99 at W ≤ 8.
- **Where:** CONTRACTS, "Cost model"; `estimate/cost.py`, `char/tables.py`, `estimate/dynamic.py`.

### D184 — A copy is costed by its L1 accesses

- **Decided:** a copy (D182) costs what its L1 accesses cost a streamer's: per burst one bank read and one bank write, a burst shorter than a word priced as a word, and each a crossbar access, the read at a read-only input's price and the write at a write-only input's (D178). The static estimate prices the copies' bursts so, the dynamic energy (D177) the DMA's L1 → L1 reads and writes in its window. The DMA engine and the core that issues the copy stay uncosted (D167), and so do the loads and stores. Amends D167, D177 and D182's price of a copy.
- **Why:** a copy's traffic is memory and interconnect accesses, priced with the tables already there and nothing new to characterize; without it the copy cost nothing on SNAX-GVSoC. Loads and stores are the same for every form of one kernel, so they cancel when forms are compared.
- **Where:** CONTRACTS, "Cost model"; `estimate/__init__.py` (`copy_bursts`), `estimate/dynamic.py`, `run/snapshots.py`.
- **Edge:** a lower bound on the copy: the engine's work per burst and the core's issue loop are left out, which favours the copy against a form that reads A once. snax-forge-experiments X95 measures the copy on the gate-level cluster against it.

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
