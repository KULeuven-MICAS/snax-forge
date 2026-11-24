# SNAX-FORGE Decisions

Why SNAX-FORGE is built the way it is: one entry per live decision, in its current form. What the system is lives in `docs/ARCHITECTURE.md`, the artefacts' fields in `docs/CONTRACTS.md`, open work in `docs/STATUS.md`.

Rules (D117):

- An entry is a choice with a real alternative that someone may want to reverse: direction, ownership, the interfaces between components, the order of work. Implementation details live in the module's docstring and its tests.
- An entry says what was decided, why, and where the detail lives, in a few short bullets; an edge case only when there is one. A change is written into the entry, not chained after it.
- Entries merged into one keep all their numbers in its heading, so a cite of any of them still finds it. A number is never changed or reused; the text of a removed entry is in git history.
- A new decision gets the next free number, under the area it belongs to. Next free number: **D198**.

Numbers retired, never reused: D1–D7, D9–D11, D13–D18, D20–D25, D27, D29–D45, D47–D50, D53, D54, D56, D58–D63, D65–D67, D69, D74, D76–D78, D80, D81, D83, D88–D97, D100, D101, D103, D104, D106, D107, D109, D110, D114, D115, D119, D124, D127, D128, D132, D133, D138–D141, D151, D156, D159, D160, D181.

## Scope

### D51 · D52 — The user owns the accelerator; everything around it is the model

- **Decided:** the user describes the accelerator through its BRM (lanes, rates, latency, II, op); everything around it (cores, DMA, streamers, crossbar, memory) is the model of the SNAX platform. One accelerator can be swapped for its RTL in a cosim (M10); the whole cluster is checked by one program on SNAX-GVSoC and on snax_cluster's RTL (D123). Neither check blocks the inner loop.
- **Why:** the user knows their block; the platform is shared and must be right for everyone, trusted as far as the RTL comparison shows.
- **Where:** ARCHITECTURE section 4.

### D116 — SNAX-GVSoC is the model

- **Decided:** the model is SNAX-GVSoC, a GVSoC model of the SNAX cluster in which a Snitch core runs the real program.
- **Why:** control cost (the core configuring, starting and polling) is measured, not declared, and the model is checked against snax_cluster's RTL by one ELF on both (D123).
- **Where:** ARCHITECTURE, "SNAX-GVSoC"; the snax-gvsoc repository.

### D19 · D8 — Text artefacts, extended by registration

- **Decided:** every artefact is text that humans and LLMs read alike (the AI thinker is a commercial LLM). Everything is extended by registration (node kinds, transforms, checks, passes, `brm/function.py`'s accelerator kinds and ops, the generator's blocks) and namespaced attrs, never by editing core classes.
- **Why:** an LLM can read and write every step, and a new kind of block or transform touches no core code.
- **Where:** the registries in each package; CLAUDE.md's principles.

### D113 · D173 — Kernels: NPBench's structure, integers first

- **Decided:** kernels keep NPBench's DaCe structure, a dtype per container so one kernel may mix them (vsum: `int32` in, `int64` out), at PolyBench's MINI and SMALL sizes. `int64` through tier C; before the paper every kernel whose streams pack (tier done and untiered) also runs `int8` in, `int32` accumulate (as OpenGeMM), and `int16` and `int32` in, as type presets of one source (`--set types=i8|i16|i32`), exact products by its own cast (D143 · D175). Tier A and E7's points stay `int64`; floats later.
- **Why:** an integer check is exact; narrow data makes streams, crossbar and banks cost what a real design's would.
- **Where:** `kernels/README.md`, PAPER section 6.1.

### D111 · D130 — A bundle for snax_cluster, one wrapper contract

- **Decided:** per design point snax-forge writes a bundle for snax_cluster (the accelerator's RTL with its shell wrapper and file list, the cluster and streamer hjson, the program) and the same hjson and headers for SNAX-GVSoC. A generated accelerator carries its own CSR manager (ReqRspManager's protocol, snax_cluster's own off) and its CSRs per group. Python checks `accelerator.json` against the datapath catalogue; Scala only builds.
- **Why:** SNAX's generators build the rest of the cluster from one hjson, so snax_cluster sees only the bundle.
- **Where:** CONTRACTS, "Bundle" and "Accelerator RTL spec".

### D118 · D126 — One image, a pinned copy of snax_cluster's software

- **Decided:** one development image, `ghcr.io/kuleuven-micas/snax-forge:main` from `container/`, holds the tools for snax-forge, snax-gvsoc and the programs; each repository keeps its own environment. snax-gvsoc builds programs from a pinned copy of snax_cluster's software in `sw/`, refreshed by `make sync-snax-cluster` (its diff is the check); a design point's program takes the cluster headers snax-forge writes.
- **Why:** the same tools everywhere, and only the RTL needs a snax_cluster checkout.
- **Where:** `container/Dockerfile`; snax-gvsoc's `sw/README.md`.

### D98 · D171 — Tests check runs loosely; walk-throughs are of two kinds

- **Decided:** a test on a SNAX-GVSoC run checks what the program computes exactly (exit 0, no data error, every counter at least its beats; firings, beats, words, bytes) and timing loosely: each counter within 10 % of the estimate's expected BUSY_COUNTER beyond its fill (D155), no window pinned. Walk-throughs under `examples/dryrun/` are untested tutorials; the others test every number they state, cycles loosely. Runs are compared in the viewer, no diff view.
- **Why:** program or platform changes move cycles without breaking anything; exact numbers go to run notes and STATUS's "The local RTL check", and a counter beyond 10 % is investigated before any number changes.
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

- **Decided:** a map over `0:X` (X one symbol) is named `lower(X)` at import (rowsum's `m` and `n`), else it keeps DaCe's name. A split dimension's pieces are numbered outer to inner (`n_1`, `n_2`, `n_3`), a tag adds `_t`, `_s` or `_tile`. The name is derived from `loop.dim` and `loop.piece` after every step, never typed; map ids do not change.
- **Why:** a variable says which dimension it runs over and how, so `A[m, n]` reads as it should, and recipes keep working.
- **Where:** CONTRACTS, "SNAX-DFG" and "Recipe".

### D182 · D164 — A read across the loop order: a mark or a copy

- **Decided:** a read across the loop order (mvt's `A[m, n]` under loops (n, m)) is served by `mark_transpose`, an annotation on the reader (not a view container), LOWER deriving the strides; it refuses an element narrower than the bank word (D142 · D174 · D180). Or by `data_transpose`: a `transpose` node copies the container transposed into a new L1 container read in loop order, `by` the DMA (later the core), emitted as `transpose_<dst>` (`l1_to_l1`), one element a burst. The estimate prices it at the DMA's RTL rate, 8/3 cycles a burst (D155), its energy by its L1 accesses (D184). Rank 2 only.
- **Why:** the mark is what a recipe's match reads, drawn once with `Aᵀ` on the edge; int8 mvt's half 2 cannot be read by strides, and an explicit node keeps the reference check exact (D28 · D79).
- **Where:** CONTRACTS, "SNAX-DFG", "Recipe", "Task list" and "Estimate"; `sandbox/transpose.py`, `lower/derive.py`, `estimate/copies.py`.

### D28 · D79 — The reference executor is exact

- **Decided:** the reference executor runs any `.snaxdfg` in NumPy, accelerated nodes through their BRM's function, a slot's value from its params or its slot connector; every sandbox step and every run is compared with it. Reductions are exact: integers first, a float reduction in the accumulation order its BRM defines.
- **Why:** the check against the executor stays exact and DaCe's CPU reference is not needed; the check against NumPy is a tolerance.
- **Where:** `dfg/execute.py`.

## BRMs and blocks

### D68 · D70 · D129 — A BRM is a hand-written JSON file

- **Decided:** a BRM has a shared part (an interface with a rate per port and a `when` naming the design params a port exists at; function, dataflow, pattern) and implementations (source, supports, timing, binding); its timing is the user's numbers (D51). A port's dataflow is a nest in a registered notation: `affine` gives shape, offset and loops, `stream` only lanes and a rate. A `binding` names its datapath: one Chisel module, one `blocks.json` entry, one SNAX-GVSoC class.
- **Why:** reductions and SIMD blocks share one interface; a new block is one of each of the three, and the user's numbers stay theirs.
- **Where:** CONTRACTS, "Block runtime model" and "Accelerator RTL spec".

### D82 — A BRM says what one lane computes

- **Decided:** a BRM gives one lane's `function.code`, or a table of codes each fixing design params and naming scalar slots (`simd`'s per `op` and `streams`, with `k`, `k1`, `k2`), matched with `+`, `*`, `min` and `max` commuting. `bind` checks the tasklet, or a fused map's tasklets joined by a local value, against it; the first match gives the instance its params and slots. `unbind` and `join_map` take it back.
- **Why:** one block serves several ops, and binding is checked, not trusted.
- **Where:** CONTRACTS, "Block runtime model".

### D143 · D175 — Arithmetic follows NumPy; types follow the program

- **Decided:** each op runs at its operands' result type, signedness from each port's container, then extends or wraps into its target. A kernel's cast (`np.sum(A.astype(OP_TYPE) * B.astype(OP_TYPE))`) is a type: the importer folds DaCe's convert map into its consumer as a value cast setting `opType`, stripped at `bind`. A fold runs at NumPy's type (`np.sum` of a narrower int at `int64`, `@` and `np.dot` at the operands'), the same bits as the hardware's cast before the tree. A literal is weak, as NEP 50 says (relu's `k`). `bind` wants an output as wide as every input.
- **Why:** model, RTL and reference compute the same bits, and the program alone gives the types.
- **Where:** CONTRACTS, "SNAX-DFG" and "Block runtime model"; `brm/dtypes.py`; snax-gvsoc's `snax/snax_types.hpp`.

### D149 — Blocks with variants, not chains

- **Decided:** the MAC is one block with design-time variants (`accumulate`, or a·b + c with `regOut` for a later systolic PE), not vmul into a lane accumulator. The lane accumulator without an op is its own block, `LaneAccumulate`; the one-lane vsum is `SerialAccumulate` (`span` = lanes); a buffer is `LinkBuffer`, not a FIFO.
- **Why:** each block is the one the RTL has, so the cost of each is measured alone.
- **Where:** CONTRACTS, "Accelerator RTL spec"; `hw/chisel/`.

### D163 · D187 — A fold's seed is a port

- **Decided:** a fold's result read only by an elementwise nest of the same op (mvt's x₁ + A y₁) is folded in as the fold's seed, a per-job port read on a job's first firing, by `fuse_seed` (inverse `unfuse_seed`). A seed has its output's lanes: one for a lane fold (`TreeAccumulate`, `SerialAccumulate`), W for a `Mac` (BRM `mac`'s `seed` port at design param `seed` 1, first firing `seed + a·b`). Only these three take one; their unseeded RTL and the group's rule are unchanged.
- **Why:** no vecadd after the fold and no "init" port on every block; the column form (mvt's x₂ + Aᵀ y₂) needs no link and no second unit.
- **Where:** CONTRACTS, "Block runtime model" and "Recipe".

### D186 — A broadcast input: one lane given to all

- **Decided:** an input of a lane-wise datapath (`SimdLanes`, `Mac`) whose memlet lacks the spatial variable is read as one lane a beat and wired to all W lanes (`ShellWrapper`'s `<unit>_<port>_bcast`, `SnaxGroup` beside `SnaxSlicer`). `bind` derives it from the memlet and refuses it where lanes do not compute alone, or on a linked port (but a held one, D197), seed or output. `accelerator.json` marks `"broadcast": true`; channels and d_in follow from one lane (D180); kept for a row it is also a hold (D190). Priced as wires.
- **Why:** the column form reads y₂[i] once a beat for all lanes; a W-lane stream would cost W times the reads and a copy.
- **Where:** `sandbox/transforms.py`, `design/streamers.py` (`is_broadcast`); CONTRACTS, "Accelerator RTL spec".

## Sandbox and recipes

### D72 — Decisions are made by hand, as recipes

- **Decided:** decisions are made in SNAX-SANDBOX: registered transforms applied in order from a recipe, each step checked by the reference executor. A sweep is one recipe with a parameter; automated search writes recipes. A transform may give `site(g, selected)`, the test of where it applies, which the editor marks and lists from; one taking several items (`encapsulate`, `fuse_map`) has a site over the list.
- **Why:** DSE decides, LOWER derives: every decision is a readable, replayable step.
- **Where:** CONTRACTS, "Recipe"; `sandbox/`.

### D73 · D150 — The mapping is spelled step by step

- **Decided:** `split` only splits, with neutral names; `tag_map` tags `tile`, `temporal` or `spatial`; `split_reduce`, `fuse_map`, `fuse_seed`, `interchange_map` and `link` each do one thing, each with its inverse, one recipe per variant. `bind` reads design params off the graph (W from the spatial bound) and refuses with the fix ("fuse_map first"). `split_map` still splits and tags at once, ids `<id>` and `<id>_s`, for old recipes (dot 4–6, matvec 4–6) binding `mult_map_s`.
- **Why:** each choice is made once, in its own step, and undone the same way.
- **Where:** CONTRACTS, "Recipe".

### D161 — A scalar argument is read from a CSR, or baked

- **Decided:** a kernel's scalar argument binds to a slot read at run time from an R/W CSR, so one ELF serves any value (axpy's alpha as K1). How it is read is its own step, `bake_scalar` with a value (baked into the hardware) or `csr`; `bind` reads an unmarked scalar from a CSR.
- **Why:** both variants are recipes with the same steps, and the editor shows the choice.
- **Where:** CONTRACTS, "Block runtime model", "Recipe" and "Accelerator RTL spec".

### D165 — Two nests on one set of hardware

- **Decided:** two nests run on one set of hardware by binding them to one instance (mvt's variant 1: both halves on one `vmul`, `vsum` and `vadd`), chosen in `bind`'s form ("share with") or by `share` for a linked group. LOWER runs the group once per nest on the same streamers; the estimate counts the area once.
- **Why:** reuse of hardware is a mapping choice, not a new block.
- **Where:** CONTRACTS, "SNAX-DFG", "Recipe" and "Estimate".

### D166 — A recipe is the unit of reuse

- **Decided:** a recipe applies another on a region of its graph matching the other's kernel (`apply_recipe`), so matvec is dot's variant on each row and mvt is matvec's on each half. The user lifts a kernel's regions as blocks by hand (`lift`); no search applies recipes unasked. The match leaves the outer loops free.
- **Why:** a larger kernel is built from the smaller ones' recipes, and the user stays in charge of what is reused.
- **Where:** CONTRACTS, "Recipe"; `sandbox/blocks.py`.

### D188 — A recipe is named by version, and says what it builds

- **Decided:** a checked-in recipe's `name` (file stem, design) is `<kernel>_v<N>` plus `_<k>` per change from its parent (`mvt_v5_1`), type presets counted as the kernel: `v<N>` a new structure, `_<k>` one change (reuse, sharing, type, layout, size), never renumbered. Its `id` says what it builds in a fixed vocabulary (`mvt_dot2seed_colseed_i8_o32`), shared by recipes building the same units, its `description` the same in one to three sentences. `recipes/custom/` is free of the rule; Save writes neither field.
- **Why:** names grown round by round (`mvt_4_i8_col_vadd`) no longer said what a recipe builds; a version keeps names short and stable, the id carries the meaning.
- **Where:** `recipes/README.md`; CONTRACTS, "Recipe".

## Accelerators

### D112 · D145 · D146 · D148 — An accelerator is groups of linked units, described once

- **Decided:** a bound graph nests accelerator ⊃ groups ⊃ units and links. A group is a stand-alone accelerator with its own START, JOBS (one per chain, D189), STATUS, BUSY_COUNTER and control unit, no ERROR CSR (an empty run still ends in a done); links of a depth (0 a wire, ≥ 1 a buffer of that many beats) join its units into chains, streamers on outer ports only. A unit fires JOBS · ∏ T over itself and the folds downstream. `link` merges groups, `encapsulate` puts groups in one accelerator. One `accelerator.json` feeds Emit, the catalogue check and SNAX-GVSoC. Links between accelerators are parked (open item 53).
- **Why:** a composition is one accelerator with one register window and wrapper, groups run apart, and model and RTL cannot be configured apart.
- **Where:** CONTRACTS, "SNAX-DFG" and "Accelerator RTL spec".

### D147 · D152 · D153 · D168 — Datapath, control and CSRs each hold their own

- **Decided:** every block is a datapath of data and enables only, under one generated control unit per group. `<tag>_csr` holds what the core writes (settings, START) and reads (STATUS), each CSR at platform width, a write to a busy group's settings held until idle. `<tag>_ctrl` holds what a firing changes (busy, counters, the II gap) and every firing test, reading settings live. `<tag>_peripherals` holds BUSY_COUNTER, left out of every figure. Programming the next task during a run is open item 123.
- **Why:** control is written once for model and RTL, and `<tag>_ctrl` plus `<tag>_datapath` compare with a whole HLS block; SnaxGroupRule's flat count of firings stays the spec.
- **Where:** CONTRACTS, "Accelerator RTL spec"; `GroupRule.scala`.

### D189 — Chains share a read: one reader, one group

- **Decided:** chains whose heads read one container with the same beats in the same order share one reader by `fork_read`: a beat is taken only when every head fires (readies ANDed). They form one group, done when every chain is, each chain with its own JOBS (`JOBS_<SINK>`) and output. LOWER refuses a fork whose heads fire unlike; a forked unit of span > 1 and a fork beside `share_ports` are refused for now. A branch to a unit down a chain goes through a buffer (D196).
- **Why:** one pass over A (bicg, mvt, later atax) reads it once for both halves; results come at their own rate (q a row, s once), and a shared beat ties the chains to one schedule.
- **Where:** CONTRACTS, "SNAX-DFG", "Recipe", "Task list" and "Accelerator RTL spec"; snax-gvsoc's `snax/snax_group_rule.py`.

### D190 — A held input: read once a row, kept for the row

- **Decided:** a banked unit's input whose memlet uses none of the bank's row loops (bicg's r[i] across s's row) is read once every R firings and kept in a register of its element's width: its streamer skips the row loops, taken on a row's first firing when `row` wraps, so no CSR. `bind` derives it from the memlet, as the broadcast (D186); `accelerator.json` marks `"hold": true`. Only a broadcast input of a banked `Mac` for now; it may come over a link instead of a streamer (D197).
- **Why:** r[i] is the same for the N/W beats of row i; read every beat it costs N/W times the reads and the slicer's pops.
- **Where:** `design/streamers.py` (`is_held`); CONTRACTS, "Accelerator RTL spec"; snax-gvsoc's `snax/snax_group_rule.py`.

### D191 — A fold's unfinished outputs stay in a bank inside the unit

- **Decided:** a `Mac` may keep R rows of W sums (`rows` R, default 1), one row a firing in turn: a job of T firings starts each row on its first R firings and gives one result on each of its last R. `bind` offers it where the output's index changes along loops inside its sum loop. Rows are registers; a `ROWS` CSR (at most R) sets the rows a pass uses; the control gives the `Mac` its `row` and `drain` pins. An output wider than R·W is cut into strips, r held (D190), q carried (D192), N a multiple of R·W for now. Seeded: D194; price: D193; a bank as memory is open item 158.
- **Why:** bicg's one pass reads A along its rows, so s's N sums are unfinished until the last row; keeping them costs flops, not R datapaths.
- **Where:** CONTRACTS, "Block runtime model" and "Accelerator RTL spec"; `brm/function.py`, `design/streamers.py` (`bank_loops`); snax-gvsoc's `snax/snax_group_rule.py`.

### D193 — A bank's price: one row loads a fire, its muxes by X97's forms

- **Decided:** a `Mac` bank of R rows is priced from the rows-1 entry plus the `bank` forms in `char.json` (X97): comb area plus 59.8 µm² per lane and row, a fire's comb energy plus W · (0.343 log₂ R + 0.316) pJ, a fire loading one row of W · 32 bits; flops are the catalogue's. Past the largest fitted read mux (2 048 bits) the lookup says `bank_beyond` (DC's mux is about 2× at 4 096 bits). Left out: held registers, gates' idle energy (R × 4.9 fJ a cycle).
- **Why:** rows-1 comb with every row loading gave 0.76 to 0.87 of DC's area and 1.05 to 3.37 times the energy; the forms give 0.99 to 1.01 and 0.90 to 1.06. The numbers live in the table.
- **Where:** `char/__init__.py` (`_banked`); CONTRACTS, "Cost model".

### D194 — A seeded bank starts each row from its seed

- **Decided:** a `Mac` bank may be seeded (D187): its `seed` port moves a beat a row, read on a job's first R firings (`first`), so row ρ starts as seed + a·b. The seed is neither held (D190) nor carried (D192): its streamer runs its memlet's loops. One W-lane mux before the adders serves every row; priced as D193's bank, seeded as unseeded (D187). One strip (N = R·W) is built; strips would carry x1's tree from L1 (D192, no mask).
- **Why:** mvt's one pass (x2 += Aᵀ y2) is bicg's with a start value; seeding keeps x2's sums in the unit, without reading x2 back for a `vadd`.
- **Where:** `brm/function.py`, `lower/derive.py`, `gen/rtl.py`, `datapath/Mac.scala`, `ctrl/GroupRule.scala`; snax-gvsoc's `snax/`; CONTRACTS, "Accelerator RTL spec".

### D192 — A fold carried across strips through L1

- **Decided:** a fold whose output is revisited by loops outside the ones it moves with (q's rows, once per strip) is carried: each visit starts from what the last wrote, the first from the identity. `carry_seed` (inverse `uncarry_seed`) sets `"carry": true`, and `bind` gives the unit its output as `seed` port (D163). The seed reader runs every job; a `CARRY` CSR, derived by LOWER, holds the jobs before the first revisit, until which the wrapper gives 0 instead. Only a fold that adds.
- **Why:** strips outermost let the row form read A as the bank's column form does (D191), for any N; masking strip 0 costs one comparison, not a task.
- **Where:** CONTRACTS, "Recipe", "Task list" and "Accelerator RTL spec"; `sandbox/seed.py`, `lower/derive.py` (`carry_jobs`).

### D196 — A fork's branch down a chain goes through a buffer

- **Decided:** `link` on a forked container buffers the branch to the one reader down a chain (atax's `Mac` behind vmul → vsum): a ring `LinkBuffer` of depth D, recorded on the fork node (`buffers`) and in `accelerator.json` (`fork_links`). A beat is taken when the other heads fire and the buffer has room; the reader pops one a firing. LOWER refuses D < T, the folds' T between the fork and the reader's held input (N/W for atax), which never ends; T ≤ D < T + L stalls, which the estimate shows. A push loads one entry of D.
- **Why:** atax's one pass reads A once for t = A x and y = Aᵀ t, but row i's `Mac` firings need t[i], which comes T beats after row i's first beat; the buffer holds those beats instead of a second read. A depth beside the fork node keeps `fork_read` as it is (Ryan, 24 Nov).
- **Where:** `sandbox/compose.py` (`link`), `lower/tasks.py`, `estimate/cycles.py`; CONTRACTS, "SNAX-DFG", "Accelerator RTL spec" and "Cost model"; snax-gvsoc's `snax/snax_group_rule.py`.

### D197 — A held input may come over a link

- **Decided:** a banked unit's held, broadcast input (D190, D186) may be fed by a link instead of a streamer: its beat is taken on a row's first firing, so `link` counts it once a row (M beats against the writer's M) and the writer's T need not be the bank's. JOBS is the last unit's firings over its T. Link depth 1; t lives in no memory.
- **Why:** atax's t[i] is made by the vsum one a row and used by the bank for the row; through L1 it would cost a pass and the per-row handshake of open item 53.
- **Where:** `sandbox/compose.py` (`beats`), `gen/accelerator.py`, `lower/c.py` (`job_writes`); CONTRACTS, "Accelerator RTL spec"; snax-gvsoc's `snax/snax_group_rule.py`.

### D157 — A design's accelerators are streamer sets of one entry

- **Decided:** a design's accelerators are streamer sets in one snax_cluster entry, `snax_forge_<design>`, each with its own streamer, START and CSRs behind `snax_csr_range_demux`.
- **Why:** not entries behind `snax_acc_mux_demux`, whose 0x7C6 select can misroute a delayed write, nor address windows (open item 108).
- **Where:** CONTRACTS, "Generated hjson files" and "Accelerator RTL spec".

## Design

### D84 · D131 · D170 — The platform is its own file

- **Decided:** the platform is a file in `platforms/`, paired with the bound graph by the design step; cluster parameters and FIFO depths are design knobs, temporal loops come from the graph. The base cluster, `platforms/snax_forge_cluster.hjson` (`design from-hjson`), has one accelerator entry, `<accelerator_slot>_<design>`, and one Snitch core driving DMA and accelerator; more cores still lower.
- **Why:** graph or platform changes alone. A second core only idled at barriers, still clocked; with one, only core 0's `obs_o` matters.
- **Where:** CONTRACTS, "Platform".

### D85 — Design checks name every problem and its fix

- **Decided:** the design checks report every problem, each with its fix, before anything is written. A limit of the model's cluster is a platform field the checks hold the design to, such as the TCDM crossbar's inputs (`l1.max_xbar_inputs`, SNAX-GVSoC's 128), with a note above `l1.warn_xbar_inputs` (64).
- **Why:** a design is refused before a run would assert, and the note marks costs extrapolated beyond the tables.
- **Where:** CONTRACTS, "Platform" and "Design checks".

### D12 · D142 · D169 · D174 · D180 — Streamers: one per port, packed and sliced words, shared movers

- **Decided:** one streamer per accelerator port; a bank word packs num_slc = 64 / b elements, little-endian. An input with W a multiple of num_slc is wire-sliced on num_ch = W / num_slc channels; with W dividing num_slc a slicer takes a word on num_ch = 1 and gives d_in = num_slc / W pops, only while its group is busy, cleared at start. An output is wired, or packed: d_out = num_slc / W results a word, sent when full or on the last push. Other W are refused with the fix; with an outer loop, d_in or d_out divides the innermost count. The flow places both. `share_ports` (off by default, sets in series, D158) puts a writer and reader on one set by `PORT_MUX_SEL`; a streamer starts after its group is idle (D179).
- **Why:** the streamer matches snax_cluster's, the slicer is safe in any start order, and shared movers save crossbar inputs.
- **Where:** `design/streamers.py`; `FifoSlicer`, `FifoPacker`, `SnaxSlicer`, `SnaxPacker`; CONTRACTS, "Platform" and "Accelerator RTL spec".

### D86 · D105 · D144 — Memory: planned by passes, chained through L1, padded in the data

- **Decided:** the memory plan comes from registered residency, layout and placement passes, contiguous by default, with pins; `padded_rows` avoids bank conflicts, set by `--set`, never a recipe. A reader in L1 waits on its container's writer. Padding is in the data: a split dimension pads to a multiple of W, a container to whole DMA beats, with zero or the fold's identity. A fold reads a tail another unit wrote only if the producer yields the identity on pads (dot 1 at W_mul = W_sum = 10), else `connect.pad` refuses (open item 106).
- **Why:** each pass is one choice; the only wait between accelerators is a real dependence; masks would cost hardware in every block.
- **Where:** CONTRACTS, "Memory plan and design point"; `lower/layout.py`, `dfg/pad.py`.

### D87 — The design point is LOWER's only input

- **Decided:** `design_point.json` holds the bound graph, the platform, the resolved streamers and the memory plan; it is checked again on load and is SNAX-LOWER's and the generator's only input.
- **Why:** what was decided is one file, and LOWER derives without deciding.
- **Where:** CONTRACTS, "Memory plan and design point".

## Lower and program

### D64 · D75 · D158 · D179 · D183 — The task list is derived; its schedule says what overlaps

- **Decided:** the task list is derived: per accelerated node `configure`, `start` and `sync`, loads and stores from the memory plan, `after` for data dependences only, one wait per component, names derived. Schedules (`--set design.schedule=`): `serial` (default) syncs on the previous accelerator before each start, configures before that sync; `sequential` syncs right after every start; `pipeline` drops the serial sync; `pipeline_compute` also gathers starts waiting only on loads; a store moves past later starts. A group never runs two tasks at once. The estimate takes the longest path through the runs (D155), control per own group (D154 · D172); the quiet build runs pipelines serially (open item 140).
- **Why:** lowering adds only needed waits. Serial $c_1 + r_1 + r_2$, pipeline $c_1 + \max(r_1, c_2 + r_2)$, pipeline_compute $c_1 + c_2 + \max(r_1, r_2)$; sequential gives each step its own cycles, to compare a copy (D182).
- **Where:** CONTRACTS, "Task list" and "Estimate"; `lower/derive.py`, `estimate/schedule.py`.

### D135 — The task list becomes C on snRuntime

- **Decided:** one program per design point: each step on the core owning its component, D64's waits as polls, the DMA waited on for dependences only, two `mcycle` reads bounding the window. The quiet build (`-DFORGE_QUIET`, `forge_quiet.elf`) waits on every start, marks each section in CSR `0x7c5` (core 0's `obs_o`) and prints nothing, in preprocessor blocks, so the default build is unchanged.
- **Why:** RTL and model run the same program; the quiet sections serve power.
- **Where:** CONTRACTS, "Generated hjson files"; `lower/c.py`.

## SNAX-GVSoC

### D120 · D121 · D122 · D125 — SNAX-GVSoC follows the RTL, by measurement

- **Decided:** the Snitch core sends CSRs 0x3C0 to 0x5FF as 32-bit IO requests to an accelerator component per core, so a SNAX block is a plain IO slave. GVSoC's iDMA model is kept and fitted to snax_cluster's iDMA RTL by measurement. The crossbar takes the SNAX ports, cores and AXI, numbered as on the RTL, and copies its sparse arbiter, checked in Verilator. Loads follow the RTL: one integer load in flight. Fetch keeps GVSoC's cache (open item 65).
- **Why:** reuse GVSoC's models where they fit and copy the RTL where cycles depend on it, each checked.
- **Where:** ARCHITECTURE, "SNAX-GVSoC"; snax-gvsoc's `docs/notes/NOTES.md`.

### D134 — Three files are the contract with SNAX-GVSoC

- **Decided:** per design point, the cluster hjson, the streamer hjson and `accelerator.json` are read by SNAX-GVSoC at run time (`snax_cfg.py`); snax-gvsoc's `forge/` fixtures are those files, checked in.
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

### D195 — Fixtures and check sets by feature

- **Decided:** snax-gvsoc's `forge/` holds one folder per integration feature (groups, links, fork, bank, hold, carry, seeds, broadcast, slicer, packer, schedules, and more), each at its smallest showing size (`FORGE_FIXTURES`); types and ops go to the datapath testbench. RTL fixtures and recorded runs use forge/'s designs, a recorded run keeping GVSoC's outputs and its inputs' sha256. `forge-set quick` is about five points, one per feature group, `full` adds one point each at i8, i16 and i32; a task's own set lives only while it is open.
- **Why:** a folder per type and op grew the sets without testing more integration; the datapath vectors check each bit for bit.
- **Where:** snax-gvsoc's `forge/README.md`; `snax_forge/flow/forge_set.py`.

## Flow and estimate

### D136 · D108 — The flow runs the loop once and decides nothing

- **Decided:** `pixi run flow` runs the inner loop once on SNAX-GVSoC: lower, gen, build when needed, sw, gvsoc, check (the program's exit code, `total_cycles` its window), into a folder named after every `--set`, with no backend flag. It also starts from a bound `.snaxdfg`, the suffix telling the forms apart.
- **Why:** one command from recipe to checked run; a hand-edited graph can be run and checked, not replayed or swept (open item 34).
- **Where:** `flow/run.py`; CONTRACTS, "Profile and trace".

### D155 — An ideal estimate, and what it is not

- **Decided:** per group, snax-gvsoc's group rule on ideal streams, groups adding through L1; unmapped work at one cycle per iteration and a copy (D182) at the DMA's measured rate, each a labelled constant; the total adds all three. The expected BUSY_COUNTER (cycles + fill) is a guide, not a check. `bank_bound`, ⌈Σ beats × num_ch / n_banks⌉, bounds the streams' L1 time.
- **Why:** a quick number before any run. A run's BUSY_COUNTER includes bank conflicts and is shortened by the streamers' head start.
- **Where:** CONTRACTS, "Estimate".

### D154 · D172 — A first-order cost model on clock-gated blocks, before synthesis

- **Decided:** a block costs its fires, each its comb energy and its register bits loading (e_load); a holding flop costs nothing, as clock gating makes it. Control is its bits × e_ctrl every cycle its own group is busy, e_ctrl measured on the clock-gated forge accelerators (5.45 fJ a bit and cycle on nangate45); its area is its flops' plus an equal comb area. The table is `snax_forge/char/<tech>.json`, snax-forge-experiments' clock-gated `char.json` (X37).
- **Why:** an early estimate (about 30 % on the large costs), no synthesis in the loop. Clock-gated (X79 · X87 · X88), a held bit costs about 0.07 fJ a cycle against 7.1 fJ ungated, so the hold term (0.004 pJ on dot 3) is dropped.
- **Where:** CONTRACTS, "Cost model"; `snax_forge/char/README.md`.

### D176 — A block priced at its width pair

- **Decided:** the block table holds each block at its width pairs (X89); a unit is looked up at its widest input and its output (a SimdLanes' at most its op's width, a Mac's at its output, unsigned as signed). Over lanes: `copy` from the pair's point, a narrow fold's `slope` (int64 cost per lane × w_in / 64), a narrow LinkBuffer the int64 one × w / 64. A FifoPacker is keyed by lane type and d_out, a FifoSlicer by lane type, lanes and d_in (D180). A seeded fold (D163) costs its `unseeded` entry until measured. A SimdLanes or Mac whose inputs differ in width is looked up at its own pair first where measured (int8 × int32, X98), in either order, else at its widest input.
- **Why:** synthesis removes logic on copied sign bits, so a narrow block costs what it computes; one point per pair and one rule over lanes keep the table small.
- **Where:** CONTRACTS, "Cost model"; `snax_forge/char/README.md`.

### D167 · D178 — The cost in five parts, the crossbar by input kind

- **Decided:** a design's cost is reported in five parts (datapath, control, streamers, crossbar, memory), with shares, the total and the total without banks. Streamers, crossbar and banks use snax-forge-experiments' clock-gated tables (`pixi run import-costs`), extrapolated with a note. A reader's crossbar inputs are read-only, a writer's write-only, an access $e_{access}\thinspace(1 - g_k(b))$, the full $e_{access}(b, b)$ less its kind's saving (`xbar_kinds.json`, X93). Cores, I-cache, DMA engine and csr are not costed; a copy's L1 accesses are (D184). nangate45 only.
- **Why:** the estimate shows the parts a user can change. The table's $e_{access}$ is a read-write input's (X71): dot 3 paid 0.67 to 0.81 of it, by kind 0.94 to 0.99 at W ≤ 8.
- **Where:** CONTRACTS, "Cost model"; `char/tables.py`, `estimate/cost.py`.

### D177 — The dynamic energy: the model's counts at the static prices

- **Decided:** a run's dynamic energy is D167's five parts with SNAX-GVSoC's counts at the static prices (D154 · D172, D176), per window between counter snapshots (core 0's writes to CSR `0x7C5`, D135; the end). A unit costs firings × span, a link or packer its producer's pushes, a slicer its pops, control its group's busy cycles, a mover its requests by kind (D178), banks the grants, a copy its L1 accesses (D184); a refused request costs a granted one's. A pipelined run (D179) is one window of the default build.
- **Why:** prices are full-rate granted requests, so static and dynamic differ only where the run differs from the graph; trends, not accuracy.
- **Where:** CONTRACTS, "Cost model"; `estimate/dynamic.py`, `run/snapshots.py`, `flow/energy.py`.

### D184 — A copy is costed by its L1 accesses

- **Decided:** a copy (D182) costs its L1 accesses: a DMA beat is one access on its superbank's port, touching all its banks, so a one-element burst is 8 bank reads and 8 bank writes plus one read-only and one write-only crossbar access (D178), in the static estimate and in the dynamic energy (D177). The DMA engine, core, loads and stores stay uncosted (D167): a lower bound; X95 measures the copy at gate level, X96 its 8 accesses each way a burst. Amends D167, D177 and D182.
- **Why:** a copy's traffic is memory and interconnect accesses, priced from existing tables.
- **Where:** CONTRACTS, "Cost model"; `estimate/__init__.py` (`copy_bursts`), `estimate/dynamic.py`; snax-gvsoc's `snax_tcdm_xbar.cpp`.

## Viewers, reports and docs

### D55 · D57 · D99 — Local viewers that only draw; reports for people and LLMs

- **Decided:** the viewers are a local server on 127.0.0.1 plus a static page: stdlib only, no build step, offline, every POST `application/json`, one vendored library for the DFG layout (D185). `view` shows runs, `view-dfg` `.snaxdfg` files, `edit-dfg` a session saved as a new recipe in `recipes/custom/`. Python answers every fold, lookup and step; the page only draws, its state in the URL hash. Reports are Markdown from dataclasses in HLS reports' layout, every number from the design point, profile or trace.
- **Why:** nothing to install, a view never disagrees with a report, and LLMs read the reports.
- **Where:** `viz/`, `report/`; CONTRACTS, "Recipe".

### D185 — The DFG page: a flat graph laid out by ELK

- **Decided:** the DFG page draws a step as a flat graph: loops as a chain of boxes, edges through the loops their subsets use, hardware keyed by what it is (`unit:<instance>`, `link:<from port>><to port>`, `slicer:`/`packer:<instance>.<port>`) and drawn once, a temporal mux and demux per task after `share`, each group one box. Python decides every box, port, edge, text and id (`viz/dfg/drawing.py`); the page measures boxes and lays them out with ELK (elkjs, vendored, in its worker), edges at right angles.
- **Why:** nested frames overlapped once links, slicers and packers sat together; only the browser knows text sizes.
- **Where:** CONTRACTS, "Recipe"; `viz/dfg/drawing.py`, `viz/static/graph.js`.

### D26 · D46 — Contracts are dataclasses, written down and checked

- **Decided:** contracts are Python dataclasses with `to_dict` / `from_dict` and plain JSON until the contract freeze (M6), versioned schemas after. They are written in `docs/CONTRACTS.md`, and every snippet in it is checked against its source.
- **Why:** one source for each artefact's form, and a document that cannot drift from the code.
- **Where:** CONTRACTS, "Conventions"; `tests/test_contracts.py`.

### D117 — The documents stay short

- **Decided:** one fact in one place; a decision only for a choice someone may want to reverse; done tasks out of STATUS at each milestone close; prose one line per paragraph, so a diff shows only what changed.
- **Why:** people and LLMs read the documents before every change; every extra line is read again.
- **Where:** the rules at the top of this file; `util/md_reflow.py`.
