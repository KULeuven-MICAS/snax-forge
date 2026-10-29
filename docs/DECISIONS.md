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
- An entry says what was decided and why, and where the detail lives, in at most three lines. An
  amendment is folded into the entry it changes, not chained after it.
- A replaced or merged entry is removed, and its number goes under Retired with the entry that
  holds it now. A number is never changed or reused, since code, tests and docs cite them; the
  text of every retired entry is in git history (tag `pre-gvsoc`; RET1's at `cad6619`, CLN1's at
  `6079d24`).
- Closing a milestone compacts this file and removes the done tasks from STATUS.
- A new decision gets the next number at the end of this file, with its area tag. Next free
  number: **D151**.

Area tags: `scope` (plan, ownership, order), `lower` (SNAX-LOWER), `gvsoc` (SNAX-GVSoC), `brm`
(SNAX-BRM), `dfg` (SNAX-DFG and importers), `ref` (reference executor), `sandbox`, `design`
(platform, checks, memory plan, design point), `flow`, `viz`, `docs`.

| Area | Decisions |
|---|---|
| scope | D8 D23 D51 D52 D91 D98 D100 D111–D113 D116 D118 D123 D126 D130 D141 D145 D146 |
| lower | D64 D75 D105 D135 D144 |
| gvsoc | D120–D122 D125 D134 D137 |
| brm | D68 D70 D82 D103 D129 D133 D143 D147 D149 |
| dfg | D19 D71 D102 D148 D150 |
| ref | D28 D79 |
| sandbox | D72 D73 |
| design | D12 D84–D87 D131 D142 |
| flow | D50 D108 D136 |
| viz | D55 D57 D99 |
| docs | D26 D46 D117 |

Retired, and where each now lives:

| Now in | Retired |
|---|---|
| D19 | D43 |
| D51 | D22 |
| D55 | D16 D54 D76 D81 |
| D57 | D56 D58 D60 D61 D96 D97 |
| D64 | D63 D66 D89 |
| D68 | D3 D5 D25 |
| D70 | D15 D69 |
| D71 | D1 D2 D77 D78 |
| D72 | D14 D27 D80 |
| D79 | D17 D20 |
| D84 | D7 |
| D87 | D74 |
| D99 | D106 |
| D100 | D109 |
| D102 | D104 |
| D105 | D101 |
| D111 | D18 |
| D113 | D9 |
| D116 | D4 D6 D10 D11 D21 D24 D29 D30 D31 D32 D33 D34 D36 D37 D40 D41 D42 D44 D47 D48 D49 D53 D62 D65 D67 D83 D88 D92 D95 D110 D114 D115 D119 D127 |
| D117 | D93 |
| D123 | D124 D140 |
| D129 | D35 |
| D130 | D138 D139 |
| D134 | D128 D132 |
| D135 | D45 |
| D136 | D90 D94 D107 |
| D137 | D38 D39 D59 |
| D142 | D13 |

---

**D8** · scope — The AI thinker is a commercial LLM, so every artefact is text that humans and LLMs
read alike.

**D12** · design — One streamer per accelerator port, named after it (D75); its width `n_ports` is
the port's lanes W, and the hjson's `num_channel` is its num_ch (D142); the two are equal only at
num_slc = 1. → `design/streamers.py`, C§13.

**D19** · dfg — Everything is extended by registration (node kinds, transforms, checks, passes, the
accelerator kinds and ops of `brm/function.py`, the generator's blocks) and namespaced attrs, never
by editing core classes.

**D23** · scope — SNAX-FORGE complements ZigZag and Stream: cycle-level and RTL-backed rather than
analytical. Parked: Stream as a proposer for DNN-shaped workloads, and an analytical estimator of
our own for the kernels Stream cannot describe (open item 4).

**D26** · docs — Contracts are Python dataclasses with `to_dict` / `from_dict` and plain JSON until
the contract freeze (M6), versioned schemas after. → C§1.

**D28** · ref — Reductions are exact: integers first, and a float reduction follows the accumulation
order its BRM defines, so the check against the reference executor stays exact and the check against
NumPy is a tolerance (D113).

**D46** · docs — The contracts are written in `docs/CONTRACTS.md`, and every snippet in it is
checked against its source (`tests/test_contracts.py`).

**D50** · flow — A run's `run.json` records what ran (the design point, the hjson files, the ELF,
the snax-gvsoc commit), so a run directory says on its own which hardware produced it. → C§7.

**D51** · scope — The user owns the accelerator, through its BRM (lanes, rates, latency, II, op);
everything around it is the model of the SNAX platform. Model cycles are trusted only as far as the
comparison with SNAX RTL shows (M16, D116).

**D52** · scope — Cosim (M10) swaps one accelerator for its RTL and checks its output, latency and
II; the whole cluster is checked by running the same program on SNAX-GVSoC and on snax_cluster RTL
(D123). Neither blocks the inner loop.

**D55** · viz — The viewers (runs, and graphs as a second mode) are a local server bound to
127.0.0.1 plus a static page: stdlib only, no build step, no library, offline, for humans; LLMs read
the reports (D99). Views read the dataclasses, not raw JSON. → `viz/`.

**D57** · viz — The run views (a schedule over the program's window, the cluster at the selected
cycle, the memory tab, data movement off the beat trace) live in the URL hash; Python answers every
fold, lookup and step and `static/` only draws, so a view never disagrees with a report.

**D64** · lower — The task list is derived from a design point: per accelerated node `configure`,
`start`, `sync` and `read` steps, loads and stores from the memory plan, `after` for data
dependences only; lowering adds only the waits correctness needs, one per component. → C§9.

**D68** · brm — A BRM is a hand-written JSON file: a shared part (an interface with a rate per port,
so reductions and elementwise blocks share one; function, dataflow, pattern) and implementations
(source, supports, timing, binding D129), its timing the user's numbers (D51). → C§10.

**D70** · brm — A port's dataflow is a nest in a registered notation. `affine` gives the operand's
shape, an offset and loops (spatial last, an innermost stride 0 repeating a beat); `stream`, for
unit BRMs, gives only lanes and a rate, the bound memlet giving the order. → C§10.

**D71** · dfg — `.snaxdfg` is SNAX-DFG's own JSON format: SDFG's concepts, one expression grammar
(`expr.py`) shared with the BRM, MatMul and Transpose first expanded into dot's shape (`sdfg/expand.py`).
Importers derive, decide nothing and name what they lack; streamers appear only at lowering. → C§11.

**D72** · sandbox — Decisions are made by hand in SNAX-SANDBOX: registered transforms on a
`.snaxdfg`, applied in order from a recipe (kernel, params, symbols, steps), each step checked by
the reference executor. A sweep is one recipe with a parameter; automated search writes recipes.

**D73** · sandbox — Loops are tagged `tile`, `temporal` or `spatial`, by `split_map` or `tag_map`,
and `bind` reads a BRM's design params off the graph (W from the spatial bound), so each is decided
once. Streamer values come from memlets through the containers' layouts.

**D75** · lower — Names are derived, never stated: streamer `<instance>_<port>`, task
`<node>_<component>`, DMA task `load_<container>` or `store_<container>`.

**D79** · ref — The reference executor runs any `.snaxdfg` in NumPy, accelerated nodes through their
BRM's function; every sandbox step and every run is compared with it, so DaCe's CPU reference is not
needed. → `dfg/execute.py`.

**D82** · brm — A BRM says what one lane computes (`function.code`), and `bind` checks the tasklet,
or a fused map's tasklets joined by a local value, against it. A bound graph records what it
replaced, so `unbind` and `join_map` take it back.

**D84** · design — The platform is its own file in `platforms/`, paired with the bound graph by the
design step, so either changes alone; its cluster parameters and the streamers' FIFO depths are
design knobs, never a template (D134), and the streamers' temporal loops come from the graph. → C§13.

**D85** · design — Design checks report every problem before anything is written, each with its fix.
→ C§14.

**D86** · design — The memory plan comes from registered residency, layout and placement passes,
contiguous by default, with pins, seeing each port's element-index stream to avoid bank conflicts
without a run; the L2 base is the program's link (its ELF's symbols, D50). → C§15.

**D87** · design — `design_point.json` holds the bound graph, the platform, the resolved streamers
and the memory plan; it is checked again on load and is SNAX-LOWER's and the generator's only input.
→ C§15.

**D91** · scope — The old SDFG → descriptor → RTL path is gone; the Chisel blocks and `Emit` are a
project of their own in `hw/chisel/`.

**D98** · scope — No run diff view: runs are compared by switching in the viewer and reading their
reports, until sweeps need one. Walk-throughs live in `examples/`, with a test checking every number
they state.

**D99** · viz — Design and run reports in the layout of HLS reports, Markdown from dataclasses, for
humans and LLMs alike. Every number comes from the design point, profile or trace, nothing grows
with cycles, and a wait is named by what it was for. → `report/`.

**D100** · scope — SNAX-FORGE is the subject of a paper, with no fixed deadline: a plan (P1) runs
alongside development, a draft (P2) follows the evidence. The evaluation climbs NPBench from kernels
to layers to blocks and models (ResNet, Transformer, NVSA). → `docs/PAPER.md`.

**D102** · dfg — A reduction is a map whose tasklet writes through a memlet with a `reduce` (an
associative op and its identity), so import, split and bind treat it as any map; bind folds every
lane of an output without the spatial variable into one, its rate from `named_rates`. → C§11.

**D103** · brm — Timing has a third number, `drain`: the cycles from a push in which no input is
taken. Port names are not Python keywords and do not start with `_`, since they appear in `code` and
in streamer names. → C§10.

**D105** · lower — Accelerators chain through L1: the reader waits on the writer that put its
container there, the only wait between them. A container the DMA moves is padded (its size and
value: D144), and the padding is kept free in both memories. → `lower/layout.py`.

**D108** · flow — The flow also starts from a bound `.snaxdfg`, the suffix telling the two forms
apart. That is how a hand-edited graph is run and checked, though it cannot be replayed or swept
(open item 34).

**D111** · scope — Per design point SNAX-FORGE writes a bundle for snax_cluster (the RTL with its
shell wrapper and file list, the hjson of D134, the program) and the same hjson and headers for
SNAX-GVSoC; SNAX's own generators build the rest, so the hjson describes the cluster once. → C§18.

**D112** · scope — A composition is one accelerator: a group's units joined by links of a depth (0 a
wire, ≥ 1 a buffer of that many beats), streamers on the outer ports only, so one register window and
one wrapper. Links between accelerators stay parked (open item 53). → C§17.

**D113** · scope — The kernels keep NPBench's DaCe structure with a dtype per container, so one
kernel may mix them (vsum: `int32` in, `int64` out), at PolyBench's MINI and SMALL sizes. Integers
first, for an exact check: `int64` through tier C, `int8` in and `int32` accumulate from tier P (as
OpenGeMM), its exact products by an explicit cast (D143); float later. → PAPER §6.1.

**D116** · scope — SNAX-GVSoC replaced SNAX-MODEL: a GVSoC model of the SNAX cluster in which a
Snitch core runs the real program, so control cost is measured, not declared, and checked against
snax_cluster's RTL (D123). M14 tried the model side first, then snax_cluster's.

**D117** · docs — The documents stay short: one fact in one place, a decision only for a choice
someone may want to reverse, each entry in its current form in at most three lines, and the done
tasks out of STATUS at every milestone close. The rules are at the top of this file.

**D118** · scope — One development image, `ghcr.io/kuleuven-micas/snax-forge:main` from
`container/`, for snax-forge, snax-gvsoc and the programs built from snax_cluster: the image holds
the tools, each repo keeps its own environment. → `container/Dockerfile`.

**D120** · gvsoc — The Snitch core sends CSRs 0x3C0 to 0x5FF as 32-bit IO requests on its `snax_csr`
port to an accelerator component per core, so a SNAX block is a plain IO slave; both sides follow
Zicsr. → ARCHITECTURE §5.6, snax-gvsoc `SG1.md`, `SG6.md`.

**D121** · gvsoc — SNAX-GVSoC keeps GVSoC's iDMA model and fits it to snax_cluster's iDMA RTL by
measurement, not a new model: the target sets the DMA and DRAM values, the model keeps the DMA's own
read path. → snax-gvsoc `SG2.md`, `SG7.md`.

**D122** · gvsoc — SNAX-GVSoC's crossbar takes the SNAX ports, cores and AXI, numbered as on the
RTL, and copies its sparse arbiter, checked against the RTL module in Verilator; the DMA takes whole
superbanks and refuses narrow winners. → snax-gvsoc `SG3.md`, `SG7.md`.

**D123** · scope — SNAX-GVSoC is checked against snax_cluster's RTL by one ELF, snax-forge's, on
both: counters free of the core's timing, against GEN2b's tiers and the cluster tests' references.
snax-forge never runs RTL; snax_cluster's own ELF is only its CI's pass/fail. → C§18.

**D125** · gvsoc — The Snitch core's loads follow the hjson and the RTL: one integer load in flight,
the RTL's load-to-load cycles, the core going on until the next load or a use of its register; fetch
keeps GVSoC's cache (open item 65). → snax-gvsoc `SG5.md`, `SG7.md`.

**D126** · scope — snax-gvsoc builds its programs from a pinned copy of snax_cluster's software in
`sw/`, refreshed by `make sync-snax-cluster`, whose diff is the check; its default config and
`test_snax_cfg` read `forge/vecadd/`, never `sw/cfg/snax_forge/`; a design point's program takes the
cluster headers snax-forge writes for it. Only the RTL needs snax_cluster. → snax-gvsoc `sw/README.md`.

**D129** · brm — A `binding` (kinds registered, D19) names its datapath and how the BRM fills it; the
CSRs are the group's (D146). A datapath is one Chisel module, one `blocks.json` entry and one SNAX-GVSoC
class by name (G1), so a new block is one of each; its latency may be an expression. → C§10, C§17.

**D130** · scope — A generated accelerator carries its own CSR manager (ReqRspManager's protocol,
snax_cluster's off), its CSRs per group (D146): one wrapper contract. Python checks `accelerator.json`
against the datapath catalogue, Scala only builds; snax_cluster sees only the bundle. → C§17, C§18.

**D131** · design — The base cluster is snax-forge's own file, `platforms/snax_forge_cluster.hjson`
(from snax_cluster's, its names snax-forge's), which `design from-hjson` reads: one accelerator, its
movers agreeing (D84), the rest small16's; `accelerator_slot` names its slot (D134). → C§13.

**D133** · brm — The Chisel Accumulator becomes `TreeAccumulate`, a datapath under the nest
controller (D147; L 1, pins `fire` and `first`, drain 0 as today): it extends its lanes by signedness
(D143), and its W → 1 adder tree makes it `chisel_adder_tree` too. → C§17, `hw/chisel/`.

**D134** · gvsoc — Three files per design point are the contract between snax-forge and SNAX-GVSoC
(the cluster and streamer hjson, `accelerator.json`, D145), read at run time (`snax_cfg.py`), so a
sweep compiles nothing; snax-gvsoc's `forge/` fixtures need no snax-forge. → C§16, C§17.

**D135** · lower — The task list becomes C on snRuntime, one accelerator per program: each step on
the core that owns its component, D64's waits as polls with a hardware barrier where another core
goes on, the DMA waited on for dependences only, two `mcycle` reads bounding the window. → C§16.

**D136** · flow — `pixi run flow` runs the inner loop once on SNAX-GVSoC, decides nothing and has no
backend flag: lower, gen, build when needed, sw, gvsoc, check (the program's exit code,
`total_cycles` its window), into a folder named after every `--set`. → `flow/run.py`, C§7.

**D137** · gvsoc — Events in two stages, so the model owns what it counts and snax-forge only names
it: SNAX-GVSoC's traces go to one `events.log`, a level turns trace names on, lines are linked by
ids; the converter renames, never recounts, and cuts the window. → `run/`, C§7.

**D141** · scope — M13, the block library, joins M7's tier A, the composed bind and BASE1's forge side:
blocks, composition and kernels designed together, dot's variants the case study. Phases B0, B1, B1x,
clean-up, VIEW2, SYN1, HWCOST, B2, tier A (B3 and B4), with RAM1 alongside. → ARCHITECTURE §8.

**D142** · design — A bank word packs num_slc = 64/b elements of a port's container dtype, little-endian
as NumPy. W lanes use num_ch = W/num_slc channels; another needs the FIFO slicer (parked: open item 93),
so `connect.num_ch` refuses it. A packed stream: word-aligned base, num_slc | S_0, innermost stride 1,
the others multiples of num_slc. The platform keeps only the word. → C§1, C§13, C§14.

**D143** · brm — Arithmetic follows NumPy: each op runs at its operands' result type, with
signedness from each port's container, and the result is then extended or wrapped into its target. A
fold's lanes are cast to its output's type before its tree. Exact products (tier P) need an explicit
cast in the kernel (open item 37). → C§16, snax-gvsoc `snax/snax_types.hpp`.

**D144** · lower — Padding is in the data, the hardware masks nothing: a split dimension pads to a
multiple of W, then the container to whole DMA beats, with zero or a max or min fold's identity; a
chain into a fold at a non-dividing size is refused (`connect.pad`). → `dfg/pad.py`, `gen/data.py`.

**D145** · scope — One description per design point, `accelerator.json`: snax-forge writes it once,
and Emit, the catalogue check and SNAX-GVSoC read it, in place of `rtl.json` and `<slot>_acc.hjson`,
so the model and the RTL cannot be configured apart. → C§16, C§17.

**D146** · scope — An accelerator is groups of linked units, each with one JOBS (a unit fires JOBS · ∏ T
over itself and the folds downstream, SDF's balance) and its own START, STATUS, COUNTER and busy and
done pins, so groups run apart; an empty run ends in a done, so no ERROR CSR. → C§17.

**D147** · brm — Every block is a datapath of data and enables only, under one control unit per group
(`<tag>_<group>_ctrl`) generated from the description, so control is written once for model and RTL,
never per variant; "shell" is the wrapper. → C§17.

**D148** · dfg — A bound graph nests accelerator ⊃ groups ⊃ units and links. `link` merges two bound
units' groups, their transient becoming a link; a group is its own accelerator until `encapsulate` puts
several in one (both undone by inverses): dot 1 is one accelerator, groups `mul` and `sum`. → C§11, C§12.

**D149** · brm — The MAC is one block with design-time variants (`accumulate`, or a·b + c with `regOut`
for a later systolic PE; no `regIn`), not vmul into a lane accumulator. The one-lane vsum is
`SerialAccumulate` (`span` = lanes), not a serializer link; a buffer is `LinkBuffer`, not a FIFO. → C§17.

**D150** · dfg — The mapping is spelled step by step: `split_reduce`, `fuse_map` and `link`, each with
its inverse, one recipe per variant, instead of binding a group of tasklets in one step. `bind` refuses
with the fix named ("fuse_map first", "split_reduce first"). → C§11, C§12.
