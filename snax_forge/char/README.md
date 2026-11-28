# The cost tables: CHAR1's and HWCOST's first-order model

What snax-forge needs to put an area and an energy beside a design's cycles before any synthesis (D154 · D172, D167, C§14). Every table comes from clock-gated synthesis in snax-forge-experiments (its X79 · X87 · X88):

- `snax_forge/char/<tech>.json`: per technology, the comb area and comb energy per fire of each catalogue block, and a few constants for registers and control (snax-forge-experiments' `char.json`).
- `snax_forge/char/<tech>/`: HWCOST's tables for the rest of the design, the streamers, the TCDM crossbar and the L1 banks (CACTI's itrs-lstp cells, X47 · X100 there), the cluster's fixed parts (`cluster_fixed.json`: the core, the DMA and the rest, area, leakage and idle floor) the DMA engine's energy per burst (`dma_copy.json`) and the terms fitted at gate level (`model_fit.json`: a refusal's fraction of a grant, measured on the blocks, the readers' own in `streamer.csv`'s `r_refused`, the streamers', crossbar's and DMA's idle floor, the DMA's energy per L2 ↔ L1 beat), and `SOURCE`, which names the commit and each of the eight files with its sha256 (`pixi run import-costs`). Only nangate45 exists.
- Every table carries leakage (µW) beside its area, read by the same rule (X101 there): a block's comb leakage and a flop's, a streamer's, the crossbar's, a bank's.
- The estimate (`snax_forge/estimate/cost.py`) combines them with counts it already has from the description: each unit's fires, its register bits and the accelerator's cycles.
- `design.md`, the DFG viewer and the editor show the result as a first-order estimate: an area to 3 significant figures and an energy to 3 decimals in its unit, display precision and not accuracy.

## What it is and is not

It is a first-order model: registers are bits that cost energy only when they load, a block is its comb energy per fire, control is its bits times one constant every cycle. It aims at the large costs and at showing how a design is costed, about 30 % of synthesis on the large costs; a small block can be much further off.

- On the clock-gated dot 3 cluster at W = 2 to 32 (snax-forge-experiments `results/cluster/`), the datapath's area is 0.999 of the netlist's at every W, its energy 1.61 to 1.63 of the run's at W = 2 to 16 and 1.06 at W = 32; the memory is within 1 % in both.
- The datapath's energy is high because the blocks are measured on random 64-bit data, which toggles more than dot's; kept so, a pessimistic price, until a kernel's own data is worth characterizing.
- Against the gated accelerators alone (vecmul, vsum, dot 4 and dot 5 at W = 8, snax-forge-experiments' `results/char/char_table.md`) the model's ctrl + datapath lands at 0.996 to 1.035 of the measured energy; its control alone at 0.57 to 1.62 per design, as a gated control unit's energy follows its counters' activity more than its bits, a small share of every kernel there (under 8 %).

The known limits (C§14):

- The control unit's comb area is taken equal to its flops' (`CTRL_COMB_PER_FLOP`), a rough early estimate, its comb energy as negligible. The csr, the cores, the I-cache and the DMA are not modelled.
- A streamer is priced per beat and a crossbar access at one price per bank count, a full crossbar's: a streamer's clock while it waits, the crossbar's per-cycle part and its conflicts are not counted, and a writer is priced as a reader (snax-forge-experiments X52).
- The crossbar's area prices a reader's channels as read-only inputs and a writer's as write-only (X71's factor form on `xbar.csv`, from `xbar_kinds.json`); the further constants a whole cluster puts on it (priority, user and strobe bits) are not, which left CSYN1's synthesized cluster's interconnect about 10 % under the estimate (snax-forge-experiments X70, X71).
- An accumulator's isolated fire costs 25 to 30 % more than a back-to-back one (its inputs change at one edge and its accumulator at the next, so its adders settle twice); the model has back-to-back fires only (X34).
- Comb energy is counted per fire only, so a unit fed by another unit is undercounted when its inputs change more often than it fires: dot 4's sum fires once but sees 256 new Mac outputs. The kernel stays within 4 % because such sums are small.
- A TreeAccumulate alone keeps the logic for its `first` pin, which dot 4 ties and firtool folds, so the table's comb area is above the block inside dot 4 by about 370 µm² on nangate45 (82 % at W = 1, 4 % at W = 32).
- A seeded TreeAccumulate, SerialAccumulate or Mac (D163 · D187) is priced as its unseeded entry (`unseeded`): the same adder and register, only the first beat's mux moved from the sum to the register; no seeded point is measured. The Mac's entries were measured before it took a seed and have no `seed` key: an unseeded Mac (`seed` false) is that entry as it is.
- The table holds the Mac in its accumulating form only, SimdLanes `add` and `mul` only with two streams and no slot from a CSR (so relu's one-stream `max` and axpy's `axpby` are not characterized), LaneAccumulate not at all, a packer with one lane in only, and a slicer with one word in only; anything else is "not characterized" (its comb counted as 0, its flops still counted) and the design note `char.missing` names it.
- At a narrow pair with $w_{in} < w_{out}$ a register's upper bits copy the sign and toggle less, so $e_{\mathit{load}}$ prices it high, by up to 2.2×; a narrow SerialAccumulate's lanes cost more energy than the slope rule says (32 % at 8/32 and W = 32, on 0.4 pJ a fire).

## Where the numbers come from

Every number is measured by CHAR1 and CHAR2 in [snax-forge-experiments](https://github.com/KULeuven-MICAS/snax-forge-experiments) (`results/char/`, X28 to X37, X89; run notes `results/char/runs/2026-11-03_char1_r0.md` to `_r4.md` and `2026-11-16_char2_r2.md` to `_r5.md`; the slicer, X94, `2026-11-17_tiera1_r5d.md`), synthesized with clock gating (X83 · X85), and `<tech>.json` here is a byte copy of its `results/char/<tech>/char.json`, which names the snax-forge and snax-forge-experiments commits it was made from.

- **The blocks** (X28, X29): 27 points at int64, each one catalogue block at full params emitted alone by snax-forge's `gen-block` (`Emit --block`, C§10), the module an accelerator holds with its asynchronous reset: SimdLanes add and mul at W = 1, 8, 32; the accumulating Mac at W = 1, 8, 32; TreeAccumulate at W = 1, 2, 4, 8, 16, 32; SerialAccumulate at W = 2, 4, 8, 16, 32; LinkBuffer at depth 1, 2, 4, 8, 16 at W = 1 and depth 1 at W = 8 and 32.
- **The types** (X89, D176): every compute block at its 10 width pairs ($w_{in}$, $w_{out}$ in 8, 16, 32, 64, $w_{out} \ge w_{in}$, the op at $w_{out}$), at one lane count $W_0$: SimdLanes at 1, the Mac and the folds at 8; each input port driven at its own width. FifoPacker with one lane in and one word out at int32, int16 and int8 (d_out 2, 4, 8); FifoSlicer with one word in at int32 (1 lane), int16 (1, 2) and int8 (1, 2, 4), d_in 2 to 8 (X94).
- **Synthesis** (X30, X79): Synopsys Design Compiler on Nangate 45 nm, a 10 ns clock, the block as the top, `compile_ultra -no_autoungroup -gate_clock`, no wire-load model; a block alone has the area of the same module inside the accelerators within 0.05 % wherever the accelerator drives its pins.
- **Activity** (X31, X32): a zero-delay Questa simulation of the mapped netlist with rand64 data (seed 0), inputs changing at the rising edge, through fixed 128-cycle windows after a lead-in: `busy` (a fire every cycle; SerialAccumulate a read every cycle; FifoPacker a push every cycle; FifoSlicer a pop every cycle, a take every d_in) and `idle` (the enable at 0, every other input held) for SimdLanes, Mac, TreeAccumulate, SerialAccumulate, FifoPacker and FifoSlicer, and `stream` (a load every cycle at full occupancy) for LinkBuffer; the VCD is cut per window and turned into a SAIF.
- **Power** (X33): PrimePower on the same netlist, library and constraints, per window the energy per cycle of each power group: comb, and the flops' (clock, register and sequential, the clock gates among them) as e_flop; e_bit is e_flop over the block's flops.
- **The table** (X37, X83 · X85): `report/char_export.py` (`pixi run char-export`) writes `char.json` from the gated rows; an integrated clock gate counts as a sequential cell, and the control factor is measured on the gated forge accelerators.

## How each field is computed

| Field | How | From |
|---|---|---|
| `flop.area_um2` | sequential area over sequential cells, the clock gates among them | the registered int64 points: 5.296378 µm² |
| `flop.e_load_fj` | the mean busy e_bit | the points whose flops all load every cycle: Mac, TreeAccumulate and SerialAccumulate busy, LinkBuffer depth 1 stream: 17.8153 fJ |
| `flop.e_hold_fj` | the mean idle e_bit | the idle points of Mac, TreeAccumulate and SerialAccumulate: 0.0655 fJ, which gating leaves of a held bit; used only through $e_{\mathit{ctrl}}$ |
| `ctrl.counter_factor` | a forge point's `ctrl/ctrl` energy per cycle over its sequential cells × e_hold, the mean over the points | the 48 int64 tier-done forge points (43 to 148): 83.20, so $e_{\mathit{ctrl}}$ = 83.20 × 0.0655 fJ = 5.450 fJ |
| a point's `comb_um2` | the block's comb area | `char_synth.csv` |
| a point's `e_comb_pj` | the comb energy per cycle of the busy window, a fire every cycle, so per fire; LinkBuffer's stream window, per load (0 at depth 1: a register alone) | `char_power.csv` |
| an entry's `lanes_rule` | `copy` where the block is lane copies (SimdLanes, Mac, LinkBuffer: their W = 8 and 32 points are within 3 % of W × the lanes-1 point), `interpolate` for the int64 TreeAccumulate and SerialAccumulate, `slope` for a narrow one, `none` for a packer | X37, X89 |
| a narrow fold's `slope` | the int64 entry's cost per lane between W = 8 and 32, times $w_{in}$ / 64: a lane is a leaf or a mux input at the input's width, while the accumulator stays at the output's | X89 |
| a packer's `flops`, `load_bits` | its held slots and counter, $(d_{out} - 1)\thinspace w + \lceil \log_2 d_{out} \rceil$; of them $(d_{out} - 1)/d_{out}$ of a slot and the counter load per push | X89 |
| a slicer's `flops`, `load_bits` | its word, counter and full flag, $64 + \lceil \log_2 d_{in} \rceil + 1$; of them $64/d_{in}$ of the word and the counter load per pop | X94 |

A block's register bits are not in the table: they are the catalogue's `flops` on its params (`hw/chisel/blocks.json`, C§10), which CHAR1 and CHAR2 checked against DC's sequential cells at every point (a packer's and a slicer's are `estimate/bits.py`'s, equal to its point's `flops`).

**The lookup** (`lookup(table, datapath, params, lanes, depth)`, `__init__.py`) takes the entry whose params equal the block's at its width pair (`pair_params`, D176: lanes, depth and unset constants aside; a SimdLanes' and a Mac's inputs at their widest input, unless the block's own mixed pair is measured (`exact_params`, int8 × int32, either order), a SimdLanes' output at most its op's width, a Mac's at its output; unsigned as signed; a packer's `d_out`, `lanes` and `dtype` as its `inLanes`, `laneType` and `words`, a slicer's `d_in`, `lanes` and `dtype` as its `words`, `laneType` and `outLanes`), then in order:

1. a point at these lanes (and depth): used as is (`measured`; a packer's or a slicer's one point, with its `load_bits`);
2. under `copy`: lanes / $W_0$ × the point at the smallest lanes $W_0$, a LinkBuffer's at the same depth (`copy`);
3. under `interpolate`: linear between the two measured lanes around it (`interpolate`);
4. under `slope`: the nearest measured point plus (lanes − its lanes) × the slope (`slope`);
5. a LinkBuffer depth no point has: linear between the two measured depths around it, then copied (`interpolate`); a LinkBuffer at a dtype with no entry: the int64 entry's value × width / 64 (`width`);
6. anything else: none (no entry with these params, or lanes or a depth outside the measured range: the int64 TreeAccumulate at lanes 64, the int64 SerialAccumulate at lanes 1, a LinkBuffer deeper than 16).

Worked example, dot at i8 and W = 32: its Mac at (8, 32) is 32 / 8 × the W = 8 point, 4 × 3416.5 = 13 666 µm² and 4 × 4.132 = 16.528 pJ a fire; a TreeAccumulate sum at (32, 32) and W = 32 is 1084.5 + 24 × 267.06 × 32 / 64 = 4289.2 µm².

## The model

Per unit and per buffer link, over one run of the accelerator:

$$E_{\mathit{unit}} = n_{\mathit{fire}}\thinspace(e_{\mathit{comb}} + n_{\mathit{bits}}\thinspace e_{\mathit{load}})$$

$$A_{\mathit{unit}} = A_{\mathit{comb}} + n_{\mathit{bits}}\thinspace A_{\mathit{flop}}$$

A packer's bits that load per push are its point's `load_bits` in place of $n_{\mathit{bits}}$ in $E$ (only one held slot loads a push, none on the push that sends).

and per group its control unit, over the accelerator's $C_{\mathit{acc}}$ cycles:

$$E_{\mathit{ctrl}} = e_{\mathit{ctrl}}\thinspace n_{\mathit{bits}}^{\mathit{ctrl}}\thinspace C_{\mathit{acc}}, \qquad A_{\mathit{ctrl}} = 2\thinspace n_{\mathit{bits}}^{\mathit{ctrl}}\thinspace A_{\mathit{flop}}$$

and, from HWCOST's tables, per port's streamer p (FIFO depth $D_p$, loops $L_p$, channels $C_p$, $W_p$ words in L1 over $W_p / C_p$ beats), the crossbar at its $n_{\mathit{in}}$ inputs and $n_{\mathit{banks}}$ banks and the banks' $N_{\mathit{rd}}$ reads and $N_{\mathit{wr}}$ writes (C§14 has the rule and every symbol):

$$E_{\mathit{L1}} = \sum_p \frac{W_p}{C_p}\thinspace e_{\mathit{beat}}(D_p, L_p, C_p) + \Big(\sum_p W_p\Big)\thinspace e_{\mathit{acc}}(n_{\mathit{banks}}, n_{\mathit{banks}}) + N_{\mathit{rd}}\thinspace e_{\mathit{rd}} + N_{\mathit{wr}}\thinspace e_{\mathit{wr}}$$

A flop that loads costs $e_{\mathit{load}}$ in that cycle; one that holds costs nothing, since clock gating stops its clock (0.0655 fJ a bit and cycle is what is left, 0.004 pJ on dot 3's datapath at W = 2). The control unit's counters and state change every cycle, so it keeps one constant per bit and cycle, $e_{\mathit{ctrl}}$. A group's cost is its units', buffer links' and control unit's added up; the design's is its groups' (the datapath and control parts, the kernel's) plus its streamers', its crossbar's and its banks' (D167).

Where the estimate gets each count:

| Symbol | What | Where |
|---|---|---|
| $n_{\mathit{fire}}$ | a unit's fires: its firings times its span (SerialAccumulate reads span beats per firing); a buffer link's loads: its producer's outputs, its firings over its T (a fold gives once per T beats) | SNAX-LOWER's firings (`lower/derive.py`), the catalogue's span (`estimate/cycles.py`) |
| $n_{\mathit{bits}}$ | the register bits of the unit's datapath or the buffer | the catalogue's `flops` on its params (`gen/rtl.py`, the estimate's `bits`) |
| $n_{\mathit{bits}}^{\mathit{ctrl}}$ | the register bits of the group's control unit firtool keeps | `estimate/bits.py`, the estimate's `ctrl_bits` |
| $C_{\mathit{acc}}$ | the accelerator's cycles, every group's, for its control unit | the group rule stepped on ideal streams (D155), `accelerator_cycles` |
| $e_{\mathit{comb}}$, $A_{\mathit{comb}}$ | the block's comb energy per fire and comb area | this table's lookup |
| $e_{\mathit{load}}$, $A_{\mathit{flop}}$, $e_{\mathit{ctrl}}$ | the technology's constants | this table's `flop` and `ctrl` ($e_{\mathit{ctrl}}$ = `counter_factor` × `e_hold_fj`) |

## A worked example: dot 5 at W = 8 on nangate45

dot 5 (`recipes/dot/dot_v5.json`, N = 2048, W = 8) is one group, `dot`: a Mac `mac` at lanes 8 feeding a SerialAccumulate `sum` at lanes 8 over a wire. The estimate gives $C_{\mathit{acc}}$ = 265 cycles.

The constants (`nangate45.json`, `flop` and `ctrl`): $e_{\mathit{load}}$ = 17.8153 fJ; $A_{\mathit{flop}}$ = 5.296378 µm²; $e_{\mathit{ctrl}}$ = 83.20 × 0.0655 fJ = 5.4496 fJ.

**The Mac.** It fires once per beat of 8 elements: $n_{\mathit{fire}}$ = 2048 / 8 = 256. Its `flops` is `lanes * width(outType) * max(accumulate, regOut)` = 8 × 64 × 1, so $n_{\mathit{bits}}$ = 512. The table has the accumulating int64 Mac measured at lanes 8: $A_{\mathit{comb}}$ = 57 131 µm², $e_{\mathit{comb}}$ = 92.181 pJ.

- per fire: 92.181 + 512 × 17.8153 fJ = 92.181 + 9.121 = 101.302 pJ;
- per task: 256 × 101.302 = 25 933 pJ, 25.933 nJ;
- area: 57 131 + 512 × 5.296 = 57 131 + 2 712 = 59 843 µm².

**The sum.** It fires once with span 8 (it reads its 8 lanes one per cycle): $n_{\mathit{fire}}$ = 1 × 8 = 8. Its `flops` is `width(outType)` = 64. The table has SerialAccumulate measured at lanes 8: $A_{\mathit{comb}}$ = 821.9 µm², $e_{\mathit{comb}}$ = 1.0340 pJ.

- per fire: 1.0340 + 64 × 17.8153 fJ = 1.0340 + 1.1402 = 2.1741 pJ;
- per task: 8 × 2.1741 = 17.4 pJ, 0.017 nJ (it holds its sum for the other 257 cycles at no cost);
- area: 821.9 + 64 × 5.296 = 821.9 + 339.0 = 1 161 µm².

**The wire** between them has no register and no cost.

**The control unit.** `ctrl_bits` is 103: `busy` 1, the Mac's `mac_job_beats_counter` and `mac_jobs_counter` 32 each and `mac_v` 1, the sum's `sum_jobs_counter` 32, `sum_v` 1 and `sum_lane` 3, and `done` 1 (`estimate/bits.py`); JOBS and the Mac's JOB_BEATS are read live from the csr, with no copy (D168).

- per task: 103 × 265 × 5.4496 fJ = 148 749 fJ, 0.149 nJ;
- area: 103 × 5.296 = 546 µm² of flops and as much comb (the rough estimate), 1 091 µm².

**The streamers, crossbar and banks** (HWCOST, D167):

- `mac_a` and `mac_b` read 2048 words each over 256 beats of 8 channels at FIFO depth 8 and one loop, a grid point of `streamer.csv`: 31 739 µm² and 22.49923 pJ a beat each, 5.760 nJ; `sum_out` writes 1 word on 1 channel, 4 556 µm² and 0.003 nJ.
- The crossbar has 8 + 8 + 1 channels plus the core and AXI, 19 inputs, on 32 banks: between the grid's 16 and 24 inputs, $t = 3/8$, 98 415 µm² in `xbar.csv`. Its 16 reader channels only read and the writer's one only writes (X71), so $98{,}415 \times (1 - 0.395299 \cdot 16/19 - 0.409746 \cdot 1/19) = 98{,}415 \times 0.64555 = 63{,}532$ µm².
- Every access is at a full crossbar's price, `xbar.csv`'s 32 inputs on 32 banks, 2.752585 pJ, less its input kind's saving at 32 banks (D178, `xbar_kinds.json`'s `energy`): a read 2.752585 × (1 − 0.331792) = 1.839299 pJ, a write 2.752585 × (1 − 0.471583) = 1.454513 pJ, so 4096 reads and 1 write cost 7.535 nJ.
- The 32 banks of 512 × 64 bits are 32 × 13 502.9 = 432 093 µm², and 4096 reads at 3.38552 pJ and 1 write at 6.45356 pJ cost 13.874 nJ.

**The total**: the kernel 25.9333 + 0.0174 + 0.1487 = 26.099 nJ per run, the streamers 11.523 nJ, the crossbar 7.535 nJ and the banks 13.874 nJ, 59.031 nJ in all; 59 843 + 1 161 + 1 091 = 62 095 µm² of datapath and control, with the streamers' 68 035 µm² and the crossbar's 63 532 µm² 193 662 µm² without the banks, 625 755 µm² with them, shown as 194 000 µm² and 626 000 µm². dot 4 at W = 8 has the kernel 26.078 nJ.

**Against synthesis.** The gated cluster priced the same way (snax-forge-experiments `results/cluster/`, dot 3 at W = 2 to 32) is the check this table has so far. In area the datapath, control and memory are within 1 %, the streamers 0.83 to 1.02 and the crossbar 1.06 to 1.12 of the netlist. In energy, at W = 2 to 16, the datapath and control are 1.5 to 1.6 times high (random data), the streamers 0.93 to 0.97, the crossbar 0.84 to 0.99 (priced by input kind; 1.26 to 1.48 at the read-write price) and the memory 0.99 (the banks then at itrs-hp, their reads 4 % dearer); at W = 32 the run stalls, which the estimate does not see.

## Adding a block, a dtype or a technology

**A block** (a new catalogue datapath, or new params of one):

- the datapath and its `flops` go into snax-forge's catalogue first (`hw/chisel/blocks.json`; `BlockFlopsSpec` checks the count);
- in snax-forge-experiments: its points in `char/points.json`; `pixi run char-rtl`; `pixi run syn run '<points>' --source char --tech <tech>`; its windows (a kind of its own needs a source in `sim/sources/char.py` and a testbench) and `pixi run pwr run '<points>' --source char --tech <tech>`; `pixi run char-export` and `pixi run char-table`, the result reviewed in `char_table.md`;
- `pixi run char-export --clk-gate` there, then `pixi run import-costs` here and `pixi run test-char`.

**A dtype** is an entry of its own: the lookup matches params exactly, so int32 points under an entry whose params name int32 leave int64's as they are. Measure it as a block above.

**A technology**: synthesize and simulate the 27 points on it in snax-forge-experiments, and SYN2's 48 accelerators too, since the counter factor comes from their control units (`results/forge/<tech>/`). Add its paths to `util/import_costs.py`'s `PATHS` if they differ, run `pixi run import-costs --tech <tech>` and add the name to `TECHS` in `__init__.py`: the estimate, `design.md` (`pixi run report DIR --tech`), the viewers' `--tech` and the page's dropdown then offer it.

`tests/char/` checks every table against the catalogue: each entry's datapath exists, its params with each point's lanes (and depth) are a valid and complete param set there, its rule is `copy` or `interpolate`, its points are unique.

## The checks and what each one shows

| Check | Where | What it shows |
|---|---|---|
| the table against the catalogue | `tests/char/` | every entry names a datapath and params the generator can build |
| `flops` against the emitted RTL | `BlockFlopsSpec` (`hw/chisel/`) | the catalogue's count is the register bits firtool emits, at random params |
| the estimate against the experiments' pricing | `tests/estimate/test_cost.py` | a consistency check: dot 3 at W = 2 to 32, each of the five parts within 1e-6 of snax-forge-experiments' own implementation on the same gated tables (its control and crossbar rescaled to CG5's constants); it says nothing about accuracy |
| the model against synthesis | snax-forge-experiments `results/cluster/` (the gated cluster) and `char_table.md`, Model (the gated accelerators alone) | the evidence: the datapath's area at 0.999 of the gated cluster's, its energy 1.6 times high on random data; the kernels at 0.996 to 1.035 of the gated accelerators' |
| a block alone against inside | snax-forge-experiments X30, R0 | the block CHAR1 measured is the block an accelerator holds: area within 0.05 % where the accelerator drives its pins |
