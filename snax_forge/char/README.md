# The cost tables: CHAR1's and HWCOST's first-order model

`snax_forge/char/<tech>.json` holds, per technology, what snax-forge needs to put an area and an energy beside a design's cycles before any synthesis: the comb area and comb energy per fire of each catalogue block, and a handful of constants for registers and control (D154, C§20). `snax_forge/char/<tech>/` holds HWCOST's three tables for the rest of the design, the streamers, the TCDM crossbar and the L1 banks (D167): only nangate45 has them, and only nangate45 is offered; `nangate15.json` stays as data. The estimate (`snax_forge/estimate/cost.py`) combines them with counts it already has from the description: each unit's fires, its register bits and the accelerator's cycles. `design.md`, the DFG viewer and the editor show the result as a first-order estimate, an area to 3 significant figures and an energy to 3 decimals in its unit; the decimals tell two close designs apart and are display precision, not accuracy.

## What it is and is not

It is a first-order model: registers are bits, a block is its comb energy per fire, control is its bits held every cycle times a factor. It aims at the large costs and at showing a designer how a design is costed, not at accuracy: about 30 % of synthesis on the large costs is the intent, and a small block can be much further off. The evidence so far: on CHAR1's checked designs (vecmul, vsum, dot 4 and dot 5 at W = 8 and 32, N = 2048, on Nangate 45 and 15 nm) the model's kernel energy (datapath and control) lands at 0.99 to 1.04 of the synthesized and simulated accelerator (snax-forge-experiments `results/char/char_table.md`, Model, "Against SYN2"); per unit the Mac is at 1.03 to 1.04, while dot 4's TreeAccumulate sum is at 0.06 to 0.19 and dot 5's SerialAccumulate sum at 0.41 to 0.78. That is evidence on eight designs per technology, not a promise for others.

The known limits (C§20):

- The control unit's comb area is taken equal to its flops' (`CTRL_COMB_PER_FLOP`), a rough early estimate, its comb energy as negligible. The csr, the cores, the I-cache and the DMA are not modelled.
- A streamer is priced per beat and a crossbar access at one price per bank count, a full crossbar's: a streamer's clock while it waits, the crossbar's per-cycle part and its conflicts are not counted, and a writer is priced as a reader (snax-forge-experiments X52).
- An accumulator's isolated fire costs 25 to 30 % more than a back-to-back one (its inputs change at one edge and its accumulator at the next, so its adders settle twice); the model has back-to-back fires only (X34).
- Deep LinkBuffers come out about 1.4 to 1.8 times high on nangate45: there the stream slope PrimePower gives is 0.56 to 0.71 of one held slot per depth (X35); nangate15's is 0.96 to 1.14.
- Comb energy is counted per fire only, so a unit fed by another unit is undercounted when its inputs change more often than it fires: dot 4's sum fires once but sees 256 new Mac outputs. The kernel stays within 4 % because such sums are small.
- A TreeAccumulate alone keeps the logic for its `first` pin, which dot 4 ties and firtool folds, so the table's comb area is above the block inside dot 4 by about 370 µm² on nangate45 (82 % at W = 1, 4 % at W = 32).
- The table holds int64 only, the Mac in its accumulating form only, and ElementwiseLanes (now SimdLanes, mapped by `DATAPATH_RENAMES` until the next export) `add` and `mul` only, two-input points (`streams` 2 and no slot from a CSR, `kCsr`, `k1Csr`, `k2Csr` false, `TABLE_DEFAULTS`, so a one-stream SimdLanes such as relu's, or axpy's axpby, is not characterized); anything else is "not characterized" (its comb counted as 0, its flops still counted) and the design note `char.missing` names it.

## Where the numbers come from

Every number is measured by CHAR1 in [snax-forge-experiments](https://github.com/KULeuven-MICAS/snax-forge-experiments) (`results/char/`, X28 to X37; run notes `results/char/runs/2026-11-03_char1_r0.md` to `_r4.md`), and `<tech>.json` here is a byte copy of its `results/char/<tech>/char.json`, which names the snax-forge and snax-forge-experiments commits it was made from.

- **The blocks** (X28, X29): 27 points at int64, each one catalogue block at full params emitted alone by snax-forge's `gen-block` (`Emit --block`, C§17), the module an accelerator holds with its asynchronous reset: ElementwiseLanes add and mul at W = 1, 8, 32; the accumulating Mac at W = 1, 8, 32; TreeAccumulate at W = 1, 2, 4, 8, 16, 32; SerialAccumulate at W = 2, 4, 8, 16, 32; LinkBuffer at depth 1, 2, 4, 8, 16 at W = 1 and depth 1 at W = 8 and 32.
- **Synthesis** (X30): Synopsys Design Compiler on Nangate 45 nm and 15 nm, a 10 ns clock, the block as the top, the settings of SYN1 and SYN2 (`compile_ultra -no_autoungroup`, no wire-load model); a block alone has the area of the same module inside SYN2's accelerators within 0.05 % wherever the accelerator drives its pins.
- **Activity** (X31, X32): a zero-delay Questa simulation of the mapped netlist with rand64 data (seed 0), inputs changing at the rising edge, through fixed 128-cycle windows after a lead-in: `busy` (a fire every cycle; SerialAccumulate a read every cycle) and `idle` (the enable at 0, every other input held) for ElementwiseLanes, Mac, TreeAccumulate and SerialAccumulate, and `stream` (a load every cycle at full occupancy) for LinkBuffer; the VCD is cut per window and turned into a SAIF.
- **Power** (X33): PrimePower on the same netlist, library and constraints, per window the energy per cycle of each power group: comb, and the flops' (clock, register and sequential) as e_flop; e_bit is e_flop over the block's flops.
- **The table** (X37): `report/char_export.py` (`pixi run char-export`) writes `char.json` from the CSVs; `char_table.md` (`pixi run char-table`) shows every number it holds and the model against SYN2.

## How each field is computed

| Field | How | From |
|---|---|---|
| `flop.area_um2` | sequential area over sequential cells | one value at all 21 registered points: 5.32 µm² on nangate45, 1.277952 µm² on nangate15 |
| `flop.e_load_fj` | the mean busy e_bit | the 17 points whose flops all load every cycle: Mac, TreeAccumulate and SerialAccumulate busy, LinkBuffer depth 1 stream (nangate45 17.1692 fJ, 6.85 % range) |
| `flop.e_hold_fj` | the mean idle e_bit | the 14 idle points of Mac, TreeAccumulate and SerialAccumulate (nangate45 7.1476 fJ, 0.17 % range) |
| `ctrl.counter_factor` | per point the control unit's e_dyn per cycle over its flops × e_hold, the mean to two decimals | SYN2's 48 accelerators, `ctrl/ctrl` (nangate45 1.06, nangate15 1.12) |
| a point's `comb_um2` | the block's comb area | `char_synth.csv` |
| a point's `e_comb_pj` | the comb energy per cycle of the busy window, a fire every cycle, so per fire; LinkBuffer's stream window, per load | `char_power.csv` |
| an entry's `lanes_rule` | `copy` where the block is lane copies (ElementwiseLanes, Mac, LinkBuffer: their W = 8 and 32 points are within 3 % of W × the lanes-1 point), `interpolate` where it is not (TreeAccumulate, SerialAccumulate) | X37 |

A block's register bits are not in the table: they are the catalogue's `flops` on its params (`hw/chisel/blocks.json`, C§17), which CHAR1 checked against DC's sequential cells at every point.

**The lookup** (`lookup(table, datapath, params, lanes, depth)`, `__init__.py`) takes the entry whose params equal the block's (as the generator fills them, lanes and depth aside), then in order:

1. a point at these lanes (and depth): used as is (`measured`);
2. under `copy`: lanes × the lanes-1 point, a LinkBuffer's at the same depth (`copy`);
3. under `interpolate`: linear between the two measured lanes around it (`interpolate`);
4. a LinkBuffer depth no point has: linear between the two measured depths around it at lanes 1, then copied (`interpolate`);
5. anything else: none (no entry with these params, or lanes or a depth outside the measured range: TreeAccumulate at lanes 64, SerialAccumulate at lanes 1, a LinkBuffer deeper than 16).

## The model

Per unit and per buffer link, over one run of the accelerator:

$$E_{\mathit{unit}} = n_{\mathit{fire}}\thinspace\big(e_{\mathit{comb}} + n_{\mathit{bits}}\thinspace(e_{\mathit{load}} - e_{\mathit{hold}})\big) + C_{\mathit{acc}}\thinspace n_{\mathit{bits}}\thinspace e_{\mathit{hold}}$$

$$A_{\mathit{unit}} = A_{\mathit{comb}} + n_{\mathit{bits}}\thinspace A_{\mathit{flop}}$$

and per group its control unit:

$$E_{\mathit{ctrl}} = f_{\mathit{ctrl}}\thinspace n_{\mathit{bits}}^{\mathit{ctrl}}\thinspace C_{\mathit{acc}}\thinspace e_{\mathit{hold}}, \qquad A_{\mathit{ctrl}} = 2\thinspace n_{\mathit{bits}}^{\mathit{ctrl}}\thinspace A_{\mathit{flop}}$$

and, from HWCOST's tables, per port's streamer p (FIFO depth $D_p$, loops $L_p$, channels $C_p$, $W_p$ words in L1 over $W_p / C_p$ beats), the crossbar at its $n_{\mathit{in}}$ inputs and $n_{\mathit{banks}}$ banks and the banks' $N_{\mathit{rd}}$ reads and $N_{\mathit{wr}}$ writes (C§20 has the rule and every symbol):

$$E_{\mathit{L1}} = \sum_p \frac{W_p}{C_p}\thinspace e_{\mathit{beat}}(D_p, L_p, C_p) + \Big(\sum_p W_p\Big)\thinspace e_{\mathit{acc}}(n_{\mathit{banks}}, n_{\mathit{banks}}) + N_{\mathit{rd}}\thinspace e_{\mathit{rd}} + N_{\mathit{wr}}\thinspace e_{\mathit{wr}}$$

A flop that loads costs $e_{\mathit{load}}$ in that cycle and one that holds $e_{\mathit{hold}}$; there is no clock gating, so every flop holds in every cycle it does not load, and $n_{\mathit{fire}}\thinspace e_{\mathit{load}} + (C_{\mathit{acc}} - n_{\mathit{fire}})\thinspace e_{\mathit{hold}}$ per bit is the form above regrouped. A group's cost is its units', buffer links' and control unit's added up; the design's is its groups' (the datapath and control parts, the kernel's) plus its streamers', its crossbar's and its banks' (D167).

Where the estimate gets each count:

| Symbol | What | Where |
|---|---|---|
| $n_{\mathit{fire}}$ | a unit's fires: its firings times its span (SerialAccumulate reads span beats per firing); a buffer link's loads: its producer's outputs, its firings over its T (a fold gives once per T beats) | SNAX-LOWER's firings (`lower/derive.py`), the catalogue's span (`estimate/cycles.py`) |
| $n_{\mathit{bits}}$ | the register bits of the unit's datapath or the buffer | the catalogue's `flops` on its params (`gen/rtl.py`, the estimate's `bits`) |
| $n_{\mathit{bits}}^{\mathit{ctrl}}$ | the register bits of the group's control unit firtool keeps | `estimate/bits.py`, the estimate's `ctrl_bits` |
| $C_{\mathit{acc}}$ | the accelerator's cycles, every group's | the group rule stepped on ideal streams (D155), `accelerator_cycles` |
| $e_{\mathit{comb}}$, $A_{\mathit{comb}}$ | the block's comb energy per fire and comb area | this table's lookup |
| $e_{\mathit{load}}$, $e_{\mathit{hold}}$, $A_{\mathit{flop}}$, $f_{\mathit{ctrl}}$ | the technology's constants | this table's `flop` and `ctrl` |

## A worked example: dot 5 at W = 8 on nangate45

dot 5 (`recipes/dot/dot_5.json`, N = 2048, W = 8) is one group, `dot`: a Mac `mac` at lanes 8 feeding a SerialAccumulate `sum` at lanes 8 over a wire. The estimate gives $C_{\mathit{acc}}$ = 265 cycles.

The constants (`nangate45.json`, `flop` and `ctrl`): $e_{\mathit{load}}$ = 17.1692 fJ, $e_{\mathit{hold}}$ = 7.1476 fJ, so $e_{\mathit{load}} - e_{\mathit{hold}}$ = 10.0216 fJ; $A_{\mathit{flop}}$ = 5.32 µm²; $f_{\mathit{ctrl}}$ = 1.06.

**The Mac.** It fires once per beat of 8 elements: $n_{\mathit{fire}}$ = 2048 / 8 = 256. Its `flops` is `lanes * width(outType) * max(accumulate, regOut)` = 8 × 64 × 1, so $n_{\mathit{bits}}$ = 512. The table has the accumulating int64 Mac measured at lanes 8: $A_{\mathit{comb}}$ = 58 325 µm², $e_{\mathit{comb}}$ = 92.128 pJ.

- per fire: 92.128 + 512 × 10.0216 fJ = 92.128 + 5.131 = 97.259 pJ;
- holding: 512 × 7.1476 fJ = 3.660 pJ per cycle;
- per task: 256 × 97.259 + 265 × 3.660 = 24 898 + 970 = 25 868 pJ, 25.9 nJ;
- area: 58 325 + 512 × 5.32 = 58 325 + 2 724 = 61 049 µm².

**The sum.** It fires once with span 8 (it reads its 8 lanes one per cycle): $n_{\mathit{fire}}$ = 1 × 8 = 8. Its `flops` is `width(outType)` = 64. The table has SerialAccumulate measured at lanes 8: $A_{\mathit{comb}}$ = 908.4 µm², $e_{\mathit{comb}}$ = 1.1216 pJ.

- per fire: 1.1216 + 64 × 10.0216 fJ = 1.1216 + 0.6414 = 1.7630 pJ;
- holding: 64 × 7.1476 fJ = 0.4574 pJ per cycle;
- per task: 8 × 1.7630 + 265 × 0.4574 = 14.1 + 121.2 = 135.3 pJ, 0.135 nJ (almost all of it holding);
- area: 908.4 + 64 × 5.32 = 908.4 + 340.5 = 1 249 µm².

**The wire** between them has no register and no cost.

**The control unit.** `ctrl_bits` is 103: `busy` 1, the Mac's `mac_job_beats_counter` and `mac_jobs_counter` 32 each and `mac_v` 1, the sum's `sum_jobs_counter` 32, `sum_v` 1 and `sum_lane` 3, and `done` 1 (`estimate/bits.py`); JOBS and the Mac's JOB_BEATS are read live from the csr, with no copy (D168).

- per task: 1.06 × 103 × 265 × 7.1476 fJ = 206 799 fJ, 0.207 nJ;
- area: 103 × 5.32 = 548 µm² of flops and as much comb (the rough estimate), 1 096 µm².

**The streamers, crossbar and banks** (HWCOST, D167). `mac_a` and `mac_b` read 2048 words each over 256 beats of 8 channels at FIFO depth 8 and one loop, a grid point of `streamer.csv`: 38 525 µm² and 39.65753 pJ a beat each, 10.152 nJ; `sum_out` writes 1 word on 1 channel, 5 469 µm² and 0.005 nJ. The crossbar has 8 + 8 + 1 channels and 2 cores and AXI, 20 inputs, on 32 banks: between the grid's 16 and 24 inputs, its area linear in inputs, $t = 0.5$, 103 432 µm²; every access at a full crossbar's price, `xbar.csv`'s 32 inputs on 32 banks, 2.739443 pJ, so 4097 accesses cost 11.223 nJ. The 32 banks of 512 × 64 bits are 32 × 13 522.7 = 432 726 µm², and 4096 reads at 3.53819 pJ and 1 write at 6.4677 pJ cost 14.499 nJ.

**The total**: the kernel 25.868 + 0.135 + 0.207 = 26.210 nJ per run, the streamers 20.310 nJ, the crossbar 11.223 nJ and the banks 14.499 nJ, 72.243 nJ in all; 61 049 + 1 249 + 1 096 = 63 394 µm² of datapath and control, with the streamers' 82 519 µm² and the crossbar's 103 432 µm² 249 345 µm² without the banks, 682 071 µm² with them, shown as 249 000 µm² and 682 000 µm². dot 4 at W = 8 has the kernel 26.160 nJ.

**Against SYN2** (snax-forge-experiments `results/forge/nangate45/`, the same design synthesized whole and simulated for one task, `char_table.md`'s Model; SYN2's design still holds CTRL1's copies of JOBS and the Mac's JOB_BEATS, so the numbers below are the model's for that 167-bit control unit, 26.339 nJ in all, until CTRL2 R5 synthesizes the new one): the kernel is 25.540 nJ, so the model is at 1.031; per part the Mac is 24.912 nJ (1.038), the sum 0.297 nJ (0.456: its inputs change on each of the Mac's 256 fires, which the model does not count) and the control unit 0.331 nJ (1.012). In area DC gives the Mac 61 051 µm² (1.000), the sum 1 282 µm² (0.974) and the control unit 1 750 µm², against the model's 1 777 µm² (1.015), its 888 µm² of flops and the placeholder's equal comb area; the csr's 1 103 µm² is outside the model. SYN2 has no L1 and no streamer, so HWCOST's parts are checked against their own tables (snax-forge-experiments `docs/cost_tables.md`) and later against the cluster synthesis (CSYN1).

## Adding a block, a dtype or a technology

**A block** (a new catalogue datapath, or new params of one): the datapath and its `flops` go into snax-forge's catalogue first (`hw/chisel/blocks.json`, `BlockFlopsSpec` checks the count). Then in snax-forge-experiments: its points in `char/points.json` (X28, X29); `pixi run char-rtl`; `pixi run syn run '<points>' --source char --tech <tech>`; its windows (a kind of its own needs a source in `sim/sources/char.py` and a testbench, X31) and `pixi run pwr run '<points>' --source char --tech <tech>`; `pixi run char-export` and `pixi run char-table`, and the result reviewed in `char_table.md`. Copy `results/char/<tech>/char.json` here byte for byte as `<tech>.json` and run `pixi run test-char`.

**A dtype** is an entry of its own: the lookup matches params exactly, so int32 points under an entry whose params name int32 leave int64's as they are. Measure it as a block above.

**A technology**: synthesize and simulate the 27 points on it in snax-forge-experiments, and SYN2's 48 accelerators too, since the counter factor comes from their control units (`results/forge/<tech>/`). Copy its `char.json` here as `<tech>.json` and add the name to `TECHS` in `__init__.py`: the estimate, `design.md` (`pixi run report DIR --tech`), the viewers' `--tech` and the page's dropdown then offer it.

`tests/char/` checks every table against the catalogue: each entry's datapath exists, its params with each point's lanes (and depth) are a valid and complete param set there, its rule is `copy` or `interpolate`, its points are unique.

## The checks and what each one shows

| Check | Where | What it shows |
|---|---|---|
| the table against the catalogue | `tests/char/` | every entry names a datapath and params the generator can build |
| `flops` against the emitted RTL | `BlockFlopsSpec` (`hw/chisel/`) | the catalogue's count is the register bits firtool emits, at random params |
| bits against SYN2 | `tests/estimate/test_bits.py` | each datapath's count matched DC's flops at SYN2's 48 points; the control unit's matched them when it was built and is held within 10 %: a check of the counts, not a promise |
| the estimate against the experiments' model | `tests/estimate/test_cost.py` | a consistency check: the code computes the experiments' formula on the same inputs, within 0.5 % (the table's rounding) on 16 designs; it says nothing about accuracy |
| the model against synthesis | snax-forge-experiments `char_table.md`, Model | the evidence: kernel energy at 0.99 to 1.04 of SYN2 on eight designs per technology, small units much further off |
| a block alone against inside | snax-forge-experiments X30, R0 | the block CHAR1 measured is the block an accelerator holds: area within 0.05 % where the accelerator drives its pins |
