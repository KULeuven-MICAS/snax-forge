# Dry run 4: mvt, eleven forms from one kernel

Every number on this page is what the tools printed when it was written; no test checks them (D98), so they can drift as the code moves.

mvt is two matrix-vector products over one matrix: $x_1 \mathrel{+}= A y_1$ and $x_2 \mathrel{+}= A^T y_2$ (`kernels/polybench/mvt.py`; `mvt_types.py` is the same with A and y at int8 and x at int32). Half 1 reads A along its rows, as a dot product per row. Half 2 needs A's columns, and that is the designer's question: how should half 2 read A, and what do the two halves share? This page runs eleven answers at 64 × 64 and W = 8, from the checked-in recipes in `recipes/mvt/` (`recipes/README.md` names each), on SNAX-GVSoC with the energy run, and sets them beside their vsim runs on snax_cluster's RTL.

## 1. The forms

Each recipe's `id` says what it builds (D188): `dot1` or `dot2` the dot group of each half (dot 1 through L1, dot 2 linked), `add`, `chain` or `seed` how x is added (a `vadd` through L1, a `vadd` linked after the fold, or the fold starting from x), `shared` both halves on one group, `dmaT` a transposed copy of A by the DMA, `col` half 2 in the column form.

| Recipe | `id` | How half 2 reads A | Accelerators | Crossbar inputs | Estimate (cycles) |
|---|---|---|---:|---:|---:|
| `mvt_v1` | `mvt_dot1add_shared` | down a column, strided | 3 | 59 | 2066 |
| `mvt_v1_2` | `mvt_dot2add_shared` | down a column | 2 | 43 | 1044 |
| `mvt_v2` | `mvt_dot2add` | down a column | 4 | 84 | 1044 |
| `mvt_v3` | `mvt_dot2chain_shared` | down a column | 1 | 20 | 1030 |
| `mvt_v3_1` | `mvt_dot2chain` | down a column, the halves pipelined | 2 | 38 | 515 |
| `mvt_v4` | `mvt_dot2seed_shared` | down a column | 1 | 20 | 1028 |
| `mvt_v4_1` | `mvt_dot2seed_shared_dmaT` | along the rows of a copy `A_T` | 1 | 20 | 1028 + copy 10 923 |
| `mvt_v4_1_1` | `mvt_dot2seed_shared_i8_o32_dmaT` | the same at int8 | 1 | 6 | 1028 + copy 10 923 |
| `mvt_v5` | `mvt_dot2seed_colseed_i8_o32` | along its rows: W `Mac` lanes on x₂, y₂ broadcast | 2 | 16 | 1028 (514 pipelined) |
| `mvt_v5_2` | `mvt_dot2seed_colchain_i8_o32` | as `mvt_v5`, the `Mac` unseeded, a `vadd` linked after it | 2 | 16 | 1029 |

The crossbar inputs are the streamer channels plus one for the core and one for AXI (D85); `mvt_v2` is above the platform's note at 64 and runs. The estimate is ideal (D155): every beat served, so it does not see half 2's bank conflicts. `mvt_v1_1`, `mvt_v1_2_1`, `mvt_v2_1` and `mvt_v5_1` are reuse twins of `mvt_v1`, `mvt_v1_2`, `mvt_v2` and `mvt_v5` (section 2) and are not run apart.

## 2. Building them

Every form starts from the same half 1. `mvt_v4`'s first eight steps make it a seeded dot 2 group (Ryan's, built in `edit-dfg`):

| Step | Transform | What it does |
|---|---|---|
| 1, 2 | `tag_map tmp0_mult_map temporal`, `split_map tmp0_mult_map_1 W_x1` | the products' row loop in time, W lanes on n |
| 3 | `bind tmp0_mult simd chisel_tiled_spatial vmul_x1` | the multiplier |
| 4, 5 | the same two on `sum_map` | the sum's loops |
| 6 | `fuse_seed sum_map_1 augassign_18_4_map` | the fold starts from x₁, so no `vadd` (D163) |
| 7 | `bind sum accumulate chisel_adder_tree vsum_x1` | the adder tree |
| 8 | `link tmp0_prod D_2 group mvt` | the products through a one-beat buffer, not L1 |

Half 2 repeats the eight after `mark_transpose` of A (step 9; half 2's memlets read `A[m, n]` with m inner), and step 18 `share`s the two groups, so one group serves both halves in turn. `mvt_v4_1` puts `data_transpose tmp2_mult_map A dma` in place of `mark_transpose`: a `transpose · dma` node copies A into `A_T` after A's load, and half 2 reads `A_T` along its rows (D182). `mvt_v4_1_1` is the same 18 steps on `mvt_types` at `i8`.

`mvt_v5` keeps half 1 and builds half 2 in the column form in nine steps: `tag_map`, `split_map` and `interchange_map` on the products' and the sum's maps so the W lanes sit on x₂'s index and the rows run in time, `fuse_seed` into `augassign_19_4_map`, `fuse_map` of the two nests, and `bind tmp4_mult_map_s mac chisel_mac mac_x2`. The bind sees that y₂'s memlet has no lane variable and broadcasts its one lane to the W lanes (D186); the `Mac` takes x₂ as its seed (D187).

The same design by reuse is four steps (`mvt_v5_1`): `lift matvec_acc` at each half, then `apply_recipe` with matvec's own recipes, `matvec_acc_v3` (the seeded dot 2 per row) on half 1 and `matvec_acc_v4` (the column form) on half 2. The sandbox checks each step against the reference:

    pixi run sandbox recipes/mvt/mvt_v5.json
    17 steps, each equal to the input graph on the reference check

    pixi run sandbox recipes/mvt/mvt_v5_1.json
    4 steps, each equal to the input graph on the reference check

    pixi run sandbox recipes/mvt/mvt_v4_1_1.json
    18 steps, each equal to the input graph on the reference check

`pixi run design` on the last step of `mvt_v5_1` and of `mvt_v5` gives the same design once the units' names are mapped, only the accelerator named `matvec_acc` in place of `mvt`; `mvt_v1_1` at `W=8` gives `mvt_v1`'s exactly.

## 3. The runs

Each form through the flow with the energy run, here `mvt_v5`:

    pixi run flow recipes/mvt/mvt_v5.json --platform platforms/snax_forge.json --set design.schedule=sequential --energy

The int64 forms that read A down a column run with `--set memory.passes.layout=padded_rows` (each row a word longer, so a column meets all 32 banks; section 4), `mvt_v1`, `mvt_v1_2` and `mvt_v2` with `--set W=8` (`W_x1=8 W_x2=8`). All twelve runs: data 0 errors (128 words). The window is SNAX-GVSoC's, from main to the check; vsim's is the same ELF on snax_cluster's RTL (open item 95); energy is dynamic, SNAX-GVSoC's counts priced per transaction (D177), against the static estimate; area is the estimate's, with the L1 banks and without them.

| # | Form | Type | Window | vsim | Dynamic (nJ) | Static (nJ) | Area (µm²) | Without banks (µm²) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `mvt_v1`, padded | int64 | 3 009 | 3 091 | 394.529 | 392.671 | 912 000 | 479 000 |
| 2 | `mvt_v1_2`, padded | int64 | 1 930 | 1 998 | 247.032 | 246.275 | 802 000 | 369 000 |
| 3 | `mvt_v2`, padded | int64 | 1 930 | 1 998 | 247.991 | 247.219 | 1 160 000 | 723 000 |
| 4 | `mvt_v3`, padded | int64 | 1 883 | – | 245.316 | 244.031 | 636 000 | 203 000 |
| 5 | `mvt_v4`, contiguous | int64 | 4 629 | 4 657 | 356.891 | 243.602 | 635 000 | 202 000 |
| 6 | `mvt_v4`, padded | int64 | 1 883 | 1 937 | 244.803 | 243.602 | 635 000 | 202 000 |
| 7 | `mvt_v3_1`, padded, `pipeline` | int64 | 1 391 | 1 397 | 246.253 | 244.031 | 831 000 | 398 000 |
| 8 | `mvt_v4_1`, padded, `sequential` | int64 | 13 003 | 13 061 | 586.167 | 584.966 | 635 000 | 202 000 |
| 9 | `mvt_v4_1_1`, `sequential` | int8 | 12 509 | 12 593 | 373.423 | 373.354 | 482 000 | 48 800 |
| 10 | `mvt_v5`, `sequential` | int8 | 1 578 | 1 646 | 27.056 | 26.997 | 562 000 | 129 000 |
| 11 | `mvt_v5`, `pipeline` | int8 | 904 | 913 | 27.056 | 26.439 | 562 000 | 129 000 |
| 12 | `mvt_v5_2`, `sequential` | int8 | 1 581 | 1 652 | 27.308 | 27.243 | 564 000 | 132 000 |

The banks are 433 000 µm² in every form (128 KiB of L1), so the column without them is the accelerators, their streamers and the crossbar. vsim is 0.4 % to 4.5 % over SNAX-GVSoC everywhere; 11 of these 12 points ran on vsim, with verdicts PERFECT or NEAR, none past 0.8 % beyond the fill.

## 4. How to read it

- **The layout comes first at int64.** In the contiguous layout a row is 64 words, so a column of A sits in one bank and half 2's W lanes queue on it: `mvt_v4` takes 4 629 cycles, half 2's counter 3 248 against 518 for half 1 (6.3×, under D86's 8×). One word of padding per row puts the column across all 32 banks: 1 883 cycles, both halves 514 and 521. The energy follows the cycles: 356.9 nJ against 244.8, the refused requests on half 2's streamers and the crossbar, which the static estimate (243.6) does not see.
- **The forms at int64, padded.** Passing the products through L1 (`mvt_v1`, three accelerators) costs 3 009 cycles and 394.5 nJ; linking them (`mvt_v1_2`, `mvt_v2`) 1 930 and 247 nJ; folding the `vadd` into the group (`mvt_v3` chained, `mvt_v4` seeded) 1 883 and 245 nJ at 202 000 µm², the smallest. A set of units per half (`mvt_v2`) buys nothing in series: the same 1 930 cycles as `mvt_v1_2` for 723 000 µm² against 369 000.
- **Pipelining the halves.** `mvt_v3_1` gives each half its own chain and starts half 2 while half 1 runs (D179): 1 391 cycles, 26 % under `mvt_v3`, for 398 000 µm² against 203 000 and about the same energy. The halves meet on the banks, which costs each counter about 40 cycles.
- **The copy.** Reading `A_T` makes half 2 as fast as half 1 in any layout, but the DMA's transposed copy is 4096 one-element bursts, 10 925 cycles: about 21× half 2, so `mvt_v4_1` takes 13 003 cycles against `mvt_v4`'s 1 883. Its energy is 341.4 nJ in both types (D184): each burst reads and writes a whole superbank of 8 banks through the DMA's wide port, whatever the element's size (X96, measured at gate level). At int8 the copy is 91 % of the run's energy (373.4 nJ), against 58 % at int64 (586.2 nJ). At int8 the copy is also the only way to read a column in the row form: 8 int8 lanes come from one bank word, which a column does not fill (D142 · D174).
- **The column form.** `mvt_v5` reads A along its rows in both halves. Half 2 keeps one sum per lane and adds over the rows, y₂[i] broadcast to the 8 lanes: 512 firings, as half 1, and y₂ read 64 words instead of 512. 1 578 cycles and 27.1 nJ against the copy's 12 509 and 373.4 nJ; without the copy `mvt_v4_1_1` would take about 1 584 cycles and 32.1 nJ, so the column form takes the same time for 16 % less energy, at 129 000 µm² of logic against 48 800 (two accelerators and a `Mac` in place of one shared group). `pipeline` runs the halves at once: 904 cycles (913 on vsim). The dynamic energy is the same as `sequential` because the energy run's quiet build still runs the halves one after the other (open item 140).
- **Seed or chain.** `mvt_v5_2` unseeds the `Mac` and adds x₂ in a linked `vadd`: 3 cycles, 0.25 nJ and 2 900 µm² more, most of it a control unit of 100 bits against 67. The seed (D187) is the form to keep.

## 5. What this gives the designer

At 64 × 64 and W = 8, best first, within each type (int64 and int8 move different data, so their energies are not compared):

| Measure | int64 | int8 |
|---|---|---|
| Cycles | `mvt_v3_1` pipelined (1 391), `mvt_v3` = `mvt_v4` (1 883), `mvt_v1_2` = `mvt_v2` (1 930), `mvt_v1` (3 009), `mvt_v4_1` (13 003) | `mvt_v5` pipelined (904), `mvt_v5` (1 578), `mvt_v5_2` (1 581), `mvt_v4_1_1` (12 509) |
| Energy | `mvt_v4` (244.8), `mvt_v3` (245.3), `mvt_v3_1` (246.3), `mvt_v1_2` (247.0), `mvt_v2` (248.0), `mvt_v1` (394.5), `mvt_v4_1` (586.2) | `mvt_v5` (27.06), `mvt_v5_2` (27.31), `mvt_v4_1_1` (373.4) |
| Area without banks | `mvt_v4_1` = `mvt_v4` (202 000), `mvt_v3` (203 000), `mvt_v1_2` (369 000), `mvt_v3_1` (398 000), `mvt_v1` (479 000), `mvt_v2` (723 000) | `mvt_v4_1_1` (48 800), `mvt_v5` (129 000), `mvt_v5_2` (132 000) |

**At int64, `mvt_v4` padded** is the one to keep: the least energy and area, and only `mvt_v3_1`'s pipeline is faster, by 26 % for twice the logic. **At int8, `mvt_v5`** wins on time and energy; `mvt_v4_1_1` is smaller only because one shared group does both halves, and its copy costs 8× the cycles and 14× the energy. The column form is the one that scales with the type: half 2 reads packed words as half 1 does, where the row form at int8 needs A copied.

How far this holds: the cycles are SNAX-GVSoC's, within 4.5 % of the RTL's; the energy is SNAX-GVSoC's counts at the cost tables' prices, within 1 % of the static estimate where the banks serve every beat and the halves run one after the other, and first order against gate level (D154). What it leaves out: the DMA engine's and the core's energy (D167; the copy at gate level is about 3× D184's price, R7g), the pipelined runs' energy (open item 140), and the column form one pass over A for both halves (open item 150).
