# Dry run 4: mvt, eleven forms from one kernel

Every number here is as the tools printed it when written; no test checks them (D98).

mvt is two matrix-vector products over one matrix, $x_1 \mathrel{+}= A y_1$ and $x_2 \mathrel{+}= A^T y_2$ (`kernels/polybench/mvt.py`; `mvt_types.py` with A and y at int8, x at int32). Half 1 reads A's rows; half 2 needs its columns. How should half 2 read A, and what do the halves share? Eleven answers from `recipes/mvt/` at 64 × 64, W = 8, on SNAX-GVSoC and vsim.

## 1. The forms

A recipe's `id` (D188): `dot1` or `dot2` each half's dot group (through L1 or linked), `add`, `chain` or `seed` how x is added (a `vadd` through L1, a linked `vadd`, or the fold starting from x), `shared` one group for both halves (D165), `dmaT` a DMA-transposed copy of A, `col` half 2 in the column form. `recipes/README.md` lists every id (checked by `tests/sandbox/test_recipe_names.py`).

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

Crossbar inputs are the streamer channels plus the core and AXI (D85). The estimate is ideal (D155), blind to half 2's bank conflicts. `mvt_v1_1`, `mvt_v1_2_1`, `mvt_v2_1` and `mvt_v5_1` are reuse twins of `mvt_v1`, `mvt_v1_2`, `mvt_v2` and `mvt_v5`, not run apart.

`mvt_v3` per half is `vmul → link → vsum → link → vadd`, the `vadd` at W = 1; `mvt_v3_1` is it without `share` (accelerators `mvt`, `mvt_1`), under D179's `--set design.schedule=serial|pipeline|pipeline_compute`; `sequential` (D183) syncs after every start. 64 is MINI's class rounded up so W = 4 and 8 divide (was 40 × 40, where D86's $n_\text{banks} / \gcd(p, n_\text{banks})$ banks for a column of pitch p gave W = 4 no conflict, W = 8 two-way, and `mvt_v2` 85 inputs, refused; 84 at 64 runs with a note).

## 2. Building them

`mvt_v4`'s first eight steps make half 1 a seeded dot 2 group:

| Step | Transform | What it does |
|---|---|---|
| 1, 2 | `tag_map tmp0_mult_map temporal`, `split_map tmp0_mult_map_1 W_x1` | the products' row loop in time, W lanes on n |
| 3 | `bind tmp0_mult simd chisel_tiled_spatial vmul_x1` | the multiplier |
| 4, 5 | the same two on `sum_map` | the sum's loops |
| 6 | `fuse_seed sum_map_1 augassign_18_4_map` | the fold starts from x₁, so no `vadd` (D163) |
| 7 | `bind sum accumulate chisel_adder_tree vsum_x1` | the adder tree |
| 8 | `link tmp0_prod D_2 group mvt` | the products through a one-beat buffer, not L1 |

Half 2 repeats them after `mark_transpose` of A (step 9, D164; `A[m, n]`, m inner), and step 18 `share`s the groups. `mvt_v4_1_1` is the same 18 steps on `mvt_types` at `i8` (`OP_TYPE`; presets `i8`, `i16`, `i32`, only `i8` run, open item 152; `--set types=`, `--set dtypes.<name>=`).

- **The copy (D182).** `data_transpose tmp2_mult_map A dma` adds a `transpose · dma` node copying A into the L1 transient `A_T` (by core: refused, open item 143): one element a burst (`l1_to_l1`), one `snrt_dma_start_2d` of 64 bursts per row, half 2 after it (D135). Estimate: 4096 bursts, 10 923 cycles, `total_cycles` 1028 → 11 951; D184 prices a beat on its whole superbank of `wide_bits` / `width_bits` = 8 banks (X96: 8 bank reads and 8 writes a burst): +327.87 nJ banks, +13.49 nJ crossbar (+40.98 nJ banks on one bank a beat, before the amendment). SNAX-GVSoC counts `copies copy_reads copy_writes` per beat (`dma-timing`: 25 088 each) and `bytes_l1_to_l1`.
- **The slicer (D180)**, the packer's mirror for narrow seeds and inputs, priced per pop from X94's six points (D176, D177). **The broadcast (D186)**: an input lacking the lane variable moves one lane, wired to every lane (`<unit>_<port>_bcast`).
- **The seeded `Mac` (D163 · D187)**: s ← (first ? seed : s) + a·b, flops unchanged, priced unseeded as `mvt_v4`'s seeded `vsum`. `mac_x2` is `Mac_i8xi8_i32_i32_n8_acc_seed`, `b` through `FifoSlicer_w1_i8_n1` and `mac_x2_b_bcast`; the 75 bundles of the quick, mvt and pack sets kept their SV byte for byte.

`mvt_v5` builds half 2 in the column form in nine steps: `tag_map`, `split_map` and `interchange_map` put the W lanes on x₂'s index with the rows in time, then `fuse_seed` into `augassign_19_4_map`, `fuse_map`, and `bind tmp4_mult_map_s mac chisel_mac mac_x2`.

By reuse (`mvt_v5_1`) it is four steps: `lift matvec_acc` at each half, then `apply_recipe` `matvec_acc_v3` (seeded dot 2 per row) on half 1 and `matvec_acc_v4` (column form, seeded; `matvec_v7` unseeded) on half 2; the column recipe on half 1 would read A down its columns, refused at int8 (`connect.pack`).

    pixi run sandbox recipes/mvt/mvt_v5.json
    17 steps, each equal to the input graph on the reference check

    pixi run sandbox recipes/mvt/mvt_v5_1.json
    4 steps, each equal to the input graph on the reference check

    pixi run sandbox recipes/mvt/mvt_v4_1_1.json
    18 steps, each equal to the input graph on the reference check

`pixi run design` gives `mvt_v5_1` and `mvt_v5` the same design up to names; `mvt_v1_1` at `W=8` gives `mvt_v1`'s exactly.

## 3. The runs

Here `mvt_v5`:

    pixi run flow recipes/mvt/mvt_v5.json --platform platforms/snax_forge.json --set design.schedule=sequential --energy

int64 forms reading A down a column add `--set memory.passes.layout=padded_rows` (a row one word longer, so a column meets all 32 banks); `mvt_v1`, `mvt_v1_2`, `mvt_v2` take `--set W=8` (`W_x1=8 W_x2=8`). All twelve: data 0 errors (128 words). vsim runs the same ELF on the RTL (open item 95); dynamic energy is SNAX-GVSoC's counts priced per transaction (D177).

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

Banks: 433 000 µm² everywhere (128 KiB). vsim is 0.4 % to 4.5 % over SNAX-GVSoC; 11 of 12 points ran on vsim, PERFECT or NEAR, none past 0.8 % beyond the fill.

- **vsim, data exact:** first round 25 bundles, 5 PERFECT, 20 NEAR (`mvt_v2`, the first above 64 crossbar inputs, with `mvt_v1_2`'s counters and windows); `mvt_v3_1`'s six points 1/5; `mvt_v4`, `mvt_v4_1` `sequential` 3/1 (`mvt_v4_1` contiguous, half 1 518 against 514); `mvt_v4_1_1` padded PERFECT, contiguous NEAR (518 against 514); `mvt_v5` (three) and `mvt_v5_2` 3/1. Full set at the close: 26 bundles, 17 PERFECT, 9 NEAR, every counter and window as before.
- **The copy:** 10 924 cycles on the RTL, first backend request to last W in the iDMA trace (10 925 counting both ends), in both types and layouts; SNAX-GVSoC 10 925.
- **`mvt_v4`, `mvt_v4_1` `sequential`:** half 2's BUSY_COUNTER 521 padded, 518 and 521 on A_T (estimate 517); windows 4792, 2021, 12 987, 13 003.
- **`mvt_v4_1_1` `sequential`:** BUSY_COUNTER 514, 517 contiguous, 519, 518 padded (estimate 514); x's sliced reader and packed writer 32 words a half; padded window 12 526, contiguous 478 under `mvt_v4_1`'s 12 987 (A's load 113 against 561); `bytes_l1_to_l1` 4096 (int64 32 768); `dma_copy` 4096 beats each way, 327.87 nJ banks, 13.49 nJ crossbar; dynamic 373.42 contiguous, 373.46 padded against static 373.35 nJ; `mvt_v4_1` 585.94 against 584.97. Before D184's amendment: 86.53, 86.57 against 86.46; `mvt_v4_1` 244.6 → 299.1 (static 298.1).

## 4. How to read it

- **Layout first at int64.** Contiguous, a column sits in one bank: `mvt_v4` takes 4 629 cycles, half 2's counter 3 248 against half 1's 518 (6.3×, under D86's 8×). Padded, 1 883, the halves 514 and 521, 244.8 nJ against 356.9: refused requests the static estimate (243.6) misses.
- **The forms, padded int64.** Through L1 (`mvt_v1`): 3 009 cycles, 394.5 nJ; linked (`mvt_v1_2`, `mvt_v2`): 1 930, 247 nJ; `vadd` folded in (`mvt_v3` chained, `mvt_v4` seeded): 1 883, 245 nJ at 202 000 µm². `mvt_v2`'s units per half buy nothing in series: 723 000 µm² against 369 000.
- **Pipelining.** `mvt_v3_1` (D179): 1 391 cycles, 26 % under `mvt_v3`, for 398 000 µm² against 203 000; the banks cost each counter about 40 cycles.
- **The copy.** With `A_T` half 2 is as fast as half 1, but the copy takes about 21× half 2: 13 003 cycles against 1 883. It costs 341.4 nJ in both types: 91 % of the run at int8 (373.4 nJ), 58 % at int64 (586.2 nJ). At int8 it is the row form's only way to a column: 8 lanes come from one bank word, which a column does not fill (D142 · D174).
- **The column form.** `mvt_v5` reads rows in both halves: half 2 keeps a sum per lane, y₂[i] broadcast, 512 firings as half 1, y₂ read 64 words instead of 512. 1 578 cycles and 27.1 nJ against the copy's 12 509 and 373.4; without the copy `mvt_v4_1_1` would take about 1 584 cycles and 32.1 nJ, so 16 % less energy at the same time, for 129 000 µm² against 48 800. `pipeline`: 904 cycles (913 on vsim), energy as `sequential`: the quiet build runs the halves in turn (open item 140).
- **Seed or chain.** `mvt_v5_2` (unseeded `Mac`, linked `vadd`): 3 cycles, 0.25 nJ, 2 900 µm² more (2 600 first measured), a control unit of 100 bits against 67. Keep the seed (D187).

## 5. What this gives the designer

64 × 64, W = 8, best first, per type (energies not compared across types):

| Measure | int64 | int8 |
|---|---|---|
| Cycles | `mvt_v3_1` pipelined (1 391), `mvt_v3` = `mvt_v4` (1 883), `mvt_v1_2` = `mvt_v2` (1 930), `mvt_v1` (3 009), `mvt_v4_1` (13 003) | `mvt_v5` pipelined (904), `mvt_v5` (1 578), `mvt_v5_2` (1 581), `mvt_v4_1_1` (12 509) |
| Energy | `mvt_v4` (244.8), `mvt_v3` (245.3), `mvt_v3_1` (246.3), `mvt_v1_2` (247.0), `mvt_v2` (248.0), `mvt_v1` (394.5), `mvt_v4_1` (586.2) | `mvt_v5` (27.06), `mvt_v5_2` (27.31), `mvt_v4_1_1` (373.4) |
| Area without banks | `mvt_v4_1` = `mvt_v4` (202 000), `mvt_v3` (203 000), `mvt_v1_2` (369 000), `mvt_v3_1` (398 000), `mvt_v1` (479 000), `mvt_v2` (723 000) | `mvt_v4_1_1` (48 800), `mvt_v5` (129 000), `mvt_v5_2` (132 000) |

**int64: `mvt_v4` padded**, least energy and area; `mvt_v3_1` is 26 % faster for twice the logic. **int8: `mvt_v5`**; `mvt_v4_1_1` is smaller, but its copy costs 8× the cycles and 14× the energy.

How far this holds: cycles within 4.5 % of the RTL's; energy within 1 % of the static estimate where the banks serve every beat and the halves run in turn, first order against gate level (D154). Left out: the DMA's and core's energy (D167; the copy at gate level is about 3× D184's price, R7g), the pipelined energy (open item 140), and the column form's one pass over A (`mvt_v6`, dry run 5).
