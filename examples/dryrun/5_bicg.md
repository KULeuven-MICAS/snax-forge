# Dry run 5: bicg, from two passes over A to one

Every number here is as the tools printed it when written; no test checks them (D98).

bicg is mvt without the add: $q = A p$ and $s = A^T r$ (`kernels/polybench/bicg.py`; `bicg_types.py` with A, p, r at int8 and q, s at int32, presets as mvt's, open item 152). q reads A's rows, s its columns, as mvt's half 2 (dry run 4). How does s read A, and must A be read twice? `recipes/bicg/` at 64 × 64 on SNAX-GVSoC and vsim, then one pass against two by W.

## 1. The forms

Each `id` (D188) names q's half, then s's: `dot2` the linked dot group, `col` the column form (W `Mac` lanes on s's index, r broadcast, D186), `colbank` with its sums in a bank (D191), `fork` A read once for both (D189), `strip` the row form in strips.

| Recipe | `id` | How s reads A | Accelerators | Crossbar inputs | Estimate (cycles) |
|---|---|---|---:|---:|---:|
| `bicg_v1` | `bicg_dot2` | down a column (the row form on Aᵀ) | 2 | 36 | 514 + 514 |
| `bicg_v2` | `bicg_dot2_col` | along its rows, the column form | 2 | 36 | 514 + 513 |
| `bicg_v2_1` | `bicg_dot2_col_i8_o32` | the same at int8 | 2 | 11 | 514 + 514 |
| `bicg_v3` | `bicg_dot2_col_fork_i8_o32` | the column form on q's read of A, at N = W (64 × 32, W = 32) | 1 | 28 | 67 |
| `bicg_v4` | `bicg_dot2_colbank_fork_i8_o32` | the column form on q's read, s's N sums in a bank of R rows | 1 | 8 | 1 027 (W = 4, R = 16) |
| `bicg_v4_1` | `bicg_dot2strip_colbank_fork_i8_o32` | the same in two strips of R·W columns, q carried across them | 1 | 9 | 1 027 (W = 4, R = 8) |

W = 8 unless named; two estimates under `sequential`, the longer one under `pipeline` (`bicg_v3`'s 67 against 66 for its two passes pipelined). Crossbar inputs as in dry run 4 (D85). The fork (`fork_read`) gives one streamer to a shared input and a JOBS per chain (`JOBS_<SINK>`); a bank has `rows` R (the `mac` BRM, default 1) and a `ROWS` CSR, a held input skipping the bank's loops; a strip's carried seed (`carry_seed`) a `CARRY` CSR, N a multiple of R·W; the seeded bank reads a seed beat a row (D194), `mvt_v6` one strip at W = 8, 16, 32 (R = 64 / W).

| Recipe | W, R | JOBS_SUM | JOBS_MAC | MAC_JOB_BEATS | ROWS | SUM_CARRY |
|---|---|---:|---:|---:|---:|---:|
| `bicg_v4` (`matvec_v8`, `matvec_colbank`) | 4, 16 | 64 | 1 | 1 024 | 16 | – |
| `bicg_v4_1` (`matvec_v9`, `matvec_dot2strip`) | 4, 8 | 128 | 2 | 512 | 8 | 64 |

The one pass (`bicg_v4`): rows of A outer, each word feeding `vmul → link → vsum` (q[i] once a row) and the `Mac` bank (R rows of W sums, row = beat mod R), r[i] held for the row (one word per 8 rows), s stored after the last row.

## 2. Building them

`bicg_v1` and `bicg_v2` are four steps by reuse (D166): `lift matvec` at q and s, then `apply_recipe` `matvec_v2` (dot group per row) on q and `matvec_v2` (`bicg_v1`) or `matvec_v7` (column form, `bicg_v2`) on s. `bicg_v2_1` is `bicg_v2` at `types=i8`; `bicg_v3` adds `fork_read A`.

`bicg_v4` uses `matvec_v8` on s (the column form rows outer, strips of R·W columns, the `Mac` with "bank rows" R, r held, D190). `bicg_v4_1` strips q too (`matvec_v9`, `S = R * W`), q[i] written once per strip, the first strip's seed masked to 0 (D192).

    pixi run sandbox recipes/bicg/bicg_v4.json
    5 steps, each equal to the input graph on the reference check

    pixi run sandbox recipes/bicg/bicg_v4_1.json
    5 steps, each equal to the input graph on the reference check

mvt's one pass, `mvt_v6` (`mvt_dot2seed_colbankseed_fork_i8_o32`), is 24 steps by hand: half 1 as `mvt_v5`, half 2 as `matvec_v8` with `fuse_seed` into x₂, then `fork_read A`; bank row ρ starts from x₂'s beat ρ (D194).

    pixi run sandbox recipes/mvt/mvt_v6.json
    24 steps, each equal to the input graph on the reference check

- **The RTL.** A fork adds no register (58 group cases); the fixture `bicg_v3_w64` (64 × 64, W = 64) counts 67. `Mac_..._r<R>` keeps R rows of W sums, the control's one new register `<unit>_row` (a result when d + ROWS ≥ JOB_BEATS); the seeded `Mac_..._acc_seed_r<R>` adds one W-lane mux, its fixture `mvt_v6` counting 515 with the estimate's 200 control bits. Existing SV stayed byte for byte (627 files at the bank, 568 at the seeded bank).
- **The bank's price (D193)**, the rows-1 `Mac` plus X97's `bank` forms: `bicg_v4`'s `mac` 16 423 µm², 11.10 pJ a fire (measured 16 515, 11.43; was 12 598, 38.69), `bicg_v4_1`'s 9 087, 9.73 (measured 9 021, 8.92).
- **Energy windows (D154 · D172, D177).** Control priced over its own group's busy cycles: dot 5 unlinked 59.811 → 59.755 nJ, `mvt_v5` `sequential` control 1.115 → 0.557 nJ; a pipelined run, the default build in one window. On `bicg_v2_1` `sequential`, one window against the quiet build's 13: streamers −0.003, crossbar −0.001 nJ (4 refused requests against 5), control −0.003 nJ (0.5 %), the rest equal; groups wait 3 cycles each on their first inputs (each accelerator 1 cycle after its streamer), 10 in the default; 25.859 against 25.866 nJ (static 25.840).
- **Fixtures**: five `forge/` designs; recorded runs `bicg_v3` and `bicg_v4_1` (40 × 16, R = 2, the smallest M LOWER's carry takes); 66 group testbench cases, 863 Chisel specs; `--trace beat` checks each output against its `rate` (D174).

## 3. The runs

Here the one pass at W = 8:

    pixi run flow recipes/bicg/bicg_v4.json --platform platforms/snax_forge.json --set W=8 --set R=8 --energy

All runs: data 0 errors; columns as in dry run 4 (the RTL check, D177), area without the 433 000 µm² of L1 banks. 64 × 64, W = 8:

| # | Form | Type | Window | vsim | Dynamic (nJ) | Static (nJ) | Without banks (µm²) |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | `bicg_v1`, contiguous, `sequential` | int64 | 4 638 | 4 685 | 355.467 | 242.500 | 380 000 |
| 2 | `bicg_v1`, contiguous, `pipeline` | int64 | 4 036 | 4 110 | 370.602 | 242.500 | 380 000 |
| 3 | `bicg_v1`, padded, `sequential` | int64 | 1 954 | 2 006 | 243.039 | 242.500 | 380 000 |
| 4 | `bicg_v1`, padded, `pipeline` | int64 | 1 325 | 1 341 | 248.865 | 242.500 | 380 000 |
| 5 | `bicg_v2`, `sequential` | int64 | 1 903 | 1 977 | 212.972 | 212.008 | 378 000 |
| 6 | `bicg_v2`, `pipeline` | int64 | 1 575 | 1 621 | 243.382 | 212.008 | 378 000 |
| 7 | `bicg_v2`, padded, `pipeline` | int64 | 1 448 | 1 481 | 226.3 | 212.008 | 378 000 |
| 8 | `bicg_v2_1`, `sequential` | int8 | 1 426 | 1 485 | 25.866 | 25.840 | 91 400 |
| 9 | `bicg_v2_1`, `pipeline` | int8 | 879 | 887 | 26.174 | 25.840 | 91 400 |
| 10 | `bicg_v4`, R = 8 | int8 | 855 | 873 | 26.447 | 26.428 | 97 000 |

At other sizes and widths:

| Form | Size, W | Window | vsim | Dynamic (nJ) | Static (nJ) | Without banks (µm²) |
|---|---|---:|---:|---:|---:|---:|
| `bicg_v2_1`, `sequential` | 40 × 64, 8 | 1 055 | 1 115 | 16.298 | 16.280 | 91 400 |
| `bicg_v2_1`, `pipeline` | 40 × 64, 8 | 648 | 661 | 16.432 | 16.280 | 91 400 |
| `bicg_v2_1`, `pipeline` | 64 × 32, 32 | 390 | 396 | 12.71 | 12.21 | 267 000 |
| `bicg_v3` | 64 × 32, 32 | 392 | 400 | 10.49 | 10.09 | 238 000 |
| `bicg_v4`, R = 16 | 64 × 64, 4 | 1 368 | 1 383 | within 2.3 % of static | 30.84 | 79 500 |
| `bicg_v4`, R = 16 | 40 × 64, 4 | 963 | 968 | within 2.3 % of static | – | 79 500 |
| `bicg_v4_1`, R = 8 | 64 × 64, 4 | 1 408 | 1 407 | within 2.3 % of static | 30.36 | 81 700 |

On vsim, data exact: the four bank points PERFECT (the carried seed's mask exact), `bicg_v3` NEAR 4.2 % (78; open items 129, 156); the 11 two-pass points 3 PERFECT, 8 NEAR, none past 3.9 % beyond the fill (open items 139, 155); `mvt_v6` PERFECT at W = 16, NEAR 0.2 % and 1.5 % at 8 and 32 (vsim 515 and 131, the rule's count, against 519 and 136); `mvt_v5` `pipeline` NEAR (0.6 % at W = 8, open item 139; within the fill at 16 and 32). The two passes at int8 fail at W = 4: s's column read steps 64 elements, so a word's two 4-lane slices are not popped in turn (`connect.pack`).

- **Two-pass counters** within 10 % of the estimate (514, 513; 322 at 40 × 64) but three, all pipelined: `bicg_v1` contiguous's `matvec` 1727, behind `matvec_1`'s one-bank column read (3263 `pipeline`), and `bicg_v2`'s 840 and 739 at int64 (530 and 534 at int8). r's reader: 4096 words in the row form, 40 at 40 × 64.
- **`bicg_v3`**: counter 72 against 67 (+7.5 %), 5 `stall_in` cycles: 1 r's slicer taking its first word once the group is busy (D180), 4 A's row 56 refused 11 cycles on banks 0–3 by p's reader, rereading p every row (open item 156). `bicg_v2_1` at 64 × 32 `pipeline`: q 81 (+22.7 % of 66: a fill of 3 and the same conflicts at rows 8 and 56), s 69 (+4.5 %).
- **The bank points**: counters within a cycle of the estimate, 1030 against 1027 (+0.3 %), 647 against 643, 1030 (`bicg_v4_1`), 517 against 515 (W = 8). At W = 4 streamers and crossbar are 11 % over static, refused requests priced as granted (p's reader refused 121 times on banks 0–7 on `bicg_v4`, 73 + 2 at 40 rows, 125 + 5 + 1 on `bicg_v4_1`; open item 4).
- **L1 words**: r 8 (`bicg_v4`), 5 (40 rows), 16 (`bicg_v4_1`, 8 a strip); q's seed 64 on `bicg_v4_1`, 32 a strip, the second strip's the first's write, the first's 0.
- **`--trace beat`**: the bank's 16 results leave in the job's last 16 cycles, `first` and `last` 64 firings each, `bicg_v4_1`'s 8 at each strip's end; r's reader takes `bicg_v4`'s 8 words at the start, `bicg_v4_1`'s second strip a word every 64 cycles.

## 4. How to read it

- **Layout first at int64.** Contiguous, `bicg_v1` takes 4 638 cycles, s's counter 3 247 against q's 514 (6.3×); padded, 1 954. `bicg_v2` reads rows in any layout: 1 903 cycles, 213.0 nJ, the least at int64, r broadcast (512 words against 4 096).
- **Pipelining at int64 meets the banks.** `bicg_v2`'s column form stays on 8 banks for 64 beats while q's reader and p sweep all 32; A's, p's and r's readers are refused on 30–36 % of requests: 1 575 cycles, 17 % under `sequential`, for 30 nJ more. Padding moves the strip a bank a beat (1 448, 226.3 nJ). The estimate misses it (open item 155); the RTL agrees.
- **int8, two passes pipelined** (`bicg_v2_1`): 879 cycles, 26.2 nJ, 91 400 µm², r read 64 words.
- **One pass at N = W.** `bicg_v3` (64 × 32, W = 32) against its twin: 392 against 390 cycles, 17 % less energy (10.49 against 12.71 nJ), 11 % less logic; s's sums fit one `Mac` row, no bank.
- **One pass with a bank.** At 64 × 64, W = 8, R = 8, `bicg_v4` is 2.7 % faster than two passes pipelined (855 against 879) and 1.0 % dearer (26.45 against 26.17 nJ; static 26.428 against 25.840, +2.3 %), one group busy 517 against two at once (530, 534). One read of A saves 5.2 nJ: 2.059 of streamers (1076 beats against 1701), 1.150 of crossbar, 2.010 of memory (1032 reads against 1600, r's 64 beats now 8). The bank's `Mac` costs 5.5 nJ: 19.450 pJ a fire against the row's 8.693 (9.958 against 4.451 nJ). Area 530 000 against 524 000 µm² with the banks; without, datapath 24 000 against 10 600, streamers 36 400 against 40 900, crossbar 34 500 against 37 800 (+6.1 % in all).
- **Strips.** `bicg_v4_1`, two strips of R·W = 32 sums: 1 408 against 1 368 cycles, 81 700 against 79 500 µm²; q read back each strip (D192).
- **What the add costs**, against the mvt twin: `bicg_v2_1` against `mvt_v5`, both pipelined, 879 against 904 cycles, 26.17 against 26.83 nJ; `bicg_v4` against `mvt_v6` at W = 8, 855 against 899, 26.45 against 27.10: about 0.65 nJ and 25–45 cycles for x₁ and x₂. At int64 `bicg_v1` padded against `mvt_v4` padded (one shared group): 1 954 against 1 883 cycles, 243.0 against 244.8 nJ.

## 5. One pass or two, by W

One pass reads A once; s's sums then live in a bank of R = N / W rows, each firing paying to pick its row. 64 × 64, int8, two passes under `pipeline`:

| Kernel | W, R | Window, one pass / two | vsim, one / two | Dynamic nJ, one / two | Change | Without banks (µm²), one / two |
|---|---|---|---|---|---:|---|
| bicg | 8, 8 | 855 / 879 | 873 / 887 | 26.447 / 26.174 | +1.0 % | 97 000 / 91 400 |
| bicg | 16, 4 | 596 / 592 | 614 / 622 | 24.245 / 25.000 | −3.0 % | 148 000 / 150 000 |
| bicg | 32, 2 | 471 / 484 | 489 / 498 | 22.470 / 24.547 | −8.5 % | 247 000 / 267 000 |
| mvt | 8, 8 | 899 / 904 | 922 / 913 | 27.101 / 26.830 | +1.0 % | 135 000 / 129 000 |
| mvt | 16, 4 | 643 / 638 | 651 / 653 | 24.899 / 25.652 | −2.9 % | 213 000 / 215 000 |
| mvt | 32, 2 | 518 / 505 | 538 / 518 | 23.207 / 25.216 | −8.0 % | 371 000 / 390 000 |

mvt by part (nJ, SNAX-GVSoC with `--energy`, static in brackets):

| W | Form | Datapath | Control | Streamers | Crossbar | Memory | Dynamic (nJ) | L1 reads |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | one pass | 16.278 | 0.566 | 3.832 | 2.134 | 4.292 | 27.101 (27.028) | 1096 |
| 8 | two passes | 10.770 | 0.578 | 5.894 | 3.286 | 6.301 | 26.830 (26.439) | 1664 |
| 16 | one pass | 14.622 | 0.285 | 3.560 | 2.141 | 4.292 | 24.899 (24.808) | 1096 |
| 16 | two passes | 10.519 | 0.295 | 5.396 | 3.254 | 6.188 | 25.652 (25.210) | 1632 |
| 32 | one pass | 13.092 | 0.147 | 3.493 | 2.184 | 4.292 | 23.207 (23.001) | 1096 |
| 32 | two passes | 10.393 | 0.152 | 5.244 | 3.296 | 6.132 | 25.216 (24.601) | 1616 |

bicg and mvt move together. With M × N = 4 096 elements of A:

- **Reading A once saves** the second read's streamer beats and crossbar grants: 5.22, 4.85 and 4.70 nJ on mvt at W = 8, 16, 32 (520 to 568 fewer L1 reads), about constant: $S \approx 5.0 / 4096 = 1.22$ pJ an element.
- **The bank costs** (D193) the plain `Mac` plus $W (0.343 \log_2 R + 0.316)$ pJ a firing (X97, fitted at R = 2 to 16), over $MN/W$ firings: $MN (0.343 \log_2 R + 0.316)$ pJ. W = 8, R = 8: $4096 \times (0.343 \times 3 + 0.316) = 5.5$ nJ; W = 16, R = 4: 4.1 nJ; W = 32, R = 2: 2.7 nJ, as measured (5.51, 4.10, 2.70 nJ: firings 512, 256, 128, the premium a fire 10.8, 16.0, 21.1 pJ). x₂'s seed adds nothing (D194: priced unseeded, 8 beats a run).

So one pass pays when $0.343 \log_2 R + 0.316 < 1.22$, $R < 2^{2.64} \approx 6$ rows: at N = 64 from W = 16. Area too: R·W·32 flops is N·32 at any W, while the dropped reader and control unit grow with W. N = 64 has no W between 8 and 16.

| Measured against the estimate | bicg | mvt |
|---|---|---|
| Counters, one pass | 261 / 262, 134 / 134 | 519, 263, 136 / 518, 262, 134 (+0.2 % to +1.5 %) |
| Counters, two passes | 269 and 267, 139 and 138 / 261, 133 (within 4.5 %) | 531 and 537, 272 and 272, 141 and 138 / 517, 261, 133 (+2.7 % to +6.0 %) |
| Estimate, one / two | −1.6 % at W = 16 (24.209 / 24.611 nJ), −6.7 % at 32 (22.403 / 24.004) | 515, 259, 131 / 514, 258, 130 cycles; 27.03, 24.81, 23.00 / 26.44, 25.21, 24.60 nJ (+2.2 %, −1.6 %, −6.5 %) |
| Two passes over it | refused 78 (W = 16) and 113 (32), A's readers 48 and 56 (one pass 7, 14): streamers 4.8 %, 7.2 %, crossbar 4.7 %, 6.7 %; L1 reads 1032 / 1568, 1552 | streamers +4.2 % to +7.7 %, crossbar +4.2 % to +7.5 % (one pass +1.1 % to +3.7 %) |
| Balance | | area without banks +4.7 %, −0.9 %, −4.9 % (estimate +4.7 %, −1.0 %, −4.8 %); windows −0.6 %, +0.8 %, +2.6 % (vsim +1.0 %, −0.3 %, +3.9 %) |

The windows stay within a few percent: the one pass is busy 72–84 cycles less but starts 43–52 later (r, five streamers) and stores q after its run (16–24 cycles); the two passes' half 2 starts 66–69 cycles after half 1 (the core's configuration, outside D179's longest path); on mvt the one pass is busy 5 to 18 cycles less than the longer half (10 to 16 on vsim). On mvt at W = 8 and 32 SNAX-GVSoC counts 4–5 cycles over the rule, vsim the rule: A, y₁, x₁ and x₂ start at multiples of 256 bytes, their first words all in bank 0 (open item 65).

## 6. What this gives the designer

At 64 × 64, best first, within each type:

| Measure | int64 (W = 8) | int8 |
|---|---|---|
| Cycles | `bicg_v1` padded pipelined (1 325), `bicg_v2` padded pipelined (1 448), `bicg_v2` pipelined (1 575), `bicg_v2` (1 903), `bicg_v1` padded (1 954), `bicg_v1` contiguous (4 036 pipelined, 4 638) | `bicg_v4` W = 32 (471), `bicg_v2_1` W = 32 pipelined (484), `bicg_v2_1` W = 16 pipelined (592), `bicg_v4` W = 16 (596), `bicg_v4` W = 8 (855), `bicg_v2_1` pipelined (879), `bicg_v2_1` (1 426) |
| Energy | `bicg_v2` (213.0), `bicg_v2` padded pipelined (226.3), `bicg_v1` padded (243.0), `bicg_v2` pipelined (243.4), `bicg_v1` padded pipelined (248.9), `bicg_v1` contiguous (355.5, 370.6 pipelined) | `bicg_v4` W = 32 (22.47), `bicg_v4` W = 16 (24.25), `bicg_v2_1` W = 32 pipelined (24.55), `bicg_v2_1` W = 16 pipelined (25.00), `bicg_v2_1` W = 8 (25.87), `bicg_v2_1` W = 8 pipelined (26.17), `bicg_v4` W = 8 (26.45) |
| Area without banks | `bicg_v2` (378 000), `bicg_v1` (380 000) | `bicg_v4` W = 4 (79 500), `bicg_v4_1` W = 4 (81 700), `bicg_v2_1` W = 8 (91 400), `bicg_v4` W = 8 (97 000) |

**int64: `bicg_v2`**, rows in any layout, r read once, least energy; pipelined, 17 % of time for 14 % of energy. **int8: the one pass (`bicg_v4`)** from W = 16: 3–9 % less energy, 1–8 % less logic, same time. At W = 8 the two passes pipelined (`bicg_v2_1`) are as good; the smallest is the one pass at W = 4. mvt likewise (`mvt_v6` against `mvt_v5`).

How far this holds: vsim within −0.1 % to +5.7 % of SNAX-GVSoC's windows; energy within 2.3 % of static except where readers are refused (open item 4), first order against gate level (D154). The bank price is X97's fit, within 14 % in energy and 5.8 % in area up to 2 048 bits a mux (D193); 4 096 or more (R = 16 at W = 8) is outside it. mvt gets one strip only: strips would carry x₁'s tree through L1.
