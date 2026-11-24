# Dry run 2: dot 1, two accelerators through L1

Every number here is as the tools printed it when written; no test checks them (D98).

dot is $\mathit{out} = \sum_i A_i \cdot B_i$ over N int64 elements (`kernels/polybench/dot.py`). In dot 1 a multiplier `acc_1` writes the products to a transient `tmp0` in L1 and an adder tree `acc_2` sums them: two accelerators, two streamer sets of the one entry `snax_forge_<design>` (D157), run in series (D158). Section 7 links them, section 9 encapsulates them.

What does the round trip through L1 cost, should the two get different lanes, and is one streamer set or two the better shell?

## 1. Start the editor on the kernel

In snax-forge's image (`docs/TOOL.md` section 1), from the folder that holds `snax-forge`:

    podman run --rm -it --network host -v "$PWD":"$PWD" -w "$PWD" ghcr.io/kuleuven-micas/snax-forge:main
    cd snax-forge && pixi install
    pixi run edit-dfg --kernel dot --set N=2048

    serving dot at http://127.0.0.1:8767/ (Ctrl-C to stop)

Step 0: `mult_map · i · 2048 · untagged` holding `mult · in1 * in2`, `sum_map · i · 2048 · untagged` holding `sum · fold +`, `tmp0` between. Containers `A: N · L1 · 16 KiB`, `B` the same, `out: 1 · L1 · 8 B`, `tmp0: N · L1 · 16 KiB`. Estimate `accelerator 0 + unmapped 4096 = total 4096` (D155), 6144 words read, 4096 written; L1 48 KiB of 128 KiB.

## 2. Eight steps

| Step | Select | Apply | What changes |
|---|---|---|---|
| 1 | `mult_map` | `split`, factor `W` with its value box set to 8 | `mult_map` runs `i_1` over 256 around `mult_map_inner` running `i_2` over 8; Bindings shows `mult_map split · W = 8 · not bound` |
| 2 | `mult_map` | `tag_map`, `temporal` | the streamers' loop |
| 3 | `mult_map_inner` | `tag_map`, `spatial` | the lanes; the total stays at 4096, since a tag alone runs nothing |
| 4 | `mult` | `bind`, `simd / chisel_tiled_spatial`, instance `acc_1`, the field's default | the log ends `C 0 → 256 · unmapped 4096 → 2048 · renamed W → W_acc_1 · 1 note` (`sum` is still unbound); total 2304 |
| 5 | `sum_map` | `split`, factor `W`, 8 | `sum_map` runs `i_3` over 256 around `sum_map_inner` running `i_4` over 8 |
| 6 | `sum_map` | `tag_map`, `temporal` | |
| 7 | `sum_map_inner` | `tag_map`, `spatial` | |
| 8 | `sum` | `bind`, `accumulate / chisel_adder_tree`, instance `acc_2`, the field's default | `C 256 → 513 · unmapped 2048 → 0 · renamed W → W_acc_2`; 0 problems, 0 notes |

A param used by one unit takes its name on the bind (`W_acc_1`, `W_acc_2`); the second split takes the next free loop names (D73).

At step 8: `chisel_adder_tree` (latency 1, II 1, span 1) or `chisel_serial` (span 8); `mac` needs `split_reduce` (D150), the element-wise patterns are refused (`'out = in1' is not one operator over distinct inputs`). Each accelerator has a dashed outline.

The Estimate after step 8:

| Group | Cycles | Bottleneck | Ramp | Stall | Expected BUSY_COUNTER | Channels | Area (µm²) | Energy (nJ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acc_1 | 256 | 256 | 0 | 0 | 259 | 24 | 56 000 | 23.393 |
| acc_2 | 257 | 256 | 1 | 0 | 259 | 9 | 4 110 | 1.740 |

The two add through L1:

$$C = \frac{N}{W_{\mathit{acc}_1}} + \frac{N}{W_{\mathit{acc}_2}} + 1 = 256 + 256 + 1 = 513.$$

The expected BUSY_COUNTER adds the fill (3 for two inputs, 2 for one). Reads 6144 words, writes 2049; `2 streamers (not costed, open item 110) · 40 CSRs: acc_1 960–981, acc_2 982–999`. The cost on nangate45:

| Accelerator | Block | Datapath | Area (µm²) | Energy (nJ) | Share |
|---|---|---|---:|---:|---:|
| acc_1 | acc_1 | SimdLanes | 55 300 | 23.137 | 45.9 % |
| acc_1 | acc_1 ctrl | control unit | 702 | 0.257 | 0.5 % |
| acc_1 | L1 accesses (placeholder, 4 × e_load per bit) | 6144 words, 393 216 bits | – | 27.005 | 53.6 % |
| acc_1 | total | | 56 000 | 50.398 | 100 % |
| acc_2 | acc_2 | TreeAccumulate | 2 710 | 1.231 | 11.5 % |
| acc_2 | acc_2 ctrl | control unit | 1 390 | 0.509 | 4.7 % |
| acc_2 | L1 accesses (placeholder, 4 × e_load per bit) | 2049 words, 131 136 bits | – | 9.006 | 83.8 % |
| acc_2 | total | | 4 110 | 10.746 | 100 % |

The design's total: 60 100 µm² and 61.144 nJ.

- **The L1 accesses are a placeholder** (D154), 4.40 pJ a 64-bit word on nangate45: `acc_1`'s 6144 words are A, B (4096) and tmp0's 2048 writes, `acc_2`'s 2049 tmp0's reads and `out`. Half of the 8193 are tmp0's; the DMA is left out.
- **Two control units.** `acc_1`'s 66 bits are vecmul's; `acc_2`'s 131 add JOB_BEATS (32), a beats counter (32) and a valid bit. Holding over all 513 cycles (no clock gating), `acc_1 ctrl` costs 0.257 nJ against vecmul's 0.128 nJ at the same 256 busy cycles.
- **The streamers and the 40 CSRs are left out** (open item 110).

## 3. Save the recipe

Save recipe, as `dot_1_dry`, writes `recipes/custom/dot_1_dry.json`: `"params": {"N": 2048, "W_acc_1": 8, "W_acc_2": 8}` and section 2's eight steps (as `vecmul_dry.json`'s, then on `sum_map`). `recipes/dot/dot_v1.json` is the same at W = 4, N = 64 with `split_map`, one `W`, instances `vmul` and `vsum`.

## 4. Sweep the two widths in Bindings

A change replays the steps and checks (`params W_acc_1=8, W_acc_2=10  ok · 8 steps replayed · 1 problem`):

| W_acc_1 | W_acc_2 | Cycles | Expected BUSY_COUNTER | Channels | Area 45 nm (µm²) | Energy 45 nm (nJ) | of which L1 | Area 15 nm (µm²) | Energy 15 nm (nJ) |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 1 | 4097 | 2051, 2051 | 3, 2 | 9 850 | 69.836 | 36.011 | 2 940 | 22.452 |
| 4 | 4 | 1025 | 515, 515 | 12, 5 | 31 400 | 62.285 | 36.011 | 9 800 | 20.135 |
| 8 | 4 | 769 | 259, 515 | 24, 5 | 59 000 | 61.925 | 36.011 | 18 700 | 20.071 |
| 8 | 8 | 513 | 259, 259 | 24, 9 | 60 100 | 61.144 | 36.011 | 18 900 | 19.823 |
| 10 | 4 | 718 | 208, 515 | 30, 5 | 72 900 | 61.735 | 36.037 | 23 100 | 19.985 |
| 10 | 8 | 462 | 208, 259 | 30, 9 | 73 900 | 60.955 | 36.037 | 23 400 | 19.738 |
| 10 | 10 | 411 | 208, 208 | 30, 11 | 74 500 | 60.812 | 36.046 | 23 500 | 19.692 |
| 16 | 4 | 641 | 131, 515 | 48, 5 | 114 000 | 61.536 | 36.011 | 36 400 | 19.928 |
| 16 | 8 | 385 | 131, 259 | 48, 9 | 115 000 | 60.756 | 36.011 | 36 600 | 19.680 |

- **The sum's lanes are nearly free**: the tree is 2 710 µm² at 8 lanes, the multiplier 55 300, so 8/4 saves 1 100 µm² (1.8 %) for 256 cycles; its control unit stays 1 390 µm².
- **Energy hardly moves** (61.9 nJ at 8/4, 60.8 at 10/10): multiplies and L1 words are fixed, only the holding shrinks. W = 1 holds 4097 cycles, 69.8 nJ.
- **A tail needs the right pad.** At W_acc_1 = 10, 205 beats; pads count: `acc_1`'s 6150 words at 10/4 and 10/8, `acc_2`'s 2051 at 10/10, the pad lanes giving $0 \cdot 0 = 0$, the add's identity (D144). At 8/10 nobody writes them: `[connect.pad] container tmp0: is written by mult and read through a tail by the fold sum: ... they are never written, so the program would have to write the fold's identity there before the fold's task, open item 106`.
- **Banks and crossbar.** At W_acc_1 = 16, 48 words a cycle from 32 banks: `acc_1: the streams move 6144 words over 32 banks, at least 192 cycles of L1 time against the ideal 128`. At 16/16, 65 channels plus two cores and AXI: `[connect.xbar_inputs] acc_1, acc_2: 68 TCDM crossbar inputs, the platform takes at most 64`.

## 5. Run it on SNAX-GVSoC

    pixi run flow recipes/custom/dot_1_dry.json --platform platforms/snax_forge.json

    flow dot_1_dry (dot_1_dry: N=2048, W_acc_1=8, W_acc_2=8; N=2048) on snax_forge -> out/flow/dot_1_dry/
      sandbox   8 steps, each equal to the input graph on the reference check -> sandbox/
      lower     tasks.json (17 steps)
      gen       snax_forge.hjson, snax_forge_dot_1_dry_streamer_acc_1.hjson, snax_forge_dot_1_dry_streamer_acc_2.hjson, accelerator.json, main.c, data.h (seed 0) -> gen/
      ...
      check     passed: 1218 cycles (main to forge_check); data 0 errors (1 words); counters mult_acc_1 259 (256 beats), sum_acc_2 259 (256 beats)
      ...

| Number | What it counts | Where it comes from |
|---|---|---|
| Expected BUSY_COUNTER | the group's ideal cycles plus the streamers' fill (3, or 2 for one input) | the estimate, a guide (D155) |
| BUSY_COUNTER | every cycle from the group's START until its busy drops: bank conflicts included, shortened when the streamers started ahead | the group's BUSY_COUNTER CSR, read by the program |
| Window | the program from `main` to `forge_check()` (D135): setup, the DMA loads, both accelerators, the core's handover, the DMA store | the core's `mcycle` |

The other points, each in a folder named after its `--set`s:

    pixi run flow recipes/custom/dot_1_dry.json --platform platforms/snax_forge.json --set W_acc_1=8 --set W_acc_2=4

    flow dot_1_dry_W_acc_1-8_W_acc_2-4 (dot_1_dry: N=2048, W_acc_1=8, W_acc_2=4; N=2048) on snax_forge -> out/flow/dot_1_dry_W_acc_1-8_W_acc_2-4/

| W_acc_1 | W_acc_2 | Expected BUSY_COUNTER | BUSY_COUNTER | Window | DMA busy |
|---:|---:|---|---|---:|---:|
| 1 | 1 | 2051, 2051 | 2048, 2049 | 4822 | 534 |
| 4 | 4 | 515, 515 | 515, 515 | 1731 | 534 |
| 8 | 4 | 259, 515 | 256, 513 | 1496 | 534 |
| 8 | 8 | 259, 259 | 259, 259 | 1218 | 534 |
| 10 | 4 | 208, 515 | 208, 515 | 1440 | 537 |
| 10 | 8 | 208, 259 | 208, 259 | 1185 | 537 |
| 10 | 10 | 208, 208 | 205, 206 | 1161 | 536 |
| 16 | 4 | 131, 515 | 194, 515 | 1409 | 534 |
| 16 | 8 | 131, 259 | 194, 259 | 1154 | 534 |

**A BUSY_COUNTER 3 under its guide.** It counts from START, written after STREAMER_START; `--trace task` puts them 1 cycle apart at 4/4 and 8/8 (`acc_1` 608, 609; `acc_2` 888, 889 at 8/8), 10 at 1/1 and 8/4 (`acc_1` 620, 630; `acc_2` 899, 909 at 8/4), and at 10/10 10 for `acc_1` (622, 632), 20 for `acc_2` (850, 870). With that head start the FIFOs are full: 256 or 513 at 8/4, not 259 or 515 (D155).

**At W_acc_1 = 16** the bound is 192 cycles; at 16/8 the streamer runs 609 to 809 and START lands at 610: 194, the 192 cycles and a fill of 2.

**The window** at 8/8, 1218 cycles: loads 77 to 603, multiply 609 to 869, handover 869 to 888 (poll `acc_1`'s STATUS and streamer busy, read the counter, barrier, start `acc_2`), sum 889 to 1149, store 1163 to 1171. vecmul's was 1200: dot 1 adds the sum and the handover but stores 8 B, not 16 KiB. The DMA's 534 to 537 cycles stay, so 10/10 is only 57 cycles under 8/8 for 14 400 µm² more.

## 6. The run in the reports and the viewer

    pixi run flow recipes/custom/dot_1_dry.json --platform platforms/snax_forge.json --trace beat

    ...
      convert   run/events.log -> profile.json, trace.jsonl (9584 events in the window; boot 6358; sources acc_1, acc_2, acc_1_a, acc_1_b, acc_1_out, acc_2_a, acc_2_out, dma)

`report/run.md`:

| Accelerator | Groups | CSRs | Streamer busy | BUSY_COUNTER |
|---|---|---|---|---|
| acc_1 | acc_1 | 960–981 | 608–870 | acc_1 259 |
| acc_2 | acc_2 | 982–999 | 888–1150 | acc_2 259 |

| Block | Group | # | Task | Accelerator | Start | Done | Cycles | Direction | Bytes |
|---|---|---:|---|---|---:|---:|---:|---|---:|
| dma | – | 0 | load_A | – | 77 | 347 | 270 | L2 → L1 | 16384 |
| dma | – | 1 | load_B | – | 92 | 603 | 511 | L2 → L1 | 16384 |
| acc_1_a | acc_1 | 0 | mult_acc_1_a | acc_1 | 608 | 866 | 258 | – | – |
| acc_1_b | acc_1 | 0 | mult_acc_1_b | acc_1 | 608 | 867 | 259 | – | – |
| acc_1_out | acc_1 | 0 | mult_acc_1_out | acc_1 | 608 | 870 | 262 | – | – |
| acc_1 | acc_1 | 0 | mult_acc_1 | acc_1 | 609 | 869 | 260 | – | – |
| acc_2_a | acc_2 | 0 | sum_acc_2_a | acc_2 | 888 | 1146 | 258 | – | – |
| acc_2_out | acc_2 | 0 | sum_acc_2_out | acc_2 | 888 | 1150 | 262 | – | – |
| acc_2 | acc_2 | 0 | sum_acc_2 | acc_2 | 889 | 1149 | 260 | – | – |
| dma | – | 2 | store_out | – | 1163 | 1171 | 8 | L1 → L2 | 64 |

`acc_2_out` waits 259 cycles for the one result.

    pixi run view out/flow/dot_1_dry_beat

- **Schedule.** Nothing overlaps; core0 writes `acc_2`'s twelve CSRs from 610 to 646, during `acc_1`.
- **The cluster view.** Boxes `acc_1` (CSRs 960–977, `acc_1_a`, `acc_1_b`, `acc_1_out`) and `acc_2` (982–994, `acc_2_a`, `acc_2_out`). At cycle 700 `acc_1` fires beat 87, `acc_1_a` reads A[720..727] on banks 16–23 (three beats ahead), `acc_1_b` B[712..719] on 8–15, `acc_1_out` writes tmp0[688..695] on 24–31. At 1000 `acc_2` fires beat 108, `acc_2_a` reading tmp0[880..887] on banks 24–31.
- **Memory.** tmp0 written 614 to 869, read 890 to 1145, never in L2: the 4096 accesses dot 2 removes. A[i] and B[i] share a bank, so `acc_1_b` waits one cycle at the start.

## 7. What a link would buy

At step 8 apply `link` on `tmp0` (depth `D` = 1, group `dot`): step 9 is dot 2, one accelerator `dot`, `acc_1 → tmp0 → acc_2` through a one-beat buffer.

| | Step 8 (dot 1) | Step 9 (linked) |
|---|---:|---:|
| Cycles | 513 | 258 |
| Expected BUSY_COUNTER | 259, 259 | 261 |
| Channels | 24 + 9 | 17 |
| Streamers, CSRs | 2, 40 | 1, 23 |
| L1 footprint | 48 KiB | 32 KiB |
| Words read, written | 6144, 2049 | 4096, 1 |
| Area (µm²) | 60 100 | 63 800 |
| Energy (nJ) | 61.144 | 45.133 |
| of which L1 accesses | 36.011 | 18.008 |

Both units run at once and tmp0 leaves L1, for a LinkBuffer (3 690 µm², 2.492 nJ); one control unit replaces two (2 090 µm² against 702 + 1 390 = 2 092). The 26 % is the L1 placeholder's (HWCOST, RAM1). One width only: W_acc_2 = 4 gives `[link.order] link 'tmp0': mult.out gives 256 beats of 8 lanes, sum.a takes 512 of 4, or another order; a link passes the beats as they are`.

**The depth**, `link tmp0 · depth · D = 1` in Bindings, at W = 8:

| D | Cycles | Expected BUSY_COUNTER | Control bits | Area 45 nm (µm²) | Energy 45 nm (nJ) | Area 15 nm (µm²) | Energy 15 nm (nJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 258 | 261 | 196 | 63 800 | 45.133 | 19 900 | 14.724 |
| 2 | 258 | 261 | 199 | 68 500 | 49.131 | 21 000 | 16.141 |
| 4 | 258 | 261 | 202 | 75 800 | 54.320 | 23 100 | 17.700 |
| 8 | 258 | 261 | 205 | 91 300 | 63.395 | 27 200 | 20.625 |

Depth absorbs nothing at a beat a cycle; each slot costs W × 64 bits, and at D = 8 the design passes dot 1 (63.395 against 61.144 nJ). It pays only before a slower consumer, as dot 6's serial sum.

## 8. What was generated

`out/flow/dot_1_dry/gen/` holds vecmul's files with a streamer file per set; the entry `snax_forge_dot_1_dry`:

```json
    "snax_acc_name": "snax_forge_dot_1_dry",
    "snax_tcdm_ports": 33,
    "sparse_interconnect_config": [[24, 1], [9, 1]],
    ...
    "snax_streamer_sets": [
     {"name": "acc_1", "snax_streamer_cfg": {"$ref": "snax_forge_dot_1_dry_streamer_acc_1.hjson"}, "snax_num_rw_csr": 2, "snax_num_ro_csr": 2},
     {"name": "acc_2", "snax_streamer_cfg": {"$ref": "snax_forge_dot_1_dry_streamer_acc_2.hjson"}, "snax_num_rw_csr": 3, "snax_num_ro_csr": 2}
    ],
    "snax_datapath_cfg": {"$ref": "accelerator.json"}
```

33 TCDM ports: the sets' 24 and 9 channels. **The CSRs**, each set's streamer's then its accelerator's, from 960 (D157, C§9):

| CSRs | Set | What |
|---|---|---|
| 960–974 | acc_1 | `acc_1_a`, `acc_1_b`, `acc_1_out`: base pointer (low, high), spatial stride, temporal bound, temporal stride, 5 each |
| 975, 976, 977 | acc_1 | STREAMER_START, STREAMER_BUSY, STREAMER_PERFORMANCE_COUNTER |
| 978, 979, 980, 981 | acc_1 | START (R/W, reads 0), JOBS (R/W, 256), STATUS (R/O, bit 0 busy, bit 1 done), BUSY_COUNTER (R/O) |
| 982–991 | acc_2 | `acc_2_a`, `acc_2_out`: the same 5 each |
| 992, 993, 994 | acc_2 | STREAMER_START, STREAMER_BUSY, STREAMER_PERFORMANCE_COUNTER |
| 995, 996, 997, 998, 999 | acc_2 | START, JOBS (1), JOB_BEATS (R/W, 256), STATUS, BUSY_COUNTER |

40 in all; the tree folds 256 beats into one job of JOB_BEATS. `include/streamer_acc_1_csr_addr_map.h` names them `ACC_1_BASE_PTR_READER_0_LOW` 960, `_HIGH` 961, `ACC_1_S_STRIDE_READER_0_0` 962, `T_BOUND` 963, `T_STRIDE` 964 (READER_1 965–969, WRITER_0 970–974) to `ACC_1_STREAMER_PERFORMANCE_COUNTER_CSR` 977, `streamer_acc_2_csr_addr_map.h` from `ACC_2_BASE_PTR_READER_0_LOW` 982 (READER_0 982–986, WRITER_0 987–991) to 994; `include/acc_1_csr.h` has `ACC_1_RW_START` 978 to `ACC_1_RO_BUSY_COUNTER` 981, `include/acc_2_csr.h` `ACC_2_RW_START` 995 to `ACC_2_RO_BUSY_COUNTER` 999. `gen/main.c` starts `acc_1` (`ACC_1_STREAMER_START_CSR`, `ACC_1_RW_START`), writes `acc_2`'s twelve set-up CSRs while it runs (D158: five per mover from `ACC_2_BASE_PTR_READER_0_LOW` 0x10008040, JOBS 1, JOB_BEATS 256), polls `ACC_1_RO_STATUS` and `ACC_1_STREAMER_BUSY_CSR` (the serial sync), reads `ACC_1_RO_BUSY_COUNTER` into `forge_counter[0]`, passes `snrt_cluster_hw_barrier()` and starts `acc_2`.

The RTL comes from `gen-rtl` (the flow makes none, D136):

    pixi run gen-rtl out/flow/dot_1_dry/design/design_point.json

`emitted.json`: `snax_forge_dot_1_dry_shell_wrapper from out/rtl/dot_1_dry/accelerator.json: 11 module(s)`, those of the file table below. snax_cluster builds a streamer per set and connects `stream2acc_0` to `_2` (`acc_1_a`, `acc_1_b`, `acc_2_a`), `acc2stream_0`, `_1` (`acc_1_out`, `acc_2_out`) and the pins `acc_1_busy_o`, `acc_1_done_o`, `acc_2_busy_o`, `acc_2_done_o`.

The CSR block's registers 0 to 8 (`acc_1` 0–3, `acc_2` 4–8) are CSRs 978–981 and 995–999. `out/rtl/dot_1_dry/sv/`, in `filelist.f`'s order:

| File | Module |
|---|---|
| `snax_forge_dot_1_dry_csr.sv` | the CSR block: both accelerators' START, JOBS, STATUS and BUSY_COUNTER and acc_2's JOB_BEATS, offsets 0 to 8 |
| `snax_forge_dot_1_dry_acc_1_ctrl.sv` | the group `acc_1`'s control unit |
| `snax_forge_dot_1_dry_acc_2_ctrl.sv` | the group `acc_2`'s control unit |
| `snax_forge_dot_1_dry_ctrl.sv` | the control units of every group, here two |
| `ElementwiseAlu_w64_mul.sv` | one 64-bit lane's multiply |
| `SimdLanes_i64xi64_i64_i64_n8_mul.sv` | the 8 lanes, the datapath block of `acc_1` |
| `snax_forge_dot_1_dry_acc_1.sv` | the unit `acc_1`, around its block |
| `TreeAccumulate_i64_i64_n8.sv` | the 8-lane adder tree and its 64-bit running sum, the datapath block of `acc_2` |
| `snax_forge_dot_1_dry_acc_2.sv` | the unit `acc_2`, around its block |
| `snax_forge_dot_1_dry_datapath.sv` | the datapaths of every unit, here two |
| `snax_forge_dot_1_dry_shell_wrapper.sv` | the top, the CSR block, the control and the datapath wired to the streamers |
| `filelist.f` | the files above, in compile order |

The adder tree:

```systemverilog
// TreeAccumulate_i64_i64_n8.sv
    else if (io_fire) begin
      ...                                     // _beat_T_12, the sum of io_in_0 to io_in_7
      if (io_first)
        acc <= _beat_T_12;
      else
        acc <= acc + _beat_T_12;
    end
```

The CSR block reads 0 at both START offsets (0 and 4; a start is a strobe, D152) and keeps a saturating BUSY_COUNTER per accelerator (`io_busy_0`, `io_busy_1`). `acc_2`'s control takes a beat when its input is valid, holding the result in `acc_2_v_0` (latency 1). `estimate/bits.py`: `acc_1`'s 66 bits are `busy` (1), `acc_1_jobs` (32), `acc_1_jobs_counter` (32), `io_done_REG` (1); `acc_2`'s 131 add `acc_2_job_beats` (32), `acc_2_job_beats_counter` (32), `acc_2_v_0` (1), its jobs `acc_2_jobs` and `acc_2_jobs_counter`.

## 9. One streamer set: dot_v1_1

At step 8, click `acc_1`'s dashed outline, shift-click `acc_2`'s: only `encapsulate` is offered (the rest say `takes one selection, not 2`), accelerator `dot`. It asks `Drop steps 9..9?`; step 9 `encapsulate dot units=['acc_1', 'acc_2']`, `C 513 → 513`, 0 problems, 0 notes. Each group keeps its CSRs and control unit; they share one streamer.

The cost stays (513, 60 100 µm², 61.144 nJ); `1 streamer (not costed, open item 110) · 37 CSRs`, 960 to 996. Saved as `dot_1_enc_dry`, the recipe gains `{"transform": "encapsulate", "params": {"accelerator": "dot", "units": ["acc_1", "acc_2"]}}`; likewise `recipes/dot/dot_v1_1.json`.

- **37 CSRs.** `streamer_dot_csr_addr_map.h` has the five movers (readers `acc_1_a`, `acc_1_b`, `acc_2_a`, writers `acc_1_out`, `acc_2_out`) at 960–984, `DOT_STREAMER_START_CSR` 985, BUSY 986, the performance counter 987; `dot_csr.h` has `DOT_RW_ACC_1_START` 988 to `DOT_RO_ACC_2_COUNTER` 996.
- **Idle movers.** One STREAMER_START starts all five, so each task zeroes the other group's: `acc_2`'s set-up is 27 writes (25 plus JOBS, JOB_BEATS) against 12.
- **Reports.** Five idle rows of 2 cycles: `mult_acc_2_a (idle)`, `mult_acc_2_out (idle)` (608 to 610), `sum_acc_1_a`, `sum_acc_1_b`, `sum_acc_1_out (idle)` (894 to 896). One Accelerators row, `dot | acc_1, acc_2 | 960–996 | 608–878, 894–1156 | acc_1 256, acc_2 259`; headings `dot ▸ acc_1`, `dot ▸ acc_2`; one box `dot` (CSRs 960–987).

At N = 2048, W = 8 the window is 1230 against 1218, the COUNTERs 256 and 259 against 259 and 259: `acc_1`'s START lands 10 cycles after the streamer's (608, 618), dropping the fill, and `acc_1` ends 6 cycles later.

**Two sets against one, model and RTL.** `dot_v1` and `dot_v1_1` (W = 4, `vmul`, `vsum`) through `pixi run forge` at N = 64 and `--set N=512`, then snax_cluster's `snax-forge-check` on vsim:

| N | Design | Window, SNAX-GVSoC | Window, RTL (vsim) | BUSY_COUNTER vmul, vsum, SNAX-GVSoC | BUSY_COUNTER, RTL | Streamers | CSRs | Estimate (nangate45) |
|---:|---|---:|---:|---|---|---:|---|---|
| 64 | dot_v1 | 264 | 282 | 19, 19 | 19, 19 | 2 | 40 (960–999) | 31 400 µm², 1.953 nJ |
| 64 | dot_v1_1 | 304 | 302 | 19, 19 | 19, 19 | 1 | 37 (960–996) | 31 400 µm², 1.953 nJ |
| 512 | dot_v1 | 584 | 597 | 131, 131 | 131, 131 | 2 | 40 (960–999) | 31 400 µm², 15.576 nJ |
| 512 | dot_v1_1 | 586 | 609 | 131, 131 | 128, 131 | 1 | 37 (960–996) | 31 400 µm², 15.576 nJ |

Every RTL run passes (`forge data: 0 errors`, counters within the fill of SNAX-GVSoC's, D123, `forge fill` in `forge_expected.txt`); `dot_v1_1` at N = 512 notes `mult_vmul: counter 128 against 131, within its fill 3`, its STREAMER_START and START 6 cycles apart (1174, 1180 in core 0's trace).

**Why two accelerators are faster.** The set-up runs during `vmul`. At N = 64 `vmul` runs 20 cycles (120 to 140): `dot_v1`'s writes end at 154, `vsum` starts at 175; `dot_v1_1`'s end at 189, `vsum` at 209: 40 cycles saved on SNAX-GVSoC, 20 on the RTL. At N = 512 `vmul` runs about 130, longer than either set-up (27 writes took 67 cycles): 2 and 12. The RTL's core timing is not modelled (open items 65, 67).

## 10. What this gives the designer

dot 1's time is the two accelerators' cycles plus the handover, its area nearly all multiplier. Give the sum the multiply's lanes; past W = 8 to 10 the DMA-bound window stops paying for area. W_acc_2 must divide N (or match at a tail); past W_acc_1 = 16 banks, then the crossbar, limit it. Two streamer sets save set-up writes (12 against 27), visible only where the first accelerator is shorter than the set-up.

At the placeholder price L1 is 58.9 % of dot 1's energy, tmp0's 4096 of 8193 words about half; a link removes them and halves the cycles for 3 700 µm²: dot 2 to 6, once memory and streamers are characterized.
