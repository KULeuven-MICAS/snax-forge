# Dry run 1: vecmul, from the kernel to SNAX-GVSoC

Every number here is as the tools printed it when written; no test checks them (D98).

vecmul is $C_i = A_i \cdot B_i$ over N int64 elements (`kernels/polybench/vecmul.py`), the first half of dot. The question: how many lanes W, given what a lane costs in area and buys in time?

## 1. Start the editor on the kernel

In snax-forge's image (`docs/TOOL.md` section 1), from the folder that holds `snax-forge`:

    podman run --rm -it --network host -v "$PWD":"$PWD" -w "$PWD" ghcr.io/kuleuven-micas/snax-forge:main
    cd snax-forge && pixi install
    pixi run edit-dfg --kernel vecmul --set N=2048

    serving vecmul at http://127.0.0.1:8767/ (Ctrl-C to stop)

Without `--set N` it stops at `vecmul: give N, e.g. --set N=2048`.

Step 0: one map, `mult_map · i · 2048 · untagged`, holding `mult · in1 * in2`. Symbols `N [2048]`; Containers `A: N · L1 · 16 KiB`, the same for B and C. The Estimate reads `accelerator 0 + unmapped 2048 = total 2048`: one cycle per unmapped iteration (D155). L1 footprint 48 KiB of 128 KiB.

## 2. Four steps

On `mult_map` the sidebar offers `split`, `split_map`, `tag_map` ("Not here" says why not `bind` or `split_reduce`).

| Step | Select | Apply | What changes |
|---|---|---|---|
| 1 | `mult_map` | `split`, factor `W` with its value box set to 8 | `mult_map` now runs `i_1` over 256, around a new map `mult_map_inner` running `i_2` over 8; the tasklet reads `A[8 * i_1 + i_2]`; Bindings shows `mult_map split · W = 8 · not bound` |
| 2 | `mult_map` | `tag_map`, `temporal` | the outer loop becomes the streamers' loop |
| 3 | `mult_map_inner` | `tag_map`, `spatial` | the inner loop will become lanes; the unmapped cycles stay at 2048, since a tag alone runs nothing |
| 4 | `mult` | `bind`, `simd / chisel_tiled_spatial`, instance `acc` | the tasklet becomes unit `acc` at 8 lanes; Bindings shows `acc · simd · W = 8`; the total drops from 2048 to 256 (accelerator 256 + unmapped 0) |

The factor names a param ("new param W = 4"; `W_2` for a second split), editable in Bindings and swept with `--set W=…`. The tags decide which loop becomes what; `split_map` does steps 1 to 3 at once (`i_t`, `i_s`, `mult_map_s`).

At step 4 only `simd` fits (latency 0, II 1, span 1); `mac` needs `fuse_map` first (D150). After the bind, one group `acc`:

| Cycles | Bottleneck | Ramp | Stall | Expected BUSY_COUNTER | Area | Energy |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 256 | 0 | 0 | 259 | 55 700 µm² | 23.265 nJ |

Cycles is $N / W$; the expected BUSY_COUNTER adds the streamers' fill of 3 (D155, C§13). The cost on nangate45:

| Block | Datapath | Area (µm²) | Energy (nJ) | Share |
|---|---|---:|---:|---:|
| acc | SimdLanes | 55 300 | 23.137 | 99.4 % |
| acc ctrl | control unit | 351 | 0.128 | 0.6 % |

## 3. Save the recipe

Save recipe, as `vecmul_dry`, runs the design checks and writes `recipes/custom/vecmul_dry.json`:

```json
{
 "name": "vecmul_dry",
 "kernel": "vecmul",
 "params": {"N": 2048, "W": 8},
 "symbols": {"N": "N"},
 "steps": [
  {"transform": "split", "params": {"map": "mult_map", "factor": "W"}},
  {"transform": "tag_map", "params": {"map": "mult_map", "kind": "temporal"}},
  {"transform": "tag_map", "params": {"map": "mult_map_inner", "kind": "spatial"}},
  {"transform": "bind", "params": {"node": "mult", "brm": "simd", "implementation": "chisel_tiled_spatial", "instance": "acc"}}
 ]
}
```

The log lists what to run next:

| Command | What it does |
|---|---|
| `pixi run sandbox recipes/custom/vecmul_dry.json` | writes the steps and checks each against the reference |
| `pixi run flow recipes/custom/vecmul_dry.json --platform platforms/snax_forge.json` | runs the recipe on SNAX-GVSoC and writes the reports |
| `pixi run forge recipes/custom/vecmul_dry.json` | the flow plus the bundle for snax_cluster's RTL |

The steps also go to `out/sandbox/vecmul_dry/` (`0_input.snaxdfg` to `4_bind.snaxdfg`); only the recipe replays (D72). Save never overwrites.

## 4. Sweep W in the editor

Change W in Bindings; the tech dropdown switches nangate45 and nangate15.

| W | Cycles | BUSY_COUNTER (expected) | Channels | Area 45 nm (µm²) | Energy 45 nm (nJ) | ctrl share | Area 15 nm (µm²) | Energy 15 nm (nJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2048 | 2051 | 3 | 7 260 | 24.022 | 4.3 % | 2 300 | 7.945 |
| 2 | 1024 | 1027 | 6 | 14 200 | 23.510 | 2.2 % | 4 510 | 7.802 |
| 4 | 512 | 515 | 12 | 28 000 | 23.254 | 1.1 % | 8 930 | 7.730 |
| 8 | 256 | 259 | 24 | 55 700 | 23.265 | 0.6 % | 17 800 | 7.769 |
| 10 | 205 | 208 | 30 | 69 500 | 23.123 | 0.4 % | 22 200 | 7.695 |

- **Area grows with W, energy does not.** SimdLanes is W one-lane multipliers, 0 flops: area $W \cdot A_1$, energy N multiplies. On nangate45 $A_1$ is 6 910 µm² of comb, about 11.3 pJ per 64-bit multiply.
- **Only the control unit changes with time**: 66 bits held every cycle, $E_{\mathit{ctrl}} = f_{\mathit{ctrl}} \cdot 66 \cdot C \cdot e_{\mathit{hold}}$, 1.024 nJ at W = 1, 0.103 nJ at W = 10.
- **W = 10** pads 2048 to 205 beats (2 pad elements, D144), lane use 99.9 %.
- **Under 1 % is the table**: SimdLanes is measured at W = 1, 8 and 32 and copied between (C§14), so W = 8 is 0.05 % above W = 4.
- **No leakage**, so W = 1 looks as cheap as W = 10 (a known limit).

## 5. Run it on SNAX-GVSoC

The flow builds SNAX-GVSoC the first time (about 6 minutes):

    pixi run flow recipes/custom/vecmul_dry.json --platform platforms/snax_forge.json

    flow vecmul_dry (vecmul_dry: N=2048, W=8; N=2048) on snax_forge -> out/flow/vecmul_dry/
      sandbox   4 steps, each equal to the input graph on the reference check -> sandbox/
      design    24 checks passed; platform base snax_forge, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (12 steps)
      gen       snax_forge.hjson, snax_forge_vecmul_dry_streamer_acc.hjson, accelerator.json, main.c, data.h (seed 0) -> gen/
      ...
      sw        sw/forge.elf (sha256 254f665a9aa34d27)
      check     passed: 1200 cycles (main to forge_check); data 0 errors (2048 words); counters mult_acc 259 (256 beats)
      ...

| Number | What it counts | Where it comes from |
|---|---|---|
| Expected BUSY_COUNTER | the group's ideal cycles plus the streamers' fill of 3 | the estimate (D155), before any run |
| BUSY_COUNTER | the cycles the group is busy, from the cycle after its start is accepted to its last output push | the group's BUSY_COUNTER CSR, read by the program |
| Window | the program from `main` to `forge_check()` (D135): setup, the DMA loads, the accelerator, polling, the DMA store | the core's `mcycle` |

The other points (`out/flow/vecmul_dry_W4/` and so on):

    pixi run flow recipes/custom/vecmul_dry.json --platform platforms/snax_forge.json --set W=4

| W | Expected BUSY_COUNTER | BUSY_COUNTER | Window | DMA busy |
|---:|---:|---:|---:|---:|
| 1 | 2051 | 2048 | 2999 | 789 |
| 2 | 1027 | 1027 | 1962 | 788 |
| 4 | 515 | 515 | 1455 | 789 |
| 8 | 259 | 259 | 1200 | 789 |
| 10 | 208 | 208 | 1169 | 792 |

The counter is the estimate plus 3 but at W = 1, 3 under (STATUS open item 104). From W = 4 to 10 the accelerator saves 307 cycles, the window only 286: every point spends about 790 cycles on DMA and 150 on the cores.

**Beyond W = 10**, before any run:

- At W = 16 the streams need $3W = 48$ words a cycle from 32 banks, 1.5 cycles a beat: `acc: the outer ports move 6144 words over 32 banks, at least 192 cycles against the ideal 128`; the run's counter is 194.
- At W = 32, 96 channels, two cores and AXI: `[connect.xbar_inputs] acc: 99 TCDM crossbar inputs, the platform takes at most 64` (design step and Save).

## 6. The run in the cluster viewer

`--trace beat` records every grant, DMA beat and firing (folder `_beat`):

    pixi run flow recipes/custom/vecmul_dry.json --platform platforms/snax_forge.json --trace beat

    ...
      convert   run/events.log -> profile.json, trace.jsonl (7753 events in the window; boot 22610; sources acc, acc_a, acc_b, acc_out, dma)

`report/run.md`'s tasks and accelerators:

| Block | # | Task | Accelerator | Start | Done | Cycles | Direction | Bytes |
|---|---:|---|---|---:|---:|---:|---|---:|
| dma | 0 | load_A | – | 89 | 359 | 270 | L2 → L1 | 16384 |
| dma | 1 | load_B | – | 94 | 615 | 521 | L2 → L1 | 16384 |
| acc_a | 0 | mult_acc_a | acc | 619 | 877 | 258 | – | – |
| acc_b | 0 | mult_acc_b | acc | 619 | 878 | 259 | – | – |
| acc_out | 0 | mult_acc_out | acc | 619 | 883 | 264 | – | – |
| acc | 0 | mult_acc | acc | 620 | 880 | 260 | – | – |
| dma | 2 | store_C | – | 894 | 1157 | 263 | L1 → L2 | 16384 |

| Accelerator | Groups | CSRs | Streamer busy | BUSY_COUNTER |
|---|---|---|---|---|
| acc | acc | 960–981 | 619–883 | acc 259 |

Load A and B (one 512-bit beat a cycle, 256 beats each), compute, store C; nothing overlaps.

    pixi run view out/flow/vecmul_dry_beat

    serving vecmul_dry_beat at http://127.0.0.1:8765/ (Ctrl-C to stop)

- **Schedule.** A lane per core, the DMA, each streamer and the accelerator; quiet stretches of 16 cycles or more fold.
- **The cluster view.** At a clicked cycle, e.g. 700, the 24 streamer lanes ask 24 banks for a word. Arrows step by event, shift-arrows by task.
- **Memory.** A[i], B[i] and C[i] share a bank, so `acc_b` stalls once at the start (the design report's note).

## 7. Two points side by side

    pixi run sandbox recipes/custom/vecmul_dry.json --set W=4 --out out/sandbox/try/vecmul_w4
    pixi run sandbox recipes/custom/vecmul_dry.json --set W=8 --out out/sandbox/try/vecmul_w8
    pixi run view-dfg out/sandbox/try/vecmul_w4/ out/sandbox/try/vecmul_w8/ --platform platforms/snax_forge.json

`vecmul_w4/4_bind` against `vecmul_w8/4_bind`: cycles 512 → 256, area 28 000 → 55 700 µm², energy 23.254 → 23.265 nJ.

## 8. What was generated

`out/flow/vecmul_dry/gen/` holds the cluster and streamer files, the headers, `main.c`, `data.h`, `manifest.json` and `accelerator.json`, from which SNAX-GVSoC and the RTL are built (D134, C§10). One entry with one streamer set `acc` (D157), named `snax_forge_<design>`:

```json
    "snax_acc_name": "snax_forge_vecmul_dry",
    "snax_tcdm_ports": 24,
    ...
    "snax_streamer_sets": [
     {
      "name": "acc",
      "snax_streamer_cfg": {"$ref": "snax_forge_vecmul_dry_streamer_acc.hjson"},
      "snax_num_rw_csr": 2,
      "snax_num_ro_csr": 2
     }
    ],
```

The RTL comes from `gen-rtl` (the flow makes none, D136):

    pixi run gen-rtl out/flow/vecmul_dry/design/design_point.json

It writes `out/rtl/vecmul_dry/accelerator.json`, runs `hw/chisel/`'s `Emit --spec` into `sv/` and lists the output in `emitted.json`. `out/rtl/vecmul_dry/sv/`, in `filelist.f`'s order (the top wires them to the streamer's `acc_a`, `acc_b`, `acc_out`, 8 lanes each):

| File | Module |
|---|---|
| `snax_forge_vecmul_dry_csr.sv` | the CSR block: START, JOBS, STATUS, BUSY_COUNTER |
| `snax_forge_vecmul_dry_acc_ctrl.sv` | the group `acc`'s control unit (`GroupCtrl`) |
| `snax_forge_vecmul_dry_ctrl.sv` | the control units of every group, here one |
| `ElementwiseAlu_w64_mul.sv` | one 64-bit lane's multiply |
| `SimdLanes_i64xi64_i64_i64_n8_mul.sv` | the 8 lanes, the datapath block of `acc` |
| `snax_forge_vecmul_dry_acc.sv` | the unit `acc`, around its block |
| `snax_forge_vecmul_dry_datapath.sv` | the datapaths of every unit, here one |
| `snax_forge_vecmul_dry_shell_wrapper.sv` | the top, the CSR block, the control and the datapath wired to the streamer |
| `filelist.f` | the files above, in compile order |

```systemverilog
// SimdLanes_i64xi64_i64_i64_n8_mul.sv, eight ElementwiseAlu_w64_mul
  input  [63:0] io_a_0,
  ...
// ElementwiseAlu_w64_mul.sv
  assign io_out = io_a * io_b;

// snax_forge_vecmul_dry_acc_ctrl.sv
  reg  [31:0] acc_jobs_counter;
  wire        acc_offer =
    busy & acc_jobs_counter != acc_jobs & io_in_0_valid & io_in_1_valid;
  wire        acc_frozen = acc_offer & ~io_out_ready;

// snax_forge_vecmul_dry_csr.sv: BUSY_COUNTER saturates
      else if (io_busy_0 & acc_counter != 32'hFFFFFFFF)
        acc_counter <= acc_counter + 32'h1;
```

**The CSRs**, the streamer's first, then the accelerator's, from 960 (D157, C§9); offsets from `accelerator.json` (D152):

| CSR | Access | Offset | Number | Bits | What |
|---|---|---:|---:|---:|---|
| `acc_a`, `acc_b`, `acc_out` | R/W | – | 960–974 | 32 | the streamer's movers: base pointer (low, high), spatial stride, temporal bound, temporal stride, 5 each |
| STREAMER_START, STREAMER_BUSY, STREAMER_PERFORMANCE_COUNTER | – | – | 975–977 | 32 | the streamer's start, busy and counter |
| START | R/W | 0 | 978 | 1 | a write with bit 0 set starts the group; it reads 0 |
| JOBS | R/W | 1 | 979 | 32 | how many jobs; here one job per beat, 256 |
| STATUS | R/O | 2 | 980 | 2 | bit 0 busy, bit 1 done (sticky, cleared by the next start) |
| BUSY_COUNTER | R/O | 3 | 981 | 32 | the busy cycles of the last run, saturating at 2³² − 1 |

`include/streamer_acc_csr_addr_map.h` and `include/acc_csr.h` name them (`ACC_BASE_PTR_READER_0_LOW` 960 to `ACC_STREAMER_PERFORMANCE_COUNTER_CSR` 977, `ACC_RW_START` 978 to `ACC_RO_BUSY_COUNTER` 981). `gen/main.c` sets the three movers (256 temporal beats each), writes `ACC_RW_JOBS` 256, `ACC_STREAMER_START_CSR`, then `ACC_RW_START`, polls `ACC_RO_STATUS` and `ACC_STREAMER_BUSY_CSR`, and reads `ACC_RO_BUSY_COUNTER`.

**The control.** `GroupCtrl` (`hw/chisel/.../ctrl/GroupCtrl.scala`) runs the group rule SNAX-GVSoC also steps (`snax_group_rule.py`): a beat only when both inputs are valid and the output ready. Its 66 flops (`estimate/bits.py`): `busy` (1), `acc_jobs` (32), `acc_jobs_counter` (32), `io_done_REG` (1).

**The datapath.** `SimdLanes` (`hw/chisel/.../datapath/SimdLanes.scala`) is W `ElementwiseAlu`s with casts at the edges, no valid, ready or registers.

## 9. What this gives the designer

Up to W = 10 the cycles fall as N / W and the area grows as W at the same energy; past it L1's 32 banks cap the streams, and at W = 32 the platform refuses the design. DMA and cores dominate the window, so W of 4 to 8 is where the lanes pay off.
