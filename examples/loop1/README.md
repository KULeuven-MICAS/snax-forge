# LOOP1: one turn of the loop by hand

This walk-through takes vecadd on `platforms/snax_alu.json` once around the
loop on SNAX-GVSoC: run it, read why it takes the cycles it takes, write down
what a change should do, make the change, and check. Every number below is
checked by `tests/flow/test_loop1.py`, so the text stays true as the code
moves. The outputs are not in git (D98): the commands make them again in a
few minutes, in snax-forge's image (`docs/RUN.md`).

Every quoted table is copied line for line from the report named above it.

## 1. Run the default

```bash
pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json --set N=512 --trace stall
```

`N=512` makes the vectors 512 elements long, so the accelerator's work is
long enough to see against the cores' and the DMA's. The folder is
`out/flow/vecadd_N512_stall/`. The run is traced at stall level so the
schedule has its detail rows and `run.md` its stalls; the cycle counts are
the same at every trace level.

## 2. Read the design report

`out/flow/vecadd_N512_stall/report/design.md` says where the contiguous
placement put the three vectors in L1: one after the other, each 16 bank
rows long over all 32 banks.

<!-- excerpt: vecadd_N512_stall/report/design.md -->
| Region | Base | End | Bytes | Share | Banks | Rows |
|---|---:|---:|---:|---:|---|---|
| A | 268435456 | 268439552 | 4096 | 3.1% | 0–31 | 0–15 |
| B | 268439552 | 268443648 | 4096 | 3.1% | 0–31 | 16–31 |
| C | 268443648 | 268447744 | 4096 | 3.1% | 0–31 | 32–47 |

Its notes spell out what that means for the accelerator's two readers:

<!-- excerpt: vecadd_N512_stall/report/design.md -->
- acc_a and acc_b: A and B lie 4096 B apart, a whole number of 256-byte bank rows, so A[i] and B[i] are in the same bank for every i.

This is a fact about addresses only. Whether it costs cycles depends on when
the two streamers ask, which only the run shows.

## 3. Read the run report

`out/flow/vecadd_N512_stall/report/run.md`:

<!-- excerpt: vecadd_N512_stall/report/run.md -->
|  |  |
|---|---|
| Window (main to forge_check, the program's mcycle) | 486 |
| Functional check | passed |
| Counters | add_acc 131 |

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| Accelerator | Firings | Target II | Achieved II | Utilisation (window) | Task span | Utilisation (span) | Starved (stall_in) | Blocked (stall_out) | Flush |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acc | 128 | 1 | 1.00 | 26.3% | 132 | 97.0% | 3 | 0 | 0 |

The accelerator fires 128 times, one firing per 4-element beat, every cycle
once it runs (achieved II 1.00): it is busy 128 of the 132 cycles of its task,
and starved for the first 3 (cycles 224–226), while its first operands are on
their way. The streamers barely wait:

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| Streamer | Held by the crossbar (cycles) | Waiting on its FIFO (cycles) | FIFO highest / depth |
|---|---:|---:|---|
| acc_a | 0 | 0 | 2 / 8 |
| acc_b | 1 | 0 | 1 / 8 |
| acc_out | 0 | 4 | 1 / 8 |

acc_b is held for one cycle, at its first request (cycle 224): both readers
ask for elements 0–3, which lie in banks 0–3 for A and for B, as the design
report warned, and the banks serve acc_a. From then on acc_b reads each beat
one cycle after acc_a, four banks behind it, and they never meet again.

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| Banks | Stalls per bank |
|---|---:|
| 0–3 | 1 |

So the accelerator is not what makes the run 486 cycles long: it is busy for
26.3% of the window.

## 4. Read the schedule

```bash
pixi run view out/flow/vecadd_N512_stall
```

The Schedule tab (`docs/RUN.md` says how to open it on a server) shows one
lane per core, the DMA, each mover and the accelerator over the window. Two
quiet stretches, where no task or copy is open, are folded:

- cycles 0–76, the setup: both cores run `main`, and core0 writes 12 of its 18
  CSR writes, the streamer's configuration, before core1 has the DMA issue
  its first copy at 77;
- cycles 439–485, the end: the cores return to `main` and core0 runs
  `forge_check`.

Between them, the run is serial. The task table of `run.md` gives the bars:

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| Block | # | Task | Start | Done | Cycles | Direction | Bytes |
|---|---:|---|---:|---:|---:|---|---:|
| dma | 0 | load_A | 77 | 154 | 77 | L2 → L1 | 4096 |
| dma | 1 | load_B | 92 | 218 | 126 | L2 → L1 | 4096 |
| acc | 0 | add_acc | 223 | 355 | 132 | – | – |
| dma | 2 | store_C | 367 | 438 | 71 | L1 → L2 | 4096 |

- The setup takes 77 cycles before the first copy.
- The two loads take cycles 77 to 218, while core0 waits at a barrier from
  84 to 221.
- The accelerator's busy stretch is 128 cycles inside its task, 223 to 355,
  while core0 polls it (131 cycles of poll).
- The store takes 71 cycles, from 367 to 438.
- The end takes the last 47 cycles.

The DMA is busy 212 cycles of the window against the accelerator's 128:

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| DMA | Transfers | Busy | Beats read | Beats written | Bytes L2 → L1 | Bytes L1 → L2 | Bytes per busy cycle |
|---|---:|---:|---:|---:|---:|---:|---:|
| dma | 3 | 212 | 192 | 192 | 8192 | 4096 | 58.0 |

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| Core | Run | Poll | Barrier | Sleep | CSR writes | CSR reads |
|---|---:|---:|---:|---:|---:|---:|
| core0 | 137 | 131 | 218 | 0 | 18 | 61 |

Click a cycle in the accelerator's bar to see the cluster under the schedule:
each lane's grant into its bank, the read data coming back a cycle later, the
FIFO slots and the firing with its operands. The right arrow steps to the next
cycle with an event, shift + right arrow to the next task boundary.

## 5. Predict, before rerunning

The change: a wider accelerator, `--set W=8`. Each firing then takes 8
elements, and each streamer 8 lanes.

Prediction, written here before the rerun:

- firings 64 (N/W, was 128);
- the accelerator's busy cycles lower than at W=4;
- the window drops by less than the busy cycles saved (the setup and the DMA
  copies stay).

## 6. Rerun at W=8

```bash
pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json --set N=512 --set W=8 --trace stall
```

The folder is `out/flow/vecadd_N512_W8_stall/`, next to the first one.
`report/run.md`:

<!-- excerpt: vecadd_N512_W8_stall/report/run.md -->
|  |  |
|---|---|
| Window (main to forge_check, the program's mcycle) | 432 |
| Functional check | passed |
| Counters | add_acc 67 |

<!-- excerpt: vecadd_N512_W8_stall/report/run.md -->
| Accelerator | Firings | Target II | Achieved II | Utilisation (window) | Task span | Utilisation (span) | Starved (stall_in) | Blocked (stall_out) | Flush |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acc | 64 | 1 | 1.00 | 14.8% | 68 | 94.1% | 3 | 0 | 0 |

<!-- excerpt: vecadd_N512_W8_stall/report/run.md -->
| Block | # | Task | Start | Done | Cycles | Direction | Bytes |
|---|---:|---|---:|---:|---:|---|---:|
| dma | 0 | load_A | 75 | 152 | 77 | L2 → L1 | 4096 |
| dma | 1 | load_B | 90 | 216 | 126 | L2 → L1 | 4096 |
| acc | 0 | add_acc | 221 | 289 | 68 | – | – |
| dma | 2 | store_C | 312 | 383 | 71 | L1 → L2 | 4096 |

## 7. Check the prediction

| Prediction | Result |
|---|---|
| firings 64 | 64 |
| busy cycles lower than at W=4 (128) | 64 |
| the window drops by less than the 64 busy cycles saved | 486 → 432, 54 cycles |

All three hold. The copies take the cycles they took at W=4 (77, 126 and 71),
since they move the same bytes; the accelerator's task is 64 cycles shorter.
The window drops by 10 cycles less than that: the setup ends 2 cycles
earlier, but the store is issued 23 cycles after the accelerator's done
instead of 12, a quiet stretch the schedule folds (cycles 293–311), and the
end is 1 cycle longer. A wider accelerator shortens only the 26.3% of the
window it was busy in; the rest is setup, copies and the cores' waits.

To flip between the two runs:

```bash
pixi run view out/flow/vecadd_N512_stall out/flow/vecadd_N512_W8_stall
```

## What this does not show

The cycle counts are SNAX-GVSoC's, checked against snax_cluster's RTL (D123),
not RTL cycles themselves; open items 65 and 67 list where the cores' timing
differs. The loads and the store are one copy each here, before and after the
accelerator's task; with a DMA per tile (M15) the copies can overlap the
accelerator's work, and a wider accelerator then buys more of the window.
