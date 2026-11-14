# LOOP1: one turn of the loop by hand

This walk-through takes vecadd on `platforms/snax_forge.json` once around the loop on SNAX-GVSoC: run it, read why it takes the cycles it takes, write down what a change should do, make the change, and check. Every number below is checked by `tests/flow/test_loop1.py`, so the text stays true as the code moves. The outputs are not in git (D98): the commands make them again in a few minutes, in snax-forge's image (`docs/TOOL.md` section 1).

Every quoted table is copied line for line from the report named above it.

## 1. Run the default

```bash
pixi run flow recipes/elementwise/vecadd.json --platform platforms/snax_forge.json --set N=512 --trace stall
```

`N=512` makes the vectors 512 elements long, so the accelerator's work is long enough to see against the cores' and the DMA's. The folder is `out/flow/vecadd_N512_stall/`. The run is traced at stall level so the schedule has its detail rows and `run.md` its stalls; the cycle counts are the same at every trace level.

## 2. Read the design report

`out/flow/vecadd_N512_stall/report/design.md` says where the contiguous placement put the three vectors in L1: one after the other, each 16 bank rows long over all 32 banks.

<!-- excerpt: vecadd_N512_stall/report/design.md -->
| Region | Base | End | Bytes | Share | Padding | Banks | Rows |
|---|---:|---:|---:|---:|---|---|---|
| A | 268435456 | 268439552 | 4096 | 3.1% | none | 0–31 | 0–15 |
| B | 268439552 | 268443648 | 4096 | 3.1% | none | 0–31 | 16–31 |
| C | 268443648 | 268447744 | 4096 | 3.1% | none | 0–31 | 32–47 |

Its notes spell out what that means for the accelerator's two readers:

<!-- excerpt: vecadd_N512_stall/report/design.md -->
- acc_a and acc_b: A and B lie 4096 B apart, a whole number of 256-byte bank rows, so A[i] and B[i] are in the same bank for every i.

This is a fact about addresses only. Whether it costs cycles depends on when the two streamers ask, which only the run shows.

## 3. Read the run report

`out/flow/vecadd_N512_stall/report/run.md`:

<!-- excerpt: vecadd_N512_stall/report/run.md -->
|  |  |
|---|---|
| Window (main to forge_check, the program's mcycle) | 471 |
| Functional check | passed |
| Data | 0 errors over 512 words |
| Counters | add_acc 128 (expected 131, a guide) |

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| Unit | Firings | Target II | Achieved II | Utilisation (window) | Task span | Utilisation (span) | Starved (stall_in) | Blocked (stall_out) | Flush |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acc | 128 | 1 | 1.00 | 27.2% | 129 | 99.2% | 0 | 0 | 0 |

The accelerator fires 128 times, one firing per 4-element beat, every cycle once it runs (achieved II 1.00): it is busy 128 of the 129 cycles of its task and never starved. Its streamers started 10 cycles before it (the task table below: the movers from 183, the accelerator from 193), the gap between the program's two start writes, so their FIFOs were full when it started. That head start is also why its counter reads 128 against the estimate's 131 (D155): the counter starts at the accelerator's START, after the streamers' fill. The streamers barely wait:

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| Streamer | Held by the crossbar (cycles) | Waiting on its FIFO (cycles) | FIFO highest / depth |
|---|---:|---:|---|
| acc_a | 0 | 1 | 8 / 8 |
| acc_b | 2 | 0 | 7 / 8 |
| acc_out | 1 | 10 | 2 / 8 |

acc_b is held at its first request (cycle 185): both readers ask for elements 0–3, which lie in banks 0–3 for A and for B, as the design report warned, and the banks serve acc_a. From then on acc_b reads each beat one cycle after acc_a. They meet once more when the accelerator starts and the full FIFOs drain together, and acc_out's first write meets acc_b in the same banks a cycle later:

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| Banks | Conflicts per bank |
|---|---:|
| 0–3 | 3 |

So the accelerator is not what makes the run 471 cycles long: it is busy for 27.2% of the window.

## 4. Read the schedule

```bash
pixi run view out/flow/vecadd_N512_stall
```

The Schedule tab (`docs/TOOL.md` section 1 says how to open it on a server) shows one lane per core, the DMA, each mover and the accelerator over the window. The cluster has one core (D170), which starts the copies, configures the accelerator and waits for it. Three quiet stretches, where no task or copy is open, are folded:

- cycles 0–33, the setup: the core runs `main` and enters `forge_core_0`, which starts the first copy at 34;
- cycles 325–355, between the accelerator's end and the store;
- cycles 428–470, the end: the core runs `forge_check`.

Between them, the run is serial. The task table of `run.md` gives the bars:

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| Block | # | Task | Accelerator | Start | Done | Cycles | Direction | Bytes |
|---|---:|---|---|---:|---:|---:|---|---:|
| dma | 0 | load_A | – | 34 | 111 | 77 | L2 → L1 | 4096 |
| dma | 1 | load_B | – | 48 | 176 | 128 | L2 → L1 | 4096 |
| acc | 0 | add_acc | acc | 193 | 322 | 129 | – | – |
| dma | 2 | store_C | – | 356 | 427 | 71 | L1 → L2 | 4096 |

- The setup takes 34 cycles before the first copy.
- The two loads take cycles 34 to 176; the core writes the streamer's and the accelerator's CSRs while they run (the DMA queues its copies), then waits for them.
- The accelerator's busy stretch is 128 cycles inside its task, 193 to 322, while the core polls it (127 cycles of poll).
- The store takes 71 cycles, from 356 to 427.
- The end takes the last 43 cycles.

The DMA is busy 213 cycles of the window against the accelerator's 128:

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| DMA | Transfers | Busy | Beats read | Beats written | Bytes L2 → L1 | Bytes L1 → L2 | Bytes per busy cycle |
|---|---:|---:|---:|---:|---:|---:|---:|
| dma | 3 | 213 | 192 | 192 | 8192 | 4096 | 57.7 |

<!-- excerpt: vecadd_N512_stall/report/run.md -->
| Core | Run | Poll | Barrier | Sleep | CSR writes | CSR reads |
|---|---:|---:|---:|---:|---:|---:|
| core0 | 344 | 127 | 0 | 0 | 18 | 46 |

Click a cycle in the accelerator's bar to see the cluster under the schedule: each lane's grant into its bank, the read data coming back a cycle later, the FIFO slots and the firing with its operands. The right arrow steps to the next cycle with an event, shift + right arrow to the next task boundary.

## 5. Predict, before rerunning

The change: a wider accelerator, `--set W=8`. Each firing then takes 8 elements, and each streamer 8 lanes.

Prediction, written here before the rerun, on what the accelerator and the DMA do, which the program's code layout does not move:

- firings 64 (N/W, was 128);
- the accelerator's busy cycles (the profile's, a firing or the II gap after it) 64, half of W=4's 128;
- each copy takes the cycles it took at W=4, since it moves the same bytes: load_A 77, load_B 128, store_C 71, the DMA busy 213 cycles.

Not predicted: the window, and the counter. The counter (`Counters` in `run.md`) starts at the accelerator's START, so it moves with the streamers' head start, which the core's timing between the two start writes sets (D155, open item 65).

## 6. Rerun at W=8

```bash
pixi run flow recipes/elementwise/vecadd.json --platform platforms/snax_forge.json --set N=512 --set W=8 --trace stall
```

The folder is `out/flow/vecadd_N512_W8_stall/`, next to the first one. `report/run.md`:

<!-- excerpt: vecadd_N512_W8_stall/report/run.md -->
|  |  |
|---|---|
| Window (main to forge_check, the program's mcycle) | 391 |
| Functional check | passed |
| Data | 0 errors over 512 words |
| Counters | add_acc 67 (expected 67, a guide) |

<!-- excerpt: vecadd_N512_W8_stall/report/run.md -->
| Unit | Firings | Target II | Achieved II | Utilisation (window) | Task span | Utilisation (span) | Starved (stall_in) | Blocked (stall_out) | Flush |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acc | 64 | 1 | 1.00 | 16.4% | 68 | 94.1% | 3 | 0 | 0 |

<!-- excerpt: vecadd_N512_W8_stall/report/run.md -->
| Block | # | Task | Accelerator | Start | Done | Cycles | Direction | Bytes |
|---|---:|---|---|---:|---:|---:|---|---:|
| dma | 0 | load_A | – | 34 | 111 | 77 | L2 → L1 | 4096 |
| dma | 1 | load_B | – | 48 | 176 | 128 | L2 → L1 | 4096 |
| acc | 0 | add_acc | acc | 183 | 251 | 68 | – | – |
| dma | 2 | store_C | – | 277 | 348 | 71 | L1 → L2 | 4096 |

<!-- excerpt: vecadd_N512_W8_stall/report/run.md -->
| DMA | Transfers | Busy | Beats read | Beats written | Bytes L2 → L1 | Bytes L1 → L2 | Bytes per busy cycle |
|---|---:|---:|---:|---:|---:|---:|---:|
| dma | 3 | 213 | 192 | 192 | 8192 | 4096 | 57.7 |

## 7. Check the prediction

| Prediction | Result |
|---|---|
| firings 64 | 64 |
| busy cycles 64, half of 128 | 64 |
| each copy as at W=4: load_A 77, load_B 128, store_C 71; the DMA busy 213 | 77, 128, 71; 213 |

All three hold: the accelerator is busy half as long, and the copies take the cycles they took at W=4, since they move the same bytes. The counter went from 128 to 67, 61 lower and not 64: here the accelerator's START came one cycle after its streamers' start, so it waited 3 cycles for its first operands (starved, 3), where at W=4 the 10-cycle head start had filled the FIFOs. The window went from 471 to 391 cycles, more than the 64 busy cycles saved, since besides the accelerator's work it moves by tens of cycles with the program's code layout (I-cache lines, open item 65). A wider accelerator shortens only the 27.2% of the window it was busy in; the rest is setup, copies and the core's waits.

The numbers here are SNAX-GVSoC's as printed when this page was last written (CORE1, one core); its test checks the claims that do not depend on timing exactly and the cycle numbers within 10 % (D171).

To flip between the two runs:

```bash
pixi run view out/flow/vecadd_N512_stall out/flow/vecadd_N512_W8_stall
```

## What this does not show

The cycle counts are SNAX-GVSoC's, checked against snax_cluster's RTL (D123), not RTL cycles themselves; open items 65 and 67 list where the cores' timing differs. The loads and the store are one copy each here, before and after the accelerator's task; with a DMA per tile (M15) the copies can overlap the accelerator's work, and a wider accelerator then buys more of the window.
