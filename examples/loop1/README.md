# LOOP1: one turn of the loop by hand

This walk-through takes vecadd on the `small16` platform once around the
loop: run it, read why it takes the cycles it takes, write down what a change
should do, make the change, and check. Every number below is checked by
`tests/flow/test_loop1.py`, so the text stays true as the code moves. The
outputs are not in git (D67): the commands make them again in a few seconds.

Every quoted table is copied line for line from the report named above it.

## 1. Run the default

```bash
pixi run flow recipes/vecadd.json --platform platforms/small16.json --trace beat
```

The folder is `out/flow/vecadd/`. The run is traced at beat level so the
memory tab can show conflicts and `run.md` can show where each word moved;
the cycle counts are the same at every trace level.

## 2. Read the design report

`out/flow/vecadd/report/design.md` says where the contiguous placement put
the three vectors in L1: one after the other, each 4 bank rows long.

<!-- excerpt: vecadd/report/design.md -->
| Region | Base | End | Bytes | Share | Banks | Rows |
|---|---:|---:|---:|---:|---|---|
| A | 0 | 512 | 512 | 6.3% | 0–15 | 0–3 |
| B | 512 | 1024 | 512 | 6.3% | 0–15 | 4–7 |
| C | 1024 | 1536 | 512 | 6.3% | 0–15 | 8–11 |

Its notes spell out what that means for the two readers of the accelerator:

<!-- excerpt: vecadd/report/design.md -->
- acc_a and acc_b: A and B lie 512 B apart, a whole number of 128-byte bank rows, so A[i] and B[i] are in the same bank for every i.

This is a fact about addresses only. Whether it costs cycles depends on when
the two streamers ask, which only the run shows.

## 3. Read the run report

`out/flow/vecadd/report/run.md`:

<!-- excerpt: vecadd/report/run.md -->
|  |  |
|---|---|
| Cycles | 85 |
| Trace | beat |

<!-- excerpt: vecadd/report/run.md -->
| Accelerator | Firings | Target II | Achieved II | Busy | Task window | Utilisation | Starved (stall_in) | Blocked (stall_out) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acc | 16 | 1 | 1.44 | 16 | 25 | 64.0% | 8 | 0 |

The accelerator could fire every cycle (target II 1) but fires every 1.44
cycles on average, and it is starved: it waits for its inputs. The memory
section says why:

<!-- excerpt: vecadd/report/run.md -->
| Banks | Conflicts per bank | Stalls per bank |
|---|---:|---:|
| 0–3 | 3 | 3 |
| 8–11 | 4 | 4 |

<!-- excerpt: vecadd/report/run.md -->
| Streamer | Held by the xbar (cycles) | Waiting on its FIFO (cycles) | Port stalls | FIFO highest / depth |
|---|---:|---:|---:|---|
| acc_b | 7 | 0 | 28 | 1 / 2 |

acc_b is held back by the crossbar for 7 cycles, on banks 0–3 and 8–11. The
data movement section names each hold (only the first three are listed per
port):

<!-- excerpt: vecadd/report/run.md -->
L1 conflicts: 28.

<!-- excerpt: vecadd/report/run.md -->
| Port | Task | Pattern | Beats | First | Last | Last at one beat a cycle | Held back (cycles) | First held back |
|---|---:|---|---:|---:|---:|---:|---:|---|
| acc_a.0 | 0 | A[4j] | 16 | 41 | 63 | 56 | 0 | – |
| acc_b.0 | 0 | B[4j] | 16 | 42 | 64 | 57 | 7 | 44: B[8] waited, bank 8 served acc_a.0 for A[8]; 47: B[16] waited, bank 0 served acc_a.0 for A[16]; 50: B[24] waited, bank 8 served acc_a.0 for A[24] |

acc_b starts one cycle after acc_a, so they normally read banks 4 apart. When
acc_a pauses, acc_b catches up, both ask for element i in the same bank, and
acc_a wins. All 28 conflicts are this case: acc_b waiting for B[i] while its
bank serves acc_a for A[i], with the same i.

To see it, open the viewer:

```bash
pixi run view out/flow/vecadd/run
```

In the Memory tab, L1 row 4 holds B[0..15], right under A's rows 0–3 in the
same 16 banks. The Conflicts mode marks the B words in banks 0–3 and 8–11
only; clicking one lists its conflict cycles and the A word that won, and
each cycle jumps the schedule there.

## 4. Predict, before rerunning

The change: start B 64 bytes later, half a bank row, with
`--set memory.B.l1.base=576`. B[i] then sits 8 banks away from A[i]. acc_a
reads 4 banks per cycle, so while acc_b stays within one beat of acc_a, the
two never ask for the same bank. C comes after B in the contiguous placement,
so it moves too.

Prediction, written here before the rerun:

- acc_b is held by the crossbar for 0 cycles;
- the accelerator's achieved II is 1.00;
- the run takes fewer than 85 cycles.

## 5. Rerun with B moved

```bash
pixi run flow recipes/vecadd.json --platform platforms/small16.json --trace beat --set memory.B.l1.base=576
```

The folder is `out/flow/vecadd_B.l1.base576/`, next to the first one.
`report/design.md` shows the pin and the new places; the placement moved C
from 1024 to 1088:

<!-- excerpt: vecadd_B.l1.base576/report/design.md -->
| Memory plan | residency=default, layout=contiguous, placement=contiguous; pins: B.l1.base=576 |

<!-- excerpt: vecadd_B.l1.base576/report/design.md -->
| Region | Base | End | Bytes | Share | Banks | Rows |
|---|---:|---:|---:|---:|---|---|
| B | 576 | 1088 | 512 | 6.3% | 0–15 | 4–8 |
| C | 1088 | 1600 | 512 | 6.3% | 0–15 | 8–12 |

`report/run.md`:

<!-- excerpt: vecadd_B.l1.base576/report/run.md -->
|  |  |
|---|---|
| Cycles | 77 |

<!-- excerpt: vecadd_B.l1.base576/report/run.md -->
| Accelerator | Firings | Target II | Achieved II | Busy | Task window | Utilisation | Starved (stall_in) | Blocked (stall_out) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acc | 16 | 1 | 1.00 | 16 | 18 | 88.9% | 1 | 0 |

<!-- excerpt: vecadd_B.l1.base576/report/run.md -->
No bank conflicts.

<!-- excerpt: vecadd_B.l1.base576/report/run.md -->
| Streamer | Held by the xbar (cycles) | Waiting on its FIFO (cycles) | Port stalls | FIFO highest / depth |
|---|---:|---:|---:|---|
| acc_b | 0 | 0 | 0 | 1 / 2 |

In the Memory tab of this run, L1 row 4 is empty in banks 0–7 and holds
B[0..7] in banks 8–15.

## 6. Check the prediction

| Prediction | Result |
|---|---|
| acc_b held 0 cycles | 0 |
| achieved II 1.00 | 1.00 |
| fewer than 85 cycles | 77 |

All three hold. The task tables show where the 8 cycles went: the
accelerator's task is 7 cycles shorter, and everything after it moves up.

<!-- excerpt: vecadd/report/run.md -->
| Block | # | Task | Start | Done | Cycles | Direction |
|---|---:|---|---:|---:|---:|---|
| acc | 0 | add_acc | 42 | 67 | 25 | – |
| dma | 2 | store_C | 71 | 83 | 12 | L1 → L2 |

<!-- excerpt: vecadd_B.l1.base576/report/run.md -->
| Block | # | Task | Start | Done | Cycles | Direction |
|---|---:|---|---:|---:|---:|---|
| acc | 0 | add_acc | 42 | 60 | 18 | – |
| dma | 2 | store_C | 63 | 75 | 12 | L1 → L2 |

To flip between the two runs:

```bash
pixi run view out/flow/vecadd/run out/flow/vecadd_B.l1.base576/run
```

## What this does not show

The cycle counts compare design points on the model; they are not RTL cycles
until the model is anchored against the SNAX cluster (D51). The move worked
because acc_b stayed within one beat of acc_a here; a longer or differently
paced workload can line the streamers up differently, so check the Conflicts
mode again after any change to the recipe or the platform.
