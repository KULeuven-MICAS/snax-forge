# EX2: dot, its graph and two accumulators

It walks the flow on SNAX-MODEL, which RET1 removed; `docs/RUN.md` is the
SNAX-GVSoC path.

This walk-through takes the `dot` kernel on the `small16` platform from its
dataflow graph to two checked runs: one recipe binds the sum to a 4-lane
adder tree, the other to the Chisel `TreeAccumulate` at one lane. It shows how to
look at the graph each recipe step makes, how to run a bound graph, and
where the reports say the two runs differ. Every number below is checked by
`tests/flow/test_dot_example.py`, so the text stays true as the code moves.
The outputs are not in git (D98): the commands make them again in a few
seconds.

Every quoted table is copied line for line from the report named above it.

## 1. Make the graphs

```bash
pixi run sandbox recipes/dot_1.json
pixi run sandbox recipes/dot_serial.json
```

Each command imports the kernel, applies its recipe step by step and writes
one graph per step to `out/sandbox/dot_1/` and `out/sandbox/dot_serial/`:

| File | What the step did |
|---|---|
| `0_input.snaxdfg` | the kernel as imported, with N = 64 |
| `1_split_map.snaxdfg` | `mult_map` split into a temporal loop and a spatial loop of W |
| `2_bind.snaxdfg` | `mult` bound to `elementwise_mul` as the instance `mul` |
| `3_split_map.snaxdfg` | `sum_map` split the same way |
| `4_bind.snaxdfg` | `sum` bound to `accumulate` as the instance `sum` |
| `5_encapsulate.snaxdfg` | dot_1 only: both groups in the accelerator `dot` (D148) |

After every step the reference executor runs the new graph and the input
graph on the same data, and they must agree. `4_bind.snaxdfg` is the bound
graph: every tasklet is an accelerator now.

The two recipes differ in two values of the last two steps. `dot_1.json` splits
`sum_map` by `W` and binds `sum` to the implementation `chisel_adder_tree`;
`dot_serial.json` splits it by 1 and binds `sum` to `chisel_accumulator`.

## 2. Look at the graphs

```bash
pixi run view-dfg out/sandbox/dot_1/
```

Open http://127.0.0.1:8766/. The five steps stand side by side, named after
their files, each drawn top to bottom in the order it runs. The number after
a map's range is its iteration count (`64×`).

In `0_input`, the graph is two maps with a container between them:

- `A` and `B` go into `mult_map` (`i in 0:N`, `64×`), whose tasklet is
  `out = in1 * in2`;
- it writes `tmp0`, drawn dashed as transient data: it only exists between
  the two maps;
- `tmp0` goes into `sum_map`, whose tasklet is `out = in1` and whose output
  reads `out[0] (reduce add)`. That label is the reduction: every iteration
  is added into the one element of `out`.

Going right, `1_split_map` shows `mult_map` as a temporal loop
`i_t in 0:N // 4` (`16×`) around a spatial loop `i_s in 0:4`. In `2_bind` the
spatial loop and the tasklet are gone and one green box, marked
`accelerated`, stands in their place: `mul = elementwise_mul`,
`impl = chisel_tiled_spatial`, `W = 4, op = mul`, and under it what it
replaces, `mult_map_s › mult`. Its ports carry 4 elements each,
`A[4 * i_t:4 * i_t + 4]`. Steps 3 and 4 do the same to the sum. In `4_bind`
the box is `sum = accumulate`, and its output reads `out[0:1]`: the `reduce`
label is gone, because the accumulator does the adding now.

To see what the two recipes disagree on, open the two bound graphs together:

```bash
pixi run view-dfg out/sandbox/dot_1/4_bind.snaxdfg out/sandbox/dot_serial/4_bind.snaxdfg
```

They show as `4_bind` and `4_bind-2`. The multiplier half is the same in
both. The sum differs:

| | `4_bind` (dot) | `4_bind-2` (dot_serial) |
|---|---|---|
| `sum_map` | `i_t_1 in 0:N // 4`, `16×` | `i_t_1 in 0:N // 1`, `64×` |
| `sum` | `impl = chisel_adder_tree`, `W = 4, op = add` | `impl = chisel_accumulator`, `W = 1, op = add` |
| input of `sum` | `tmp0[4 * i_t_1:4 * i_t_1 + 4]` | `tmp0[i_t_1:i_t_1 + 1]` |

So the adder tree takes 4 products per firing and fires 16 times; the serial
accumulator takes 1 and fires 64 times.

## 3. Run dot from its bound graph

The flow takes a bound graph in place of a recipe (D108): a file ending in
`.snaxdfg` goes straight to the design step.

```bash
pixi run flow out/sandbox/dot_1/4_bind.snaxdfg --platform platforms/small16.json
```

The folder is `out/flow/dot_1/`. `report/design.md` names the graph it came
from and the two accelerators:

<!-- excerpt: dot_1/report/design.md -->
| Kernel | dot |
| Graph | out/sandbox/dot_1/4_bind.snaxdfg |

<!-- excerpt: dot_1/report/design.md -->
| Instance | Kind | BRM | Implementation | Params | Latency | Target II | Drain | Ports |
|---|---|---|---|---|---:|---:|---:|---|
| mul | elementwise | elementwise_mul | chisel_tiled_spatial | W=4, op=mul | 0 | 1 | 0 | a ← mul_a; b ← mul_b; out ← mul_out |
| sum | reduce | accumulate | chisel_adder_tree | W=4, op=add | 1 | 1 | 0 | a ← sum_a; out ← sum_out |

`report/run.md`:

<!-- excerpt: dot_1/report/run.md -->
|  |  |
|---|---|
| Cycles | 99 |
| Functional check | passed: out (1 elements, from l2) equals the reference and REF1 |

The task table reads in the order things happen: A and B are loaded into L1,
the multiplier runs, the accumulator runs, the result is stored.

<!-- excerpt: dot_1/report/run.md -->
| Block | # | Task | Start | Done | Cycles | Direction |
|---|---:|---|---:|---:|---:|---|
| dma | 0 | load_A | 11 | 23 | 12 | L2 → L1 |
| dma | 1 | load_B | 24 | 36 | 12 | L2 → L1 |
| mul | 0 | mult_mul | 42 | 67 | 25 | – |
| sum | 0 | sum_sum | 72 | 91 | 19 | – |
| dma | 2 | store_out | 93 | 98 | 5 | L1 → L2 |

The accumulator does not start until the multiplier has written all of
`tmp0` into L1. SNAX-MODEL's controller waits for that, and the report names it:

<!-- excerpt: dot_1/report/run.md -->
- mul → sum: 17 cycles waiting on mul_out, which sum_a reads

<!-- excerpt: dot_1/report/run.md -->
| Accelerator | Firings | Target II | Achieved II | Busy | Task window | Utilisation | Starved (stall_in) | Blocked (stall_out) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mul | 16 | 1 | 1.44 | 16 | 25 | 64.0% | 8 | 0 |
| sum | 16 | 1 | 1.00 | 16 | 19 | 84.2% | 1 | 0 |

The multiplier fires every 1.44 cycles instead of every cycle. That is the
bank conflict between its two readers that `examples/loop1/README.md` takes
apart on vecadd; A and B sit at the same places here.

## 4. Run dot_serial the same way

```bash
pixi run flow out/sandbox/dot_serial/4_bind.snaxdfg --platform platforms/small16.json
```

The folder is `out/flow/dot_serial/`. `report/design.md`:

<!-- excerpt: dot_serial/report/design.md -->
| Instance | Kind | BRM | Implementation | Params | Latency | Target II | Drain | Ports |
|---|---|---|---|---|---:|---:|---:|---|
| mul | elementwise | elementwise_mul | chisel_tiled_spatial | W=4, op=mul | 0 | 1 | 0 | a ← mul_a; b ← mul_b; out ← mul_out |
| sum | reduce | accumulate | chisel_accumulator | W=1, op=add | 1 | 1 | 0 | a ← sum_a; out ← sum_out |

<!-- excerpt: dot_serial/report/design.md -->
| sum_a | sum.a | read | 1 | 2 | 1 | tmp0 |

The multiplier is the same. The accumulator has one lane, so its reader
`sum_a` has one lane too, and its result leaves as the next input comes in:
no drain (D133).

`report/run.md`:

<!-- excerpt: dot_serial/report/run.md -->
|  |  |
|---|---|
| Cycles | 147 |
| Functional check | passed: out (1 elements, from l2) equals the reference and REF1 |

<!-- excerpt: dot_serial/report/run.md -->
| Block | # | Task | Start | Done | Cycles | Direction |
|---|---:|---|---:|---:|---:|---|
| mul | 0 | mult_mul | 42 | 67 | 25 | – |
| sum | 0 | sum_sum | 72 | 139 | 67 | – |
| dma | 2 | store_out | 141 | 146 | 5 | L1 → L2 |

<!-- excerpt: dot_serial/report/run.md -->
- mul → sum: 17 cycles waiting on mul_out, which sum_a reads

<!-- excerpt: dot_serial/report/run.md -->
| Accelerator | Firings | Target II | Achieved II | Busy | Task window | Utilisation | Starved (stall_in) | Blocked (stall_out) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mul | 16 | 1 | 1.44 | 16 | 25 | 64.0% | 8 | 0 |
| sum | 64 | 1 | 1.00 | 64 | 67 | 95.5% | 1 | 0 |

## 5. Compare

| | dot | dot_serial |
|---|---:|---:|
| `sum` lanes | 4 | 1 |
| `sum` firings | 16 | 64 |
| `sum_sum` task (cycles) | 19 | 67 |
| chaining wait `mul → sum` (cycles) | 17 | 17 |
| total (cycles) | 99 | 147 |

The two runs are the same up to cycle 72, when `sum_sum` starts in both. The
48 cycles between 99 and 147 are all in that one task: 48 more firings at
one a cycle. Both accumulators fire every cycle once they run (achieved II
1.00), so neither is held back; the serial one simply has four times as many
firings to do.

To flip between the two runs in the run viewer:

```bash
pixi run view out/flow/dot_1/run out/flow/dot_serial/run
```

## What this does not show

The cycle counts compare design points on SNAX-MODEL; they are not RTL
cycles. The model anchored against the SNAX cluster is SNAX-GVSoC (D51,
D123), which the flow runs from FLOW3 on. The adder tree is the Chisel
`TreeAccumulate` at W lanes (D133); it runs on SNAX-GVSoC as snax-gvsoc's
`forge/vsum_tree` (vsum at W = 4), and no RTL run stands behind the 19
cycles here. The two accelerators are chained through L1 on SNAX-MODEL; on
SNAX-GVSoC they are one accelerator of two groups since M13's B1
(`recipes/dot_1.json`, D148), and M13's B3 makes such a chain one composed
accelerator (D112).

The bound-graph flow writes to `out/flow/dot_1/`, the folder a recipe flow of
`recipes/dot_1.json` also uses. Give one of them `--name` to keep both.
