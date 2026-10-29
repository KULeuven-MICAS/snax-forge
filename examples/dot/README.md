# EX2: dot on SNAX-GVSoC, six ways

dot is `out = Σ A[i] · B[i]` over N elements (`kernels/` `dot`). Six recipes build it as one accelerator `dot` in the platform's one slot, from two groups in L1 (dot 1) to one group of two units joined by a link (dot 2 to dot 6, D112, D147), and the flow runs each on SNAX-GVSoC: the program SNAX-LOWER writes checks the result and reads each group's COUNTER (D136). Every number below is checked by `tests/flow/test_dot_example.py`, so the text stays true as the code moves. The outputs are not in git (D98).

## 1. Run one

In snax-forge's image (`docs/RUN.md` section 1):

```bash
pixi run flow recipes/dot_4.json --platform platforms/snax_forge.json
```

The folder is `out/flow/dot_4/`. Its `check` line gives the program's own check of the result and the group's COUNTER:

    check     passed: 437 cycles (main to forge_check); data 0 errors (1 words); counters mult_dot 133 (128 beats)

Any of the six runs the same way, at the recipe's N = 512 and W = 4; `--set W=8` or `--set N=64` gives another point.

## 2. The six

| Recipe | Groups | Units, in chain order | Link | COUNTER | C§17 | Fill |
|---|---|---|---|---:|---:|---:|
| `dot_1` | `mul`, `sum`, a task each | `mul` ElementwiseLanes; `sum` TreeAccumulate | none: `tmp0` in L1 | 131, 131 | 128, 129 | 3, 2 |
| `dot_2` | `dot` | `mul` ElementwiseLanes → `sum` TreeAccumulate | `tmp0`, a buffer of depth 1 | 133 | 130 | 3 |
| `dot_3` | `dot` | `mul` ElementwiseLanes → `sum` TreeAccumulate | `tmp0`, a wire (depth 0) | 132 | 129 | 3 |
| `dot_4` | `dot` | `mac` Mac → `sum` TreeAccumulate | `out_partial`, a wire (depth 0) | 133 | 130 | 3 |
| `dot_5` | `dot` | `mac` Mac → `sum` SerialAccumulate | `out_partial`, a wire (depth 0) | 136 | 133 | 3 |
| `dot_6` | `dot` | `mac` Mac → `sum` SerialAccumulate | `out_partial`, a buffer of depth 1 | 137 | 134 | 3 |

COUNTER is the group's busy cycles at N = 512, W = 4 (T = N / W = 128 beats, one job). C§17 gives the accelerator alone on ideal streams:

    C2 = JT + 2, C3 = JT + 1, C4 = JT + 2, C5 = J(T + W − 1) + 2, C6 = (J − 1)·max(T, W) + T + W + 2

and, for dot 1's two groups, B1's rule: `mul` (L 0) its firings, JT; `sum` (L 1) JT + 1. The rest is the fill: the cycles before the first beats reach the accelerator. The streamers' start takes 2, and a unit that reads two ports loses one more on its first beat, as A and B lie a whole number of bank rows apart (A[0] and B[0] in the same bank): 3 for every unit reading A and B, 2 for dot 1's `sum`, which reads `tmp0` alone. The second reader's one crossbar stall (`mul_b` or `mac_b`, `stall_xbar` 1 in `run/profile.json`) is that first beat's.

What each builds:

- dot 1: the multiplier writes `tmp0` to L1, then the adder tree reads it back; two groups of one accelerator (D148), a task each.
- dot 2, dot 3: the same two units in one group, the products passed on in place of `tmp0`: through a buffer of one beat (dot 2, one cycle more) or a wire (dot 3).
- dot 4: a MAC sums each lane over the job's T beats and gives one beat of W partials, which the tree adds up (its T fixed at 1).
- dot 5, dot 6: the MAC's partials summed one lane a cycle (span W), the W − 1 lanes after the last beat the cycles over dot 4: through a wire (dot 5, where from the second job on the MAC waits while the sum reads) or a buffer of one beat (dot 6, where it goes on, one cycle more at one job).

## 3. What this does not show

The counts are SNAX-GVSoC's; snax_cluster's RTL runs the same ELF from the bundle (`pixi run forge`, `docs/RUN.md` section 8). Bank and port statistics per group and the beat-level trace of a linked group are parked to SYN1 (`docs/STATUS.md` open item 96): `--trace beat` is refused for dot 2 to dot 6 (`convert.level`).
