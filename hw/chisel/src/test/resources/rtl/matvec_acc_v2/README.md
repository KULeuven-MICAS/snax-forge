# matvec_acc_v2

matvec with two groups in one accelerator `matvec`: `mul` (`SimdLanes`, 4 lanes of int64) and `sum` (a seeded `SerialAccumulate` of one lane), at M = 2, N = 16: its `accelerator.json` and streamer hjson are snax-gvsoc's `forge/matvec_acc_v2/`. Its two groups have a task each, so it has no `vectors.json`; `ShellEmitSpec` emits it and `TwoGroupsSpec` drives its groups one at a time.
