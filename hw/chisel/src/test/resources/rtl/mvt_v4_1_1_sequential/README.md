# mvt_v4_1_1_sequential

mvt at int8 in, int32 out, its half 2 reading `A_T`, A copied transposed by the DMA, one accelerator `mvt_1` run once per half, at M = N = 16: its `accelerator.json` and streamer hjson are snax-gvsoc's `forge/mvt_v4_1_1_sequential/`. It has two tasks, so no `vectors.json`; `ShellEmitSpec` emits it.
