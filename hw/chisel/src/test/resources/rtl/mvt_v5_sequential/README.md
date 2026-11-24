# mvt_v5_sequential

mvt in two accelerators, `mvt` (a seeded `dot_v2` chain) and `mac_x2` (a seeded `Mac` of 8 lanes, y2 broadcast), at M = N = 16: its `accelerator.json` and streamer hjsons are snax-gvsoc's `forge/mvt_v5_sequential/`. Its two accelerators have a task each, so it has no `vectors.json`; `ShellEmitSpec` emits it.
