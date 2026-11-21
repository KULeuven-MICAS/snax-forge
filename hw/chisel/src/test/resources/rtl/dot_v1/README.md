# dot_v1

Dot 1's fixture, two accelerators `vmul` and `vsum_v1`, each its own streamer set (D157): its `accelerator.json` and streamer hjsons are snax-gvsoc's `forge/dot_v1/`. Its two accelerators have a task each, so it has no `vectors.json`; `ShellEmitSpec` emits it, and `TwoGroupsSpec` drives the encapsulated form, `../dot_v1_1/`.
