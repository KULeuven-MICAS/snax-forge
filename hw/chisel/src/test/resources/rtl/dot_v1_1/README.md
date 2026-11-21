# dot_v1_1

Dot 1 encapsulated, one accelerator `dot` of two groups `vmul` and `vsum_v1` in one streamer set (D148): its `accelerator.json` and streamer hjson are snax-gvsoc's `forge/dot_v1_1/`. Its two groups have a task each, so it has no `vectors.json`; `TwoGroupsSpec` drives each group itself.
