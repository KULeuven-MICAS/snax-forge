# dot_1_enc

Dot 1 encapsulated, one accelerator `dot` of two groups `vmul` and `vsum` in one streamer set (D148): its `accelerator.json` and streamer hjson are snax-gvsoc's `forge/dot_1_enc/`. Its two groups have a task each, so it has no `vectors.json`; `TwoGroupsSpec` drives each group itself.
