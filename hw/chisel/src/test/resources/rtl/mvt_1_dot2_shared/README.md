# mvt_1_dot2_shared

mvt 1 with dot 2's form, its groups `mvt` (vmul_x1 into vsum_x1) and `vadd` on one streamer set whose movers they share (`share_ports`, D169), at W = 8: its `accelerator.json` (with `movers` and `port_mux`) and streamer hjson are snax-gvsoc's `forge/mvt_1_dot2_shared/`. It has no `vectors.json`; `ShellEmitSpec` emits it and `PortMuxSpec` drives its mux.
