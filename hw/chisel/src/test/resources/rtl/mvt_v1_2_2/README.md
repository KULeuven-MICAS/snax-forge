# mvt_v1_2_2

mvt 1 with dot 2's form, its groups `mvt` (vmul_x1 into vsum_x1) and `vadd` on one streamer set whose movers they share (`share_ports`), at W = 8, M = N = 8: its `accelerator.json` (with `movers` and `port_mux`) and streamer hjson are snax-gvsoc's `forge/mvt_v1_2_2/`. It has no `vectors.json`; `ShellEmitSpec` emits it and `PortMuxSpec` drives its mux.
