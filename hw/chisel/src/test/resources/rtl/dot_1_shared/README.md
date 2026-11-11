# dot_1_shared

Dot 1 with its two groups `vmul` and `vsum` on one streamer set whose movers they share (`share_ports`, D169): its `accelerator.json` (with `movers` and `port_mux`) and streamer hjson are snax-gvsoc's `forge/dot_1_shared/`. Its two groups have a task each, so it has no `vectors.json`; `ShellEmitSpec` emits it (the movers at the wrapper, `<tag>_dot_port_mux` inside), `PortMuxSpec` drives its mux and `ShellEndToEndSpec` runs `main.c`'s two tasks through the wrapper.
