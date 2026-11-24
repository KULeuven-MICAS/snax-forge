# dot_v1_2

Dot 1 with its two groups `vmul` and `vsum` on one streamer set whose movers they share (`share_ports`), at N = 64, W = 4: its `accelerator.json` (with `movers` and `port_mux`) and streamer hjson are here, written by `tests/gen/test_rtl.py` (snax-gvsoc's `forge/` does not hold it). Its two groups have a task each, so it has no `vectors.json`; `ShellEmitSpec` emits it (the movers at the wrapper, `<tag>_dot_port_mux` inside), `PortMuxSpec` drives its mux and `ShellEndToEndSpec` runs `main.c`'s two tasks through the wrapper.
