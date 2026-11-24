# mvt_v5_sequential at trace level task

A recorded SNAX-GVSoC run, so CI converts real events without GVSoC: mvt_v5 at M = N = 16, schedule sequential: two accelerators in two streamer sets, `mvt` (a seeded `dot_v2` chain) and `mac_x2` (a seeded `Mac` of 8 lanes, y2 broadcast), a task each; not at level stall, whose conversion refuses it (STATUS, open item 162). Made in snax-forge's image, the submodule checked out, by

    pixi run flow recipes/mvt/mvt_v5.json --platform platforms/snax_forge.json --set M=16 --set N=16 --set design.schedule=sequential --trace task

and copied from each flow's folder: `run/events.log` as `events_task.log`, `check.json` (the profile's `total_cycles` and `functional_check`) and `elf_symbols.json` (`run.json`'s), the ELF the same for every level (sha256 `8c3ec55ac92d71fc25a68b9d254c280685a8d6300fb6c911eee12a7f0bd7b7c1`). The run: 521 cycles, data 0 errors (32 words), counters `tmp1_mult_mvt` 34 (32 beats), `tmp4_mult_mac_x2` 36 (32 beats).

The generated files the run was made on are not kept: `tests/run/helpers.py`'s `recorded_files` makes them again from the recipe at these sizes (`FORGE_FIXTURES`, the forge/ folder of the same name) and checks each against its sha256 here; when one differs, record the run again by the commands above and write the new hashes here.

- `accelerator.json`: sha256 `92073b0b2d9a773b913cea6b9b7caa48d737a0fdf43b907cbb5227ad39b7894f`
- `design_point.json`: sha256 `b2fca73ac1579383e96d62a63a7d25f3321d26ce6718644970541320aa320bd1`
- `manifest.json`: sha256 `3e6465c9d09a97eb70f346a24b0380618eb6225823760ba15cd479e29a2bc503`
- `snax_forge.hjson`: sha256 `020480c858386afa1b7ed3fe6903d1777bee6eb5b371f3b2cb253824c5a47985`
- `snax_forge_mvt_v5_streamer_mac_x2.hjson`: sha256 `7156b7f26f9e83739da32647d1175082e029d19be17edfa1566f8aaa0a53caf4`
- `snax_forge_mvt_v5_streamer_mvt.hjson`: sha256 `94b9be75d427a2b380a47a3b09a2e070c4624204a6e3c341cf542bf46526861d`
- `tasks.json`: sha256 `2e049de7731d257be43d3d2a2768aa02f4be51fe7433a4afdca302a0851fc7fd`
