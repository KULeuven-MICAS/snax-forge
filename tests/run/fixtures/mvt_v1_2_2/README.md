# mvt_v1_2_2 at trace levels task and stall

A recorded SNAX-GVSoC run, so CI converts real events without GVSoC: mvt_v1_2_2 at W = 8, M = N = 8: one accelerator `mvt_vadd` of two groups, `mvt` (`vmul_x1` linked into `vsum_x1`) and `vadd`, on one streamer set whose movers they share through the port mux, four tasks. Made in snax-forge's image, the submodule checked out, by

    pixi run flow recipes/mvt/mvt_v1_2_2.json --platform platforms/snax_forge.json --set W=8 --set M=8 --set N=8 --trace task
    pixi run flow recipes/mvt/mvt_v1_2_2.json --platform platforms/snax_forge.json --set W=8 --set M=8 --set N=8 --trace stall

and copied from each flow's folder: `run/events.log` as `events_task.log` and `events_stall.log`, `check.json` (the profile's `total_cycles` and `functional_check`) and `elf_symbols.json` (`run.json`'s), the ELF the same for every level (sha256 `b054cf1f6d641defd5484bcd8b1096088edfeaf494025ba7fb0e2ffd94145815`). The run: 509 cycles, data 0 errors (16 words), counters `tmp0_mult_mvt` 15 (8 beats), `augassign_18_4_vadd` 4 (1 beats), `tmp2_mult_mvt` 14 (8 beats), `augassign_19_4_vadd` 4 (1 beats).

The generated files the run was made on are not kept: `tests/run/helpers.py`'s `recorded_files` makes them again from the recipe at these sizes (`FORGE_FIXTURES`, the forge/ folder of the same name) and checks each against its sha256 here; when one differs, record the run again by the commands above and write the new hashes here.

- `accelerator.json`: sha256 `64b87633dc57ce1562f8acc3b6b3881ee8bcc4a98622d411bb1888005d536931`
- `design_point.json`: sha256 `dd1c3c948275805aa2b1aeb6f3754378dafea6adbc5697aa857a08914fd7b5a6`
- `manifest.json`: sha256 `2d366c5d6b6c2de25bbf8c9b67e0e08261dadb1c5f2f05830b0f2e965f7c0222`
- `snax_forge.hjson`: sha256 `10d7006bc425aa30bd544dac8305c7815337d5469a4b7392ad8edd4afa477789`
- `snax_forge_mvt_v1_2_2_streamer_mvt_vadd.hjson`: sha256 `79eb57f669b11581fcc835e9ebd7b35a78a51423a67b861f7c8b4487ca5684ab`
- `tasks.json`: sha256 `c0f68381136311baf4a1ccbb3af9a8115f494315407cf80eaf0da513f2e71d77`
