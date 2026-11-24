# vecadd_v1 at trace levels task and stall

A recorded SNAX-GVSoC run, so CI converts real events without GVSoC: vecadd at N = 64, W = 4: one accelerator `acc` of one group of one unit (`SimdLanes`, add), a task. Made in snax-forge's image, the submodule checked out, by

    pixi run flow recipes/elementwise/vecadd_v1.json --platform platforms/snax_forge.json --trace task
    pixi run flow recipes/elementwise/vecadd_v1.json --platform platforms/snax_forge.json --trace stall

and copied from each flow's folder: `run/events.log` as `events_task.log` and `events_stall.log`, `check.json` (the profile's `total_cycles` and `functional_check`) and `elf_symbols.json` (`run.json`'s), the ELF the same for every level (sha256 `0727a196a0226295cc99040063603b20b884d856a63b8290966839e14b620168`). The run: 209 cycles, data 0 errors (64 words), counters `add_acc` 19 (16 beats).

The generated files the run was made on are not kept: `tests/run/helpers.py`'s `recorded_files` makes them again from the recipe at these sizes (`FORGE_FIXTURES`, the forge/ folder of the same name) and checks each against its sha256 here; when one differs, record the run again by the commands above and write the new hashes here.

- `accelerator.json`: sha256 `f3d22e18be5b767c99dad612922346249a9f2bc62dfcabcaf59faf13db9a01f6`
- `design_point.json`: sha256 `aac82ac44fbf54a3417b79780b00c704d7a3d3e69fe49b79f5aa142d87582e3c`
- `manifest.json`: sha256 `b0e0da53a65b2613d2252b19386fff8f2d0fa82f7810822ac949bd743865666c`
- `snax_forge.hjson`: sha256 `56db9f212f4bc5c18f7e8374e91fa2fe1b56f1103b207b7854ba8d67c1e0d4e7`
- `snax_forge_vecadd_v1_streamer_acc.hjson`: sha256 `f42084873266f3c37a3785c7a6ddd049f2b99f64e0c9f434c64492d1d726b161`
- `tasks.json`: sha256 `15013da7aaafef96fed06e8b4840d31328ba0874cfd2d132650ac1e486079376`
