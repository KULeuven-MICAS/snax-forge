# dot 1 at trace levels task and stall

A recorded SNAX-GVSoC run of an accelerator of two groups (D148), so CI converts it without GVSoC (`tests/run/test_fixture.py`, `test_dot1.py`): dot 1, the accelerator `dot` of groups `mul` and `sum`, a task each. Made in M13 B1 R3 in snax-forge's image with snax-gvsoc at `f960b066`:

    pixi run flow recipes/dot_1.json --platform platforms/snax_forge.json --trace task

and copied from `out/flow/dot_1_task/` as vecadd's (`../vecadd/README.md`): `run/events.log`, the three description files of `gen/`, `tasks.json`, `design/design_point.json` (its `graph_from` made relative), `check.json` (the profile's `total_cycles` and `functional_check`, each task's group with it) and `elf_symbols.json` (`run.json`'s, the ELF `sw/forge.elf` sha256 `a34fbf71a179b4ff`). Made again the same way when snax-gvsoc's grammar (`docs/EVENTS.md`) or the generated files change. `accelerator.json` written again in CTRL1 (D152, every CSR with its width) by the same command at snax-gvsoc `c061e32b`: `events.log`, `tasks.json`, the two hjson files and the ELF came out byte-identical.

`events_stall.log` (DRY1-2 R2) is the same recipe at `--trace stall`, by `pixi run flow recipes/dot_1.json --platform platforms/snax_forge.json --trace stall` at snax-forge `dry1-2-r1` and snax-gvsoc `c061e32b`: its ELF (sha256 `a34fbf71a179b4ff`), `tasks.json`, the three description files and `elf_symbols.json` are the task run's, its window 304 cycles too; only its `run/events.log` is kept, so the converter reads both groups' beat and stall lines without GVSoC (`tests/run/test_dot1.py`).
