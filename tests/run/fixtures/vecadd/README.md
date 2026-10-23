# vecadd at trace levels task and stall

A recorded SNAX-GVSoC run, so CI converts real events without GVSoC
(`tests/run/test_fixture.py`, no expected numbers). Made in snax-forge's
image with snax-gvsoc at `dd475a00` (LNK2b S1, the generated accelerator
named `snax_forge`; both `events.log` files, `tasks.json` and the ELF came
out byte-identical to `5d23a1da`'s, FLOW3 round 3 G1, the `class` lines):

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --trace task
    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --trace stall

and copied from `out/flow/vecadd_task/`: `run/events.log`, the three hjson
files of `gen/`, `tasks.json`, `design/design_point.json` (its `graph_from`
made relative) and `check.json` (the profile's `total_cycles` and
`functional_check`, what the flow's check stage gives the converter); from
`out/flow/vecadd_stall/` its `run/events.log` as `events_stall.log` (its
other files are the task run's). `elf_symbols.json` is what `run.json`
records of the same ELF (`sw/forge.elf`, sha256 `b6b4b1c70d8c8986`): the
L2 containers' symbols (`run/elf.py`), added in the round 2a fix-up. Made
again in CLN1 R2a at snax-gvsoc `8c9a47c3`: only the source hjson's path
changed (`design_point.json`, `snax_forge.hjson`), every other file the same. Made again the same way when snax-gvsoc's
grammar (`docs/EVENTS.md`) changes.
