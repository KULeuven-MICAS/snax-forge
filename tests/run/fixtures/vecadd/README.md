# vecadd at trace levels task and stall

A recorded SNAX-GVSoC run, so CI converts real events without GVSoC
(`tests/run/test_fixture.py`, no expected numbers). Made in snax-forge's
image with snax-gvsoc at `fec4c219` (FLOW3 2a step 4):

    pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json --trace task
    pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json --trace stall

and copied from `out/flow/vecadd_task/`: `run/events.log`, the three hjson
files of `gen/`, `tasks.json`, `design/design_point.json` (its `graph_from`
made relative) and `check.json` (the profile's `total_cycles` and
`functional_check`, what the flow's check stage gives the converter); from
`out/flow/vecadd_stall/` its `run/events.log` as `events_stall.log` (its
other files are the task run's). `elf_symbols.json` is what `run.json`
records of the same ELF (`sw/forge.elf`, sha256 `b6b4b1c70d8c8986`): the
L2 containers' symbols (`run/elf.py`), added in the round 2a fix-up. Made again the same way when snax-gvsoc's
grammar (`docs/EVENTS.md`) changes.
