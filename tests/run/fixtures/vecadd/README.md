# vecadd at trace level task

A recorded SNAX-GVSoC run, so CI converts real events without GVSoC
(`tests/run/test_fixture.py`, no expected numbers). Made in snax-forge's
image with snax-gvsoc at `5a8bfbd4` (FLOW3 1b step 3):

    pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json --trace task

and copied from `out/flow/vecadd_task/`: `run/events.log`, the three hjson
files of `gen/`, `tasks.json`, `design/design_point.json` (its `graph_from`
made relative) and `check.json` (the profile's `total_cycles` and
`functional_check`, what the flow's check stage gives the converter). Made
again the same way when snax-gvsoc's grammar (`docs/EVENTS.md`) changes.
