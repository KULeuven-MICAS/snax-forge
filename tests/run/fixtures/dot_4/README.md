# dot 4 at trace level task

A recorded SNAX-GVSoC run of an accelerator of one linked group (D148), so CI
converts it without GVSoC (`tests/run/test_dot4.py`): dot 4, the group `dot`
of units `mac` and `sum` wired by `out_partial`, one SNAX-GVSoC component
`snax_acc0/group_dot` named after the group, one task `mult_dot`. Made in
M13 B1x F3 in snax-forge's image with snax-gvsoc at `253f0147`, at the
recipe's point (N = 512, W = 4):

    pixi run flow recipes/dot_4.json --platform platforms/snax_forge.json --trace task

and copied from `out/flow/dot_4_task/` as dot 1's (`../dot_1/README.md`):
`run/events.log`, the three description files of `gen/`, `tasks.json`,
`design/design_point.json` (its `graph_from` made relative), `check.json` (the
profile's `total_cycles` and `functional_check`) and `elf_symbols.json`
(`run.json`'s, the ELF `sw/forge.elf` sha256 `cbb37f12894adbef`). The run:
437 cycles, COUNTER 133 = C§17's C4 130 + the streamers' fill 3. Made again the
same way when snax-gvsoc's grammar (`docs/EVENTS.md`) or the generated files
change.
