# gvsoc/targets

snax-forge's own GVSoC targets: Python modules gvrun reads with
`--target-dir` (`make run` in `gvsoc/` passes this folder), on the
components snax-gvsoc installs. Nothing here is compiled: a target here may
use only components already built into the install (`make build`).

| Target | What |
|---|---|
| `alu_probe` | snax-gvsoc's `snax_alu` under another name, its accelerator and streamer template copied in (`AluProbeAcc`), kept as a smoke test (GEN3 step 0b) |
| `tiled_add_sparse_probe` | snax-gvsoc's `snax_tiled_add` accelerator on `snax_alu_sparse`'s sparse interconnect: a component tree no built target has (GEN3 round 1b) |

A target here has no platform tree in the install (`libplatform_tree_<target>.so`,
built only for `TARGETS`): gvrun prints one line on stderr, `Warning: target
'<target>': no platform tree, using the JSON config (...)`, and runs it on
the JSON config path. That needs snax-gvsoc from GEN3 round 1b on (the
sibling checkout, `SNAX_GVSOC=../../snax-gvsoc`); the submodule's pin,
`c911569`, still stops with `Missing platform_tree library`.
`tiled_add_sparse_probe` also needs snax-gvsoc from GEN3 step 2 on, built
with `snax_tiled_add` in `TARGETS` (its `SnaxShell` and `SnaxUnit`):

    make build SNAX_GVSOC=../../snax-gvsoc TARGETS="snax_alu;snax_alu_sparse;snax_tiled_add"
    make sw SNAX_GVSOC=../../snax-gvsoc APPS="snax-alu snax-shell"
    make run SNAX_GVSOC=../../snax-gvsoc TARGET=alu_probe ELF=../../snax-gvsoc/build/sw/apps/snax-alu/100/snax-alu.elf
    make run SNAX_GVSOC=../../snax-gvsoc TARGET=tiled_add_sparse_probe ELF=../../snax-gvsoc/build/sw/apps/snax-shell/snax-shell.elf
