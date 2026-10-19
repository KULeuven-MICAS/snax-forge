# gvsoc/targets

snax-forge's own GVSoC targets: Python modules gvrun reads with
`--target-dir` (`make run` in `gvsoc/` passes this folder), on the
components snax-gvsoc installs. Nothing here is compiled: a target here may
use only components already built into the install (`make build`).

A target here, or one snax-forge generates, never takes the name of an
installed target: gvrun puts the install's `targets` ahead of this folder on
its path, so an installed `snax_alu` is run even when a `snax_alu.py` is
here (tried: a `snax_alu.py` that stops on import, in a `--target-dir`, was
never loaded). The same holds for any module a target imports. snax-forge's
generator writes no targets since LNK2a: a design point runs on the
installed `snax_hjson` with its cluster file (`--param hjson=`).

| Target | What |
|---|---|
| `alu_probe` | snax-gvsoc's `snax_alu` under another name, its accelerator and streamer template copied in (`AluProbeAcc`), kept as a smoke test (GEN3 step 0b) |
| `tiled_add_sparse_probe` | snax-gvsoc's `snax_tiled_add` accelerator on `snax_alu_sparse`'s sparse interconnect: a component tree no built target has (GEN3 round 1b) |

A target here has no platform tree in the install (`libplatform_tree_<target>.so`,
built only for `TARGETS`): gvrun prints one line on stderr, `Warning: target
'<target>': no platform tree, using the JSON config (...)`, and runs it on
the JSON config path. The submodule's pin (`21830e3`) runs both probes;
`tiled_add_sparse_probe` uses `snax_tiled_add`'s `SnaxShell` and `SnaxUnit`,
built by default (`TARGETS`). From `gvsoc/`:

    make build
    make sw APPS="snax-alu snax-shell"
    make run TARGET=alu_probe ELF=snax-gvsoc/build/sw/apps/snax-alu/100/snax-alu.elf
    make run TARGET=tiled_add_sparse_probe ELF=snax-gvsoc/build/sw/apps/snax-shell/snax-shell.elf

Each prints the warning, then `Accelerator Cycles: 25` and `Number of
errors: 0`, and `snax-shell errors: 0`. `SNAX_GVSOC=../../snax-gvsoc` (on
each command, the ELFs under `../../snax-gvsoc/build/sw/`) points at a
sibling checkout instead.
