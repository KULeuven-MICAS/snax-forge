# gvsoc/targets

snax-forge's own GVSoC targets: Python modules gvrun reads with
`--target-dir` (`make run` in `gvsoc/` passes this folder), on the
components snax-gvsoc installs. Nothing here is compiled: a target here may
use only components already built into the install (`make build`).

| Target | What |
|---|---|
| `alu_probe` | snax-gvsoc's `snax_alu` under another name, its accelerator and streamer template copied in (`AluProbeAcc`), kept as a smoke test (GEN3 step 0b) |

A target here has no platform tree in the install (`libplatform_tree_<target>.so`,
built only for `TARGETS`), and gvrun stops on that: `Missing platform_tree
library for target 'alu_probe'`. So `alu_probe` does not run yet; snax-gvsoc
`GEN3.md` (Open) has the way out.
