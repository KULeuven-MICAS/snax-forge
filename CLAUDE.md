# SNAX-FORGE

Read before designing or changing a component, and only what the task needs (D117):

- `docs/ARCHITECTURE.md`: section 4 (the overview) and the section of the component you change.
- `docs/DECISIONS.md`: the entries cited there. Do not contradict one; propose a new one with the
  next free number, following the rules at the top of that file.
- `docs/CONTRACTS.md`: the section of the artefact you produce or read. A snippet in it is checked
  against the file it came from, so update both together.
- `docs/STATUS.md`: the task you work on and its open items. Update it with every change.
- `docs/PAPER.md`: only for paper work (P1).

Keep the documents short (D117): one fact in one place, a decision only for a choice someone may
want to reverse, implementation details in module docstrings and tests.

## Principles
- The BRM is the user's source of truth; the model's accelerator and the RTL come from it and are
  checked against each other by one program on both (D116).
- The model is SNAX-GVSoC, checked against snax_cluster RTL.
- DSE decides, LOWER derives, the model measures. Never mix roles.
- All artefacts (DFG, BRM, recipe, platform, design point, task list, program, profile, trace) are
  serialisable text and must round-trip.
- Extend by registration and namespaced attrs, never by editing core classes.
- Build the vecadd vertical slice before generalising.

## Commands
- Environment: `pixi install`; `pixi run check` checks DaCe and its pins.
- Tests: `pixi run test` (all), or one block: `test-lower`, `test-brm`, `test-dfg`,
  `test-sandbox`, `test-design`, `test-flow`, `test-run`, `test-report`, `test-gen`,
  `test-kernels`. CI runs `pixi run -e ci test`.
- Lint: `pixi run lint` (CI runs it); `pixi run fmt` fixes.
- The whole path, on SNAX-GVSoC (D136, in the image, submodule checked out): `pixi run flow
  recipes/vecadd.json --platform platforms/snax_alu.json [--set W=8] [--set platform.PATH=VALUE]
  [--set memory.PATH=VALUE] [--trace LEVEL[:SOURCES]] [--vcd]`, everything in `out/flow/<name>/`
  (`<name>_<level>` above `off`); it builds GVSoC when needed. The run's events become
  `run/profile.json` and, from `task` (then `beat`, `stall`, `finegrained`), `run/trace.jsonl`
  (`snax_forge/run/`, D137); `--vcd` adds gtkwave's `run/all.vcd`.
  From a bound graph instead of a recipe (no sandbox, D108): `pixi run flow
  out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_alu.json [--kernel K]`.
  The single steps (`forge`, `import-dfg`, `check-dfg`, `sandbox`, `design`, `lower`) are in
  README.md section 2.
- Generator: `pixi run gen-gvsoc out/design/<name>/design_point.json` writes a design point's
  hjson files, headers, `data.h` and `main.c` to `out/gvsoc/<name>/`.
- SNAX-GVSoC: `gvsoc/Makefile` (`make build`, `make sw`, `make run`), from the container's plain
  shell or through pixi (`pixi run make -C gvsoc ...`); `docs/RUN.md` walks through it.
- Kernels: `pixi run kernels` lists the staged kernels in `kernels/` (tier, dtype, whether the
  SDFG builds).
- Viewers: `pixi run view DIR ...` (flow runs, port 8765), `pixi run view-dfg FILE|DIR ...`
  (graphs, port 8766).
- Reports: the flow writes `report/design.md` and `report/run.md`; `pixi run report DIR` writes
  them again for a flow folder.
- Chisel: `pixi run -e hw chisel-test`, `chisel-gen`; `hw/chisel/` is a project of its own.
- Clean slate: `pixi run clean` (`--dry-run` lists first; it keeps the GVSoC build in
  `out/snax-gvsoc/`, `--all` removes it too).

## Conventions
- Python 3.11, type hints on all public functions, dataclasses for artefacts.
- One package per component under `snax_forge/` (ARCHITECTURE.md section 5 lists them). Recipes
  live in `recipes/`, platforms in `platforms/`, BRMs in `snax_forge/brm/library/`.
- Generated files (`.snaxdfg`, design points, runs) go under `out/`, not in git; test fixtures are
  the exception (`tests/dfg/fixtures/`, in the form `Graph.to_json` writes).
- Value expressions (BRM fields, SNAX-DFG subsets, ranges, shapes) are one grammar in
  `snax_forge/expr.py`: ints, names, `+ - * //`, parsed with `ast`, stored canonical.
- Every new artefact type gets a `to_dict` / `from_dict` and a round-trip test. The generator's
  hjson files have schemas (`snax_forge/gen/schema/`); versioned schemas for the contracts come
  with M6 (F2), not before.
- Shared test helpers live in `tests/<package>/helpers.py`, not copied per file.
- Walk-throughs live in `examples/<name>/README.md`; a test checks every number one states, and
  their outputs are not checked in.
