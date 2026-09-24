# SNAX-FORGE

Architecture skeleton and decision log: `docs/ARCHITECTURE.md`.
Read it before designing or changing any component. Do not contradict a logged decision (D-numbers); propose a new one instead.
What the model-side artefacts mean, field by field: `docs/CONTRACTS.md`. Its
section 8 holds the rules a new block kind must follow; a snippet in it is
checked against the file it came from, so update both together.

## Principles
- Model first: Python models are the source of truth; RTL is checked against them.
- DSE decides, LOWER derives command sequences, MODEL measures. Never mix roles.
- All artefacts (DFG, BRM, design point, control program, profile, trace) are
  serialisable text (JSON/YAML) and must round-trip.
- Extend by registration and namespaced attrs, never by editing core classes.
- Build the vecadd vertical slice before generalising.

## Commands
- Environment: `pixi install`
- Tests: `pixi run test` (everything under tests/), `pixi run test-model` (SNAX-MODEL only), `pixi run test-lower` (SNAX-LOWER only), `pixi run test-brm` (SNAX-BRM only), `pixi run test-dfg` (SNAX-DFG only; both also run tests/test_expr.py, the shared expressions); CI runs `pixi run -e ci test`
- SDFG of a kernel: `pixi run forge <kernel>` writes `out/sdfg/<kernel>.raw.sdfg` and `.simplified.sdfg` (the input of the SNAX-DFG importer, D71)
- Viewer: `pixi run view DIR [DIR ...]` serves model output directories at http://127.0.0.1:8765/ (D55); the DFG viewer, `pixi run view-dfg FILE [FILE ...]`, comes with VIS5 (D76)
- Clean slate: `pixi run clean` removes out/, caches and Chisel build trees (`pixi run clean --dry-run` lists them first)
- Lint/format: `pixi run lint` (ruff check + format check, CI runs it), `pixi run fmt` to fix

## Conventions
- Python 3.x, type hints on all public functions, dataclasses for artefacts.
- One package per component under `snax_forge/`: snax_model and viz (built), lower (task list → program built, D64), brm (format, instances, the affine notation and the library built, D68, D70; BRMs in snax_forge/brm/library/), dfg (the `.snaxdfg` format built, D77; importers and the reference executor next, D71), then sandbox (transforms and recipes, D72) and dse (automated search, M8).
- Generated `.snaxdfg` files and design points go under `out/`, not in git (D71); test fixtures are the exception (tests/dfg/fixtures/, kept in the form `Graph.to_json` writes).
- Value expressions (BRM fields, SNAX-DFG subsets, ranges and shapes) are one grammar in `snax_forge/expr.py` (D68, D77): ints, names, `+ - * //`, parsed with `ast`, stored canonical.
- Scenarios: one folder per scenario under `scenarios/`, with the `scenario.py` that makes it and a hand-written `tasks.json` it lowers into the program (D64–D66). `pixi run scenarios` (`python scenarios/make.py`, `--check` to compare) writes `scenario.json`, the `.npy` data and the cluster files; they are generated, ignored by git and never edited by hand (D67). `pixi run model-run` and the test session write them first.
- Every new artefact type gets a `to_dict` / `from_dict` and a round-trip test. Versioned JSON schemas in `schemas/` come from M6 on (F2), not before (D26).
- Shared test helpers live in `tests/<package>/helpers.py`, not copied per file.