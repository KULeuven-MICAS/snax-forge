# SNAX-FORGE

Read before designing or changing a component, and only what the task needs (D117):

- `docs/ARCHITECTURE.md`: section 4 (the overview) and the section of the component you change.
- `docs/DECISIONS.md`: the entries cited there. Do not contradict one; propose a new one with the next free number, following the rules at the top of that file.
- `docs/CONTRACTS.md`: the section of the artefact you produce or read. A snippet in it is checked against the file it came from, so update both together.
- `docs/STATUS.md`: the task you work on and its open items. Update it with every change.
- `docs/PAPER.md`: only for paper work.
- `docs/TOOL.md`: every command, with what it reads and writes and an example.

Keep the documents short (D117): one fact in one place, a decision only for a choice someone may want to reverse, implementation details in module docstrings and tests.

## Principles

- The BRM is the user's source of truth; the model's accelerator and the RTL come from it and are checked against each other by one program on both (D116).
- The model is SNAX-GVSoC, checked against snax_cluster's RTL.
- DSE decides, LOWER derives, the model measures. Never mix roles.
- Every artefact (DFG, BRM, recipe, platform, design point, task list, program, profile, trace) is serialisable text and round-trips.
- Extend by registration and namespaced attrs, never by editing core classes.

## Commands

The whole list is `docs/TOOL.md`; the ones used most:

- `pixi install`, then `pixi run check` (DaCe and its pins).
- `pixi run test` (all, on every core; `-m "not slow"` leaves out the DaCe builds, the compiles and the largest design), or one block: `test-lower`, `test-brm`, `test-dfg`, `test-sandbox`, `test-char`, `test-estimate`, `test-design`, `test-flow`, `test-run`, `test-report`, `test-gen`, `test-kernels`, `test-viz`, `test-docs`. CI runs `pixi run -e ci test`.
- `pixi run lint` (CI runs it) and `pixi run fmt` (ruff's fixes and format, then the Markdown rule). Always the pixi tasks, never ruff or pytest directly, so the versions are the pinned ones.
- `pixi run flow RECIPE --platform platforms/snax_forge.json [--set ...] [--trace LEVEL]` runs a recipe on SNAX-GVSoC into `out/flow/<name>/` (D136; in the image, the submodule checked out). TOOL.md section 2.
- `pixi run forge RECIPE` and `pixi run forge-set quick|full|mvt` make bundles for snax_cluster's RTL check (the sets are listed in `snax_forge/flow/forge_set.py`). TOOL.md section 6.
- `pixi run sandbox`, `design`, `lower`, `gen-gvsoc`, `gen-rtl`, `bundle`: one stage alone. TOOL.md section 3.
- `pixi run view`, `view-dfg`, `edit-dfg` (ports 8765, 8766, 8767; every POST needs `Content-Type: application/json`). TOOL.md sections 4 and 5.
- `pixi run -e hw chisel-test`, `chisel-gen`, and `chisel-check` after any Scala edit; `hw/chisel/` is a project of its own.
- `pixi run clean` (`--dry-run` lists first; `--all` also removes the GVSoC build and the editor's sessions).

## Conventions

- The local RTL check: a change to the emitted RTL or to the program (`main.c`) needs the quick set checked on vsim before its PR merges, with no FAIL (D123; a NEAR or WARNING gets an open item), and a phase's close needs the full set; the numbers go into open item 95's table. TOOL.md section 6 runs it.
- Python 3.11, type hints on all public functions, dataclasses for artefacts.
- One package per component under `snax_forge/` (ARCHITECTURE section 5 lists them). Recipes live in `recipes/<category>/` (the editor's Save writes `recipes/custom/`), platforms in `platforms/`, BRMs in `snax_forge/brm/library/`.
- Generated files (`.snaxdfg`, design points, runs) go under `out/`, not in git; test fixtures are the exception (`tests/dfg/fixtures/`, in the form `Graph.to_json` writes).
- Value expressions (BRM fields, SNAX-DFG subsets, ranges, shapes) are one grammar in `snax_forge/expr.py`: ints, names, `+ - * //`, parsed with `ast`, stored canonical.
- Every new artefact type gets a `to_dict` / `from_dict` and a round-trip test. The generator's hjson files have schemas (`snax_forge/gen/schema/`); versioned schemas for the contracts come with M6, not before.
- Shared test helpers live in `tests/<package>/helpers.py`, not copied per file.
- Every Chisel module gets a seeded random spec against a Scala model written from its documented behaviour (`hw/chisel/README.md`, Tests).
- No task, round or X tags in code, tests or messages, and a CONTRACTS section is cited by its title (`CONTRACTS, "Recipe"`); Markdown may cite `C§n`. A comment `# NB (Dn): ...` (`// NB` in Scala and C) marks the few spots where the model must match the RTL or a choice looks wrong but is not; keep those. An error message says the fix, not a tag.
- A module docstring is at most about 15 lines: what it does, its entry points, the contract it reads or writes by name. Format detail lives in CONTRACTS, history in git.
- Walk-throughs live in `examples/`: those under `examples/dryrun/` are tutorials with no test, their numbers as printed when written; the others keep a test checking every number they state (D98). Their outputs are not checked in.
- Markdown prose is one line per paragraph, list item and table row, never hard-wrapped (`util/md_reflow.py`, checked by `pixi run lint`); headings, code blocks, tables, display maths and HTML stay as they are. Sections open with two or three sentences, then bullets and tables; a Mermaid diagram where a structure reads better as a picture. Code that writes Markdown (the reports) follows the same rule.
