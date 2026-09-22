# SNAX-FORGE

Architecture skeleton and decision log: `docs/ARCHITECTURE.md`.
Read it before designing or changing any component. Do not contradict a logged decision (D-numbers); propose a new one instead.

## Principles
- Model first: Python models are the source of truth; RTL is checked against them.
- DSE decides, LOWER derives command sequences, MODEL measures. Never mix roles.
- All artefacts (DFG, BRM, design point, control program, profile, trace) are
  serialisable text (JSON/YAML) and must round-trip.
- Extend by registration and namespaced attrs, never by editing core classes.
- Build the vecadd vertical slice before generalising.

## Commands
- Environment: `pixi install`
- Tests: `pixi run test` (everything under tests/), `pixi run test-model` (SNAX-MODEL only); CI runs `pixi run -e ci test`
- Lint/format: `pixi run lint` (ruff check + format check, CI runs it), `pixi run fmt` to fix

## Conventions
- Python 3.x, type hints on all public functions, dataclasses for artefacts.
- One package per component under `snax_forge/`: dfg, brm, dse, lower, model, viz.
- Every new artefact type gets a JSON schema in `schemas/` and a round-trip test.