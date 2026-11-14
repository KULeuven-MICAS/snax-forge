# Housekeeping

- Use the pixi tasks (`pixi run fmt`, `pixi run lint`, `pixi run test ...`), never ruff or pytest directly, so the versions match the pinned environment.
- A hook formats each Python file you edit, and another runs `pixi run lint` when you stop and sends its errors back: fix them before finishing.
