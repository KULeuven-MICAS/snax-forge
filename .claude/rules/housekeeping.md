# Housekeeping
- Format: `pixi run fmt`. Lint: `pixi run lint`.
- Always use the pixi tasks, never call ruff/black/pytest directly, so versions match the pinned environment.
- Formatting and linting also run automatically via hooks; if a hook reports lint errors, fix them before finishing.
- After finishing your work, make sure to run the formatting and linting tasks to ensure code quality.