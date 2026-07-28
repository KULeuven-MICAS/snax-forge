from pathlib import Path


def _repo_root() -> Path:
    """
    Walk up to the directory containing pixi.toml.

    Depth-independent: survives moving this module between package levels,
    which counting parents[] does not.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pixi.toml").is_file():
            return parent
    raise RuntimeError(f"no pixi.toml found above {Path(__file__).resolve()}")
