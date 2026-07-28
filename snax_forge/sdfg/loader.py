from __future__ import annotations

import importlib.util
from pathlib import Path

from .spec import KernelSpec


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


KERNEL_ROOT = _repo_root() / "kernels"


def kernel_paths() -> dict[str, Path]:
    """
    Return a dict of kernel names to their source paths.
    """
    return {p.stem: p for p in sorted(KERNEL_ROOT.rglob("*.py")) if not p.stem.startswith("_")}


def load(name: str) -> KernelSpec:
    """
    Load a kernel module by name and return its SPEC.

    Raises KeyError if the kernel is not found, AttributeError if the module
    does not define a SPEC."""
    paths = kernel_paths()
    if name not in paths:
        raise KeyError(f"unknown kernel {name!r}; available: {sorted(paths)}")

    path = paths[name]
    spec = importlib.util.spec_from_file_location(f"_kernel_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "SPEC"):
        raise AttributeError(f"{path} defines no SPEC")
    return module.SPEC
