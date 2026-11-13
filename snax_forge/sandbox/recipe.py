"""Recipes: an ordered list of transforms applied to an imported graph.

``Recipe`` (``Recipe.load``, ``to_dict``) and its ``Step`` read and write a recipe, a JSON file
in a category folder of ``recipes/`` (CONTRACTS, "Recipe"). ``find_recipe`` finds one by name,
``recipe_paths`` lists them all. A transform's int params take an int or an expression over the
recipe params (snax_forge/expr.py); ``Recipe.with_params`` (``--set W=8``) overrides a param, so
one recipe covers a sweep, and the recipe written next to a run has the values that run used.

An unknown dtype is an error here; an unknown container one at the import, which knows the
kernel's. Whether a step's transform exists and takes its params is checked when the recipe
runs (run.py), where the transforms are registered.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from snax_forge import expr
from snax_forge.common import JsonFile, check_keys, plain, repo_root
from snax_forge.expr import ExprError, Value


class SandboxError(ValueError):
    """A recipe or a transform step that is malformed or cannot be applied."""


RECIPE_ROOT = repo_root() / "recipes"


def recipe_paths(root: Path = RECIPE_ROOT) -> dict[str, list[Path]]:
    """Every recipe under ``root``, one folder deep: name -> its files (more than one is an
    error for ``find_recipe``). A ``.json`` directly in ``root`` is an error."""
    flat = sorted(p.name for p in root.glob("*.json"))
    if flat:
        raise SandboxError(
            f"{root}: recipes live in a folder by category, not directly in it: {flat}"
        )
    paths: dict[str, list[Path]] = {}
    for p in sorted(root.glob("*/*.json")):
        paths.setdefault(p.stem, []).append(p)
    return paths


def find_recipe(name: str, root: Path = RECIPE_ROOT) -> Path:
    """The file of the recipe ``name``: ``root/<folder>/<name>.json``, one folder deep."""
    found = recipe_paths(root).get(name, [])
    if not found:
        raise SandboxError(f"no recipe {name!r} in any folder of {root}")
    if len(found) > 1:
        where = [str(p.relative_to(root)) for p in found]
        raise SandboxError(f"recipe {name!r} is in {len(found)} folders of {root}: {where}")
    return found[0]


def _keys(d: Any, required: tuple[str, ...], optional: tuple[str, ...], what: str) -> None:
    try:
        check_keys(d, [*required, *optional], what, required)
    except (TypeError, ValueError) as e:
        raise SandboxError(str(e)) from None


@dataclass
class Step:
    """One transform and its params, as written (int params may be expressions)."""

    transform: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"transform": self.transform, "params": plain(self.params)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Step:
        _keys(d, ("transform",), ("params",), what)
        params = d.get("params", {})
        if not isinstance(params, Mapping):
            raise SandboxError(f"{what}.params: must be an object")
        return cls(d["transform"], dict(params))


@dataclass
class Recipe(JsonFile):
    name: str
    kernel: str
    params: dict[str, int | str] = field(default_factory=dict)
    symbols: dict[str, Value] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for what, v in (("name", self.name), ("kernel", self.kernel)):
            if not isinstance(v, str) or not v:
                raise SandboxError(f"recipe {what}: must be a non-empty string, got {v!r}")
        for k, v in self.params.items():
            if not isinstance(k, str) or not k.isidentifier():
                raise SandboxError(f"{self.name}.params: {k!r} is not a valid name")
            if type(v) not in (int, str):
                raise SandboxError(f"{self.name}.params.{k}: must be an int or a string")
        for s, v in self.symbols.items():
            self.value(v, f"{self.name}.symbols.{s}")
        from snax_forge.brm import INT_DTYPES

        for c, dt in self.dtypes.items():
            if dt not in INT_DTYPES:
                raise SandboxError(
                    f"{self.name}.dtypes.{c}: {dt!r} is not a NumPy integer dtype "
                    f"({list(INT_DTYPES)})"
                )

    def value(self, v: Any, what: str) -> int:
        """An int, or an expression over the recipe params, as an int."""
        try:
            expr.check(v, what)
            x = expr.evaluate(v, self.params)
        except ExprError as e:
            raise SandboxError(str(e)) from None
        if type(x) is not int:
            raise SandboxError(f"{what}: {v!r} is not an int")
        return x

    def bound_symbols(self) -> dict[str, int]:
        return {s: self.value(v, f"{self.name}.symbols.{s}") for s, v in self.symbols.items()}

    def with_params(self, values: Mapping[str, int | str]) -> Recipe:
        """The recipe with some params set (``--set``); each must already be a param."""
        unknown = sorted(set(values) - set(self.params))
        if unknown:
            raise SandboxError(
                f"{self.name}: no param {unknown[0]!r} (params: {list(self.params)})"
            )
        return replace(self, params={**self.params, **values})

    # --- files ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kernel": self.kernel,
            "params": dict(self.params),
            "symbols": dict(self.symbols),
            **({"dtypes": dict(self.dtypes)} if self.dtypes else {}),
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Recipe:
        _keys(d, ("name", "kernel"), ("params", "symbols", "dtypes", "steps"), "recipe")
        name = d["name"]
        for key in ("params", "symbols", "dtypes"):
            if not isinstance(d.get(key, {}), Mapping):
                raise SandboxError(f"{name}.{key}: must be an object")
        steps = d.get("steps", [])
        if not isinstance(steps, (list, tuple)):
            raise SandboxError(f"{name}.steps: must be a list")
        return cls(
            name,
            d["kernel"],
            dict(d.get("params", {})),
            dict(d.get("symbols", {})),
            [Step.from_dict(s, f"{name}.steps[{i}]") for i, s in enumerate(steps)],
            dict(d.get("dtypes", {})),
        )
