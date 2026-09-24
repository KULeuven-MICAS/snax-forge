"""Recipes (SBX1, D72, D80): an ordered list of transforms applied to an imported graph.

A recipe is a hand-written JSON file, kept under ``recipes/``:

    {
     "name":    "vecadd",
     "kernel":  "vecadd",                 the kernel whose import it starts from
     "params":  {"W": 4},                 recipe params, each an int or a string
     "symbols": {"N": 64},                the graph's symbols, ints or expressions
                                          over the params
     "steps": [
      {"transform": "split_map", "params": {"map": "add_map", "factor": "W"}},
      {"transform": "bind",      "params": {"node": "add", "brm": "elementwise_add", ...}}
     ]
    }

**Params and sweeps.** A recipe param is a named value the steps refer to.
A transform declares which of its own params are ints (``split_map``'s
``factor``); those take an int or an expression over the recipe params
(snax_forge/expr.py), everything else is taken as written. ``--set W=8`` on
the command line (``Recipe.with_params``) overrides a param, so one recipe
covers a sweep: the same steps, one run per value (D72; the sweep runner
itself is DSE5). The recipe written next to a run's output has the values
that run used.

Every field is written; ``params`` and ``symbols`` default to empty, unknown
keys are errors (D26, D41). Whether a step's transform exists and takes its
params is checked when the recipe runs (run.py), where the transforms are
registered.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from snax_forge import expr
from snax_forge.expr import ExprError, Value
from snax_forge.snax_model.config import check_keys, plain, to_json


class SandboxError(ValueError):
    """A recipe or a transform step that is malformed or cannot be applied."""


def _keys(d: Any, required: tuple[str, ...], optional: tuple[str, ...], what: str) -> None:
    if not isinstance(d, Mapping):
        raise SandboxError(f"{what}: must be an object, got {d!r}")
    try:
        check_keys(d, [*required, *optional], what)
    except ValueError as e:
        raise SandboxError(str(e)) from None
    missing = [k for k in required if k not in d]
    if missing:
        raise SandboxError(f"{what}: missing key {missing[0]!r}")


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
class Recipe:
    name: str
    kernel: str
    params: dict[str, int | str] = field(default_factory=dict)
    symbols: dict[str, Value] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)

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
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Recipe:
        _keys(d, ("name", "kernel"), ("params", "symbols", "steps"), "recipe")
        name = d["name"]
        for key in ("params", "symbols"):
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
        )

    def to_json(self) -> str:
        return to_json(self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())
