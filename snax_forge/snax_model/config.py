"""Configuration dataclasses and the JSON format of SNAX-MODEL (MOD10, D26, D44).

What this holds
---------------
Two things every configuration class needs and nothing else does:

* ``Config``, a mixin giving a flat dataclass ``to_dict`` / ``from_dict``.
  Every field is written, defaults included, so a written file says
  everything and a missing key takes the class default. Unknown keys are an
  error, so a typo never silently falls back to a default.
* ``to_json``, the one JSON writer used for every file the model writes:
  insertion order, small containers on one line, final newline. It is
  deterministic, and ``json.loads`` gives the object back (D44).

Until the M6 freeze these are plain dataclasses and plain JSON; versioned
schemas are F2's job (D26).

This module imports nothing from the package, so every config class can use
it: ``L1Config`` (mem.py), ``L2Config`` (l2.py), ``StreamerConfig``
(streamer.py), ``DmaConfig`` (dma.py) and ``ControllerConfig`` (ctrl.py).
``AccelConfig`` is *not* one of them: it holds a Python function, so an
accelerator is described by a registered kind plus parameters instead (D43,
CONTRACTS.md section 4).

Scenario-level classes (``ClusterConfig``, ``ComponentSpec``,
``RegisterMapSpec``, ``MemInit``, ``Scenario``) keep their own ``to_dict`` /
``from_dict`` in scenario.py: they are not flat, and their errors name the
scenario.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

# =============================================================================
# Plain values and key checks
# =============================================================================


def plain(v: Any) -> Any:
    """Tuples -> lists, mappings -> dicts, recursively: JSON-ready values."""
    if isinstance(v, Mapping):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return v


def check_keys(d: Mapping[str, Any], allowed: Sequence[str], what: str = "") -> None:
    """Reject unknown keys. ``what`` prefixes the message when it is given."""
    unknown = sorted(set(d) - set(allowed))
    if unknown:
        head = f"{what}: " if what else ""
        raise ValueError(f"{head}unknown keys {unknown} (allowed: {list(allowed)})")


# =============================================================================
# The mixin
# =============================================================================


class Config:
    """Flat dataclass <-> plain dict (D26). Mixed into every config class."""

    def to_dict(self) -> dict[str, Any]:
        """Every field, in declaration order, as JSON-ready values."""
        return {f.name: plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Any:
        """Inverse of ``to_dict``; missing keys take the class default.

        Raises ValueError for unknown keys and for values the class itself
        rejects; scenario.py adds the name of the field it was reading.
        """
        check_keys(d, [f.name for f in fields(cls)])
        return cls(**d)


# =============================================================================
# JSON writing (D44)
# =============================================================================

JSON_WIDTH = 100  # line length of written JSON, as ruff's for the code


def _scalar(v: Any) -> bool:
    return not isinstance(v, (dict, list, tuple))


def _json(v: Any, indent: int, col: int) -> str:
    """One value, its block indented by ``indent``, starting at column ``col``.

    A container goes on one line if it ends within JSON_WIDTH, and a list of
    scalars always does (a histogram stays one line); otherwise one item per
    line, one space deeper.
    """
    flat = json.dumps(v)
    if _scalar(v) or (isinstance(v, (list, tuple)) and all(map(_scalar, v))):
        return flat
    if col + len(flat) <= JSON_WIDTH or not v:
        return flat
    pad = " " * (indent + 1)
    if isinstance(v, dict):
        items = []
        for k, x in v.items():
            head = f"{pad}{json.dumps(k)}: "
            items.append(head + _json(x, indent + 1, len(head)))
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [pad + _json(x, indent + 1, len(pad)) for x in v]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def to_json(obj: Any) -> str:
    """The one JSON format of every written file (D44)."""
    return _json(obj, 0, 0) + "\n"
