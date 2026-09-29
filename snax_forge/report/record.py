"""Plain dataclass records with ``to_dict`` / ``from_dict`` for the reports (REP1, D99).

A report is a tree of records: every field is a plain value (int, float,
str, bool, None), a list or dict of plain values, a record, or a list of
records. ``to_dict`` writes every field in declaration order; ``from_dict``
rebuilds the tree from the type hints, rejecting unknown keys, so a report
round-trips exactly (D26's rule for artefacts).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping
from typing import Any


def _plain(v: Any) -> Any:
    if isinstance(v, Record):
        return v.to_dict()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def _record_type(hint: Any) -> type[Record] | None:
    """The Record class inside ``hint`` (``X``, ``X | None``), if any."""
    if isinstance(hint, type) and issubclass(hint, Record):
        return hint
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        for a in typing.get_args(hint):
            t = _record_type(a)
            if t is not None:
                return t
    return None


def _build(hint: Any, v: Any) -> Any:
    if v is None:
        return None
    rec = _record_type(hint)
    if rec is not None and isinstance(v, Mapping):
        return rec.from_dict(v)
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        for a in typing.get_args(hint):
            if a is not type(None):
                return _build(a, v)
    if origin is list:
        (inner,) = typing.get_args(hint) or (Any,)
        return [_build(inner, x) for x in v]
    if origin is dict:
        args = typing.get_args(hint)
        inner = args[1] if len(args) == 2 else Any
        return {k: _build(inner, x) for k, x in v.items()}
    return v


class Record:
    """Mixin for report dataclasses (module doc)."""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)}  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Any:
        hints = typing.get_type_hints(cls)
        names = [f.name for f in dataclasses.fields(cls)]  # type: ignore[arg-type]
        unknown = sorted(set(d) - set(names))
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown keys {unknown}")
        return cls(**{k: _build(hints[k], d[k]) for k in names if k in d})
