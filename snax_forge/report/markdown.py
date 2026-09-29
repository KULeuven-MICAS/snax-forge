"""Markdown helpers shared by the design and run reports (REP1, D99).

Tables, number and range formatting, one definition each, so both reports
write a number the same way.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


def table(cols: list[str], rows: list[list[Any]], right: Collection[int] = ()) -> list[str]:
    """A Markdown table; ``right`` holds the columns aligned right. None is written "–"."""

    def cell(v: Any) -> str:
        if v is None:
            return "–"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v).replace("|", "\\|")

    out = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---:" if i in right else "---" for i in range(len(cols))) + "|",
    ]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return out


def spans(xs: Sequence[int]) -> str:
    """[0, 1, 2, 3, 8] -> "0–3, 8"."""
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}–{xs[j]}")
        i = j + 1
    return ", ".join(out)


def span_text(x: Sequence[int] | None) -> str:
    """[4, 8] -> "4–8", [4, 4] -> "4", None -> "–"."""
    if not x:
        return "–"
    return str(x[0]) if x[0] == x[1] else f"{x[0]}–{x[1]}"


def kv(d: dict[str, Any]) -> str:
    """{"W": 4, "N": 64} -> "W=4, N=64", {} -> "none"."""
    return ", ".join(f"{k}={v}" for k, v in d.items()) or "none"


def pct(x: float) -> str:
    """A share as a percentage with one decimal, halves rounded up (as the viewer shows it)."""
    return f"{Decimal(str(100 * x)).quantize(Decimal('0.1'), ROUND_HALF_UP)}%"


def kib(n: int) -> str:
    """Bytes as "16 KiB" when a whole number of KiB, else "600 B"."""
    return f"{n // 1024} KiB" if n >= 1024 and n % 1024 == 0 else f"{n} B"


__all__ = ["kib", "kv", "pct", "span_text", "spans", "table"]
