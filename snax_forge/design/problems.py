"""What the design checks report (DP1a, D85): one ``Problem`` per thing that does not fit.

A problem has a ``code`` (``<stage>.<check>``, CONTRACTS.md section 14),
``where`` (the platform field, graph node or streamer it is about), a
``message`` saying why it fails and, where there is one, a ``fix``: the
``--set`` or recipe change that removes it. ``DesignError`` carries every
problem found, so one run shows all of them rather than the first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    """One check that failed; see the module doc."""

    code: str
    where: str
    message: str
    fix: str | None = None

    def __str__(self) -> str:
        text = f"[{self.code}] {self.where}: {self.message}"
        return text + (f"\n    fix: {self.fix}" if self.fix else "")


def report(problems: Sequence[Problem], head: str = "design check") -> str:
    """``head: n problems`` and one indented entry per problem."""
    n = len(problems)
    lines = [f"{head}: {n} problem{'' if n == 1 else 's'}"]
    lines += [" " + str(p).replace("\n", "\n ") for p in problems]
    return "\n".join(lines)


class DesignError(ValueError):
    """A platform, graph or pairing of both that does not fit; holds every problem."""

    def __init__(self, problems: Sequence[Problem], head: str = "design check") -> None:
        self.problems = list(problems)
        super().__init__(report(self.problems, head))

    @property
    def codes(self) -> list[str]:
        return [p.code for p in self.problems]
