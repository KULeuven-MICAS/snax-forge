"""The numbers the tests pin: what a design point's estimate and a recorded run's profile give,
checked against a snapshot so a changed coefficient, count or cycle fails a test even where line
coverage does not see it.

    estimate_numbers(est)        the estimate's cycles, bits and cost terms (``ESTIMATE``), by path
    profile_numbers(profile)     a profile's cycles and firings per accelerator and streamer
    totals(est)                  the estimate's totals and cost parts, one recipe's line in TOTALS
    check(got, path)             ``got`` equal to the snapshot at ``path`` (floats to 1e-9)

``pixi run python -m tests.numbers`` writes every snapshot again (the composites' under
tests/composite/numbers/, the recipes' totals in tests/estimate/totals.json); run it, in the
pixi environment with the submodule checked out, only for a change meant to move the numbers.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import REPO

COMPOSITE = REPO / "tests" / "composite" / "numbers"
TOTALS = REPO / "tests" / "estimate" / "totals.json"
WRITE = "pixi run python -m tests.numbers"

ESTIMATE = re.compile(
    r"(total_cycles|accelerator_cycles|copy_cycles|l1_footprint|cost\..+"
    r"|groups\[\d+\]\.(cycles|ctrl_bits|bits\..+|cost\.(area|energy)"
    r"|cost\.(units|links)\.[^.]+\.(area\.total|energy\.(fire|task))"
    r"|cost\.ctrl\.(area\.total|energy)))"
)
PROFILE = re.compile(r"total_cycles|(accelerators|streamers)\.[^.]+\.(cycles\.[a-z_]+|firings)")
TOTAL = re.compile(r"total_cycles|l1_footprint|cost\.(area|area_design|energy|kernel_energy)"
                   r"|cost\.parts\.[a-z]+\.(area|energy|leak)|cost\.leakage\.(power|energy)")  # fmt: skip


def flat(x: Any, path: str = "") -> Iterator[tuple[str, int | float]]:
    """Every number in ``x`` (dicts and lists, booleans left out) by its dotted path."""
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        yield path, x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{path}[{i}]")


def _pick(d: dict[str, Any], keep: re.Pattern[str]) -> dict[str, int | float]:
    return {k: v for k, v in flat(d) if keep.fullmatch(k)}


def estimate_numbers(est: Any) -> dict[str, int | float]:
    return _pick(est.to_dict(), ESTIMATE)


def profile_numbers(profile: Any) -> dict[str, int | float]:
    return _pick(profile.to_dict(), PROFILE)


def totals(est: Any) -> dict[str, int | float]:
    return _pick(est.to_dict(), TOTAL)


def check(got: dict[str, int | float], want: dict[str, int | float], what: str) -> None:
    """``got`` has ``want``'s paths and numbers, floats to 1e-9 relative; the message names the
    paths that differ and the command that writes the snapshots again."""
    missing, extra = sorted(want.keys() - got.keys()), sorted(got.keys() - want.keys())
    wrong = [
        k for k in sorted(want.keys() & got.keys()) if got[k] != pytest.approx(want[k], rel=1e-9)
    ]
    if missing or extra or wrong:
        lines = [f"{k}: {got[k]} (snapshot {want[k]})" for k in wrong[:20]]
        lines += [f"{k}: gone" for k in missing[:10]] + [f"{k}: new" for k in extra[:10]]
        pytest.fail(f"{what} differs from its snapshot; if the change is meant, `{WRITE}` writes "
                    "it again:\n  " + "\n  ".join(lines))  # fmt: skip


def composite_numbers(walked: Any) -> dict[str, dict[str, int | float]]:
    """A composite walk's numbers: its estimate's and, per recorded level, its profile's."""
    from snax_forge.run.convert import convert
    from snax_forge.run.sources import all_sources

    out = {"estimate": estimate_numbers(walked.estimate)}
    for level, rec in walked.recorded.items():
        cluster, tasks, chk, lines, regions, _ = rec
        profile, _ = convert(lines, cluster, tasks, chk["total_cycles"], level,
                             chk["functional_check"], regions, all_sources(cluster))  # fmt: skip
        out[f"profile_{level}"] = profile_numbers(profile)
    return out


def check_composite(walked: Any) -> None:
    want = json.loads((COMPOSITE / f"{walked.design}.json").read_text())
    got = composite_numbers(walked)
    assert sorted(got) == sorted(want), (
        f"{walked.design}: parts {sorted(got)}, snapshot {sorted(want)}"
    )
    for part in want:
        check(got[part], want[part], f"{walked.design}'s {part}")


def recipe_totals(name: str) -> dict[str, int | float]:
    """The recipe ``name``'s last graph estimated on the default platform (its totals)."""
    from snax_forge.design.platform import Platform
    from snax_forge.estimate import estimate
    from tests.sandbox.helpers import recipe_graphs

    platform = Platform.load(REPO / "platforms" / "snax_forge.json")
    return totals(estimate(recipe_graphs(name)[1][-1], platform))


def _write_json(path: Path, data: dict[str, Any], one_line: bool = False) -> None:
    if one_line:  # one entry a line: a changed recipe is one changed line
        body = ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in data.items())
        path.write_text("{\n" + body + "\n}\n")
    else:
        path.write_text(json.dumps(data, indent=1) + "\n")


def main() -> None:
    import tempfile

    from tests.composite.helpers import walk
    from tests.helpers import FORGE_FIXTURES
    from tests.sandbox.helpers import RECIPES

    COMPOSITE.mkdir(exist_ok=True)
    for design in FORGE_FIXTURES:
        with tempfile.TemporaryDirectory() as tmp:
            _write_json(COMPOSITE / f"{design}.json", composite_numbers(walk(Path(tmp), design)))
        print(f"wrote {COMPOSITE.relative_to(REPO)}/{design}.json", file=sys.stderr)
    _write_json(TOTALS, {n: recipe_totals(n) for n in RECIPES}, one_line=True)
    print(f"wrote {TOTALS.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
