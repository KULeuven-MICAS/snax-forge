"""Shared helpers for the SNAX-LOWER tests.

The components are the design points': vecadd at W = 4 on small16 (the DMA,
``acc_a``, ``acc_b``, ``acc_out`` and ``acc``) and vsum (``acc_a``,
``acc_out``, ``acc``). Task lists in the rule tests are written as plain dicts,
the way a person or an LLM writes them.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from snax_forge.design import DesignPoint, run_checks
from snax_forge.dfg import Graph
from snax_forge.lower import Configure, Start, Sync, TaskList, placed_waits
from snax_forge.sandbox import Recipe, apply_recipe
from tests.design.helpers import FIXTURES, design

REPO = Path(__file__).resolve().parents[2]
# vecadd's task list on snax_forge, as derived and checked in for tests/run (C§9)
VECADD_TASKS = REPO / "tests" / "run" / "fixtures" / "vecadd" / "tasks.json"


@cache
def vecadd_point() -> DesignPoint:
    """vecadd at W = 4 on small16."""
    d = design()
    assert not run_checks(d)
    return DesignPoint.of(d, "vecadd")


@cache
def vsum_point() -> DesignPoint:
    """vsum bound by its recipe, on small16."""
    recipe = Recipe.load(REPO / "recipes" / "vsum.json")
    g = apply_recipe(recipe, Graph.load(FIXTURES / "vsum.snaxdfg"))[-1].graph
    d = design(g)
    assert not run_checks(d)
    return DesignPoint.of(d, "vsum")


def contiguous(base: int, n: int) -> dict[str, Any]:
    """DMA side values: n consecutive wide beats from byte ``base``."""
    return {"base": base, "bounds": [n], "strides": [64]}


def dma(task: str, direction: str = "l2_to_l1", src: int = 0, dst: int = 0, **kw: Any) -> dict:
    """A configure step for vecadd's DMA: 8 beats from ``src`` to ``dst``, its L2 side A's
    array (``symbol``) in place of a base."""
    a = {"symbol": "A", "bounds": [8], "strides": [64]}
    values = {"direction": direction, "src": contiguous(src, 8), "dst": contiguous(dst, 8)}
    values["src" if direction == "l2_to_l1" else "dst"] = a
    return {"op": "configure", "task_name": task, "type": "dma", "component": "dma",
            "values": values, **kw}  # fmt: skip


def stream(task: str, comp: str, word: int = 0, nb: int = 16, **kw: Any) -> dict:
    """A configure step for one of vecadd's 4-lane streamers: nb beats from ``word``."""
    values = {"base": word * 8, "temporal_bounds": [nb], "temporal_strides": [32],
              "spatial_strides": [8]}  # fmt: skip
    return {"op": "configure", "task_name": task, "type": "streamer", "component": comp,
            "values": values, **kw}  # fmt: skip


def add(task: str, nb: int = 16, **kw: Any) -> dict:
    """A configure step for vecadd's adder."""
    return {"op": "configure", "task_name": task, "type": "accel", "component": "acc",
            "values": {"n": nb}, **kw}  # fmt: skip


def start(*tasks: str) -> dict:
    return {"op": "start", "tasks": list(tasks)}


def sync(task: str, mode: str = "poll") -> dict:
    return {"op": "sync", "task": task, "mode": mode}


def task_list(*steps: dict, name: str = "t") -> TaskList:
    return TaskList.from_dict({"name": name, "steps": list(steps)})


def shape(*steps: dict, point: DesignPoint | None = None) -> list[str]:
    """A task list with its placed waits, in short: a configure is "C<component>" (several of
    one component in a row collapse), a wait "W<component>:<mode>", a start "S<component>" per
    task, a read "R<reg>"."""
    tl = task_list(*steps)
    waits: dict[int, list[str]] = {}
    for w in placed_waits(tl, point or vecadd_point()):
        waits.setdefault(w.step, []).append(f"W{w.component}:{w.mode}")
    conf = tl.configured()
    out: list[str] = []
    for i, s in enumerate(tl.steps):
        if isinstance(s, Configure):
            if not out or out[-1] != f"C{s.component}":
                out.append(f"C{s.component}")
        elif isinstance(s, Start):
            out += waits.get(i, [])
            out += [f"S{conf[t].component}" for t in s.tasks]
        elif isinstance(s, Sync):
            out += waits[i]
        else:
            out.append(f"R{s.reg}")
    return out
