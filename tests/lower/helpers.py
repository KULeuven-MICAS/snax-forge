"""Shared helpers for the SNAX-LOWER tests (LOW1b).

Clusters come from the checked-in cluster files, so the tests lower against
exactly what the scenarios run on. Task lists in the rule tests are written
as plain dicts, the way a person or an LLM writes them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from snax_forge.lower import TaskList, lower_program
from snax_forge.snax_model.scenario import ClusterConfig

REPO = Path(__file__).resolve().parents[2]
SCEN = REPO / "scenarios"
# Every scenario with a hand-written task list (fmul is scheduled by hand, D65).
TASK_SCENARIOS = tuple(sorted(p.parent.name for p in SCEN.glob("*/tasks.json")))


def cluster(name: str = "alu4") -> ClusterConfig:
    return ClusterConfig.load(SCEN / "clusters" / f"{name}.json")


def contiguous(base: int, n: int) -> dict[str, Any]:
    """DMA side values: n consecutive wide beats from byte ``base``."""
    return {"base": base, "bounds": [n], "strides": [64]}


def dma(task: str, direction: str = "l2_to_l1", src: int = 0, dst: int = 0, **kw: Any) -> dict:
    """A configure step for alu4's DMA: 8 beats from ``src`` to ``dst``."""
    values = {"direction": direction, "src": contiguous(src, 8), "dst": contiguous(dst, 8)}
    return {"op": "configure", "task_name": task, "type": "dma", "component": "dma",
            "values": values, **kw}  # fmt: skip


def stream(task: str, comp: str, word: int = 0, nb: int = 16, **kw: Any) -> dict:
    """A configure step for one of alu4's 4-lane streamers: nb beats from ``word``."""
    values = {"base": word * 8, "temporal_bounds": [nb], "temporal_strides": [32],
              "spatial_strides": [8]}  # fmt: skip
    return {"op": "configure", "task_name": task, "type": "streamer", "component": comp,
            "values": values, **kw}  # fmt: skip


def add(task: str, nb: int = 16, **kw: Any) -> dict:
    """A configure step for alu4's adder."""
    return {"op": "configure", "task_name": task, "type": "accel", "component": "acc",
            "values": {"n": nb}, **kw}  # fmt: skip


def start(*tasks: str) -> dict:
    return {"op": "start", "tasks": list(tasks)}


def sync(task: str, mode: str = "poll") -> dict:
    return {"op": "sync", "task": task, "mode": mode}


def task_list(*steps: dict, name: str = "t") -> TaskList:
    return TaskList.from_dict({"name": name, "steps": list(steps)})


def lowered(*steps: dict, cl: str = "alu4") -> list[dict]:
    """The program of a task list, as command dicts."""
    return [c.to_dict() for c in lower_program(task_list(*steps), cluster(cl))]


def shape(program: list[dict]) -> list[str]:
    """A program in short: writes of one block collapse to "C<block>", starts to
    "S<block>", waits to "W<block>:<mode>", reads to "R<reg>"."""
    out: list[str] = []
    for c in program:
        if c["op"] == "wait":
            out.append(f"W{c['block']}:{c['mode']}")
        elif c["op"] == "csr_read":
            out.append(f"R{c['reg']}")
        else:
            block, reg = c["reg"].split(".", 1)
            if reg == "start":
                out.append(f"S{block}")
            elif not out or out[-1] != f"C{block}":
                out.append(f"C{block}")
    return out
