"""The dot walk-through: every number and name examples/dot/README.md states, checked.

Its table (section 2) is checked twice. On the host: each COUNTER is C§17's cycles at the
README's point (N = 512, W = 4: J 1, T 128) plus the fill it states, and the C§17 column is
the formula's. On SNAX-GVSoC (marker ``gvsoc``, D134: in the image): dot 1–6 run at N = 64
(T 16), each checks itself (``forge data: 0 errors``), each group's COUNTER is C§17's cycles
plus the same fill, the second reader's one crossbar stall is the fill's, and each row's
groups, units and link are its run's ``accelerator.json``.
"""

from __future__ import annotations

import json
import re

import pytest

from snax_forge.flow import run_flow, summary
from snax_forge.run import read_run
from tests.design.helpers import REPO, SNAX_FORGE

from .helpers import needs_gvsoc

README = REPO / "examples" / "dot" / "README.md"
NAMES = [f"dot_{k}" for k in range(1, 7)]
SMALL = 64  # the runs' N
W = 4
# The fill: 2 cycles of the streamers' start, one more for a unit reading A and B
FILL = {"dot_1": (3, 2), **{n: (3,) for n in NAMES[1:]}}
ROW = re.compile(
    r"^\| `(dot_\d)` \| (.+?) \| (.+?) \| (.+?) \| ([\d, ]+) \| ([\d, ]+) \| ([\d, ]+) \|$"
)


def c17(name: str, n: int, w: int = W, j: int = 1) -> tuple[int, ...]:
    """C§17's cycles of each group at N = n: dot 1's two by B1's rule (mul JT, sum JT + 1), the
    linked groups' by their formula."""
    t = n // w
    return {
        "dot_1": (j * t, j * t + 1),
        "dot_2": (j * t + 2,),
        "dot_3": (j * t + 1,),
        "dot_4": (j * t + 2,),
        "dot_5": (j * (t + w - 1) + 2,),
        "dot_6": ((j - 1) * max(t, w) + t + w + 2,),
    }[name]


def table() -> dict[str, tuple[str, ...]]:
    rows = (ROW.match(line) for line in README.read_text().splitlines())
    return {m.group(1): m.groups()[1:] for m in rows if m}


def numbers(cell: str) -> tuple[int, ...]:
    return tuple(int(x) for x in cell.split(", "))


def test_the_table_is_c17_plus_the_fill():
    t = table()
    assert list(t) == NAMES
    for n, (_, _, _, counter, formula, fill) in t.items():
        assert numbers(formula) == c17(n, 512) and numbers(fill) == FILL[n], n
        assert numbers(counter) == tuple(a + b for a, b in zip(c17(n, 512), FILL[n], strict=True))


def test_the_readme_gives_the_command_and_dot_4s_check_line():
    text = README.read_text()
    assert "\npixi run flow recipes/dot_4.json --platform platforms/snax_forge.json\n" in text
    assert f"counters mult_dot {table()['dot_4'][3]} (128 beats)" in text


@pytest.fixture(scope="module")
def flows(tmp_path_factory):
    base = tmp_path_factory.mktemp("dot")
    recipes = REPO / "recipes"
    return {n: run_flow(recipes / f"{n}.json", SNAX_FORGE, recipe_sets={"N": SMALL}, out=base / n)
            for n in NAMES}  # fmt: skip


@pytest.mark.gvsoc
@needs_gvsoc
@pytest.mark.parametrize("name", NAMES)
def test_each_runs_and_checks_itself(flows, name):
    f = flows[name]
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0
    assert "forge data: 0 errors (1 words)\n" in (f.out / "run" / "stdout.txt").read_text()


@pytest.mark.gvsoc
@needs_gvsoc
@pytest.mark.parametrize("name", NAMES)
def test_each_counter_is_c17_plus_the_fill(flows, name):
    tasks = flows[name].check["tasks"]
    counters = tuple(x["counter"] for x in tasks.values())
    want = tuple(a + b for a, b in zip(c17(name, SMALL), FILL[name], strict=True))
    assert counters == want, tasks


@pytest.mark.gvsoc
@needs_gvsoc
@pytest.mark.parametrize("name", NAMES)
def test_the_second_readers_one_stall_is_the_fills(flows, name):
    streamers = read_run(flows[name].out / "run").profile.streamers
    held = {k: s.cycles["stall_xbar"] for k, s in streamers.items() if not s.write}
    second = "mul_b" if name in ("dot_1", "dot_2", "dot_3") else "mac_b"
    assert held[second] == 1 and sum(held.values()) == 1, held


@pytest.mark.gvsoc
@needs_gvsoc
@pytest.mark.parametrize("name", NAMES)
def test_each_row_is_its_runs_description(flows, name):
    acc = json.loads((flows[name].out / "gen" / "accelerator.json").read_text())
    groups, units, link, *_ = table()[name]
    if name == "dot_1":
        assert [g["name"] for g in acc["groups"]] == ["mul", "sum"]
        assert groups == "`mul`, `sum`, a task each" and link == "none: `tmp0` in L1"
        assert units == "; ".join(f"`{u['name']}` {u['datapath']}" for g in acc["groups"]
                                  for u in g["units"])  # fmt: skip
        return
    (g,) = acc["groups"]
    ((k,),) = [g["links"]]
    assert groups == f"`{g['name']}`" == "`dot`"
    assert units == " → ".join(f"`{u['name']}` {u['datapath']}" for u in g["units"])
    kind = f"a buffer of depth {k['depth']}" if k["depth"] else "a wire (depth 0)"
    assert link == f"`{k['name']}`, {kind}"
