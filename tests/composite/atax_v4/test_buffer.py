"""The fork's buffered branch: A read once, given to mul.a and through a ring of
D = 4 beats to mac.a. D = T + 2 runs at full rate, D = T stalls, D below T never ends; the ring
is three registers of the control unit, in its bits as in the RTL's."""

import json

from tests.composite.atax_v4.test_hold import TS, rule
from tests.composite.helpers import RTL_FIXTURES
from tests.helpers import requires_submodule

pytestmark = requires_submodule


def test_a_is_forked_and_mac_a_takes_it_through_a_fork_link(walked):
    (acc,) = walked.accelerator["accelerators"]
    assert acc["inputs"][0] == ["mul.a", "mac.a"]
    (g,) = walked.groups
    assert g["fork_links"] == [{"name": "A", "to": "mac.a", "depth": 4}]


def test_the_ring_depth_against_the_row(walked):
    def end(depth):
        return rule(walked, depth=depth).run(1, TS, lambda c: (1, 1), lambda c: 1, 400)[1]

    assert end(4) is not None and end(5) == end(4)  # T + 2: full rate
    assert end(2) > end(4)  # T: two cycles a row of stall
    assert end(1) is None  # below T the fork waits for room the bank's first t never frees


def test_the_control_units_bits_hold_the_ring(walked):
    regs = json.loads((RTL_FIXTURES / "ctrl_registers.json").read_text())["atax_v4"]
    (mine,) = regs.values()
    ring = {n: b for n, b in mine if "_fork_" in n}
    assert ring == {"mul_mac_fork_occupancy": 3, "mul_mac_fork_wslot": 2, "mul_mac_fork_rslot": 2}
    (g,) = walked.estimate.groups
    assert g.ctrl_bits == sum(b for _, b in mine)
