"""run/profile.py: the profile round-trips through its dict (C§7)."""

import json

from snax_forge.run import (
    AccelProfile,
    BankProfile,
    CoreProfile,
    DmaProfile,
    FifoProfile,
    PortProfile,
    Profile,
    StreamerProfile,
    WaitProfile,
)


def full() -> Profile:
    return Profile(
        total_cycles=226,
        accelerators={"acc": AccelProfile(
            {"busy": 16, "stall_in": 3, "stall_out": 0, "flush": 0, "idle": 207}, 16, 16 / 226)},
        streamers={"acc_a": StreamerProfile(
            False, {"busy": 16, "stall_xbar": 0, "stall_fifo": 0, "idle": 210},
            [f"acc_a.lane{i}" for i in range(4)], FifoProfile("acc_a.fifo", 8, [2, 2, 2, 2]))},
        cores={
            "core0": CoreProfile({"run": 150, "poll": 19, "barrier": 57, "sleep": 0}, 18, 13,
                                 [WaitProfile("acc", 134, 152), WaitProfile("barrier", 113, 131)]),
            "core1": CoreProfile(None, 0, 0),
        },
        dmas={"dma": DmaProfile(3, 54, 24, 24, 1024, 512)},
        ports={"acc_a.lane0": PortProfile(16, 0, 0), "core0": PortProfile(322, 0, 0, True)},
        banks=BankProfile([6] * 32, [6] * 32, [0] * 32),
        functional_check={"passed": True, "tasks": {"add_acc": {"counter": 19, "beats": 16}}},
    )  # fmt: skip


def test_round_trip():
    p = full()
    assert Profile.from_dict(json.loads(json.dumps(p.to_dict()))) == p


def test_the_least_profile_round_trips():
    p = Profile(total_cycles=7)
    d = json.loads(json.dumps(p.to_dict()))
    assert Profile.from_dict(d) == p
    assert d["banks"] is None and d["cores"] == {}
