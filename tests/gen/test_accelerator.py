"""accelerator.json: the description nests accelerators ⊃ groups ⊃ units and
round-trips, its CSRs are in CSR order from offset 0 over the accelerators, and one helper names
them for the header and the program; the tag is the entry's name."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from snax_forge.gen import Accelerator, Csr, Description, Group, Link, Port, Unit, describe
from snax_forge.gen.blocks import GenError
from tests.helpers import bound_vsum

from .helpers import (
    bound_recipe,
    one_unit,
    point,
    validate,
    with_groups,
)


def own(base: int, rw_units: int) -> list[Csr]:
    """A group's own CSRs from offset ``base``, ``rw_units`` R/W CSRs of its units between."""
    return [Csr("START", "rw", base, 1), Csr("JOBS", "rw", base + 1, 32),
            Csr("STATUS", "ro", base + 2 + rw_units, 2), Csr("BUSY_COUNTER", "ro", base + 3 + rw_units, 32)]  # fmt: skip


def mul_sum(v: str = "") -> tuple[Unit, Unit]:
    """dot's units: ``<v>mul`` (an SimdLanes unit) and ``<v>sum`` (a TreeAccumulate unit
    with JOB_BEATS at offset 6); dot 1's are vmul and vsum, dot_v7's mul and sum."""
    mul = Unit(f"{v}mul", "simd", "SimdLanes",
               {p: Port(p, "out" if p == "out" else "in", "int64", 4) for p in ("a", "b", "out")},
               {"op": "mul"}, 0, 1, [],
               "int64")  # fmt: skip
    total = Unit(f"{v}sum", "accumulate", "TreeAccumulate",
                 {"a": Port("in", "in", "int64", 4), "out": Port("out", "out", "int64", 1)}, {"op": "add"}, 1, 1,
                 [Csr("JOB_BEATS", "rw", 6, 32)])  # fmt: skip
    return mul, total


def dot() -> Description:
    """dot_v1_1 as the table has it: one accelerator dot (encapsulated), groups vmul
    and vsum, in that order."""
    mul, total = mul_sum("v")
    acc = Accelerator("dot", ["vmul.a", "vmul.b", "vsum.a"], ["vmul.out", "vsum.out"],
                      [Group("vmul", own(0, 0), [mul]), Group("vsum", own(4, 1), [total])])  # fmt: skip
    return Description("out/design/dot_v1_1/design_point.json", "snax_forge_dot_v1_1", [acc])


def dot1() -> Description:
    """dot 1 as the and the tables have it: two accelerators, vmul and vsum, each its
    streamer set, their CSRs contiguous over the two from offset 0."""
    mul, total = mul_sum("v")
    return Description("out/design/dot_v1/design_point.json", "snax_forge_dot_v1", [
        Accelerator("vmul", ["vmul.a", "vmul.b"], ["vmul.out"], [Group("vmul", own(0, 0), [mul])]),
        Accelerator("vsum", ["vsum.a"], ["vsum.out"], [Group("vsum", own(4, 1), [total])]),
    ])  # fmt: skip


def dot_v7() -> Description:
    """dot_v7: the same two groups as two accelerators, mul and sum, two streamer sets,
    their CSRs contiguous over the two from offset 0."""
    mul, total = mul_sum()
    return Description("out/design/dot_v7/design_point.json", "snax_forge_dot_v7", [
        Accelerator("mul", ["mul.a", "mul.b"], ["mul.out"], [Group("mul", own(0, 0), [mul])]),
        Accelerator("sum", ["sum.a"], ["sum.out"], [Group("sum", own(4, 1), [total])]),
    ])  # fmt: skip


def dot4() -> Description:
    """dot 4 as the block has it: one group dot, a Mac unit (JOB_BEATS) wired into a
    TreeAccumulate unit of fixed T 1 by the link out_partial."""
    mac = Unit("mac", "mac", "Mac", {"a": Port("a", "in", "int64", 4), "b": Port("b", "in", "int64", 4),
               "out": Port("out", "out", "int64", 4)}, {"accumulate": True, "regOut": False}, 1, 1,
               [Csr("JOB_BEATS", "rw", 2, 32)], "int64")  # fmt: skip
    total = Unit("sum", "accumulate", "TreeAccumulate",
                 {"a": Port("in", "in", "int64", 4), "out": Port("out", "out", "int64", 1)},
                 {"op": "add", "T": 1}, 1, 1)  # fmt: skip
    own = [Csr("START", "rw", 0, 1), Csr("JOBS", "rw", 1, 32), Csr("STATUS", "ro", 3, 2),
           Csr("BUSY_COUNTER", "ro", 4, 32)]  # fmt: skip
    acc = Accelerator("dot", ["mac.a", "mac.b"], ["sum.out"],
                      [Group("dot", own, [mac, total], [Link("out_partial", "mac.out", "sum.a", 0)])])  # fmt: skip
    return Description("out/design/dot_v4/design_point.json", "snax_forge_dot_v4", [acc])


def test_the_description_round_trips_and_meets_its_schema():
    for acc in (one_unit(), one_unit(3), dot(), dot1(), dot4(), dot_v7()):
        assert Description.from_dict(json.loads(acc.to_json())) == acc
        validate("accelerator", acc.to_dict())
    assert list(dot().to_dict()) == ["design_point", "tag", "accelerators"]
    assert list(dot().to_dict()["accelerators"][0]) == ["name", "inputs", "outputs", "groups"]


def test_one_helper_names_every_csr():
    """<ACC>_<RW|RO>_[<GROUP>_][<UNIT>_]<NAME>: the group part dropped with one group, the
    unit part with one unit in the group."""
    assert one_unit(1).defines() == {"ACC_RW_START": 0, "ACC_RW_JOBS": 1, "ACC_RW_F0": 2,
                                     "ACC_RO_STATUS": 3, "ACC_RO_BUSY_COUNTER": 4}  # fmt: skip
    acc = dot()
    assert acc.find("vmul", "START") == "DOT_RW_VMUL_START"
    assert acc.find("vsum", "JOB_BEATS") == "DOT_RW_VSUM_JOB_BEATS"
    assert acc.find("vsum", "BUSY_COUNTER") == "DOT_RO_VSUM_BUSY_COUNTER"
    with pytest.raises(KeyError):
        acc.find("vmul", "JOB_BEATS")  # the fold's CSR is the sum group's
    assert dot1().find("vsum", "JOB_BEATS") == "VSUM_RW_JOB_BEATS"  # the dot 1
    # a one-group accelerator has no group part: each its own name
    assert dot_v7().defines() == {
        "MUL_RW_START": 0, "MUL_RW_JOBS": 1, "MUL_RO_STATUS": 2, "MUL_RO_BUSY_COUNTER": 3,
        "SUM_RW_START": 4, "SUM_RW_JOBS": 5, "SUM_RW_JOB_BEATS": 6, "SUM_RO_STATUS": 7,
        "SUM_RO_BUSY_COUNTER": 8,
    }  # fmt: skip
    assert dot_v7().find("sum", "START") == "SUM_RW_START"


@pytest.mark.parametrize(
    ("start", "text"),
    [({"offset": 1}, "not contiguous"), ({"width": None}, "has a width"),
     ({"width": 33}, "1 to 32 bits"), ({"width": 32}, "START has width 32, not 1")],
    ids=["offsets", "width", "wide", "start"],
)  # fmt: skip
def test_a_reader_checks_the_offsets_and_widths(start, text):
    """Offsets are stored, never recomputed: a reader checks them contiguous from 0 in CSR
    order; every CSR has a width of 1 to 32 bits, START's 1 (START edited by ``start``)."""
    acc = one_unit()
    g = acc.groups[0]
    edited = replace(g, csrs=[replace(g.csrs[0], **start), *g.csrs[1:]])
    with pytest.raises(ValueError, match=text):
        with_groups(acc, [edited])


def test_offsets_run_on_over_the_accelerators():
    """The second accelerator's offsets continue the first's: from 0 again is refused,
    as are two accelerators or two groups of one name."""
    serial = dot_v7()
    _, total = mul_sum()
    again = replace(total, csrs=[Csr("JOB_BEATS", "rw", 2, 32)])
    with pytest.raises(ValueError, match="not contiguous"):
        with_groups(serial, [Group("sum", own(0, 1), [again])], 1)
    first = serial.accelerators[0]
    with pytest.raises(ValueError, match=r"accelerator names \['mul'\] are used twice"):
        replace(serial, accelerators=[first, replace(serial.accelerators[1], name="mul")])
    with pytest.raises(ValueError, match=r"group names \['mul'\] are used twice"):
        with_groups(serial, [replace(serial.accelerators[1].groups[0], name="mul")], 1)


def test_the_widths_come_from_the_platform(tmp_path):
    """START 1, STATUS 2, JOBS and JOB_BEATS acc_csr.job_width, BUSY_COUNTER its
    busy_counter_width
    (job_width when null); JOBS and every JOB_BEATS share one width, every BUSY_COUNTER one."""
    from snax_forge.design import DesignPoint, Platform

    from .helpers import SNAX_FORGE

    def widths(*sets: tuple[str, int | None]) -> dict[str, int]:
        plat = Platform.load(SNAX_FORGE).with_changes(list(sets))
        path = point(tmp_path, name="vsum_v1", graph=bound_vsum(), platform=plat)
        acc = describe(DesignPoint.load(path), "p")
        assert acc.job_width == plat.acc_csr.job_width
        return {c.name: c.width for _, _, c in acc.csrs()}

    assert widths() == {"START": 1, "JOBS": 32, "JOB_BEATS": 32, "STATUS": 2, "BUSY_COUNTER": 32}
    assert widths(("acc_csr.job_width", 16)) == {
        "START": 1, "JOBS": 16, "JOB_BEATS": 16, "STATUS": 2, "BUSY_COUNTER": 16}  # fmt: skip
    assert widths(("acc_csr.job_width", 16), ("acc_csr.busy_counter_width", 8))["BUSY_COUNTER"] == 8
    acc = one_unit(1)
    g = acc.groups[0]
    with pytest.raises(ValueError, match="STATUS has width 32, not 2"):
        with_groups(acc, [replace(g, csrs=[*g.csrs[:2], replace(g.csrs[2], width=32), g.csrs[3]])])
    unit = replace(g.units[0], csrs=[Csr("JOB_BEATS", "rw", 2, 16)])
    with pytest.raises(ValueError, match=r"JOBS and JOB_BEATS have the widths \[16, 32\]"):
        with_groups(acc, [replace(g, units=[unit])])


def _edit(links=None, inputs=None, outputs=None, ports=None) -> None:
    """dot4 with its links, outer ports or the sum unit's ports replaced."""
    desc = dot4()
    (acc,) = desc.accelerators
    g = acc.groups[0]
    units = g.units if ports is None else [g.units[0], replace(g.units[1], ports=ports)]
    edited = replace(acc, inputs=acc.inputs if inputs is None else inputs,
                     outputs=acc.outputs if outputs is None else outputs,
                     groups=[replace(g, units=units, links=g.links if links is None else links)])  # fmt: skip
    replace(desc, accelerators=[edited])


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        ({"links": [Link("x", "sum.a", "mac.a", 0)]}, "gen.link"),  # in to in
        ({"links": [Link("x", "mac.out", "sum.out", 0)]}, "gen.link"),  # out to out
        ({"links": [Link("x", "mac.out", "nope.a", 0)]}, "gen.link"),  # no such unit
        ({"links": [Link("x", "mac.out", "sum.a", -1)]}, "gen.link"),  # depth
        ({"ports": {"a": Port("in", "in", "int32", 4), "out": Port("out", "out", "int64", 1)}},
         "gen.link"),  # dtypes differ
        ({"ports": {"a": Port("in", "in", "int64", 2), "out": Port("out", "out", "int64", 1)}},
         "gen.link"),  # lanes differ
        ({"ports": {"a": Port("in", "in", "int64", 4, d_in=2),
                    "out": Port("out", "out", "int64", 1)}}, "gen.link"),  # a linked port sliced
        ({"ports": {"a": Port("in", "in", "int64", 4, broadcast=True),
                    "out": Port("out", "out", "int64", 1)}}, "gen.link"),  # a linked broadcast
        ({"links": [Link("x", "mac.out", "sum.a", 0), Link("y", "mac.out", "sum.a", 1)]},
         "gen.link_port"),
        ({"inputs": ["mac.a", "mac.b", "sum.a"]}, "gen.outer"),
        ({"outputs": ["mac.out", "sum.out"]}, "gen.outer"),
        ({"inputs": ["mac.b", "mac.a"]}, "gen.outer"),  # order
        ({"links": [], "inputs": ["mac.a", "mac.b", "sum.a"], "outputs": ["mac.out", "sum.out"]},
         "gen.group"),
    ],
    ids=["in-in", "out-out", "unit", "depth", "dtype", "lanes", "sliced", "broadcast", "port", "outer-in",
         "outer-out", "outer-order", "group"],
)  # fmt: skip
def test_a_reader_checks_the_links(edit, code):
    """the refusals of links and groups, on construction as on from_dict."""
    with pytest.raises(GenError) as e:
        _edit(**edit)
    assert e.value.code == code


def test_a_group_is_a_chain_or_chains_a_fork_joins():
    """gen.chain: dot 4's units in the other order, the link going back (SNAX-GVSoC's SnaxGroup
    models a chain in unit order, or chains a fork joins)."""
    acc = dot4()
    (g,) = acc.groups
    with pytest.raises(GenError) as e:
        with_groups(acc, [replace(g, units=g.units[::-1])])
    assert e.value.code == "gen.chain" and "chains a fork joins" in str(e.value)


def test_dot_serial_is_two_accelerators(tmp_path):
    """dot_v7 (no encapsulate): two accelerators mul and sum in the graph's order, each one
    group and one streamer set, its own outer ports; the offsets run on, 0 to 8."""
    from snax_forge.design import DesignPoint

    graph = bound_recipe("dot_v7", "dot")
    desc = describe(DesignPoint.load(point(tmp_path, name="dot_v7", graph=graph)), "p")
    assert desc.tag == "snax_forge_dot_v7"
    assert [(a.name, a.inputs, a.outputs, [g.name for g in a.groups]) for a in desc.accelerators] == [
        ("mul", ["mul.a", "mul.b"], ["mul.out"], ["mul"]),
        ("sum", ["sum.a"], ["sum.out"], ["sum"]),
    ]  # fmt: skip
    assert desc.defines() == dot_v7().defines()
    assert desc.inputs == ["mul.a", "mul.b", "sum.a"]  # the wrapper's order, set after set
    validate("accelerator", desc.to_dict())


@pytest.mark.parametrize("name", ["dot_1_W_acc_1-10", "dot 1", "dot.1"])
def test_a_tag_that_is_no_identifier_is_refused(tmp_path, name):
    """gen.tag: the entry's name <accelerator_slot>_<design> names modules, files and the
    cluster's entry, so a design point whose name makes it no identifier is refused, naming the
    fix (a flow's ``--set W_acc_1=10`` names its design point ``…_W_acc_1-10``)."""
    from snax_forge.design import DesignPoint

    p = replace(DesignPoint.load(point(tmp_path, name="vecadd")), name=name)
    with pytest.raises(GenError) as e:
        describe(p, "p")
    assert e.value.code == "gen.tag"
    assert f"'snax_forge_{name}'" in str(e.value) and "letters, digits and _ only" in str(e.value)


def test_a_platform_without_a_slot_has_no_tag(tmp_path):
    """gen.tag: the entry is named after the platform's accelerator_slot."""
    from snax_forge.design import DesignPoint

    p = DesignPoint.load(point(tmp_path, name="vecadd"))
    p = replace(p, platform=replace(p.platform, accelerator_slot=None))
    with pytest.raises(GenError) as e:
        describe(p, "p")
    assert e.value.code == "gen.tag" and "no accelerator_slot" in str(e.value)
