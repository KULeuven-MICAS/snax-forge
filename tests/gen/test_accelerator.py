"""accelerator.json (D145, D157, C§17): the description nests accelerators ⊃ groups ⊃ units and
round-trips, its CSRs are in CSR order from offset 0 over the accelerators, and one helper names
them for the header and the program; the tag is the entry's name."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from snax_forge.gen import Accelerator, Csr, Description, Group, Link, Port, Unit, describe
from snax_forge.gen.blocks import GenError
from tests.helpers import bound_vsum

from .helpers import bound_dot1, bound_recipe, one_unit, point, validate, with_groups


def own(base: int, rw_units: int) -> list[Csr]:
    """A group's own CSRs from offset ``base``, ``rw_units`` R/W CSRs of its units between."""
    return [Csr("START", "rw", base, 1), Csr("JOBS", "rw", base + 1, 32),
            Csr("STATUS", "ro", base + 2 + rw_units, 2), Csr("COUNTER", "ro", base + 3 + rw_units, 32)]  # fmt: skip


def mul_sum() -> tuple[Unit, Unit]:
    """dot 1's units: mul (an ElementwiseLanes unit) and sum (a TreeAccumulate unit with
    JOB_BEATS at offset 6)."""
    mul = Unit("mul", "elementwise_mul", "ElementwiseLanes",
               {p: Port(p, "out" if p == "out" else "in", "int64", 4) for p in ("a", "b", "out")},
               {"op": "mul"}, 0, 1, [],
               "int64")  # fmt: skip
    total = Unit("sum", "accumulate", "TreeAccumulate",
                 {"a": Port("in", "in", "int64", 4), "out": Port("out", "out", "int64", 1)}, {"op": "add"}, 1, 1,
                 [Csr("JOB_BEATS", "rw", 6, 32)])  # fmt: skip
    return mul, total


def dot() -> Description:
    """dot 1 as C§17's table has it: one accelerator dot (encapsulated, D148), groups mul and
    sum, in that order."""
    mul, total = mul_sum()
    acc = Accelerator("dot", ["mul.a", "mul.b", "sum.a"], ["mul.out", "sum.out"],
                      [Group("mul", own(0, 0), [mul]), Group("sum", own(4, 1), [total])])  # fmt: skip
    return Description("out/design/dot_1/design_point.json", "snax_forge_dot_1", [acc])


def dot_serial() -> Description:
    """dot_serial (D157): the same two groups as two accelerators, mul and sum, two streamer sets,
    their CSRs contiguous over the two from offset 0."""
    mul, total = mul_sum()
    return Description("out/design/dot_serial/design_point.json", "snax_forge_dot_serial", [
        Accelerator("mul", ["mul.a", "mul.b"], ["mul.out"], [Group("mul", own(0, 0), [mul])]),
        Accelerator("sum", ["sum.a"], ["sum.out"], [Group("sum", own(4, 1), [total])]),
    ])  # fmt: skip


def dot4() -> Description:
    """dot 4 as C§17's block has it: one group dot, a Mac unit (JOB_BEATS) wired into a
    TreeAccumulate unit of fixed T 1 by the link out_partial."""
    mac = Unit("mac", "mac", "Mac", {"a": Port("a", "in", "int64", 4), "b": Port("b", "in", "int64", 4),
               "out": Port("out", "out", "int64", 4)}, {"accumulate": True, "regOut": False}, 1, 1,
               [Csr("JOB_BEATS", "rw", 2, 32)], "int64")  # fmt: skip
    total = Unit("sum", "accumulate", "TreeAccumulate",
                 {"a": Port("in", "in", "int64", 4), "out": Port("out", "out", "int64", 1)},
                 {"op": "add", "T": 1}, 1, 1)  # fmt: skip
    own = [Csr("START", "rw", 0, 1), Csr("JOBS", "rw", 1, 32), Csr("STATUS", "ro", 3, 2),
           Csr("COUNTER", "ro", 4, 32)]  # fmt: skip
    acc = Accelerator("dot", ["mac.a", "mac.b"], ["sum.out"],
                      [Group("dot", own, [mac, total], [Link("out_partial", "mac.out", "sum.a", 0)])])  # fmt: skip
    return Description("out/design/dot_4/design_point.json", "snax_forge_dot_4", [acc])


def test_the_description_round_trips_and_meets_its_schema():
    for acc in (one_unit(), one_unit(3), dot(), dot4(), dot_serial()):
        assert Description.from_dict(json.loads(acc.to_json())) == acc
        validate("accelerator", acc.to_dict())
    assert list(dot().to_dict()) == ["design_point", "tag", "accelerators"]
    assert list(dot().to_dict()["accelerators"][0]) == ["name", "inputs", "outputs", "groups"]


def test_csr_order_and_counts():
    """Per group START, JOBS, the units' R/W, STATUS, COUNTER; rw = Σ_g (2 + Σ_u (f_u + k_u)),
    ro = 2G (C§17's table: dot 1 is 5, 4)."""
    acc = dot()
    order = [
        ("mul", "START", 0), ("mul", "JOBS", 1), ("mul", "STATUS", 2), ("mul", "COUNTER", 3),
        ("sum", "START", 4), ("sum", "JOBS", 5), ("sum", "JOB_BEATS", 6), ("sum", "STATUS", 7),
        ("sum", "COUNTER", 8),
    ]  # fmt: skip
    assert [(g.name, c.name, c.offset) for g, _, c in acc.csrs()] == order
    (a,) = acc.accelerators
    assert (a.rw, a.ro) == (5, 4)
    # two accelerators: the same offsets, contiguous over both; each its own counts
    serial = dot_serial()
    assert [(g.name, c.name, c.offset) for g, _, c in serial.csrs()] == order
    assert [(a.rw, a.ro) for a in serial.accelerators] == [(2, 2), (3, 2)]
    (u,) = one_unit().accelerators
    (u1,) = one_unit(1).accelerators
    assert ((u.rw, u.ro), (u1.rw, u1.ro)) == ((2, 2), (3, 2))


def test_one_helper_names_every_csr():
    """<ACC>_<RW|RO>_[<GROUP>_][<UNIT>_]<NAME>: the group part dropped with one group, the
    unit part with one unit in the group."""
    assert one_unit(1).defines() == {"ACC_RW_START": 0, "ACC_RW_JOBS": 1, "ACC_RW_F0": 2,
                                     "ACC_RO_STATUS": 3, "ACC_RO_COUNTER": 4}  # fmt: skip
    acc = dot()
    assert acc.find("mul", "START") == "DOT_RW_MUL_START"
    assert acc.find("sum", "JOB_BEATS") == "DOT_RW_SUM_JOB_BEATS"
    assert acc.find("sum", "COUNTER") == "DOT_RO_SUM_COUNTER"
    with pytest.raises(KeyError):
        acc.find("mul", "JOB_BEATS")  # the fold's CSR is the sum group's
    # a one-group accelerator has no group part: each its own name (D157)
    assert dot_serial().defines() == {
        "MUL_RW_START": 0, "MUL_RW_JOBS": 1, "MUL_RO_STATUS": 2, "MUL_RO_COUNTER": 3,
        "SUM_RW_START": 4, "SUM_RW_JOBS": 5, "SUM_RW_JOB_BEATS": 6, "SUM_RO_STATUS": 7,
        "SUM_RO_COUNTER": 8,
    }  # fmt: skip
    assert dot_serial().find("sum", "START") == "SUM_RW_START"


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
    """The second accelerator's offsets continue the first's (D157): from 0 again is refused,
    as are two accelerators or two groups of one name."""
    serial = dot_serial()
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
    """D152: START 1, STATUS 2, JOBS and JOB_BEATS acc_csr.job_width, COUNTER its counter_width
    (job_width when null); JOBS and every JOB_BEATS share one width, every COUNTER one."""
    from snax_forge.design import DesignPoint, Platform

    from .helpers import SNAX_FORGE

    def widths(*sets: tuple[str, int | None]) -> dict[str, int]:
        plat = Platform.load(SNAX_FORGE).with_changes(list(sets))
        path = point(tmp_path, name="vsum", graph=bound_vsum(), platform=plat)
        acc = describe(DesignPoint.load(path), "p")
        assert acc.job_width == plat.acc_csr.job_width
        return {c.name: c.width for _, _, c in acc.csrs()}

    assert widths() == {"START": 1, "JOBS": 32, "JOB_BEATS": 32, "STATUS": 2, "COUNTER": 32}
    assert widths(("acc_csr.job_width", 16)) == {
        "START": 1, "JOBS": 16, "JOB_BEATS": 16, "STATUS": 2, "COUNTER": 16}  # fmt: skip
    assert widths(("acc_csr.job_width", 16), ("acc_csr.counter_width", 8))["COUNTER"] == 8
    acc = one_unit(1)
    g = acc.groups[0]
    with pytest.raises(ValueError, match="STATUS has width 32, not 2"):
        with_groups(acc, [replace(g, csrs=[*g.csrs[:2], replace(g.csrs[2], width=32), g.csrs[3]])])
    unit = replace(g.units[0], csrs=[Csr("JOB_BEATS", "rw", 2, 16)])
    with pytest.raises(ValueError, match=r"JOBS and JOB_BEATS have the widths \[16, 32\]"):
        with_groups(acc, [replace(g, units=[unit])])


def test_vsums_unit_has_job_beats(tmp_path):
    """A fold (an output of a named rate) gets JOB_BEATS between its group's JOBS and STATUS."""
    from snax_forge.design import DesignPoint

    acc = describe(DesignPoint.load(point(tmp_path, name="vsum", graph=bound_vsum())), "p")
    ((_, unit),) = [(g, u) for g in acc.groups for u in g.units]
    assert [c.to_dict() for c in unit.csrs] == [
        {"name": "JOB_BEATS", "access": "rw", "offset": 2, "width": 32}
    ]
    assert list(acc.defines()) == ["ACC_RW_START", "ACC_RW_JOBS", "ACC_RW_JOB_BEATS",
                                   "ACC_RO_STATUS", "ACC_RO_COUNTER"]  # fmt: skip


def test_a_linked_group_names_its_units_csrs():
    """One group: no group part; two units: the unit part on JOB_BEATS only (C§17's dot 2–6)."""
    acc = dot4()
    assert list(acc.defines()) == ["DOT_RW_START", "DOT_RW_JOBS", "DOT_RW_MAC_JOB_BEATS",
                                   "DOT_RO_STATUS", "DOT_RO_COUNTER"]  # fmt: skip
    (a,) = acc.accelerators
    assert (a.rw, a.ro) == (3, 2)
    assert acc.find("sum", "START") == "DOT_RW_START"
    assert acc.find("mac", "JOB_BEATS") == "DOT_RW_MAC_JOB_BEATS"


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
        ({"links": [Link("x", "mac.out", "sum.a", 0), Link("y", "mac.out", "sum.a", 1)]},
         "gen.link_port"),
        ({"inputs": ["mac.a", "mac.b", "sum.a"]}, "gen.outer"),
        ({"outputs": ["mac.out", "sum.out"]}, "gen.outer"),
        ({"inputs": ["mac.b", "mac.a"]}, "gen.outer"),  # order
        ({"links": [], "inputs": ["mac.a", "mac.b", "sum.a"], "outputs": ["mac.out", "sum.out"]},
         "gen.group"),
    ],
    ids=["in-in", "out-out", "unit", "depth", "dtype", "lanes", "port", "outer-in", "outer-out",
         "outer-order", "group"],
)  # fmt: skip
def test_a_reader_checks_the_links(edit, code):
    """C§17's refusals of links and groups, on construction as on from_dict."""
    with pytest.raises(GenError) as e:
        _edit(**edit)
    assert e.value.code == code


def test_a_group_is_a_chain_until_tier_a():
    """gen.chain: dot 4's units in the other order, the link going back (SNAX-GVSoC's SnaxGroup
    models a chain in unit order)."""
    acc = dot4()
    (g,) = acc.groups
    with pytest.raises(GenError) as e:
        with_groups(acc, [replace(g, units=g.units[::-1])])
    assert e.value.code == "gen.chain" and "until tier A" in str(e.value)


def test_a_span_unit_reads_from_a_link_until_b2():
    """gen.span: dot 5's SerialAccumulate (span 4) fed by the streamer instead of the MAC."""
    serial = Unit("sum", "accumulate", "SerialAccumulate",
                  {"a": Port("in", "in", "int64", 4), "out": Port("out", "out", "int64", 1)},
                  {"T": 1}, 1, 1, [], None, 4)  # fmt: skip
    own = [Csr("START", "rw", 0, 1), Csr("JOBS", "rw", 1, 32), Csr("STATUS", "ro", 2, 2),
           Csr("COUNTER", "ro", 3, 32)]  # fmt: skip
    with pytest.raises(GenError) as e:
        acc = Accelerator("sum", ["sum.a"], ["sum.out"], [Group("sum", own, [serial])])
        Description("p", "snax_forge_p", [acc])
    assert e.value.code == "gen.span" and "until B2" in str(e.value)


def test_dot_1_is_one_accelerator_of_two_groups(tmp_path):
    """dot 1 (encapsulate, D148): one accelerator dot, its two groups mul and sum, one streamer
    set: its inputs and outputs every unlinked port, group by group; offsets 0 to 8; the tag the
    entry's name (D131, D157)."""
    from snax_forge.design import DesignPoint

    desc = describe(DesignPoint.load(point(tmp_path, name="dot_1", graph=bound_dot1())), "p")
    assert desc.tag == "snax_forge_dot_1"
    (acc,) = desc.accelerators
    assert (acc.name, acc.inputs, acc.outputs) == ("dot", ["mul.a", "mul.b", "sum.a"],
                                                   ["mul.out", "sum.out"])  # fmt: skip
    assert [g.name for g in acc.groups] == ["mul", "sum"]
    assert [c.offset for _, _, c in desc.csrs()] == list(range(9))
    assert desc.find("sum", "JOB_BEATS") == "DOT_RW_SUM_JOB_BEATS"
    validate("accelerator", desc.to_dict())


def test_dot_serial_is_two_accelerators(tmp_path):
    """dot_serial (no encapsulate): two accelerators mul and sum in the graph's order, each one
    group and one streamer set, its own outer ports; the offsets run on, 0 to 8 (D157)."""
    from snax_forge.design import DesignPoint

    graph = bound_recipe("dot_serial", "dot")
    desc = describe(DesignPoint.load(point(tmp_path, name="dot_serial", graph=graph)), "p")
    assert desc.tag == "snax_forge_dot_serial"
    assert [(a.name, a.inputs, a.outputs, [g.name for g in a.groups]) for a in desc.accelerators] == [
        ("mul", ["mul.a", "mul.b"], ["mul.out"], ["mul"]),
        ("sum", ["sum.a"], ["sum.out"], ["sum"]),
    ]  # fmt: skip
    assert desc.defines() == dot_serial().defines()
    assert desc.inputs == ["mul.a", "mul.b", "sum.a"]  # the wrapper's order, set after set
    validate("accelerator", desc.to_dict())


@pytest.mark.parametrize("name", ["dot_1_W_acc_1-10", "dot 1", "dot.1"])
def test_a_tag_that_is_no_identifier_is_refused(tmp_path, name):
    """gen.tag: the entry's name <accelerator_slot>_<design> names modules, files and the
    cluster's entry, so a design point whose name makes it no identifier is refused, naming the
    fix (a flow's ``--set W_acc_1=10`` names its design point ``…_W_acc_1-10``, C§7)."""
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
