"""accelerator.json (D145, C§17): the description round-trips, its CSRs are in CSR order from
offset 0, and one helper names them for the header and the program."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from snax_forge.gen import Accelerator, Csr, Group, Link, Port, Unit, describe
from snax_forge.gen.blocks import GenError

from .helpers import bound_vsum, one_unit, point, validate


def dot() -> Accelerator:
    """dot 1 as C§17's table has it: accelerator dot, groups mul (an ElementwiseLanes unit) and
    sum (a TreeAccumulate unit with JOB_BEATS), in that order."""

    def own(base: int, rw_units: int) -> list[Csr]:
        return [Csr("START", "rw", base, 32), Csr("JOBS", "rw", base + 1, 32),
                Csr("STATUS", "ro", base + 2 + rw_units), Csr("COUNTER", "ro", base + 3 + rw_units)]  # fmt: skip

    mul = Unit("mul", "elementwise_mul", "ElementwiseLanes",
               {p: Port(p, "out" if p == "out" else "in", "int64", 4) for p in ("a", "b", "out")},
               {"op": "mul"}, 0, 1, [],
               "int64")  # fmt: skip
    total = Unit("sum", "accumulate", "TreeAccumulate",
                 {"a": Port("in", "in", "int64", 4), "out": Port("out", "out", "int64", 1)}, {"op": "add"}, 1, 1,
                 [Csr("JOB_BEATS", "rw", 6, 32)])  # fmt: skip
    return Accelerator("dot", "design/dot/design_point.json", "snax_forge",
                       ["mul.a", "mul.b", "sum.a"], ["mul.out", "sum.out"],
                       [Group("mul", own(0, 0), [mul]), Group("sum", own(4, 1), [total])])  # fmt: skip


def dot4() -> Accelerator:
    """dot 4 as C§17's block has it: one group dot, a Mac unit (JOB_BEATS) wired into a
    TreeAccumulate unit of fixed T 1 by the link out_partial."""
    mac = Unit("mac", "mac", "Mac", {"a": Port("a", "in", "int64", 4), "b": Port("b", "in", "int64", 4),
               "out": Port("out", "out", "int64", 4)}, {"accumulate": True, "regOut": False}, 1, 1,
               [Csr("JOB_BEATS", "rw", 2, 32)], "int64")  # fmt: skip
    total = Unit("sum", "accumulate", "TreeAccumulate",
                 {"a": Port("in", "in", "int64", 4), "out": Port("out", "out", "int64", 1)},
                 {"op": "add", "T": 1}, 1, 1)  # fmt: skip
    own = [Csr("START", "rw", 0, 32), Csr("JOBS", "rw", 1, 32), Csr("STATUS", "ro", 3),
           Csr("COUNTER", "ro", 4)]  # fmt: skip
    return Accelerator("dot", "out/design/dot_4/design_point.json", "snax_forge",
                       ["mac.a", "mac.b"], ["sum.out"],
                       [Group("dot", own, [mac, total], [Link("out_partial", "mac.out", "sum.a", 0)])])  # fmt: skip


def test_the_description_round_trips_and_meets_its_schema():
    for acc in (one_unit(), one_unit(3), dot(), dot4()):
        assert Accelerator.from_dict(json.loads(acc.to_json())) == acc
        validate("accelerator", acc.to_dict())


def test_csr_order_and_counts():
    """Per group START, JOBS, the units' R/W, STATUS, COUNTER; rw = Σ_g (2 + Σ_u (f_u + k_u)),
    ro = 2G (C§17's table: dot 1 is 5, 4)."""
    acc = dot()
    assert [(g.name, c.name, c.offset) for g, _, c in acc.csrs()] == [
        ("mul", "START", 0), ("mul", "JOBS", 1), ("mul", "STATUS", 2), ("mul", "COUNTER", 3),
        ("sum", "START", 4), ("sum", "JOBS", 5), ("sum", "JOB_BEATS", 6), ("sum", "STATUS", 7),
        ("sum", "COUNTER", 8),
    ]  # fmt: skip
    assert (acc.rw, acc.ro) == (5, 4)
    assert (one_unit().rw, one_unit().ro) == (2, 2)
    assert (one_unit(1).rw, one_unit(1).ro) == (3, 2)


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


@pytest.mark.parametrize(
    ("start", "text"),
    [({"offset": 1}, "not contiguous"), ({"width": None}, "has a width")],
    ids=["offsets", "width"],
)
def test_a_reader_checks_the_offsets_and_widths(start, text):
    """Offsets are stored, never recomputed: a reader checks them contiguous from 0 in CSR
    order; an rw CSR has a width, an ro one none (START edited by ``start``)."""
    acc = one_unit()
    g = acc.groups[0]
    edited = replace(g, csrs=[replace(g.csrs[0], **start), *g.csrs[1:]])
    with pytest.raises(ValueError, match=text):
        replace(acc, groups=[edited])


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
    assert (acc.rw, acc.ro) == (3, 2)
    assert acc.find("sum", "START") == "DOT_RW_START"
    assert acc.find("mac", "JOB_BEATS") == "DOT_RW_MAC_JOB_BEATS"


def _edit(links=None, inputs=None, outputs=None, ports=None) -> None:
    """dot4 with its links, outer ports or the sum unit's ports replaced."""
    acc = dot4()
    g = acc.groups[0]
    units = g.units if ports is None else [g.units[0], replace(g.units[1], ports=ports)]
    replace(acc, inputs=acc.inputs if inputs is None else inputs,
            outputs=acc.outputs if outputs is None else outputs,
            groups=[replace(g, units=units, links=g.links if links is None else links)])  # fmt: skip


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
        replace(acc, groups=[replace(g, units=g.units[::-1])])
    assert e.value.code == "gen.chain" and "until tier A" in str(e.value)


def test_a_span_unit_reads_from_a_link_until_b2():
    """gen.span: dot 5's SerialAccumulate (span 4) fed by the streamer instead of the MAC."""
    serial = Unit("sum", "accumulate", "SerialAccumulate",
                  {"a": Port("in", "in", "int64", 4), "out": Port("out", "out", "int64", 1)},
                  {"T": 1}, 1, 1, [], None, 4)  # fmt: skip
    own = [Csr("START", "rw", 0, 32), Csr("JOBS", "rw", 1, 32), Csr("STATUS", "ro", 2),
           Csr("COUNTER", "ro", 3)]  # fmt: skip
    with pytest.raises(GenError) as e:
        Accelerator("sum", "p", "snax_forge", ["sum.a"], ["sum.out"], [Group("sum", own, [serial])])
    assert e.value.code == "gen.span" and "until B2" in str(e.value)
