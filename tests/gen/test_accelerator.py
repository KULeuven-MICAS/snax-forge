"""accelerator.json (D145, C§17): the description round-trips, its CSRs are in CSR order from
offset 0, and one helper names them for the header and the program."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from snax_forge.gen import Accelerator, Csr, Group, Port, Unit, describe
from snax_forge.gen.schemas import validate

from .helpers import bound_vsum, one_unit, point


def dot() -> Accelerator:
    """dot 1 as C§17's table has it: accelerator dot, groups mul (an ElementwiseLanes unit) and
    sum (a TreeAccumulate unit with JOB_BEATS), in that order."""

    def own(base: int, rw_units: int) -> list[Csr]:
        return [Csr("START", "rw", base, 32), Csr("JOBS", "rw", base + 1, 32),
                Csr("STATUS", "ro", base + 2 + rw_units), Csr("COUNTER", "ro", base + 3 + rw_units)]  # fmt: skip

    mul = Unit("mul", "elementwise_mul", "ElementwiseLanes",
               {p: Port(p, "int64", 4) for p in ("a", "b", "out")}, {"op": "mul"}, 0, 1, [],
               "int64")  # fmt: skip
    total = Unit("sum", "accumulate", "TreeAccumulate",
                 {"a": Port("in", "int64", 4), "out": Port("out", "int64", 1)}, {}, 1, 1,
                 [Csr("JOB_BEATS", "rw", 6, 32)])  # fmt: skip
    return Accelerator("dot", "design/dot/design_point.json", "snax_forge",
                       ["mul.a", "mul.b", "sum.a"], ["mul.out", "sum.out"],
                       [Group("mul", own(0, 0), [mul]), Group("sum", own(4, 1), [total])])  # fmt: skip


def test_the_description_round_trips_and_meets_its_schema():
    for acc in (one_unit(), one_unit(3), dot()):
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
    # Two units in one group (B3) keep the unit part on the units' own CSRs only
    sum_group = acc.groups[1]
    two = replace(sum_group, units=[*sum_group.units, replace(sum_group.units[0], name="b",
                                                                csrs=[])])  # fmt: skip
    both = replace(acc, groups=[acc.groups[0], two])
    assert both.find("sum", "JOB_BEATS") == "DOT_RW_SUM_SUM_JOB_BEATS"
    assert both.find("b", "START") == "DOT_RW_SUM_START"


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
