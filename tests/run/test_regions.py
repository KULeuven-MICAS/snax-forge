"""run/regions.py: the design point's regions, and the elements and values of a word in one."""

import json
import struct

import pytest

from snax_forge.run import Region, region_at, regions_of
from tests.lower.helpers import vecadd_point


def word(*values: int, fmt: str = "<q") -> bytes:
    return b"".join(struct.pack(fmt, v) for v in values)


def test_vecadd_regions_in_the_points_order():
    point = vecadd_point()
    regions = regions_of(point)
    assert [(r.container, r.memory) for r in regions] == [
        (name, mem) for name, lays in point.memory.layouts.items() for mem in lays
    ]
    assert {r.container for r in regions} == {"A", "B", "C"}
    for r in regions:
        lay = point.memory.layouts[r.container][r.memory]
        assert (r.base, r.shape, r.strides, r.layout) == (lay.base, lay.shape, lay.strides, lay)
        assert r.dtype == point.graph.containers[r.container].dtype == "int64"
    b = next(r for r in regions if (r.container, r.memory) == ("B", "l1"))
    assert region_at(regions, "l1", b.base + 8) == b
    assert region_at(regions, "l1", max(r.span()[1] for r in regions if r.memory == "l1")) is None


def test_a_word_names_its_element_and_value():
    r = Region("A", "l1", 0x10000000, (64,), (8,), "int64")
    assert r.span() == (0x10000000, 0x10000200)
    assert r.elems(0x10000008, 8) == [1]
    assert r.vals(0x10000008, word(-461)) == [-461]
    # a DMA beat of eight words: eight elements
    assert r.elems(0x10000040, 64) == list(range(8, 16))
    assert r.vals(0x10000040, word(*range(8))) == list(range(8))


def test_packed_and_strided_regions():
    # two int32 elements in a word
    r = Region("X", "l1", 0x100, (4,), (4,), "int32")
    assert r.elems(0x100, 8) == [0, 1]
    assert r.vals(0x100, word(-2, 7, fmt="<i")) == [-2, 7]
    # a sub-word access sees only its own element
    assert r.elems(0x104, 4) == [1]
    # 2 x 3, rows 32 bytes apart: flat indexes row-major over the shape
    m = Region("M", "l1", 0x200, (2, 3), (32, 8), "int64")
    assert m.elems(0x220, 8) == [3]
    assert m.elems(0x218, 8) == []  # the padding after row 0
    assert m.span() == (0x200, 0x238)


def test_round_trip():
    r = Region("M", "l1", 0x200, (2, 3), (32, 8), "int64")
    assert Region.from_dict(json.loads(json.dumps(r.to_dict()))) == r


@pytest.mark.parametrize("addr", [0x1FF, 0x238])
def test_outside_is_no_element(addr):
    m = Region("M", "l1", 0x200, (2, 3), (32, 8), "int64")
    assert not m.contains(addr) and m.elems(addr, 1) == []
