"""One L2 size per container (``lower.layout.l2_bytes``): data.h's array, the ELF's symbol
(run/elf.py) and the DMA's copies (run/convert.py) agree, a tail's padded rows included (D144).

rowsum_3x7 splits A's rows of 7 by W = 4, so they are padded to 8: A, M x 7 int64, takes
M x 8 words in whole 64-byte beats. At M = 3 that is 192 bytes, as the 168 unpadded ones in
beats are; from M = 8 the padded rows take more.
"""

import numpy as np
import pytest

from snax_forge.design.point import DesignPoint
from snax_forge.gen.data import arrays
from snax_forge.lower import task_list
from snax_forge.lower.tasks import Configure
from snax_forge.run import RunError, cluster_from, regions_of
from snax_forge.run.convert import l2_copies
from snax_forge.run.elf import l2_symbols
from snax_forge.run.trace import Trace
from tests.gen.helpers import designed

from .helpers import L2, files, point_symbols

L1 = 0x10000000
SIZES = {3: 192, 8: 512, 16: 1024}  # M -> A's L2 bytes


@pytest.fixture(scope="module", params=sorted(SIZES))
def rowsum(request, tmp_path_factory):
    """(M, the design point of rowsum_3x7 at M)."""
    m = request.param
    tmp = tmp_path_factory.mktemp(f"rowsum_m{m}")
    return m, DesignPoint.load(tmp / designed(tmp, "rowsum_3x7", f"rowsum_m{m}", [f"M={m}"]))


def linked(point):
    """The symbols data.h's arrays link to: one after another from L2's base, each its
    declared bytes."""
    out, at = {}, L2
    for a in arrays(point):
        n = len(a.values) * np.dtype(point.graph.containers[a.container].dtype).itemsize
        out[a.name] = (at, n)
        at += n
    return out


def copy(task, frm, size):
    """A DMA issue of ``size`` bytes from L2 address ``frm`` into L1."""
    return {"t": 0, "k": "issue", "src": "dma", "task": task, "id": 0, "dir": "l2_to_l1",
            "from": frm, "to": L1, "size": size, "from_stride": 0, "to_stride": 0, "reps": 1}  # fmt: skip


def test_data_h_and_the_elf_check_agree(rowsum):
    m, point = rowsum
    syms = linked(point)
    assert syms["A"][1] == SIZES[m]
    assert l2_symbols(point, syms)["A"] == {"addr": syms["A"][0], "size": SIZES[m]}
    assert point_symbols(point)["A"][1] == SIZES[m]


def test_a_symbol_of_another_size_names_the_padded_count(rowsum):
    m, point = rowsum
    syms = linked(point)
    syms["A"] = (syms["A"][0], SIZES[m] - 64)
    with pytest.raises(RunError) as e:
        l2_symbols(point, syms)
    assert e.value.code == "elf.size"
    assert (f"container A: its symbol is {SIZES[m] - 64} bytes, its {m * 8} int64 elements "
            f"padded to 64-byte beats are {SIZES[m]}") in str(e.value)  # fmt: skip


def test_the_dma_copies_its_whole_array(rowsum):
    m, point = rowsum
    (dma,) = [s for s in task_list(point).steps
              if isinstance(s, Configure) and s.type == "dma" and s.values["src"].get("symbol") == "A"]  # fmt: skip
    moved = dma.values["src"]["bounds"][0] * dma.values["src"]["strides"][0]
    assert moved == SIZES[m]
    regions = [r for r in regions_of(point, l2_symbols(point, linked(point)))
               if r.container == "A" and r.memory == "l2"]  # fmt: skip
    (a,) = regions
    assert a.size == SIZES[m]
    cluster = cluster_from(*files())
    assert l2_copies(Trace("task", 1, 0, ["dma"], [copy("load_A", a.base, moved)]), regions,
                     cluster) == []  # fmt: skip
    short = Trace("task", 1, 0, ["dma"], [copy("load_A", a.base, moved - 64)])
    assert l2_copies(short, regions, cluster) == [
        (
            f"container A: its copies touch L2 [('{a.base:#x}', '{a.base + moved - 64:#x}')], its "
            f"array ({m * 8} int64 elements, its padded shape in 64-byte beats) is "
            f"[{a.base:#x}, {a.base + moved:#x})"
        )
    ]
