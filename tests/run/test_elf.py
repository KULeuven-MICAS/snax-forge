"""run/elf.py: the program's L2 data from its ELF's symbols, and what they must be."""

import pytest

from snax_forge.run import RunError
from snax_forge.run.elf import l2_symbols, read_symbols
from tests.lower.helpers import vecadd_point

from .helpers import point_symbols, tiny_elf


@pytest.mark.parametrize("wide", [False, True])
def test_the_symbol_table_reads(tmp_path, wide):
    syms = {"A": (0x80002900, 512), "C_golden": (0x80002D00, 512), "main": (0x800001E0, 64)}
    (tmp_path / "x.elf").write_bytes(tiny_elf(syms, wide))
    assert read_symbols(tmp_path / "x.elf") == syms


def test_the_l2_containers_and_their_padded_sizes(tmp_path):
    point = vecadd_point()
    syms = point_symbols(point)
    (tmp_path / "x.elf").write_bytes(tiny_elf(syms))
    got = l2_symbols(point, read_symbols(tmp_path / "x.elf"))
    assert got == {c: {"addr": syms[c][0], "size": 512} for c in ("A", "B", "C")}


@pytest.mark.parametrize(
    "change, code, said",
    [
        (lambda s: {k: v for k, v in s.items() if k != "C"}, "elf.missing",
         "container C lives in L2, the ELF has no symbol 'C'"),
        (lambda s: {**s, "B": (s["B"][0], 504)}, "elf.size",
         ("container B: its symbol is 504 bytes, its 64 int64 elements padded to 64-byte beats "
          "are 512")),
    ],
)  # fmt: skip
def test_a_missing_or_wrong_sized_symbol_is_named(change, code, said):
    point = vecadd_point()
    with pytest.raises(RunError) as e:
        l2_symbols(point, change(point_symbols(point)))
    assert e.value.code == code and said in str(e.value)


def test_not_an_elf(tmp_path):
    (tmp_path / "x.elf").write_bytes(b"\x7fELF fake")
    with pytest.raises(RunError) as e:
        read_symbols(tmp_path / "x.elf")
    assert e.value.code == "elf.file"
