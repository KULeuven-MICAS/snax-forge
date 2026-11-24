"""run/elf.py: the program's L2 data from its ELF's symbols, and what they must be."""

import pytest

from snax_forge.run import RunError
from snax_forge.run.elf import read_symbols


def test_not_an_elf(tmp_path):
    (tmp_path / "x.elf").write_bytes(b"\x7fELF fake")
    with pytest.raises(RunError) as e:
        read_symbols(tmp_path / "x.elf")
    assert e.value.code == "elf.file"
