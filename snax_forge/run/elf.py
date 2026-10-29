"""The program's L2 data, read from its ELF (C§7, D86): where the linker put each container.

    read_symbols(elf) -> {name: (addr, size)}
    l2_symbols(point, symbols) -> {container: {"addr", "size"}}

The design point's memory plan gives every container its L1 base, and the program follows it;
in L2 it gives residency and shape only, since ``data.h`` declares each container that lives
in L2 as a C array under its name (gen/data.py) and the linker places it. So the flow reads
the arrays' symbols once the program is linked (its ``sw`` stage) into ``run.json``'s
``elf_symbols``, and ``regions_of`` takes the L2 regions from them.

``read_symbols`` reads the ELF's symbol table with the standard library (32- and 64-bit,
little-endian; ``.symtab`` and its string table), every named symbol with its value and size.
``l2_symbols`` keeps the design point's containers in L2: each must have a symbol of its
L2 bytes, its padded shape (a tail's pads included, D144) in whole DMA beats, as gen/data.py
declares it (D105, ``lower.layout.l2_bytes``; the platform's ``l2.beat_bits``). Other symbols (``C_golden``, the runtime's) are not regions. A
file that is not such an ELF, a container with no symbol, or one of another size is a
RunError (``elf.<rule>``), which the flow reports as its ``sw`` stage's.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from snax_forge.dfg.pad import padded_shape
from snax_forge.lower.layout import l2_bytes

from .events import RunError

SHT_SYMTAB = 2


def read_symbols(elf: str | Path) -> dict[str, tuple[int, int]]:
    """Every named symbol of an ELF's ``.symtab``: name -> (value, size)."""
    data = Path(elf).read_bytes()
    if data[:4] != b"\x7fELF" or data[5] != 1 or data[4] not in (1, 2):
        raise RunError("elf.file", f"{elf}: not a little-endian ELF")
    wide = data[4] == 2
    hdr = "<16xHHIQQQIHHHHHH" if wide else "<16xHHIIIIIHHHHHH"
    fields = struct.unpack_from(hdr, data, 0)
    shoff, shentsize, shnum = fields[5], fields[10], fields[11]
    sect = "<IIQQQQIIQQ" if wide else "<IIIIIIIIII"
    sections = [struct.unpack_from(sect, data, shoff + i * shentsize) for i in range(shnum)]
    out: dict[str, tuple[int, int]] = {}
    for s in sections:
        if s[1] != SHT_SYMTAB:
            continue
        offset, size, link, entsize = s[4], s[5], s[6], s[9]
        strtab = sections[link]
        names = data[strtab[4] : strtab[4] + strtab[5]]
        sym = "<IBBHQQ" if wide else "<IIIBBH"
        for i in range(size // entsize):
            e = struct.unpack_from(sym, data, offset + i * entsize)
            name_off, value, n = (e[0], e[4], e[5]) if wide else (e[0], e[1], e[2])
            name = names[name_off : names.index(b"\0", name_off)].decode()
            if name:
                out[name] = (value, n)
    if not out:
        raise RunError("elf.symtab", f"{elf}: no symbol table")
    return out


def l2_symbols(point: Any, symbols: dict[str, tuple[int, int]]) -> dict[str, dict[str, int]]:
    """The design point's containers in L2 and their arrays' symbols (module doc)."""
    beat = point.platform.l2.beat_bits // 8
    out = {}
    for name, lays in point.memory.layouts.items():
        if "l2" not in lays:
            continue
        want = l2_bytes(point, name)
        if name not in symbols:
            raise RunError("elf.missing", f"container {name} lives in L2, the ELF has no symbol "
                           f"{name!r} (data.h declares it)")  # fmt: skip
        addr, size = symbols[name]
        if size != want:
            n = int(np.prod(padded_shape(point.graph, name)))
            raise RunError("elf.size", f"container {name}: its symbol is {size} bytes, its "
                           f"{n} {point.graph.containers[name].dtype} elements padded to "
                           f"{beat}-byte beats are {want}")  # fmt: skip
        out[name] = {"addr": addr, "size": size}
    return out
