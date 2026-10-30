"""The flip-flops of every group control unit (``<tag>_<group>_ctrl``) of the RTL fixtures.

    pixi run python util/ctrl_regs.py [--sv DIR] [--out FILE] [FIXTURE ...]

For each fixture under ``hw/chisel/src/test/resources/rtl/`` (all, or those named), its
``accelerator.json`` is elaborated through ``Emit --spec`` (one call for all) into ``DIR/<fixture>/`` (default ``out/ctrl_regs``); then every module of the
emitted SV whose name ends in ``_ctrl`` is read for firtool's ``reg`` declarations, each a
register of its width (an unpacked array counts each element). Printed, and written to FILE:
per module one line per register with its name and width, the module's registers and bits;
per fixture its totals; and a summary table. ``--no-emit`` reads what ``DIR`` holds already.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "hw/chisel/src/test/resources/rtl"

_MODULE = re.compile(r"^module\s+(\w+)\s*[(;#].*?^endmodule", re.DOTALL | re.MULTILINE)
_REG = re.compile(
    r"^\s*reg\s*(?:\[(\d+):(\d+)\])?\s*(\w+)\s*(?:\[(\d+):(\d+)\])?\s*;", re.MULTILINE
)


def emit(fixtures: list[str], sv: Path) -> None:
    """``Emit --spec`` on each fixture into ``sv/<fixture>``, in one call (gen/rtl.py's
    ``elaborate_all``)."""
    sys.path.insert(0, str(REPO))
    from snax_forge.gen.rtl import elaborate_all

    elaborate_all([(FIXTURES / f / "accelerator.json", sv / f) for f in fixtures])


def registers(text: str) -> dict[str, list[tuple[str, int]]]:
    """Per ``*_ctrl`` module of an SV file, its registers: (name, width), in declaration order."""
    out = {}
    for m in _MODULE.finditer(text):
        if not m.group(1).endswith("_ctrl"):
            continue
        regs = []
        for r in _REG.finditer(m.group(0)):
            hi, lo, name, a, b = r.groups()
            width = abs(int(hi) - int(lo)) + 1 if hi is not None else 1
            count = abs(int(a) - int(b)) + 1 if a is not None else 1
            regs += [(name, width)] * count
        out[m.group(1)] = regs
    return out


def table(fixtures: list[str], sv: Path) -> str:
    lines, summary = [], []
    for f in fixtures:
        mods = {}
        for p in sorted((sv / f / "sv").glob("*.sv")):
            mods |= registers(p.read_text())
        n = bits = 0
        lines.append(f"== {f}")
        for mod, regs in mods.items():
            if not regs:
                continue  # <tag>_ctrl holds the groups' modules only
            w = sum(x for _, x in regs)
            lines.append(f"  {mod}: {len(regs)} registers, {w} bits")
            lines += [f"    {name:<28} {x:>3}" for name, x in regs]
            summary.append(f"{f:<16} {mod:<30} {len(regs):>4} {w:>6}")
            n, bits = n + len(regs), bits + w
        lines.append(f"  total: {n} registers, {bits} bits")
    head = f"{'fixture':<16} {'module':<30} {'regs':>4} {'bits':>6}"
    return "\n".join([head, *summary, "", *lines]) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="util/ctrl_regs.py", description=__doc__.splitlines()[0])
    ap.add_argument("fixtures", nargs="*", help="fixture names (default: every one)")
    ap.add_argument("--sv", type=Path, default=REPO / "out/ctrl_regs", help="where Emit writes")
    ap.add_argument("--out", type=Path, help="also write the table here")
    ap.add_argument("--no-emit", action="store_true", help="read the SV already in --sv")
    a = ap.parse_args(argv)
    fixtures = a.fixtures or sorted(p.name for p in FIXTURES.iterdir() if p.is_dir())
    sv = a.sv.resolve()
    if not a.no_emit:
        emit(fixtures, sv)
    text = table(fixtures, sv)
    print(text, end="")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
