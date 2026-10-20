"""One line per kernel under kernels/: where it lives, its tier and dtype, how far it gets.

    python util/kernels.py   (pixi run kernels)

Columns: the kernel, its directory, its tier (the ``Tier X`` line of its
docstring, PAPER §6.1; ``-`` when it has none), its dtypes (those of its
descriptors), whether its simplified SDFG builds, and whether the SNAX-DFG
importer accepts it, else the class of what it raised and its first
line. The importer has no error codes yet; the class names the error. It
only reports: the exit status is always 0.
"""

from __future__ import annotations

import ast
import re

from snax_forge.dfg.import_sdfg import import_sdfg
from snax_forge.sdfg.build import build
from snax_forge.sdfg.loader import KERNEL_ROOT, kernel_paths, load

TIER = re.compile(r"^Tier (\S+) \(PAPER", re.MULTILINE)


def _error(e: Exception) -> str:
    first = str(e).strip().splitlines()
    return f"{type(e).__name__}: {first[0] if first else ''}"


def row(name: str, path) -> list[str]:
    doc = ast.get_docstring(ast.parse(path.read_text())) or ""
    tier = TIER.search(doc)
    spec = load(name)
    dtypes = sorted({getattr(d, "dtype", d).type.__name__ for d in spec.descriptors.values()})
    try:
        sdfg = build(spec, simplify=True)
    except Exception as e:  # noqa: BLE001
        built, imported = f"no ({_error(e)})", "-"
    else:
        built = "yes"
        try:
            import_sdfg(sdfg, spec.name)
            imported = "yes"
        except Exception as e:  # noqa: BLE001
            imported = _error(e)
    return [
        name,
        str(path.parent.relative_to(KERNEL_ROOT)),
        tier.group(1) if tier else "-",
        "+".join(dtypes),
        built,
        imported,
    ]


def main() -> int:
    rows = [["kernel", "dir", "tier", "dtype", "sdfg", "import-dfg"]]
    rows += [row(name, path) for name, path in sorted(kernel_paths().items())]
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
