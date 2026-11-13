"""The BRM library: hand-written BRMs, one JSON file each (CONTRACTS, "Block runtime model").

``library/<name>.json`` holds the BRM called ``name``; the file name and the BRM's ``name`` must
agree. ``load_brm(name)`` reads one and ``library_names`` lists them: ``simd``, ``accumulate``,
``mac`` and ``lane_accumulate``. Files are kept in the form ``Brm.to_json`` writes, every field
included, so a hand edit that leaves that form shows up in tests/brm/test_library.py. A file
is read and checked once per version (its mtime): every caller shares the one ``Brm``, which no
caller changes.
"""

from __future__ import annotations

from pathlib import Path

from .brm import Brm, BrmError

LIBRARY = Path(__file__).resolve().parent / "library"

_LOADED: dict[Path, tuple[int, Brm]] = {}  # a file -> (its mtime_ns, its BRM)


def library_names() -> list[str]:
    """The names of every BRM in the library, sorted."""
    return sorted(p.stem for p in LIBRARY.glob("*.json"))


def load_brm(name: str) -> Brm:
    """The library BRM called ``name``."""
    path = LIBRARY / f"{name}.json"
    if not path.is_file():
        raise BrmError(f"no BRM {name!r} in the library ({library_names()})")
    mtime = path.stat().st_mtime_ns
    if path in _LOADED and _LOADED[path][0] == mtime:
        return _LOADED[path][1]
    brm = Brm.load(path)
    if brm.name != name:
        raise BrmError(f"{path.name}: holds BRM {brm.name!r}, the file name says {name!r}")
    _LOADED[path] = (mtime, brm)
    return brm
