"""Helpers shared by the scenario generators (scenarios/*/scenario.py).

Sizes of the checked-in clusters and the two layout helpers the Python-built
programs and tests use. Loaded through ``scenarios/`` on the import path,
which scenarios/make.py sets up.
"""

from __future__ import annotations

from snax_forge.snax_model import DmaPattern, StreamerRegs

WORD = 8  # bytes per bank word (64-bit banks)
BEAT = 64  # bytes per wide beat (512 bits)
LANES = 4


def contiguous(base: int, n: int) -> DmaPattern:
    """n consecutive wide beats from byte ``base``."""
    return DmaPattern(base, (n,), (BEAT,))


def unit(word: int, n_beats: int, lanes: int) -> StreamerRegs:
    """Streamer task over n_beats contiguous beats of ``lanes`` words from word ``word``."""
    return StreamerRegs(word * WORD, (n_beats,), (lanes * WORD,), (lanes,), (WORD,))
