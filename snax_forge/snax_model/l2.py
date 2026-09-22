"""L2 memory for SNAX-MODEL (MOD6, D13, D29, D34).

What this models
----------------
A flat memory behind the DMA. Only the DMA accesses it, one wide beat at a
time. There are no banks and no arbitration.

Which RTL it stands for
-----------------------
In the SNAX RTL simulation (target/common/test, testharness.sv.tpl) the
cluster's wide AXI port goes to ``tb_memory_axi``: an atomics filter, an
``axi_cut``, ``axi_to_reg`` with ``FULL_BW`` and a DPI memory that reads
combinationally and is always ready. So the L2 is a fixed latency set by a
few registers, with one beat per cycle; there is no DRAM model. Here that
is ``read_latency``: cycles from the DMA's read to the data in the DMA's
buffer, with the AXI path lumped in. ANC1 measures the real value.

AXI read and write channels are independent, so one read and one write per
cycle are allowed. A second read or a second write in the same cycle is a
bug in the DMA and raises ``L2AccessError``, as a same-bank double access
does in the L1 (D30).

Who calls what, per cycle
-------------------------
The L2 is a shared state element touched by its requester (a ``Stateful``,
like the L1): ``read`` and ``write`` call ``cluster.touch(self)``, so the
scheduler calls ``commit`` at the end of that cycle.

    MEMORY     DMA:  read(cycle, addr, tag) -> ready cycle,  write(cycle, addr, beat)
    RESPONSE   DMA:  resp(cycle)  (read data due now)

Timing rules (RTL-like, D29), as in mem.py:

* a read samples the storage as it was at the end of the previous cycle;
* a write lands in storage in ``commit``, at the end of the cycle;
* read data for a read in cycle ``t`` is visible in ``t + read_latency``.

Data layout (D13)
-----------------
Storage is ``[words, elems_per_word]``, with words of ``width_bits`` (the
L1 bank width), so a beat moves between L1 and L2 unchanged: a beat is
``beat_bits / width_bits`` consecutive words, shape [words_per_beat, epw].
Addresses are byte addresses starting at ``base_addr`` (0 by default, like
the L1), and every access is one aligned beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .sched import SimulationError


class L2AccessError(SimulationError):
    """Two reads or two writes reached the L2 in the same cycle."""


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class L2Config:
    """Size and timing of the L2. Frozen: the storage is built once from it."""

    size_bytes: int = 1 << 20  # placeholder size, 1 MiB
    base_addr: int = 0  # byte address of the first L2 word
    read_latency: int = 1  # cycles from read to data in the DMA buffer; ANC1 sets it
    beat_bits: int = 512  # one access; must equal L1Config.wide_bits
    width_bits: int = 64  # word = the L1 bank word (D13)
    dtype: str = "int64"  # element type, as in the L1
    elems_per_word: int = 1  # as in the L1

    def __post_init__(self) -> None:
        if self.width_bits % 8 or self.width_bits < 8:
            raise ValueError("width_bits must be a positive multiple of 8")
        if self.beat_bits < self.width_bits or self.beat_bits % self.width_bits:
            raise ValueError("beat_bits must be a multiple of width_bits")
        if self.size_bytes < 1 or self.size_bytes % self.beat_bytes:
            raise ValueError(f"size_bytes must be a positive multiple of {self.beat_bytes}")
        if self.read_latency < 0:
            raise ValueError("read_latency must be >= 0")
        if self.elems_per_word < 1:
            raise ValueError("elems_per_word must be >= 1")
        elem_bits = np.dtype(self.dtype).itemsize * 8
        if elem_bits * self.elems_per_word > self.width_bits:
            raise ValueError(
                f"{self.elems_per_word} x {self.dtype} does not fit in {self.width_bits} bits"
            )

    @property
    def word_bytes(self) -> int:
        return self.width_bits // 8

    @property
    def beat_bytes(self) -> int:
        """Bytes per access; also the alignment of every access."""
        return self.beat_bits // 8

    @property
    def words_per_beat(self) -> int:
        return self.beat_bits // self.width_bits

    @property
    def n_words(self) -> int:
        return self.size_bytes // self.word_bytes


@dataclass
class L2Resp:
    """Read data leaving the L2."""

    tag: Any  # copied from the read
    data: np.ndarray  # the beat, shape (words_per_beat, elems_per_word)
    issued: int  # cycle of the read


# =============================================================================
# The L2 memory
# =============================================================================


@dataclass(eq=False)  # compared by identity, like every touched element
class L2Memory:
    """Flat memory of words, accessed one beat at a time.

    State, split the RTL way:

    * committed state: ``data`` [words, epw], ``_resp`` read data waiting
      to leave, keyed by its ready cycle;
    * this cycle's uncommitted state: ``_now``, ``_read_now``,
      ``_write_now``, ``_writes``;
    * statistics (for MOD8): ``reads``, ``writes`` (beats).
    """

    cluster: Any  # anything with touch(elem)
    cfg: L2Config = field(default_factory=L2Config)

    def __post_init__(self) -> None:
        c = self.cfg
        self.data = np.zeros((c.n_words, c.elems_per_word), dtype=c.dtype)
        self.reads = 0
        self.writes = 0
        self._now: int | None = None
        self._read_now = False
        self._write_now = False
        self._writes: list[tuple[int, np.ndarray]] = []  # (first word, beat)
        self._resp: dict[int, L2Resp] = {}  # one read per cycle, fixed latency

    # -------------------------------------------------------------------------
    # Addressing
    # -------------------------------------------------------------------------

    def word_of(self, addr: int) -> int:
        """Index of the first word of the beat at ``addr``. Raises if misaligned or outside."""
        c = self.cfg
        off = addr - c.base_addr
        if off % c.beat_bytes:
            raise SimulationError(
                f"L2 address {addr:#x} is not aligned to a {c.beat_bits}-bit beat"
            )
        if not 0 <= off <= c.size_bytes - c.beat_bytes:
            raise SimulationError(f"L2 address {addr:#x} is outside L2")
        return off // c.word_bytes

    # -------------------------------------------------------------------------
    # Timed port: use during a run
    # -------------------------------------------------------------------------

    def _enter(self, cycle: int) -> None:
        if self._now is not None and self._now != cycle:
            raise SimulationError("L2 was not committed between cycles")
        self._now = cycle
        self.cluster.touch(self)

    def read(self, cycle: int, addr: int, tag: Any = None) -> int:
        """Read one beat in ``cycle``. Returns the cycle its data is ready."""
        w = self.word_of(addr)
        self._enter(cycle)
        if self._read_now:
            raise L2AccessError(f"cycle {cycle}: second L2 read (addr {addr:#x})")
        self._read_now = True
        ready = cycle + self.cfg.read_latency
        # Sample now (end-of-previous-cycle state); copy so later writes don't change it.
        beat = self.data[w : w + self.cfg.words_per_beat].copy()
        self._resp[ready] = L2Resp(tag, beat, cycle)
        self.reads += 1
        return ready

    def write(self, cycle: int, addr: int, beat: Any) -> None:
        """Write one beat in ``cycle``; it lands in storage at the end of the cycle."""
        w = self.word_of(addr)
        self._enter(cycle)
        if self._write_now:
            raise L2AccessError(f"cycle {cycle}: second L2 write (addr {addr:#x})")
        self._write_now = True
        self._writes.append((w, self._beat(beat)))
        self.writes += 1

    def resp(self, cycle: int) -> L2Resp | None:
        """Read data leaving the L2 in ``cycle``, or None. Read in Phase.RESPONSE."""
        return self._resp.get(cycle)

    def next_response(self, cycle: int) -> int | None:
        """Earliest cycle > ``cycle`` with read data, or None (committed state, D29)."""
        later = [t for t in self._resp if t > cycle]
        return min(later) if later else None

    def commit(self) -> None:
        """End of cycle: apply writes, drop delivered read data."""
        for w, beat in self._writes:
            self.data[w : w + self.cfg.words_per_beat] = beat
        self._writes.clear()
        if self._now is not None:
            for t in [t for t in self._resp if t <= self._now]:
                del self._resp[t]
        self._now = None
        self._read_now = self._write_now = False

    # -------------------------------------------------------------------------
    # Backdoor: setup and inspection outside a run. No timing, no counters.
    # Addresses here only need to be word aligned.
    # -------------------------------------------------------------------------

    def _word_index(self, addr: int, n: int = 1) -> int:
        c = self.cfg
        off = addr - c.base_addr
        if off % c.word_bytes:
            raise SimulationError(f"L2 address {addr:#x} is not word aligned")
        if not 0 <= off <= c.size_bytes - n * c.word_bytes:
            raise SimulationError(f"L2 range at {addr:#x} ({n} words) is outside L2")
        return off // c.word_bytes

    def load(self, addr: int, words: Any) -> None:
        """Write consecutive words from ``addr``. ``words``: [n] or [n, epw]."""
        arr = np.asarray(words, dtype=self.cfg.dtype).reshape(-1, self.cfg.elems_per_word)
        w = self._word_index(addr, len(arr))
        self.data[w : w + len(arr)] = arr

    def dump(self, addr: int, n: int) -> np.ndarray:
        """``n`` consecutive words from ``addr``, shape [n, epw]."""
        w = self._word_index(addr, n)
        return self.data[w : w + n].copy()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _beat(self, x: Any) -> np.ndarray:
        """A beat as shape (words_per_beat, epw); a scalar fills it."""
        c = self.cfg
        arr = np.asarray(x, dtype=c.dtype)
        if arr.ndim == 0:
            return np.full((c.words_per_beat, c.elems_per_word), arr, dtype=c.dtype)
        if arr.shape == (c.words_per_beat,) and c.elems_per_word == 1:  # one value per word
            arr = arr.reshape(c.words_per_beat, 1)
        if arr.shape != (c.words_per_beat, c.elems_per_word):
            raise SimulationError(
                f"beat must have shape {(c.words_per_beat, c.elems_per_word)}, got {arr.shape}"
            )
        return arr
