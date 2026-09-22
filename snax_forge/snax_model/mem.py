"""L1 memory banks for SNAX-MODEL (MOD2, D13, D29).

What this models
----------------
The L1 (TCDM in SNAX) is a set of single-port SRAM banks. Each bank can do
one access per cycle: either one read or one write. Many masters (streamer
ports, DMA) share the banks through the interconnect.

What this does NOT model
------------------------
Arbitration. Choosing which master gets a bank when several want it, and
stalling the others, is the interconnect's job (MOD3, a separate module).
The interconnect only passes one access per bank per cycle to this file. If
two accesses still reach the same bank in one cycle, the interconnect has a
bug, so ``request`` raises ``BankConflictError`` instead of guessing.

Who calls what, per cycle
-------------------------
The cycle phases come from ``sched.Phase``:

    REQUEST    masters drive requests            (MOD3 / MOD4 / MOD6)
    ARBITRATE  interconnect picks one per bank   (MOD3)
    MEMORY     interconnect calls L1Memory.request for each winner
    RESPONSE   interconnect reads L1Memory.responses and routes data back

The L1 is a shared state element (a ``Stateful`` in sched.py, like a FIFO),
not a ``Component``: it is never ticked and has no ``next_wake``. Instead,
``request`` calls ``cluster.touch(self)``, so the scheduler calls
``commit`` at the end of that cycle. Because the L1 is only ever changed by
someone who is awake and calling it, it can never "sleep through" a request.

Timing rules (RTL-like, D29)
----------------------------
* A read samples the storage as it was at the end of the previous cycle.
  A write issued in the same cycle, to another bank, does not affect it.
* A write is buffered and only lands in storage in ``commit``, at the end
  of the cycle. A read in the next cycle sees it.
* Read data for a request in cycle ``t`` becomes visible in cycle
  ``t + read_latency``. With latency 1 this is the registered SRAM output.
* The read latency pipeline is stored as "responses keyed by the cycle they
  become ready". That gives the same timing as a shift register of length
  ``read_latency``, without ticking every cycle to shift it.

Data layout (D13)
-----------------
The unit of access is one bank word (``width_bits`` wide). In v1 a word holds
one element. Storage is already ``[banks, rows, elems_per_word]`` and writes
take a per-element strobe, so packing several elements into a word (e.g.
8 x int8 in 64 bits) can be switched on later without changing the interface.

Addresses are byte addresses of whole words, starting at ``base_addr``
(0 by default).

Bank groups (MOD6, D33)
-----------------------
A master port can be wider than a bank. A port of ``w`` bits covers
``g = w / width_bits`` consecutive banks in an aligned group, and each of
its accesses is one access on every bank of the group. In SNAX the DMA
uses a 512-bit port: 8 banks of 64 bits, one "superbank".
``wide_bits`` is the widest port allowed (512 in SNAX). ``group_banks``
checks a port width and ``group_of`` checks a wide access. The L1 itself
still sees one access per bank: the interconnect splits a wide grant into
``g`` calls to ``request``, so the D30 check stays as it is.

Whether ``n_banks`` is a multiple of ``g`` is checked when a port of that
width is added (``Xbar.add_port``), not here: an L1 with 4 banks is fine as
long as no 512-bit port is attached to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .config import Config
from .sched import SimulationError

# Default sizes follow snax_cluster target/snitch_cluster/cfg/snax_alu_cluster.hjson:
# 128 KiB TCDM, 32 banks, 64-bit data. The base address defaults to 0 so the
# model does not depend on where a given cluster maps its TCDM.


class BankConflictError(SimulationError):
    """Two accesses reached the same bank in the same cycle.

    Should never happen once the interconnect (MOD3) is in place; it means
    arbitration let two masters through to one bank.
    """


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class L1Config(Config):
    """Size and timing of the L1. Frozen: it describes the storage array,
    which is built once from it, so it must not change afterwards.
    """

    n_banks: int = 32  # number of banks
    width_bits: int = 64  # bits per bank word
    rows: int = 512  # words per bank; 32 * 512 * 8 B = 128 KiB
    read_latency: int = 1  # cycles from read request to data out
    dtype: str = "int64"  # element type (any NumPy dtype name)
    elems_per_word: int = 1  # elements packed in one word; 1 in v1
    base_addr: int = 0  # byte address of the first L1 word
    wide_bits: int = 512  # widest port (the DMA's): one superbank of 8 banks in SNAX

    def __post_init__(self) -> None:
        # Reject configs that cannot describe a real memory.
        if self.n_banks < 1 or self.rows < 1:
            raise ValueError("n_banks and rows must be >= 1")
        if self.width_bits % 8:
            raise ValueError("width_bits must be a multiple of 8")
        if self.read_latency < 0:
            raise ValueError("read_latency must be >= 0")
        if self.elems_per_word < 1:
            raise ValueError("elems_per_word must be >= 1")
        # All elements of one word must fit in the bank width.
        elem_bits = np.dtype(self.dtype).itemsize * 8
        if elem_bits * self.elems_per_word > self.width_bits:
            raise ValueError(
                f"{self.elems_per_word} x {self.dtype} does not fit in {self.width_bits} bits"
            )
        # The widest port must itself be a valid port width.
        _group_size(self.wide_bits, self.width_bits, "wide_bits")

    # Derived values are properties, not fields, so they can never
    # disagree with the fields they come from.

    @property
    def word_bytes(self) -> int:
        """Bytes per bank word; also the address step between words."""
        return self.width_bits // 8

    @property
    def size_bytes(self) -> int:
        """Total L1 size in bytes."""
        return self.n_banks * self.rows * self.word_bytes

    def group_banks(self, width_bits: int) -> int:
        """Banks covered by one access of a ``width_bits`` port (1 for a narrow port).

        The width must be a power-of-two multiple of the bank width and at
        most ``wide_bits``: 64 -> 1, 128 -> 2, 256 -> 4, 512 -> 8 banks.
        """
        g = _group_size(width_bits, self.width_bits, "port width")
        if width_bits > self.wide_bits:
            raise ValueError(f"port width {width_bits} is wider than wide_bits = {self.wide_bits}")
        return g


def _group_size(width_bits: int, bank_bits: int, what: str) -> int:
    """``width_bits / bank_bits`` if it is a power of two >= 1, else ValueError."""
    g, rest = divmod(width_bits, bank_bits)
    if width_bits < 1 or rest or g & (g - 1):
        raise ValueError(
            f"{what} {width_bits} must be a power-of-two multiple of the bank width {bank_bits}"
        )
    return g


# =============================================================================
# Address mapping: word index <-> (bank, row)
# =============================================================================


class AddressMap(Protocol):
    """How word indices are spread over the banks.

    A "word index" is ``(addr - base_addr) / word_bytes``: 0 for the first
    word of L1, 1 for the next, and so on. Any class with these two methods
    can be passed to ``L1Memory`` (no subclassing needed).
    """

    def decode(self, word: int) -> tuple[int, int]:
        """Word index -> (bank, row)."""
        ...

    def encode(self, bank: int, row: int) -> int:
        """(bank, row) -> word index. Inverse of ``decode``."""
        ...


@dataclass(frozen=True)
class WordInterleaved:
    """Consecutive words go to consecutive banks (the SNAX TCDM layout).

    With 4 banks, words 0 1 2 3 are row 0 of banks 0 1 2 3, words 4 5 6 7
    are row 1, and so on. A unit-stride stream therefore touches every bank
    once before it comes back to the first one.
    """

    n_banks: int

    def decode(self, word: int) -> tuple[int, int]:
        return word % self.n_banks, word // self.n_banks

    def encode(self, bank: int, row: int) -> int:
        return row * self.n_banks + bank


# =============================================================================
# Requests and responses
# =============================================================================


@dataclass
class BankReq:
    """One word access, as the interconnect hands it to a bank.

    ``tag`` is not used here. It is carried into the response so that the
    interconnect can route read data back to the right master.
    """

    addr: int  # byte address of the word, aligned to word_bytes
    write: bool = False  # False = read, True = write
    wdata: Any = None  # writes only: scalar, or one value per element
    strb: Any = None  # writes only: which elements to write; None = all
    tag: Any = None  # opaque to the L1


@dataclass
class BankResp:
    """Read data leaving a bank."""

    bank: int  # bank that served the read
    tag: Any  # copied from the request
    data: np.ndarray  # the word, shape (elems_per_word,)
    issued: int  # cycle the read was served


@dataclass
class _PendingWrite:
    """A write accepted this cycle, applied in ``commit``."""

    bank: int
    row: int
    wdata: np.ndarray  # shape (elems_per_word,)
    strb: np.ndarray  # bool, shape (elems_per_word,)


# =============================================================================
# The L1 memory
# =============================================================================


@dataclass(eq=False)  # eq=False: compare by identity, like the scheduler's touch()
class L1Memory:
    """All L1 banks together.

    State, split the RTL way:

    * committed state, which anyone may read at any time:
        ``data``      the storage, [banks, rows, elems_per_word]
        ``_resp``     read data waiting to leave the banks
    * this cycle's uncommitted state, cleared in ``commit``:
        ``_now``      cycle being served (None between cycles)
        ``_busy``     banks already accessed this cycle
        ``_writes``   writes to apply at the end of the cycle
    * statistics (for MOD8), never read by the model itself:
        ``reads``, ``writes``   per-bank access counts
    """

    cluster: Any  # anything with touch(elem): a Cluster or a Scheduler
    cfg: L1Config = field(default_factory=L1Config)
    amap: AddressMap | None = None  # None = WordInterleaved(cfg.n_banks)

    def __post_init__(self) -> None:
        c = self.cfg
        if self.amap is None:
            self.amap = WordInterleaved(c.n_banks)
        # Hot-path copies of derived config values, bound once (D48): locate()
        # runs several times per access, and properties are not free.
        self._wb = c.word_bytes
        self._size = c.size_bytes
        self._base = c.base_addr
        self._decode = self.amap.decode
        self._all_strb = np.ones(c.elems_per_word, dtype=bool)  # "write every element"
        self.data = np.zeros((c.n_banks, c.rows, c.elems_per_word), dtype=c.dtype)
        self.reads = np.zeros(c.n_banks, dtype=np.int64)
        self.writes = np.zeros(c.n_banks, dtype=np.int64)
        self._now: int | None = None
        self._busy: set[int] = set()
        self._writes: list[_PendingWrite] = []
        # ready cycle -> {bank -> response}. At most one response per bank
        # per ready cycle, because a bank serves one access per cycle and
        # the latency is fixed.
        self._resp: dict[int, dict[int, BankResp]] = {}

    # -------------------------------------------------------------------------
    # Addressing
    # -------------------------------------------------------------------------

    def locate(self, addr: int) -> tuple[int, int]:
        """(bank, row) of a word address. Raises on misaligned or out-of-range."""
        c = self.cfg
        off = addr - self._base  # byte offset inside L1
        if off % self._wb:
            raise SimulationError(f"L1 address {addr:#x} is not word aligned")
        if not 0 <= off < self._size:
            raise SimulationError(f"L1 address {addr:#x} is outside L1")
        bank, row = self._decode(off // self._wb)
        # Guard against a custom address map that returns nonsense.
        if not (0 <= bank < c.n_banks and 0 <= row < c.rows):
            raise SimulationError(f"address map gave bank {bank}, row {row} for {addr:#x}")
        return bank, row

    def bank_of(self, addr: int) -> int:
        """Bank of a word address. The interconnect uses this to arbitrate."""
        return self.locate(addr)[0]

    def group_of(self, addr: int, width_bits: int) -> tuple[int, ...]:
        """Banks accessed by a ``width_bits`` access at ``addr``, in lane order.

        Lane i of the access is the word at ``addr + i * word_bytes``. The
        access must be aligned to its width, and its words must fill one
        aligned bank group in one row. With the word-interleaved map every
        aligned address passes; the check guards custom address maps.
        Raises SimulationError otherwise (misaligned, outside L1, or a map
        that scatters the words).
        """
        c = self.cfg
        g = c.group_banks(width_bits)
        if g == 1:
            return (self.bank_of(addr),)
        if (addr - c.base_addr) % (g * c.word_bytes):
            raise SimulationError(f"L1 address {addr:#x} is not aligned to {width_bits} bits")
        where = [self.locate(addr + i * c.word_bytes) for i in range(g)]
        banks = tuple(b for b, _ in where)
        first = banks[0]
        if first % g or banks != tuple(range(first, first + g)) or len({r for _, r in where}) != 1:
            raise SimulationError(
                f"L1 address {addr:#x}: the address map does not put this {width_bits}-bit "
                f"access in one aligned group of {g} banks"
            )
        return banks

    def addr_of(self, bank: int, row: int) -> int:
        """Byte address of (bank, row). Inverse of ``locate``."""
        c = self.cfg
        return c.base_addr + self.amap.encode(bank, row) * c.word_bytes

    # -------------------------------------------------------------------------
    # Timed port: use during a run
    # -------------------------------------------------------------------------

    def request(self, cycle: int, req: BankReq) -> int:
        """Serve one access in ``cycle``. Returns the bank that served it.

        Call in Phase.MEMORY, after arbitration, at most once per bank per
        cycle.
        """
        # Every cycle with accesses must end in commit() before the next
        # cycle's accesses. If not, _busy and _writes would mix two cycles.
        if self._now is not None and self._now != cycle:
            raise SimulationError("L1 was not committed between cycles")

        bank, row = self.locate(req.addr)

        # One access per bank per cycle. The interconnect should make this
        # impossible; if it happens anyway, stop instead of hiding the bug.
        if bank in self._busy:
            raise BankConflictError(
                f"cycle {cycle}: second access to bank {bank} (addr {req.addr:#x})"
            )
        self._now = cycle
        self._busy.add(bank)
        self.cluster.touch(self)  # makes the scheduler call commit() at cycle end

        if req.write:
            # Buffer the write; storage changes only in commit().
            self._writes.append(
                _PendingWrite(bank, row, self._word(req.wdata), self._strobe(req.strb))
            )
            self.writes[bank] += 1
        else:
            # Sample storage now (end-of-previous-cycle state) and hold the
            # data until it is due. copy(): later writes must not change it.
            ready = cycle + self.cfg.read_latency
            resp = BankResp(bank, req.tag, self.data[bank, row].copy(), cycle)
            self._resp.setdefault(ready, {})[bank] = resp
            self.reads[bank] += 1
        return bank

    def resp(self, bank: int, cycle: int) -> BankResp | None:
        """Read data leaving ``bank`` in ``cycle``, or None. Read in Phase.RESPONSE."""
        return self._resp.get(cycle, {}).get(bank)

    def responses(self, cycle: int) -> list[BankResp]:
        """All read data leaving the banks in ``cycle``, in bank order."""
        by_bank = self._resp.get(cycle, {})
        return [by_bank[b] for b in sorted(by_bank)]

    def next_response(self, cycle: int) -> int | None:
        """Earliest cycle > ``cycle`` in which read data leaves a bank, or None.

        For the requester's ``next_wake``: it must be awake in that cycle to
        pick the data up. Depends only on committed state, so asking again
        before that cycle gives the same answer (D29).
        """
        later = [t for t in self._resp if t > cycle]
        return min(later) if later else None

    def commit(self) -> None:
        """End of cycle: apply writes, drop delivered read data, free the banks.

        Called by the scheduler, only in cycles where request() touched us.
        """
        # Apply buffered writes, only to the strobed elements of each word.
        for w in self._writes:
            self.data[w.bank, w.row, w.strb] = w.wdata[w.strb]
        self._writes.clear()

        # Read data due in this cycle or earlier has had its RESPONSE phase.
        # (Data due in a cycle without accesses is dropped at the next
        # commit instead; harmless, since resp() looks up exact cycles.)
        if self._now is not None:
            for t in [t for t in self._resp if t <= self._now]:
                del self._resp[t]

        self._busy.clear()
        self._now = None

    # -------------------------------------------------------------------------
    # Backdoor: setup and inspection outside a run.
    # No timing, no counters, no conflict checks.
    # -------------------------------------------------------------------------

    def peek(self, addr: int) -> np.ndarray:
        """The word at ``addr``, shape (elems_per_word,)."""
        bank, row = self.locate(addr)
        return self.data[bank, row].copy()

    def poke(self, addr: int, word: Any) -> None:
        """Overwrite the word at ``addr`` (scalar or one value per element)."""
        bank, row = self.locate(addr)
        self.data[bank, row] = self._word(word)

    def load(self, addr: int, words: Any) -> None:
        """Write consecutive words starting at ``addr``. ``words``: [n] or [n, epw].

        "Consecutive" means consecutive addresses; with the default map they
        land in consecutive banks.
        """
        arr = np.asarray(words, dtype=self.cfg.dtype).reshape(-1, self.cfg.elems_per_word)
        for i, w in enumerate(arr):
            self.poke(addr + i * self.cfg.word_bytes, w)

    def dump(self, addr: int, n: int) -> np.ndarray:
        """Read ``n`` consecutive words starting at ``addr``, shape [n, epw].

        Every run dumps the whole L1 (MOD9), so with the default map this is
        one reshape instead of a Python loop over every word (D48). A custom
        address map falls back to the loop.
        """
        c = self.cfg
        if n > 0 and isinstance(self.amap, WordInterleaved) and self.amap.n_banks == c.n_banks:
            self.locate(addr)  # same checks as peek: alignment and range
            self.locate(addr + (n - 1) * c.word_bytes)
            w0 = (addr - c.base_addr) // c.word_bytes
            # [banks, rows, epw] -> word order (row * n_banks + bank) of the map.
            flat = self.data.transpose(1, 0, 2).reshape(-1, c.elems_per_word)
            return flat[w0 : w0 + n].copy()
        return np.stack([self.peek(addr + i * c.word_bytes) for i in range(n)])

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _word(self, x: Any) -> np.ndarray:
        """Turn a scalar or sequence into one word: shape (elems_per_word,)."""
        epw = self.cfg.elems_per_word
        arr = np.asarray(x, dtype=self.cfg.dtype)
        if arr.ndim == 0:  # scalar: fill every element with it
            arr = np.full(epw, arr, dtype=self.cfg.dtype)
        if arr.shape != (epw,):
            raise SimulationError(f"word must have {epw} elements, got shape {arr.shape}")
        return arr

    def _strobe(self, strb: Any) -> np.ndarray:
        """Turn a strobe into a bool mask of shape (elems_per_word,); None = all."""
        epw = self.cfg.elems_per_word
        if strb is None:
            return self._all_strb  # shared, never written to
        arr = np.asarray(strb, dtype=bool)
        if arr.shape != (epw,):
            raise SimulationError(f"strobe must have {epw} entries, got shape {arr.shape}")
        return arr
