"""TCDM interconnect for SNAX-MODEL (MOD3, D29, D30).

What this models
----------------
The narrow TCDM interconnect between the master ports (streamer ports; see
below for the DMA) and the L1 banks. Every master port can reach every bank.
Each bank has its own arbiter, and at most one master per bank gets through
per cycle. Masters to distinct banks proceed in parallel.

Which RTL it copies
-------------------
``snax_alu_cluster.hjson`` sets ``tcdm.sparse_interconnect: true``, which
selects the Chisel ``SparseInterconnect`` in snax_cluster
(hw/chisel/src/main/scala/snax/sparse_interconnect/). ``rr_arb_tree`` is only
used by the other topology (snitch_tcdm_interconnect), so it is NOT copied.
Per bank, SparseInterconnect has ``PriorityRoundRobinArbiter`` wrapping
``RoundRobinArbiter``:

* Priority: only requests with the highest ``tcdm_priority`` (1 bit in RTL)
  take part; round-robin runs among them.
* Pointer: ``previous = RegNext(selection, N-1)``. It takes the value of the
  selection every cycle, granted or not. With no request the selection is
  ``PriorityEncoder(all zero) = N-1``, so one idle cycle on a bank resets
  its pointer and the lowest index wins next.
* Selection: if ``lock`` is set and ``previous`` still requests, select it
  again. Otherwise the lowest requester with index > ``previous``, else the
  lowest requester (wrap around).
* Lock: ``lock := valid && !ready``. Set only when the bank refused the
  selected request (the DMA owns the bank that cycle, see below). If that
  master drops its request, the lock has no effect.
* Fairness: under continuous contention a requester waits at most N-1
  cycles. It is not fair across idle cycles (pointer reset).
* Port order: snaxgen orders the inputs as accelerator TCDM ports, XDMA,
  cores, AXI. Port 0 here is the first accelerator port; ports must be
  added in the RTL order for tie-breaks to match.

Not modelled: the sparse "access granularity" (a port reaching only a subset
of banks). snax_alu uses [[12, 1]], i.e. a full crossbar.

Latency
-------
The request path is combinational (grant in the request cycle). The response
mux uses ``RegNext(bankSelect)`` and assumes the SRAM answers after exactly
one cycle; ``register_tcdm_cuts`` defaults to false. So the interconnect adds
no latency, and read latency stays in ``L1Config.read_latency`` (1 for SNAX).
The RTL also raises p_valid for writes; here only reads return data.

DMA
---
In RTL the DMA does not use this interconnect. It has a wide path per
superbank and ``mem_wide_narrow_mux`` gives it absolute priority: narrow
requests to that superbank see ready = 0. ``block_bank`` is the input for
that (MOD6); it is what makes ``lock`` reachable.

Who calls what, per cycle
-------------------------
    REQUEST    master:  request(cycle, port, req, prio)   (valid)
               DMA:     block_bank(cycle, bank)           (bank not ready)
    ARBITRATE  xbar:    per-bank selection and grants
    MEMORY     xbar:    L1Memory.request for each winner
    RESPONSE   master:  granted(cycle, port)              (ready)
               master:  rdata(cycle, port)                (read data due now)

Waking (D29)
------------
The xbar is a Component, not a touched element: arbitration needs all
requests of the cycle, so it must run at a fixed point after REQUEST. It is
awake exactly when one of its owners (the components behind its ports, plus
drivers such as the DMA) is awake: next_wake is the minimum of their
next_wake answers. Those depend only on committed state, so the minimum does
too. The xbar's own state never needs a wake of its own:

* pointers only matter in cycles with requests (an owner is awake);
* a set lock implies a refused request, which the hold rule forces its
  owner to drive again next cycle;
* outstanding reads are collected by their owner, which wakes for them.

Idle cycles reset every pointer and lock, whether they are ticked (skipping
off) or skipped (``on_gap``), so both give the same result.

Hold rule (valid/ready)
-----------------------
A refused request must be driven again, unchanged, in the next cycle.
Nothing is queued inside the xbar or the L1. With ``check_hold`` (default)
dropping or changing a refused request raises ``HoldViolation``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .mem import BankReq, BankResp, L1Memory
from .sched import Component, Phase, SimulationError


class HoldViolation(SimulationError):
    """A master dropped or changed a request that was refused last cycle."""


@dataclass(frozen=True)
class Port:
    """One master port of the interconnect."""

    index: int  # arbitration index: lower wins after an idle cycle
    owner: Component  # component that drives it; wakes the xbar
    name: str


@dataclass
class _Req:
    """A request driven this cycle (a wire)."""

    req: BankReq
    bank: int
    prio: int


class Xbar(Component):
    """Narrow TCDM interconnect: per-bank round-robin, one grant per bank per cycle.

    State, split the RTL way:

    * committed state, read by anyone at any time:
        ``prev``    per-bank round-robin pointer (last selection), [n_banks]
        ``lock``    per-bank lock (selection was refused), [n_banks]
        ``_held``   per port: request refused last cycle, for the hold check
        ``_due``    per port: (ready cycle, bank) of outstanding reads
    * this cycle's wires, cleared in ``commit``:
        ``_now``, ``_req``, ``_blocked``, ``_grant``, ``_arbitrated``,
        ``_sel``, ``_lock_next``, ``_new_due``
    * statistics (for MOD8), never read by the model itself:
        per bank: ``bank_grants``, ``bank_conflicts`` (cycles with >= 2
        requesters), ``bank_stalls`` (refused requests), ``bank_blocked``
        (cycles refused because the bank was not ready)
        per port: ``port_grants``, ``port_stalls`` (cycles refused)
    """

    phases = (Phase.ARBITRATE, Phase.MEMORY)

    def __init__(self, name: str, mem: L1Memory, check_hold: bool = True) -> None:
        super().__init__(name)
        self.mem = mem
        self.check_hold = check_hold
        self.ports: list[Port] = []
        self._owners: list[Component] = []  # unique, in order of first use
        self._frozen = False

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def add_port(self, owner: Component, name: str | None = None) -> int:
        """Add a master port driven by ``owner``. Returns its index.

        Add ports in RTL input order: the index decides round-robin
        tie-breaks.
        """
        if self._frozen:
            raise SimulationError(f"{self.name}: ports cannot be added after the run started")
        idx = len(self.ports)
        self.ports.append(Port(idx, owner, name or f"{owner.name}.{idx}"))
        self.add_driver(owner)
        return idx

    def add_driver(self, owner: Component) -> None:
        """Wake the xbar whenever ``owner`` is awake, without giving it a port.

        For components that only call ``block_bank`` (the DMA in MOD6).
        """
        if all(o is not owner for o in self._owners):
            self._owners.append(owner)

    def _freeze(self) -> None:
        """Size the state arrays once the port count is final."""
        if self._frozen:
            return
        self._frozen = True
        n, nb = len(self.ports), self.mem.cfg.n_banks
        # Committed state. N-1 is the RTL reset value: "start from 0".
        self.prev = np.full(nb, n - 1, dtype=np.int64)
        self.lock = np.zeros(nb, dtype=bool)
        self._held: list[_Req | None] = [None] * n
        self._due: list[deque[tuple[int, int]]] = [deque() for _ in range(n)]
        # Statistics.
        self.bank_grants = np.zeros(nb, dtype=np.int64)
        self.bank_conflicts = np.zeros(nb, dtype=np.int64)
        self.bank_stalls = np.zeros(nb, dtype=np.int64)
        self.bank_blocked = np.zeros(nb, dtype=np.int64)
        self.port_grants = np.zeros(n, dtype=np.int64)
        self.port_stalls = np.zeros(n, dtype=np.int64)
        self._clear_wires()

    def _clear_wires(self) -> None:
        n, nb = len(self.ports), self.mem.cfg.n_banks
        self._now: int | None = None
        self._req: list[_Req | None] = [None] * n
        self._blocked: set[int] = set()
        self._grant = np.zeros(n, dtype=bool)
        self._arbitrated = False
        # With no request a bank selects N-1 and does not lock: the default.
        self._sel = np.full(nb, n - 1, dtype=np.int64)
        self._lock_next = np.zeros(nb, dtype=bool)
        self._new_due: list[tuple[int, int, int]] = []  # (port, ready, bank)

    def _enter(self, cycle: int) -> None:
        """Called by every wire driver: the wires must belong to ``cycle``."""
        self._freeze()
        if self._now is not None and self._now != cycle:
            raise SimulationError(f"{self.name} was not committed between cycles")
        self._now = cycle

    # -------------------------------------------------------------------------
    # Master side
    # -------------------------------------------------------------------------

    def request(self, cycle: int, port: int, req: BankReq, prio: int = 0) -> None:
        """Drive ``req`` on ``port`` in ``cycle`` (valid). Call in Phase.REQUEST.

        At most one request per port per cycle. A refused request must be
        driven again, unchanged, next cycle (see the hold rule).
        """
        self._enter(cycle)
        if self._arbitrated:
            raise SimulationError(f"{self.name}: request on port {port} after arbitration")
        if self._req[port] is not None:
            raise SimulationError(f"{self.name}: second request on port {port} in cycle {cycle}")
        r = _Req(req, self.mem.bank_of(req.addr), prio)  # bank_of validates the address
        held = self._held[port]
        if self.check_hold and held is not None and not _same(held, r):
            raise HoldViolation(
                f"cycle {cycle}: port {self.ports[port].name} changed a refused request"
            )
        self._req[port] = r

    def block_bank(self, cycle: int, bank: int) -> None:
        """Bank ``bank`` is not ready for narrow requests in ``cycle``.

        Call in Phase.REQUEST. The RTL case is a DMA access to the bank's
        superbank (mem_wide_narrow_mux); MOD6 drives it.
        """
        self._enter(cycle)
        if self._arbitrated:
            raise SimulationError(f"{self.name}: block_bank after arbitration")
        self._blocked.add(bank)

    def granted(self, cycle: int, port: int) -> bool:
        """Whether ``port``'s request was accepted in ``cycle`` (ready).

        A wire from ARBITRATE: read it in MEMORY or RESPONSE of the same
        cycle. False if the port did not request.
        """
        if not self._frozen or self._now != cycle or self._req[port] is None:
            return False
        if not self._arbitrated:
            raise SimulationError(f"{self.name}: grant read before arbitration")
        return bool(self._grant[port])

    def rdata(self, cycle: int, port: int) -> BankResp | None:
        """Read data for ``port`` leaving the banks in ``cycle``, or None.

        Read in Phase.RESPONSE. The tag is the one the master put in its
        BankReq. At most one per port per cycle: one grant per cycle and a
        fixed latency.
        """
        if not self._frozen:
            return None
        bank = None
        for ready, b in self._due[port]:  # committed: reads from earlier cycles
            if ready == cycle:
                bank = b
                break
        if bank is None and self._now == cycle:  # wire: latency-0 read this cycle
            for p, ready, b in self._new_due:
                if p == port and ready == cycle:
                    bank = b
        if bank is None:
            return None
        resp = self.mem.resp(bank, cycle)
        # The L1 carries (port, user tag); anything else is a routing bug.
        if resp is None or resp.tag[0] != port:
            raise SimulationError(f"{self.name}: lost read data for port {port} in cycle {cycle}")
        return replace(resp, tag=resp.tag[1])

    def next_rdata(self, cycle: int, port: int) -> int | None:
        """Earliest cycle > ``cycle`` with read data for ``port``, or None.

        For the master's next_wake. Depends only on committed state (D29).
        """
        if not self._frozen:
            return None
        for ready, _ in self._due[port]:  # in issue order, so in ready order
            if ready > cycle:
                return ready
        return None

    # -------------------------------------------------------------------------
    # Component interface
    # -------------------------------------------------------------------------

    def tick(self, cycle: int, phase: Phase) -> None:
        if phase == Phase.ARBITRATE:
            self._enter(cycle)
            self._arbitrate(cycle)
        elif phase == Phase.MEMORY:
            self._serve(cycle)

    def _arbitrate(self, cycle: int) -> None:
        """Per-bank PriorityRoundRobinArbiter; writes grants and next state."""
        if self.check_hold:
            for p, held in enumerate(self._held):
                if held is not None and self._req[p] is None:
                    raise HoldViolation(
                        f"cycle {cycle}: port {self.ports[p].name} dropped a refused request"
                    )

        # Requesters per bank, in ascending port order.
        by_bank: dict[int, list[int]] = {}
        for p, r in enumerate(self._req):
            if r is not None:
                by_bank.setdefault(r.bank, []).append(p)

        for b, ps in by_bank.items():
            if len(ps) > 1:
                self.bank_conflicts[b] += 1
            # Priority mask: only the highest priority takes part.
            top = max(self._req[p].prio for p in ps)
            valid = [p for p in ps if self._req[p].prio == top]
            prev = int(self.prev[b])
            if self.lock[b] and prev in valid:
                sel = prev  # locked: keep the refused selection
            else:
                later = [p for p in valid if p > prev]
                sel = later[0] if later else valid[0]  # next in turn, else wrap
            self._sel[b] = sel  # becomes prev at commit, granted or not
            ready = b not in self._blocked
            if ready:
                self._grant[sel] = True
                self.bank_grants[b] += 1
                self.port_grants[sel] += 1
            else:
                self._lock_next[b] = True
                self.bank_blocked[b] += 1
            refused = [p for p in ps if not self._grant[p]]
            self.bank_stalls[b] += len(refused)
            for p in refused:
                self.port_stalls[p] += 1
        self._arbitrated = True

    def _serve(self, cycle: int) -> None:
        """Send each winner to its bank; remember where read data will appear."""
        lat = self.mem.cfg.read_latency
        for p in np.flatnonzero(self._grant):
            p = int(p)
            r = self._req[p]
            # Wrap the tag so the response can be routed back (the RTL uses
            # a registered bank select instead; same effect with latency 1).
            bank = self.mem.request(cycle, replace(r.req, tag=(p, r.req.tag)))
            if not r.req.write:
                self._new_due.append((p, cycle + lat, bank))

    def commit(self, cycle: int) -> None:
        """End of cycle: pointers and locks take their next value, wires clear."""
        self._freeze()
        self.prev[:] = self._sel
        self.lock[:] = self._lock_next
        for p, r in enumerate(self._req):
            self._held[p] = r if (r is not None and not self._grant[p]) else None
        for p, ready, bank in self._new_due:
            self._due[p].append((ready, bank))
        for q in self._due:  # drop data whose RESPONSE phase has passed
            while q and q[0][0] <= cycle:
                q.popleft()
        self._clear_wires()

    def next_wake(self, cycle: int) -> int | None:
        """Awake exactly when an owner is awake (see "Waking" in the module doc)."""
        self._freeze()
        wakes = [w for o in self._owners if (w := o.next_wake(cycle)) is not None]
        return min(wakes) if wakes else None

    def on_gap(self, start: int, stop: int) -> None:
        """Skipped cycles had no requests: every bank was idle, so reset.

        Same as ticking with no requests: selection N-1, no lock.
        """
        self._freeze()
        if self.check_hold and any(h is not None for h in self._held):
            raise HoldViolation(f"cycle {start}: a refused request was dropped (owner slept)")
        self.prev[:] = len(self.ports) - 1
        self.lock[:] = False
        self._held = [None] * len(self.ports)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Totals for a quick look; MOD8 builds the real profile."""
        self._freeze()
        return {
            "grants": int(self.bank_grants.sum()),
            "conflict_cycles": int(self.bank_conflicts.sum()),
            "stalls": int(self.bank_stalls.sum()),
            "blocked": int(self.bank_blocked.sum()),
            "stalls_per_port": {p.name: int(self.port_stalls[p.index]) for p in self.ports},
            "worst_bank": int(np.argmax(self.bank_stalls)) if self.bank_stalls.any() else None,
        }


def _same(a: _Req, b: _Req) -> bool:
    """Whether two driven requests are identical (the hold rule)."""
    x, y = a.req, b.req
    return (
        x.addr == y.addr
        and x.write == y.write
        and a.prio == b.prio
        and x.tag == y.tag
        and _eq(x.wdata, y.wdata)
        and _eq(x.strb, y.strb)
    )


def _eq(u: Any, v: Any) -> bool:
    if u is None or v is None:
        return u is v
    return bool(np.array_equal(np.asarray(u), np.asarray(v)))
