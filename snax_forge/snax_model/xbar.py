"""TCDM interconnect for SNAX-MODEL (MOD3, MOD6, D29, D30, D31, D33).

What this models
----------------
The TCDM interconnect between the master ports (streamer ports, the DMA) and
the L1 banks. Every master port can reach every bank. Masters to distinct
banks proceed in parallel.

Ports have a width (D33). A narrow port (the bank width, 64 bits in SNAX)
accesses one bank. A port of ``w`` bits accesses the ``g = w / 64`` banks
of an aligned group at once: 128 bits = 2 banks, 256 = 4, 512 = 8 (one
superbank, the DMA). A wide request is granted only if every bank of its
group is free this cycle, and a grant is one access on each of those
banks, so the L1 still sees at most one access per bank per cycle (D30).

Which RTL it copies
-------------------
Narrow ports: ``snax_alu_cluster.hjson`` sets ``tcdm.sparse_interconnect:
true``, which selects the Chisel ``SparseInterconnect`` in snax_cluster
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
  selected request (a wider port owns the bank that cycle, see below). If
  that master drops its request, the lock has no effect.
* Fairness: under continuous contention a requester waits at most N-1
  cycles. It is not fair across idle cycles (pointer reset).
* Port order: snaxgen orders the inputs as accelerator TCDM ports, XDMA,
  cores, AXI. Port 0 here is the first accelerator port; ports must be
  added in the RTL order for tie-breaks to match.

Wide port: in snitch_cluster.sv the DMA reaches each superbank through
``mem_wide_narrow_mux`` with ``sel_wide_i = DMA q_valid``. In a cycle in
which the DMA drives a request to a superbank, all 8 narrow ports of that
superbank see ``q_ready = 0``, and the wide request is granted at once
(the module asserts this). The narrow arbiter still selects and updates
its pointer, and the refusal sets its lock. So: wider wins, absolutely,
per cycle and per bank group; narrow ports to other superbanks, and to the
same superbank in cycles the DMA does not use it, are not affected.

The policy (D33), all in ``_priority``:

* widths are served widest first; a request is refused if a wider grant
  already took a bank of its group (counted in ``port_stalls_wider``);
* among requests of equal width on the same group, the D31 arbiter above
  decides, with one pointer and one lock per (width, group). For 64-bit
  ports that is exactly the per-bank arbiter of MOD3.

Only 64 (streamers) and 512 (DMA) are used in SNAX; 128 and 256 are hooks.
A share-based priority (N wide and M narrow accesses per group) is open
item 6: only ``_priority`` would change.

Not modelled: the sparse "access granularity" (a port reaching only a subset
of banks). snax_alu uses [[12, 1]], i.e. a full crossbar.

Latency
-------
The request path is combinational (grant in the request cycle). The response
mux uses ``RegNext(bankSelect)`` and assumes the SRAM answers after exactly
one cycle; ``register_tcdm_cuts`` defaults to false. So the interconnect adds
no latency, and read latency stays in ``L1Config.read_latency`` (1 for SNAX).
The RTL also raises p_valid for writes; here only reads return data.

Who calls what, per cycle
-------------------------
    REQUEST    master:  request(cycle, port, req, prio)   (valid)
    ARBITRATE  xbar:    _priority: grants per bank group
    MEMORY     xbar:    L1Memory.request for each bank of each winner
    RESPONSE   master:  granted(cycle, port)              (ready)
               master:  rdata(cycle, port)                (read data due now)

A wide request is a ``BankReq`` at the group's first word; for a write,
``wdata`` / ``strb`` hold one entry per lane (shape [g] or [g, epw], or a
scalar for all lanes). Wide read data has shape [g, epw]; narrow read data
keeps shape [epw].

Waking (D29)
------------
The xbar is a Component, not a touched element: arbitration needs all
requests of the cycle, so it must run at a fixed point after REQUEST. It is
awake exactly when one of its port owners is awake: next_wake is the
minimum of their next_wake answers. Those depend only on committed state,
so the minimum does too. The xbar's own state never needs a wake of its own:

* pointers only matter in cycles with requests (an owner is awake);
* a set lock implies a refused request, which the hold rule forces its
  owner to drive again next cycle;
* outstanding reads are collected by their owner, which wakes for them.

Trace (MOD8, D39)
-----------------
At beat level, ``commit`` emits one event per port that requested in the
cycle: ``grant`` or ``stall`` (``wider`` from the ``_wider`` wire). These are
the L1 accesses: the grant is served in the same cycle, one access on each
bank of its group, so there is no separate access event. ``row`` comes from
the L1 address map (a pure lookup). The events come from the wires, the
statistics from ``_priority``; the tests compare the two.

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
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mem import BankReq, BankResp, L1Memory
from .sched import Component, Phase, SimulationError
from .trace import Grant, Stall


class HoldViolation(SimulationError):
    """A master dropped or changed a request that was refused last cycle."""


@dataclass(frozen=True)
class Port:
    """One master port of the interconnect."""

    index: int  # arbitration index: lower wins after an idle cycle
    owner: Component  # component that drives it; wakes the xbar
    name: str
    width_bits: int = 64  # port width; one bank for a narrow port
    group: int = 1  # banks per access: width_bits / bank width


@dataclass
class _Req:
    """A request driven this cycle (a wire)."""

    req: BankReq
    banks: tuple[int, ...]  # banks of its group, in lane order
    grp: int  # group index at this port's width: banks[0] // group
    prio: int
    wdata: list[Any]  # per lane (writes), or [None] * group
    strb: list[Any]  # per lane (writes), or [None] * group


class Xbar(Component):
    """TCDM interconnect: per-group arbitration, one access per bank per cycle.

    State, split the RTL way:

    * committed state, read by anyone at any time:
        ``prev[g]``  round-robin pointer (last selection) per group of g banks
        ``lock[g]``  lock (selection was refused) per group of g banks
                     (``prev[1]`` / ``lock[1]`` are the per-bank ones of MOD3)
        ``_held``    per port: request refused last cycle, for the hold check
        ``_due``     per port: (ready cycle, banks) of outstanding reads
    * this cycle's wires, cleared in ``commit``:
        ``_now``, ``_req``, ``_grant``, ``_wider`` (refused by a wider
        grant), ``_arbitrated``, ``_sel``, ``_lock_next``, ``_new_due``.
        ``_sel`` / ``_lock_next`` are sparse: only the (width, group) pairs
        that had a request this cycle; every other group takes the idle
        default in ``commit``.
    * statistics (for MOD8), never read by the model itself:
        per bank: ``bank_grants`` (accesses; a wide grant counts on each of
        its banks), ``bank_conflicts`` (cycles with >= 2 requests covering
        the bank), ``bank_stalls`` (refused requests covering the bank),
        ``bank_blocked`` (arbitrations refused because a wider grant used
        the bank)
        per port: ``port_grants``, ``port_stalls`` (cycles refused),
        ``port_stalls_wider`` (the part of ``port_stalls`` lost to a wider
        grant; the rest is same-width contention)

    Everything touched every cycle (pointers, locks, wires, the counters) is
    a plain Python list of ints or bools, not a NumPy array: at cluster sizes
    (tens of banks and ports) the array call overhead dominates, and this is
    the hottest path of the model (D48). The counters are still *read* as
    arrays through the properties below, so nothing outside changes. Arrays
    stay where they hold data (``L1Memory.data``) or address streams.
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

    def add_port(
        self, owner: Component, name: str | None = None, width_bits: int | None = None
    ) -> int:
        """Add a master port driven by ``owner``. Returns its index.

        ``width_bits`` defaults to the bank width (64 bits: a narrow port).
        It must be a power-of-two multiple of the bank width, at most
        ``L1Config.wide_bits``, and ``n_banks`` must be a multiple of the
        group it covers (ValueError otherwise).

        Add ports in RTL input order: the index decides round-robin
        tie-breaks among ports of equal width.
        """
        if self._frozen:
            raise SimulationError(f"{self.name}: ports cannot be added after the run started")
        cfg = self.mem.cfg
        width = cfg.width_bits if width_bits is None else width_bits
        g = cfg.group_banks(width)
        if cfg.n_banks % g:
            raise ValueError(
                f"{self.name}: a {width}-bit port covers {g} banks; "
                f"n_banks = {cfg.n_banks} is not a multiple of {g}"
            )
        idx = len(self.ports)
        self.ports.append(Port(idx, owner, name or f"{owner.name}.{idx}", width, g))
        if all(o is not owner for o in self._owners):
            self._owners.append(owner)
        return idx

    def _freeze(self) -> None:
        """Size the state arrays once the port count is final."""
        if self._frozen:
            return
        self._frozen = True
        n, nb = len(self.ports), self.mem.cfg.n_banks
        # Group sizes in use, widest first: the order of _priority.
        self._groups = sorted({p.group for p in self.ports} | {1}, reverse=True)
        # Committed state. N-1 is the RTL reset value: "start from 0".
        self.prev = {g: [n - 1] * (nb // g) for g in self._groups}
        self.lock = {g: [False] * (nb // g) for g in self._groups}
        # The value a group takes in a cycle without a request, copied in bulk.
        self._prev_idle = {g: list(v) for g, v in self.prev.items()}
        self._lock_idle = {g: list(v) for g, v in self.lock.items()}
        self._held: list[_Req | None] = [None] * n
        self._due: list[deque[tuple[int, tuple[int, ...]]]] = [deque() for _ in range(n)]
        # Statistics (read as arrays through the properties below).
        self._bank_grants = [0] * nb
        self._bank_conflicts = [0] * nb
        self._bank_stalls = [0] * nb
        self._bank_blocked = [0] * nb
        self._port_grants = [0] * n
        self._port_stalls = [0] * n
        self._port_stalls_wider = [0] * n
        self._clear_wires()

    # Counters: kept as lists, read as arrays (the model never reads them).

    @property
    def bank_grants(self) -> np.ndarray:
        return np.asarray(self._bank_grants, dtype=np.int64)

    @property
    def bank_conflicts(self) -> np.ndarray:
        return np.asarray(self._bank_conflicts, dtype=np.int64)

    @property
    def bank_stalls(self) -> np.ndarray:
        return np.asarray(self._bank_stalls, dtype=np.int64)

    @property
    def bank_blocked(self) -> np.ndarray:
        return np.asarray(self._bank_blocked, dtype=np.int64)

    @property
    def port_grants(self) -> np.ndarray:
        return np.asarray(self._port_grants, dtype=np.int64)

    @property
    def port_stalls(self) -> np.ndarray:
        return np.asarray(self._port_stalls, dtype=np.int64)

    @property
    def port_stalls_wider(self) -> np.ndarray:
        return np.asarray(self._port_stalls_wider, dtype=np.int64)

    def _clear_wires(self) -> None:
        n = len(self.ports)
        self._now: int | None = None
        self._req: list[_Req | None] = [None] * n
        self._grant = [False] * n
        self._wider = [False] * n
        self._arbitrated = False
        # Sparse: (width, group) -> selection / lock, only where there was a
        # request. Every other group takes the idle default (N-1, no lock).
        self._sel: dict[tuple[int, int], int] = {}
        self._lock_next: set[tuple[int, int]] = set()
        self._new_due: list[tuple[int, int, tuple[int, ...]]] = []  # (port, ready, banks)

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

        At most one request per port per cycle. A wide port's address must be
        aligned to its width. A refused request must be driven again,
        unchanged, next cycle (see the hold rule).
        """
        self._enter(cycle)
        if self._arbitrated:
            raise SimulationError(f"{self.name}: request on port {port} after arbitration")
        if self._req[port] is not None:
            raise SimulationError(f"{self.name}: second request on port {port} in cycle {cycle}")
        p = self.ports[port]
        banks = self.mem.group_of(req.addr, p.width_bits)  # validates address and alignment
        if req.write:
            wdata, strb = _lanes(req.wdata, p.group, "wdata"), _lanes(req.strb, p.group, "strb")
        else:
            wdata = strb = [None] * p.group
        r = _Req(req, banks, banks[0] // p.group, prio, wdata, strb)
        held = self._held[port]
        if self.check_hold and held is not None and not _same(held, r):
            raise HoldViolation(
                f"cycle {cycle}: port {self.ports[port].name} changed a refused request"
            )
        self._req[port] = r

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
        fixed latency. A wide port gets data of shape [group, epw], lane i
        from its i-th bank; ``bank`` is the group's first bank.
        """
        if not self._frozen:
            return None
        banks = None
        for ready, bs in self._due[port]:  # committed: reads from earlier cycles
            if ready == cycle:
                banks = bs
                break
        if banks is None and self._now == cycle:  # wire: latency-0 read this cycle
            for p, ready, bs in self._new_due:
                if p == port and ready == cycle:
                    banks = bs
        if banks is None:
            return None
        resps = [self.mem.resp(b, cycle) for b in banks]
        # The L1 carries (port, user tag); anything else is a routing bug.
        if any(r is None or r.tag[0] != port for r in resps):
            raise SimulationError(f"{self.name}: lost read data for port {port} in cycle {cycle}")
        first = resps[0]
        if len(resps) == 1:
            return BankResp(first.bank, first.tag[1], first.data, first.issued)
        return BankResp(first.bank, first.tag[1], np.stack([r.data for r in resps]), first.issued)

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
        """Hold check and conflict counts, then the policy."""
        if self.check_hold:
            for p, held in enumerate(self._held):
                if held is not None and self._req[p] is None:
                    raise HoldViolation(
                        f"cycle {cycle}: port {self.ports[p].name} dropped a refused request"
                    )
        cover: dict[int, int] = {}  # requests per bank, this cycle only
        for r in self._req:
            if r is not None:
                for b in r.banks:
                    cover[b] = cover.get(b, 0) + 1
        for b, c in cover.items():
            if c >= 2:
                self._bank_conflicts[b] += 1
        self._priority(cycle)
        self._arbitrated = True

    # --- Policy (D33). A share-based priority manager (open item 6) replaces this.

    def _priority(self, cycle: int) -> None:
        """Who wins in ``cycle``: wider first, then D31 per (width, group).

        Writes the grants, the selections (next pointers) and the next
        locks, and the grant/stall statistics.
        """
        by_group: dict[tuple[int, int], list[int]] = {}  # (g, group) -> ports, ascending
        for p, r in enumerate(self._req):
            if r is not None:
                by_group.setdefault((self.ports[p].group, r.grp), []).append(p)

        taken: set[int] = set()  # banks granted this cycle
        for g, grp in sorted(by_group, key=lambda k: (-k[0], k[1])):  # widest first
            ps = by_group[(g, grp)]
            banks = range(grp * g, grp * g + g)
            sel = self._select(g, grp, ps)
            self._sel[(g, grp)] = sel  # becomes prev at commit, granted or not
            if any(b in taken for b in banks):  # a wider grant owns a bank: not ready
                self._lock_next.add((g, grp))
                for b in banks:
                    self._bank_blocked[b] += 1
                for p in ps:
                    self._port_stalls_wider[p] += 1
                    self._wider[p] = True
            else:
                self._grant[sel] = True
                taken.update(banks)
                for b in banks:
                    self._bank_grants[b] += 1
                self._port_grants[sel] += 1
            for p in ps:
                if not self._grant[p]:
                    self._port_stalls[p] += 1
                    for b in banks:
                        self._bank_stalls[b] += 1

    def _select(self, g: int, grp: int, ps: list[int]) -> int:
        """PriorityRoundRobinArbiter of one (width, group) among requesters ``ps``."""
        top = max(self._req[p].prio for p in ps)  # priority mask
        valid = [p for p in ps if self._req[p].prio == top]
        prev = self.prev[g][grp]
        if self.lock[g][grp] and prev in valid:
            return prev  # locked: keep the refused selection
        later = [p for p in valid if p > prev]
        return later[0] if later else valid[0]  # next in turn, else wrap

    def _serve(self, cycle: int) -> None:
        """Send each winner to its banks; remember where read data will appear."""
        lat = self.mem.cfg.read_latency
        wb = self.mem.cfg.word_bytes
        for p, granted in enumerate(self._grant):
            if not granted:
                continue
            r = self._req[p]
            # Wrap the tag so the response can be routed back (the RTL uses
            # a registered bank select instead; same effect with latency 1).
            tag = (p, r.req.tag)
            if len(r.banks) == 1:
                q = r.req  # same request with the routing tag (no dataclasses.replace: hot)
                self.mem.request(cycle, BankReq(q.addr, q.write, q.wdata, q.strb, tag))
            else:  # one access per bank of the group, lane i at word i
                for i in range(len(r.banks)):
                    sub = BankReq(r.req.addr + i * wb, r.req.write, r.wdata[i], r.strb[i], tag)
                    self.mem.request(cycle, sub)
            if not r.req.write:
                self._new_due.append((p, cycle + lat, r.banks))

    def commit(self, cycle: int) -> None:
        """End of cycle: pointers and locks take their next value, wires clear."""
        self._freeze()
        if self._trace is not None and self._trace.beat:
            self._emit(cycle)
        for g in self._groups:  # every group idle by default, then this cycle's
            self.prev[g][:] = self._prev_idle[g]
            self.lock[g][:] = self._lock_idle[g]
        for (g, grp), sel in self._sel.items():
            self.prev[g][grp] = sel
        for g, grp in self._lock_next:
            self.lock[g][grp] = True
        for p, r in enumerate(self._req):
            self._held[p] = r if (r is not None and not self._grant[p]) else None
        for p, ready, banks in self._new_due:
            self._due[p].append((ready, banks))
        for q in self._due:  # drop data whose RESPONSE phase has passed
            while q and q[0][0] <= cycle:
                q.popleft()
        self._clear_wires()

    def _emit(self, cycle: int) -> None:
        """Beat-level trace: one grant or stall per requesting port, in port order."""
        tr = self._trace
        for p, r in enumerate(self._req):
            if r is None:
                continue
            port = self.ports[p]
            kw = {
                "port": port.name,
                "mem": "l1",
                "w": bool(r.req.write),
                "addr": int(r.req.addr),
                "banks": tuple(int(b) for b in r.banks),
                "row": int(self.mem.locate(r.req.addr)[1]),  # same row on every bank
            }
            if self._grant[p]:
                tr.emit(Grant(cycle, self.name, **kw))
            else:
                tr.emit(Stall(cycle, self.name, wider=bool(self._wider[p]), **kw))

    def next_wake(self, cycle: int) -> int | None:
        """Awake exactly when an owner is awake (see "Waking" in the module doc)."""
        self._freeze()
        wakes = [w for o in self._owners if (w := o.next_wake(cycle)) is not None]
        return min(wakes) if wakes else None

    def on_gap(self, start: int, stop: int) -> None:
        """Skipped cycles had no requests: every group was idle, so reset.

        Same as ticking with no requests: selection N-1, no lock.
        """
        self._freeze()
        if self.check_hold and any(h is not None for h in self._held):
            raise HoldViolation(f"cycle {start}: a refused request was dropped (owner slept)")
        for g in self._groups:
            self.prev[g][:] = self._prev_idle[g]
            self.lock[g][:] = self._lock_idle[g]
        self._held = [None] * len(self.ports)


def _lanes(x: Any, g: int, what: str) -> list[Any]:
    """Split a write's ``wdata`` / ``strb`` into one entry per lane.

    For a narrow port the value is passed through unchanged. For a wide
    port: None or a scalar applies to every lane; otherwise the first axis
    must have ``g`` entries.
    """
    if g == 1:
        return [x]
    if x is None or np.ndim(x) == 0:
        return [x] * g
    arr = np.asarray(x)
    if arr.shape[0] != g:
        raise SimulationError(f"{what} has {arr.shape[0]} lanes, the port has {g}")
    return [arr[i] for i in range(g)]


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
