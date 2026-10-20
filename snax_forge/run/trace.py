"""The trace of a flow run (C§7, D137).

SNAX-GVSoC's ``events`` lines, renamed into C§7 and cut to the program's window: t = 0 is
main's entry on the accelerator's core, and events outside ``[0, total_cycles)`` are dropped.
Every event is a flat dict of ``t``, ``k``, ``src`` and the kind's own fields:

    task  start    task  (t = the cycle the start is taken; busy from t + 1)
    task  done     task, then firings, first, last for an accelerator  (t = first idle cycle)
    task  csr      pc, csr, block, op (r/w), value, held (a held write's cycles)
    task  poll     pc, csr, block, value, count, last  (t = the first read)
    task  barrier  last  (t = the barrier's read; last = the cycle it is released)
    task  state    mode (run/sleep)  (t = the first cycle in it)
    task  func     name, pc  (t = the function's first instruction)
    task  issue    task, id, dir, from, to, size, from_stride, to_stride, reps  (a DMA copy)
    task  done     task, id  (a DMA copy; t = the cycle it completes)
"""

from __future__ import annotations

LEVELS = ("off", "task", "beat")
