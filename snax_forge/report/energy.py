"""``energy.md``: a flow's energy run (flow/energy.py) for people and LLMs.

``render_energy(run)`` gives each window of the energy run with its energy in the five parts,
the static estimate's one run of the accelerator beside it, and the counts each part was priced
on, the run windows' (or the whole run's) against the static estimate's. Every number is
``energy.json``'s.
"""

from __future__ import annotations

from snax_forge.estimate.cost import PARTS, dec3
from snax_forge.flow.energy import EnergyRun

ENERGY_MD = "energy.md"
CAVEAT = ("First-order and dynamic: the model's counts in each window priced at the static "
          "estimate's prices per transaction (a refused request at a granted one's price, held "
          "and idle cycles free); the tables were measured on random data. Aimed at trends over a "
          "sweep, not at a number.")  # fmt: skip
COUNTS = (("fires", "datapath: fires, loads and pushes"),
          ("cycles", "control: busy cycles"),
          ("beats", "streamers: beats"),
          ("requests", "crossbar: requests"),
          ("reads", "memory: reads"),
          ("writes", "memory: writes"))  # fmt: skip


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _count(x: float) -> str:
    return f"{x:g}" if isinstance(x, int) or x == int(x) else f"{x:.2f}"


def render_energy(run: EnergyRun) -> str:
    """The report (module doc)."""
    heads = ["Window", "Kind", "Label", "Cycles", *PARTS, "total"]
    out = [f"# Energy by window on {run.tech}", "", CAVEAT, "", "Energies in nJ.", "",
           _row(heads), _row(["---"] * len(heads))]  # fmt: skip
    for w, e in zip(run.windows, run.energies, strict=True):
        cells = [str(w.index), w.kind, w.label, str(w.cycles)]
        out.append(_row(cells + [dec3(e.parts[k]) or "" for k in PARTS] + [dec3(e.total) or ""]))
    total = run.total
    out.append(_row(["all", "", "", str(sum(w.cycles for w in run.windows))]
                    + [dec3(total[k]) or "" for k in PARTS] + [dec3(sum(total.values())) or ""]))  # fmt: skip
    s = run.static
    out.append(_row(["static", "", "one run of the accelerator", str(s.cycles)]
                    + [dec3(s.parts[k]) or "" for k in PARTS] + [dec3(s.total) or ""]))  # fmt: skip
    runs = run.runs()
    if runs:
        heads = ["Count", *(f"window {w.index}" for w, _ in runs), "static"]
        what = "the run windows were" if runs[0][0].kind == "run" else "the whole run was"
        out += ["", f"## The counts {what} priced on", "",
                _row(heads), _row(["---"] * len(heads))]  # fmt: skip
        for key, label in COUNTS:
            out.append(_row([label, *(_count(getattr(e, key)) for _, e in runs),
                             _count(getattr(s, key))]))  # fmt: skip
    return "\n".join(out) + "\n"
