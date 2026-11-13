"""The register bits of a group's control unit, from its description alone.

    ctrl_registers(units, links, owner, fixed, job_width, declared=False) -> list[(str, int)]
    ctrl_bits(units, links, owner, fixed, job_width) -> int
    accelerator_registers(accelerator) -> {module: list[(str, int)]}

The group description is the one ``cycles.py``'s ``group_cycles`` steps: ``units`` with
``name``, ``L``, ``ii`` and ``span``; ``links`` ``{from, to, to_input, depth}`` by unit name;
``owner`` the unit with the outer output. ``fixed`` gives per unit its T when fixed at design
time (a fold's ``params.T``, 1 for a unit without a fold), None when a JOB_BEATS CSR sets it,
read live by the control unit; ``job_width`` is the platform's ``acc_csr.job_width``. A count of a
T,
⌈log₂ T⌉ bits for a fixed T and ``job_width`` for one from a JOB_BEATS, is Chisel's ``log2Ceil``.

**The registers** of ``<tag>_<group>_ctrl``, every one ``GroupRule.scala`` and
``GroupCtrl.scala`` declare (no copy of JOBS or a JOB_BEATS, read live from the
csr), by their ``suggestName``; those marked *dropped* are declared but never written, so
firtool removes them and DC never sees them:

    busy                          1          GroupRule.scala:200
    per unit k, in unit order:
      <u>_job_beats_counter       the count of its T, none at a fixed T of 1  (:222, :231)
      <u>_<j>_job_beats_counter   the count of T_j, per unit j after k (nearest first) whose T
                                  is not a fixed 1, unless k borrows (L 0 with a wire out)  (:237)
      <u>_jobs_counter            job_width  :240, unless k borrows
      <u>_gap                     ⌈log₂(II + 1)⌉ at II > 1  :242
      <u>_v                       max(L, 1)  :243, the pipeline's valid bits; dropped at L 0
      <u>_lane                    ⌈log₂ span⌉  :244, none at span 1
    per link, wires included:
      <from>_<to>_occupancy       ⌈log₂(depth + 1)⌉  :255
      <from>_<to>_wslot, _rslot   ⌈log₂ max(depth, 2)⌉ each  :258, :261; dropped for a wire
                                  and for a buffer of depth 1 (one slot)
    done                          1          GroupCtrl.scala:92, the done pulse registered

``ctrl_registers`` lists what firtool keeps (``declared=True`` every declared one), and
``ctrl_bits`` sums it: DC keeps what firtool emits, so it matched the synthesized ctrl flops at all 48
points when it was built (``util/ctrl_regs.py`` on the fixtures; ``tests/estimate/test_bits.py``
holds it within 10 %).

**Against the emitted RTL**: ``accelerator_registers`` reads each group's description from an
``accelerator.json`` as hw/chisel's ``AccCtrl.desc`` does; ``util/ctrl_regs.py --json``
writes it for every RTL fixture into ``ctrl_registers.json`` beside them (``registers_json``),
which hw/chisel's ``CtrlRegistersSpec`` compares, register by register, with each emitted
``<tag>_<group>_ctrl``, and ``tests/estimate/test_bits.py`` checks the file is current.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def _log2_ceil(x: int) -> int:
    """Chisel's ``log2Ceil``: 0 for 1, else ⌈log₂ x⌉."""
    return max(x - 1, 0).bit_length()


def ctrl_registers(units: Sequence[Mapping[str, Any]], links: Sequence[Mapping[str, Any]],
                   owner: str, fixed: Mapping[str, int | None],
                   job_width: int, declared: bool = False) -> list[tuple[str, int]]:  # fmt: skip
    """The registers of the group's control unit firtool keeps with their bits, in the module
    doc's order; ``declared`` every one the description declares, the dropped ones too."""
    succ = {e["from"]: e["to"] for e in links}

    def after(k: str) -> list[str]:
        out = []
        while k in succ:
            k = succ[k]
            out.append(k)
        return out

    def count(t: int | None) -> int:
        return job_width if t is None else _log2_ceil(t)

    by_name = {u["name"]: u for u in units}
    regs = [("busy", 1)]
    for u in units:
        k, t = u["name"], fixed[u["name"]]
        borrows = u["L"] == 0 and any(e["from"] == k and e["depth"] == 0 for e in links)
        if t != 1:
            regs.append((f"{k}_job_beats_counter", count(t)))
        if not borrows:
            regs += [(f"{k}_{j}_job_beats_counter", count(fixed[j])) for j in after(k)
                     if fixed[j] != 1]  # fmt: skip
            regs.append((f"{k}_jobs_counter", job_width))
        if u["ii"] > 1:
            regs.append((f"{k}_gap", _log2_ceil(u["ii"] + 1)))
        if u["L"] or declared:
            regs.append((f"{k}_v", max(u["L"], 1)))
        if u["span"] > 1:
            regs.append((f"{k}_lane", _log2_ceil(u["span"])))
    for e in links:
        name = f"{by_name[e['from']]['name']}_{by_name[e['to']]['name']}"
        regs.append((f"{name}_occupancy", _log2_ceil(e["depth"] + 1)))
        if e["depth"] > 1 or declared:
            regs += [(f"{name}_{s}", _log2_ceil(max(e["depth"], 2))) for s in ("wslot", "rslot")]
    regs.append(("done", 1))
    return [(n, b) for n, b in regs if b]


def ctrl_bits(units: Sequence[Mapping[str, Any]], links: Sequence[Mapping[str, Any]], owner: str,
              fixed: Mapping[str, int | None], job_width: int) -> int:  # fmt: skip
    """The register bits of the group's control unit firtool keeps (``ctrl_registers``
    summed)."""
    return sum(b for _, b in ctrl_registers(units, links, owner, fixed, job_width))


# =============================================================================
# Against the emitted RTL (hw/chisel's CtrlRegistersSpec)
# =============================================================================


def accelerator_registers(accelerator: Mapping[str, Any]) -> dict[str, list[tuple[str, int]]]:
    """Per group of an ``accelerator.json``, its control unit ``<tag>_<group>_ctrl`` and
    ``ctrl_registers`` on the group as the RTL reads it (hw/chisel's ``AccCtrl.desc``): each
    unit's timing, its T None with a JOB_BEATS CSR, else its ``params.T`` (1 without one), the
    links by unit, and ``job_width`` the JOBS CSR's width."""
    out = {}
    for acc in accelerator["accelerators"]:
        for g in acc["groups"]:
            units = [{"name": u["name"], "L": u["timing"]["latency"], "ii": u["timing"]["ii"],
                      "span": u["timing"]["span"]} for u in g["units"]]  # fmt: skip
            links = [{"from": e["from"].split(".")[0], "to": e["to"].split(".")[0],
                      "depth": e["depth"]} for e in g["links"]]  # fmt: skip
            fixed = {u["name"]: None if any(c["name"] == "JOB_BEATS" for c in u["csrs"])
                     else u.get("params", {}).get("T", 1) for u in g["units"]}  # fmt: skip
            owner = next(u["name"] for u in units if u["name"] not in {e["from"] for e in links})
            job_width = next(c["width"] for c in g["csrs"] if c["name"] == "JOBS")
            regs = ctrl_registers(units, links, owner, fixed, job_width)
            out[f"{accelerator['tag']}_{g['name']}_ctrl"] = regs
    return out


def registers_json(registers: Mapping[str, Mapping[str, Sequence[tuple[str, int]]]]) -> str:
    """Per fixture its ``accelerator_registers``, as ``ctrl_registers.json``'s text: a control
    unit per line, its registers as ``[name, bits]`` pairs in ``ctrl_registers``' order."""
    lines = ["{"]
    for i, (fixture, units) in enumerate(registers.items()):
        lines.append(f"  {json.dumps(fixture)}: {{")
        for j, (module, regs) in enumerate(units.items()):
            comma = "," if j < len(units) - 1 else ""
            lines.append(f"    {json.dumps(module)}: {json.dumps([list(r) for r in regs])}{comma}")
        lines.append("  }" + ("," if i < len(registers) - 1 else ""))
    return "\n".join([*lines, "}"]) + "\n"
