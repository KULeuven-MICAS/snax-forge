"""A flow run's directory, written and read back: what the reports and the viewer open.

The directory is CONTRACTS, "Profile and trace": ``run.json`` (what ran, written by
flow/run.py), ``profile.json`` (profile.py), above trace ``off`` ``trace.jsonl`` and
``trace_meta.json`` (trace.py), and ``events.log``, SNAX-GVSoC's events (events.py), which a
passing flow leaves as ``events.log.gz`` (``gzip_events``) and ``read_events`` reads either way.
``write_run`` writes the profile and trace and removes a trace an earlier run left when there is
none now, so the folder matches ``run.json``; ``read_run`` reads all four back.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snax_forge.common import to_json

from .events import RunError
from .profile import Profile
from .trace import Trace

RUN_FILE = "run.json"
PROFILE_FILE = "profile.json"
TRACE_FILE = "trace.jsonl"
TRACE_META_FILE = "trace_meta.json"
EVENTS_FILE = "events.log"


def gzip_events(run_dir: Path) -> Path:
    """``events.log`` of ``run_dir`` as ``events.log.gz`` (the same bytes for the same
    events: no timestamp), the plain file removed; the new path."""
    plain = Path(run_dir) / EVENTS_FILE
    packed = plain.with_name(EVENTS_FILE + ".gz")
    packed.write_bytes(gzip.compress(plain.read_bytes(), mtime=0))
    plain.unlink()
    return packed


@dataclass
class RunOutputs:
    """A run directory read back."""

    run: dict[str, Any]
    profile: Profile | None  # None only in tests that build a run without its summary
    trace: Trace | None

    @property
    def elf_symbols(self) -> dict[str, dict[str, int]] | None:
        """run.json's ``elf_symbols`` (regions.py's L2 regions), None in a run without them."""
        syms = self.run.get("elf_symbols")
        return None if syms is None else {k: dict(v) for k, v in syms.items()}


def write_run(out: str | Path, profile: Profile, trace: Trace | None) -> Path:
    """The profile and, when there is one, the trace into ``out``; returns ``out``."""
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / PROFILE_FILE).write_text(to_json(profile.to_dict()))
    if trace is None:
        for name in (TRACE_FILE, TRACE_META_FILE):
            (d / name).unlink(missing_ok=True)
    else:
        (d / TRACE_META_FILE).write_text(to_json(trace.meta()))
        (d / TRACE_FILE).write_text("".join(json.dumps(e) + "\n" for e in trace.events))
    return d


def read_run(out: str | Path) -> RunOutputs:
    """A run directory (module doc); a RunError (``outputs.missing``) when it is not one."""
    d = Path(out)
    for name in (RUN_FILE, PROFILE_FILE):
        if not (d / name).is_file():
            raise RunError("outputs.missing", f"{d}: no {name}; not a flow run's run/ directory")
    trace = None
    if (d / TRACE_META_FILE).is_file():
        meta = json.loads((d / TRACE_META_FILE).read_text())
        lines = (d / TRACE_FILE).read_text().splitlines() if (d / TRACE_FILE).is_file() else []
        trace = Trace.from_dict({**meta, "events": [json.loads(x) for x in lines if x.strip()]})
    return RunOutputs(
        json.loads((d / RUN_FILE).read_text()),
        Profile.from_dict(json.loads((d / PROFILE_FILE).read_text())),
        trace,
    )
