"""The generator's own schemas: the streamer file, ``accelerator.json`` and ``rtl.json``.

    schema(name) -> dict        name: "streamer", "accelerator" or "rtl"
    validate(name, value)       raises jsonschema.ValidationError

JSON Schema 2020-12, in ``schema/<name>.schema.json``: what gvsoc.py writes
into ``<slot>_streamer.hjson`` (the streamer template) and
``accelerator.json`` (the accelerator's one description, accelerator.py), each
a whole file, and what rtl.py writes into ``rtl.json`` (until B1's R2). What
``accelerator.json``'s schema cannot say (the CSR offsets contiguous in CSR
order) ``Accelerator`` checks. The cluster
file is snax_cluster's own and checked against snax_cluster's schema (the
copy in snax-gvsoc ``sw/docs/schema/``), where the submodule is checked out
(tests/gen/test_schemas.py). What a schema cannot say (the movers' lists of
one length, a port per mover, the slot's counts) SNAX-GVSoC's reader checks
(snax-gvsoc ``snax/snax_cfg.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMAS = Path(__file__).parent / "schema"
NAMES = ("streamer", "accelerator", "rtl")


def schema(name: str) -> dict[str, Any]:
    """The schema of the ``name`` file."""
    if name not in NAMES:
        raise ValueError(f"no schema {name!r} (have {list(NAMES)})")
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text())


def validate(name: str, value: Any) -> None:
    """Raise jsonschema.ValidationError unless ``value`` is a ``name`` file's content."""
    Draft202012Validator(schema(name)).validate(value)
