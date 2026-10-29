"""The generator's own schemas: the streamer file and ``accelerator.json``.

    schema(name) -> dict        name: "streamer" or "accelerator"

JSON Schema 2020-12, in ``schema/<name>.schema.json``: what gvsoc.py writes
into ``<slot>_streamer.hjson`` (the streamer template) and
``accelerator.json`` (the accelerator's one description, accelerator.py), each
a whole file (gen-rtl writes the same accelerator.json). What
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

SCHEMAS = Path(__file__).parent / "schema"
NAMES = ("streamer", "accelerator")


def schema(name: str) -> dict[str, Any]:
    """The schema of the ``name`` file."""
    if name not in NAMES:
        raise ValueError(f"no schema {name!r} (have {list(NAMES)})")
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text())
