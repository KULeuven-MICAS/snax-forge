"""The generator's own schemas: the streamer file and ``accelerator.json``.

``schema(name)`` loads ``schema/<name>.schema.json`` (JSON Schema 2020-12), ``name`` being
"streamer" or "accelerator": the streamer template gvsoc.py writes and the accelerators' one
description (accelerator.py). What a schema cannot say, ``Accelerator`` (the CSR offsets
contiguous) or SNAX-GVSoC's reader (snax-gvsoc ``snax/snax_cfg.py``: the movers' lists of one
length, a port per mover, the slot's counts) checks. The cluster file is checked against
snax_cluster's own schema (tests/gen/test_schemas.py).
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
