"""SNAX-FORGE visualiser (M4a, D54, D55).

A local server plus a static viewer, for humans: ``python -m snax_forge.viz
DIR [DIR ...]`` serves the model output directories given (``snax_model``'s
``run ... --out DIR``) at http://127.0.0.1:8765/. LLMs read the profile and,
later, VIS7's summary instead.

* api.py: loading runs and every answer of the API, as plain functions;
* server.py: the stdlib HTTP server around them;
* static/: the viewer (index.html, plain JS modules, CSS; no build step and
  no external file, so it works offline).

VIS1 is the profile report; the schedule (VIS2) and the cluster view (VIS3)
use the same API.
"""

from .api import RunSet, RunView, fifo_windows, load_run
from .server import make_server

__all__ = ["RunSet", "RunView", "fifo_windows", "load_run", "make_server"]
