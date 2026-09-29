"""SNAX-FORGE visualiser (M4a, M4b, D54, D55).

A local server plus a static viewer, for humans: ``python -m snax_forge.viz
DIR [DIR ...]`` serves the model output directories given (``snax_model``'s
``run ... --out DIR``) at http://127.0.0.1:8765/. LLMs read the profile and
the reports (``snax_forge.report``, D99) instead.

* api.py: loading runs and every answer of the API, as plain functions;
* memory.py: the memory layout of a run, folded (VIS4a, D96);
* movement.py: data movement read off a beat trace (VIS4b, D97);
* server.py: the stdlib HTTP server around them;
* static/: the viewer (index.html, plain JS modules, CSS; no build step and
  no external file, so it works offline).

The tabs are the profile report (VIS1), the schedule (VIS2) with the cluster
view (VIS3) under it, and the memory tab (VIS4a, VIS4b), all on the same API.
The DFG viewer (VIS5, D81) is the subpackage dfg/, a second mode of the same
server for ``.snaxdfg`` files (``python -m snax_forge.viz.dfg``).
"""

from .api import RunSet, RunView, fifo_windows, load_run
from .server import make_server

__all__ = ["RunSet", "RunView", "fifo_windows", "load_run", "make_server"]
