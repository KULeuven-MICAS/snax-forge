"""SNAX-FORGE visualiser (D55).

A local server plus a static viewer, for humans: ``python -m snax_forge.viz
DIR [DIR ...]`` serves the flow runs given (``out/flow/<name>/`` or its
``run/``, C§7) at http://127.0.0.1:8765/. LLMs read the profile and the
reports (``snax_forge.report``, D99) instead.

* api.py: loading runs and every answer of the API, as plain functions, each
  route one query's dict;
* index.py: a run's events grouped by their ids and lookups once at load
  (``RunIndex``), and the memory tab's marks, added by registration;
* memory.py: the memory tab's layout, folded rows and marks on them (D57);
* movement.py: data movement as queries on the index: journey, residency,
  patterns, conflicts (D57);
* schedule.py, cluster.py: the schedule's lanes, folded segments and detail
  rows, the cluster at one cycle and the stepping between cycles (D57);
* server.py: the stdlib HTTP server around them;
* static/: the viewer (index.html, plain JS modules, CSS; no build step and
  no external file, so it works offline). It draws the API's answers and
  computes nothing of its own.

The tabs are the profile report, the schedule with the cluster view under
it, and the memory tab, all on the same API. The DFG viewer (D55) is the
subpackage dfg/, a second mode of the same server for ``.snaxdfg`` files
(``python -m snax_forge.viz.dfg``).
"""

from .api import RunSet, RunView, load_run
from .server import make_server

__all__ = ["RunSet", "RunView", "load_run", "make_server"]
