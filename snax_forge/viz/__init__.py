"""SNAX-FORGE visualiser: a local server and a static viewer of flow runs, for humans.

``python -m snax_forge.viz DIR [DIR ...]`` serves the flow runs given at
http://127.0.0.1:8765/; LLMs read the profile and the reports (``snax_forge.report``) instead.
api.py answers each route as a plain function; index.py groups a run's events once; memory.py,
movement.py, schedule.py and cluster.py are the tabs' queries; server.py is the stdlib server;
static/ is the viewer (plain JS, no build step, no external file). The DFG viewer is the
subpackage dfg/ (``python -m snax_forge.viz.dfg``).
"""

from .api import RunSet, RunView, load_run
from .server import make_server

__all__ = ["RunSet", "RunView", "load_run", "make_server"]
