"""The DFG viewer and editor: ``.snaxdfg`` graphs in the browser.

The visualiser's server in two modes: view mode, ``python -m snax_forge.viz.dfg
FILE|DIR ...`` (pixi ``view-dfg``), the files as a read-only history of steps, with Reload;
edit mode, ``python -m snax_forge.viz.dfg.edit`` (pixi ``edit-dfg``), a sandbox session
(sandbox/session.py) edited step by step.

* api.py: loading files, the view of one step (its drawing, Details, the Estimate) and two
  steps compared;
* drawing.py: a step's drawing, the boxes, ports, edges and outlines the page lays out (D185);
* server.py: the routes of both modes, on the viewer's Handler and static files;
* __main__.py and edit.py: the two command lines;
* static/dfg.html, static/dfg.js, static/graph.js and static/edit.js: the page; graph.js lays
  each step out with ELK (static/vendor/, in its worker) and draws it.
"""

from .api import GraphSet, graph_view
from .server import make_dfg_server

__all__ = ["GraphSet", "graph_view", "make_dfg_server"]
