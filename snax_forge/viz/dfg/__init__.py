"""The DFG viewer (VIS5, D76, D81): ``.snaxdfg`` files in the browser.

A second mode of the visualiser's server: ``python -m snax_forge.viz.dfg
FILE|DIR ...`` (pixi ``view-dfg``) shows graphs side by side, drawn top to
bottom in execution order, with Reload for the edit-reload-look loop.

* api.py: loading files and the view of one graph (rows, nodes, edges);
* server.py: the routes, on the viewer's Handler and static files;
* static/dfg.html and static/dfg.js: the page; boxes are HTML laid out by
  CSS, the edges one SVG layer per panel.
"""

from .api import GraphSet, graph_view
from .server import make_dfg_server

__all__ = ["GraphSet", "graph_view", "make_dfg_server"]
