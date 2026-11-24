"""A Mac's bank of R rows and its held input: ``bind``'s ``rows``, the editor's
"bank rows" box, and what the design, LOWER, the generator and the estimate make of it, on
``matvec_v8`` and ``bicg_v4``."""

from __future__ import annotations

from snax_forge.sandbox.session import Session
from tests.helpers import recipe_path, steps


def bound(name: str = "matvec_v8", **params: int):
    return steps(name, **params)[-1]


def mac(g):
    (n,) = [n for n, _ in g.walk() if n.kind == "accelerated" and n.attrs["instance"] == "mac"]
    return n


# =============================================================================
# bind
# =============================================================================


# =============================================================================
# The editor
# =============================================================================


def test_the_bind_list_offers_the_bank_with_the_rows_the_loops_give(roots):
    s = Session.from_recipe(recipe_path("matvec_v8"), roots=roots)
    k = s.n - 1
    entries = s.sites(k, "mult_map_inner_s")["bind"]["list"]
    (m,) = [e for e in entries if e["brm"] == "mac"]
    assert m["ok"] and m["bank"] == 8 and m["params"]["params"] == {"rows": 8}
    assert all("bank" not in e for e in entries if e["brm"] != "mac")
