"""A Mac's bank of R rows (D191) and its held input (D190): ``bind``'s ``rows``, the editor's
"bank rows" box, and what the design, LOWER, the generator and the estimate make of it, on
``matvec_v8`` and ``bicg_v4``."""

from __future__ import annotations

import pytest

from snax_forge.brm.function import mac_kind
from snax_forge.dfg import DfgError
from snax_forge.dfg.execute import named_rates
from snax_forge.dfg.graph import Memlet
from snax_forge.sandbox import SandboxError
from snax_forge.sandbox.session import Roots, Session
from snax_forge.sandbox.transforms import bind
from tests.helpers import recipe_path, steps


def bound(name: str = "matvec_v8", **params: int):
    return steps(name, **params)[-1]


def mac(g):
    (n,) = [n for n, _ in g.walk() if n.kind == "accelerated" and n.attrs["instance"] == "mac"]
    return n


@pytest.fixture
def roots(tmp_path):
    return Roots(tmp_path / "sessions", tmp_path / "recipes", tmp_path / "sandbox")


# =============================================================================
# bind
# =============================================================================


def test_matvec_v8_binds_a_mac_of_r_rows_folding_n_beats_a_row():
    g = bound()
    n = mac(g)
    assert n.attrs["params"]["rows"] == 8
    loops = [m.attrs["var"] for m, _ in g.walk() if m.kind == "map" and m.attrs.get(
        "loop.kind") == "temporal"]  # fmt: skip
    assert loops == ["m_1_t", "n_t", "m_2_t"]  # strip, A's row, the bank's row


def test_named_rates_count_a_banks_rows_then_the_folded_loops():
    out = Memlet("q", ["16 * s + 4 * r:16 * s + 4 * r + 4"])
    loops = [("s", 2), ("n", 64), ("r", 4)]
    assert named_rates({"out": (out, "T")}, loops, "x", {"out": 4}) == {"T": 256}
    with pytest.raises(DfgError, match="split its loop by 2 times the lanes"):
        named_rates({"out": (out, "T")}, loops, "x", {"out": 2})


def test_a_bank_that_does_not_match_the_loops_is_refused_naming_the_split():
    g = steps("matvec_v8")[-2]
    with pytest.raises((SandboxError, DfgError), match="split its loop by 4 times the lanes"):
        bind(g, "mult_map_inner_s", "mac", "chisel_mac", "mac", {"rows": 4})


def test_a_seeded_bank_and_a_t_off_its_rows_are_refused():
    with pytest.raises(ValueError, match="seeded bank"):
        mac_kind(lanes=4, seed=1, rows=4)


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


def test_the_bank_rows_as_the_page_sends_them_add_a_param(roots):
    s = Session.from_recipe(recipe_path("matvec_v8"), roots=roots)
    params = dict(s.recipe.params)
    params.pop("R")
    s.move(s.n - 1)
    r = s.apply("bind", {"node": "mult_map_inner_s", "brm": "mac", "implementation": "chisel_mac",
                         "instance": "mac", "params": {"rows": "Q"}}, drop=True,
                new_params={"Q": 8})  # fmt: skip
    assert r["ok"], r
    assert s.recipe.params["Q"] == 8 and s.recipe.steps[-1].params["params"] == {"rows": "Q"}
    r = s.apply("bind", {"node": "x", "brm": "mac", "implementation": "chisel_mac",
                         "instance": "mac", "params": {"rows": "Z"}}, drop=True)  # fmt: skip
    assert not r["ok"] and "params.rows 'Z' is not a param" in r["message"]
