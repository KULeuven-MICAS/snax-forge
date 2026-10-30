"""The sites (D72): where each transform applies, tested against the transforms themselves.

Every recipe's steps are accepted by their transform's site on the graph before the step, with
the params the site fills equal to the step's; wherever a site refuses, the transform refuses
with the same error; and ``split_map`` is ``split`` then two ``tag_map`` steps, file for file
(D73). Every recipe is imported (DaCe), hence slow.
"""

from __future__ import annotations

import pytest

from snax_forge.sandbox import TRANSFORMS, SandboxError, split, split_map
from snax_forge.sandbox.run import step_params
from snax_forge.sandbox.transforms import tag_map

from .helpers import RECIPES, SELECTED, default_calls, graph, recipe_graphs, selections

pytestmark = pytest.mark.slow

FIXTURES = [
    "vecadd",
    "vecadd_split",
    "vecadd_accelerated",
    "vecmul",
    "vsum",
    "dot",
    "dot_accelerated",
]


def test_every_transform_has_a_site():
    assert {n for n, spec in TRANSFORMS.items() if spec.site} == set(SELECTED)


@pytest.mark.parametrize("name", RECIPES)
def test_the_site_accepts_every_recipe_step(name):
    r, graphs = recipe_graphs(name)
    for i in range(len(r.steps)):
        transform, params = step_params(r, i)
        site, key = TRANSFORMS[transform].site, SELECTED[transform]
        for selected in params[key] if key == "units" else [params[key]]:
            filled = site(graphs[i], selected)
            for k, v in filled.items():
                want = [selected] if k == "units" else params[k]
                assert v == want, (
                    f"{name} step {i + 1} ({transform}): {k} {v!r}, the step's {want!r}"
                )


def _refusals_match(g):
    """Every transform, every node and container its site refuses: the same error from it."""
    refused = 0
    for name in SELECTED:
        spec = TRANSFORMS[name]
        for selected in selections(g):
            try:
                spec.site(g, selected)
                continue
            except SandboxError as e:
                want = (type(e), str(e), getattr(e, "code", None))
            refused += 1
            for params in default_calls(g, name, selected):
                with pytest.raises(SandboxError) as got:
                    spec.fn(g, **params)
                assert (type(got.value), str(got.value), getattr(got.value, "code", None)) == want
    assert refused


@pytest.mark.parametrize("fixture", FIXTURES)
def test_a_site_refuses_as_its_transform_on_the_fixtures(fixture):
    _refusals_match(graph(fixture))


@pytest.mark.parametrize("name", RECIPES)
def test_a_site_refuses_as_its_transform_on_every_step(name):
    for g in recipe_graphs(name)[1]:
        _refusals_match(g)


@pytest.mark.parametrize("name", RECIPES)
def test_split_map_is_split_then_two_tags(name):
    r, graphs = recipe_graphs(name)
    for i in range(len(r.steps)):
        transform, params = step_params(r, i)
        if transform != "split_map":
            continue
        g = split(graphs[i], params["map"], params["factor"])
        (inner,) = g.node(params["map"]).body
        g = tag_map(tag_map(g, params["map"], "temporal"), inner.id, "spatial")
        assert g.to_json() == split_map(graphs[i], **params).to_json() == graphs[i + 1].to_json()
