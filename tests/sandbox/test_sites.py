"""The sites: where each transform applies, tested against the transforms themselves.

Every recipe's steps are accepted by their transform's site on the graph before the step, with
the params the site fills equal to the step's (a suggested name aside, ``FREE_NAMES``);
wherever a site refuses, the transform refuses with the same error; and ``split_map`` is ``split``
then two ``tag_map`` steps, file for file
once ``split``'s neutral names are mapped to ``split_map``'s. Every recipe is imported (DaCe), hence
slow.
"""

from __future__ import annotations

import pytest

from snax_forge.sandbox import TRANSFORMS, SandboxError, split, split_map
from snax_forge.sandbox.run import step_params
from snax_forge.sandbox.transforms import tag_map

from .helpers import (
    RECIPES,
    SELECTED,
    default_calls,
    graph,
    recipe_graphs,
    renamed,
    selections,
    tagged_names,
)

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

# Params a site fills with a suggested name only (link's group: the ends' group, else the next
# free one); the editor's form lets the user type another, which the transform accepts when
# it is free, so a saved recipe may hold any such name (Ryan's ry_mvt_1: "mvt2" for "mvt_1");
# share_ports' accelerator likewise (dot_1_shared's "dot" for "vmul_vsum").
FREE_NAMES = {("link", "group"), ("share_ports", "accelerator")}


def test_every_transform_has_a_site():
    assert {n for n, spec in TRANSFORMS.items() if spec.site} == set(SELECTED)


@pytest.mark.parametrize("name", RECIPES)
def test_the_site_accepts_every_recipe_step(name):
    r, graphs = recipe_graphs(name)
    for i in range(len(r.steps)):
        transform, params = step_params(r, i)
        site, key = TRANSFORMS[transform].site, SELECTED[transform]
        if key not in params:  # apply_recipe on a region, not a lifted block: no site fills it
            continue
        for selected in params[key] if key in ("units", "nodes") else [params[key]]:
            filled = site(graphs[i], selected)
            for k, v in filled.items():
                if (transform, k) in FREE_NAMES:  # a default; recipe_graphs applied the step's
                    continue
                want = [selected] if k in ("units", "nodes") else params[k]
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
        assert inner.id.endswith("_inner") and "loop.kind" not in inner.attrs
        g = tag_map(tag_map(g, params["map"], "temporal"), inner.id, "spatial")
        names = tagged_names(graphs[i], params["map"], params["factor"])
        assert renamed(g, names) == split_map(graphs[i], **params).to_json()
        assert split_map(graphs[i], **params).to_json() == graphs[i + 1].to_json()
