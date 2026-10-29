"""SNAX-SANDBOX: decisions made by hand on a SNAX-DFG (ARCHITECTURE.md 5.4, D72).

Recipes (recipe.py), an ordered list of registered transforms with their
params, applied to an imported graph; the transforms ``split_map``,
``bind`` and ``encapsulate`` and their inverses ``unbind``, ``join_map`` and
``unencapsulate`` (transforms.py); ``split_reduce``, ``fuse_map`` and ``link``
with their inverses ``join_reduce``, ``unfuse_map`` and ``unlink``
(compose.py, D150);
the pattern matchers ``bind`` uses to recognise a BRM's pattern
(patterns.py), among them ``reduce``, with which ``bind`` binds a folding
output as one lane, its named rate from ``named_rates`` (D102); and the
runner (run.py), which binds the recipe's symbols, applies every step,
checks each step against the reference executor and writes every step's
graph. The command line is ``python -m snax_forge.sandbox RECIPE`` (pixi
``sandbox``).
"""

from .compose import fuse_map, join_reduce, link, split_reduce, unfuse_map, unlink
from .patterns import PATTERNS, match, register_pattern
from .recipe import Recipe, SandboxError, Step
from .run import Result, apply_recipe, bind_recipe_symbols, write_steps
from .transforms import (
    TRANSFORMS,
    bind,
    encapsulate,
    join_map,
    register_transform,
    split_map,
    unbind,
    unencapsulate,
)

__all__ = [
    "PATTERNS",
    "TRANSFORMS",
    "Recipe",
    "Result",
    "SandboxError",
    "Step",
    "apply_recipe",
    "bind",
    "bind_recipe_symbols",
    "encapsulate",
    "fuse_map",
    "join_map",
    "join_reduce",
    "link",
    "match",
    "register_pattern",
    "register_transform",
    "split_map",
    "split_reduce",
    "unbind",
    "unencapsulate",
    "unfuse_map",
    "unlink",
    "write_steps",
]
