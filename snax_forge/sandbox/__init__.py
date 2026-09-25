"""SNAX-SANDBOX: decisions made by hand on a SNAX-DFG (ARCHITECTURE.md 5.4, D72).

Built so far, SBX1 (D80): recipes (recipe.py), an ordered list of registered
transforms with their params, applied to an imported graph; the transforms
``split_map`` and ``bind`` and their inverses ``unbind`` and ``join_map``
(transforms.py); the pattern matchers ``bind``
uses to recognise a BRM's pattern (patterns.py); and the runner
(run.py), which binds the recipe's symbols, applies every step, checks each
step against the reference executor and writes every step's graph. The
command line is ``python -m snax_forge.sandbox RECIPE`` (pixi ``sandbox``).
"""

from .patterns import PATTERNS, match, register_pattern
from .recipe import Recipe, SandboxError, Step
from .run import Result, apply_recipe, bind_recipe_symbols, write_steps
from .transforms import TRANSFORMS, bind, join_map, register_transform, split_map, unbind

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
    "join_map",
    "match",
    "register_pattern",
    "register_transform",
    "split_map",
    "unbind",
    "write_steps",
]
