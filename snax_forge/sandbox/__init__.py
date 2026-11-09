"""SNAX-SANDBOX: decisions made by hand on a SNAX-DFG (ARCHITECTURE.md 5.4, D72).

Recipes (recipe.py), an ordered list of registered transforms with their
params, applied to an imported graph; the transforms ``split``,
``split_map``, ``tag_map``, ``bind`` and ``encapsulate`` and their inverses
``join_map``, ``untag_map``, ``unbind`` and ``unencapsulate``, and
``interchange_map``, its own inverse (transforms.py); ``split_reduce``,
``fuse_map`` and ``link`` with their inverses ``join_reduce``,
``unfuse_map`` and ``unlink`` (compose.py, D150);
``bake_scalar`` and its inverse ``unbake_scalar`` (scalar.py);
``fuse_seed`` and its inverse ``unfuse_seed`` (seed.py), a fold taking the elementwise nest after it as its seed;
``mark_transpose`` and its inverse ``unmark_transpose`` (transpose.py), an input read across its loop order marked; ``share`` and its inverse ``unshare`` (share.py), a group run on another's units;
each with its site, the graph-shape test of where it applies (D72);
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
from .scalar import bake_scalar, unbake_scalar
from .seed import fuse_seed, unfuse_seed
from .share import share, unshare
from .transforms import (
    TRANSFORMS,
    bind,
    encapsulate,
    interchange_map,
    join_map,
    register_transform,
    split,
    split_map,
    unbind,
    unencapsulate,
)
from .transpose import mark_transpose, unmark_transpose

__all__ = [
    "PATTERNS",
    "TRANSFORMS",
    "Recipe",
    "Result",
    "SandboxError",
    "Step",
    "apply_recipe",
    "bake_scalar",
    "bind",
    "bind_recipe_symbols",
    "encapsulate",
    "fuse_map",
    "fuse_seed",
    "interchange_map",
    "join_map",
    "join_reduce",
    "link",
    "mark_transpose",
    "match",
    "register_pattern",
    "register_transform",
    "share",
    "split",
    "split_map",
    "split_reduce",
    "unbake_scalar",
    "unbind",
    "unencapsulate",
    "unfuse_map",
    "unfuse_seed",
    "unlink",
    "unmark_transpose",
    "unshare",
    "write_steps",
]
