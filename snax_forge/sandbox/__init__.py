"""SNAX-SANDBOX: decisions made by hand on a SNAX-DFG (ARCHITECTURE.md 5.4).

A recipe (recipe.py) is an ordered list of registered transforms with their params, applied
to an imported graph (CONTRACTS, "Recipe"). Each transform has its inverse and a site, the
graph-shape test of where it applies: split, tag, bind, interchange and encapsulate
(transforms.py); split_reduce, fuse_map and link (compose.py); bake_scalar (scalar.py);
fuse_seed (seed.py); mark_transpose (transpose.py); share (share.py); share_ports
(share_ports.py); lift and apply_recipe (blocks.py). ``bind`` recognises a BRM's pattern
through the matchers of patterns.py. The runner (run.py) binds the recipe's symbols, applies
every step, checks each against the reference executor and writes every step's graph. The
command line is ``python -m snax_forge.sandbox RECIPE`` (pixi ``sandbox``); the editor's
session is session.py.
"""

from .blocks import lift, unlift
from .compose import fuse_map, join_reduce, link, split_reduce, unfuse_map, unlink
from .patterns import PATTERNS, match, register_pattern
from .recipe import Recipe, SandboxError, Step
from .run import Result, apply_recipe, bind_recipe_symbols, write_steps
from .scalar import bake_scalar, unbake_scalar
from .seed import fuse_seed, unfuse_seed
from .share import share, unshare
from .share_ports import share_ports, unshare_ports
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
    "lift",
    "link",
    "mark_transpose",
    "match",
    "register_pattern",
    "register_transform",
    "share",
    "share_ports",
    "split",
    "split_map",
    "split_reduce",
    "unbake_scalar",
    "unbind",
    "unencapsulate",
    "unfuse_map",
    "unfuse_seed",
    "unlift",
    "unlink",
    "unmark_transpose",
    "unshare",
    "unshare_ports",
    "write_steps",
]
