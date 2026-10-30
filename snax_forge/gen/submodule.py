"""snax-gvsoc's Python, loaded by path from the submodule (``gvsoc/snax-gvsoc``, D134).

    snax_group_rule() -> module    the group control rule, snax/snax_group_rule.py (GroupRule)

The module is loaded once. Without the submodule it is a ``GenError`` ``gen.submodule`` that
names the fix.
"""

from __future__ import annotations

import functools
import importlib.util
from types import ModuleType

from snax_forge.common import repo_root

from .blocks import GenError

SUBMODULE = repo_root() / "gvsoc" / "snax-gvsoc"


@functools.cache
def snax_group_rule() -> ModuleType:
    """snax-gvsoc's ``snax/snax_group_rule.py``: the group control rule over edges, the same as
    the C++ SnaxGroupRule and the Chisel GroupRule (the shared group cases check all three)."""
    path = SUBMODULE / "snax" / "snax_group_rule.py"
    if not path.is_file():
        raise GenError("gen.submodule", f"{path} is not there: the submodule gvsoc/snax-gvsoc is "
                                        "not checked out (git submodule update --init)")  # fmt: skip
    spec = importlib.util.spec_from_file_location("snax_group_rule", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
