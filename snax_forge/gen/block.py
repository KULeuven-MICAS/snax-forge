"""Catalogue blocks alone -> their RTL, for characterising each block on its own.

``main`` (``pixi run gen-block LIST.json``) reads a JSON list of ``{"datapath", "params",
"out"}``, ``out`` from the working directory. ``check_entry`` checks every entry against the
catalogue ``hw/chisel/blocks.json`` before any is built (``gen.block``, ``gen.catalogue``,
``gen.params``, each a ``GenError``); ``emit_blocks`` builds them all in one java call on
rtl.py's cached classpath, ``Emit --block`` each. The output is CONTRACTS, "Accelerator RTL
spec". What only the Chisel constructor knows (an ``opType`` narrower than an input) is refused
there, ``gen.elaborate``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from snax_forge.common import cwd_relative

from .blocks import GenError
from .gvsoc import cli
from .rtl import CHISEL, EMITTED_JSON, catalogue, check_params, emit_classpath, listed

KEYS = {"datapath", "params", "out"}


def check_entry(entry: Any, cat: Mapping[str, Any] | None = None) -> None:
    """One entry of a block list against the catalogue; see the module doc."""
    if (
        not isinstance(entry, Mapping)
        or set(entry) != KEYS
        or not isinstance(entry["params"], Mapping)
    ):
        raise GenError("gen.block", f"{entry!r}: an entry is {{datapath, params, out}}, params an object")  # fmt: skip
    cat = catalogue() if cat is None else cat
    name, params = entry["datapath"], entry["params"]
    if name not in cat:
        raise GenError("gen.catalogue", f"datapath {name!r} is not in the catalogue {sorted(cat)}")
    sig = cat[name]["params"]
    missing, extra = sorted(set(sig) - set(params)), sorted(set(params) - set(sig))
    if missing or extra:
        raise GenError(
            "gen.params", f"{name}: params {list(sig)}; missing {missing}, extra {extra}"
        )
    check_params(name, params, sig)


def emit_blocks(entries: list[Mapping[str, Any]]) -> list[list[Path]]:
    """Every entry checked (``check_entry``), then all through ``Emit --block`` in one java call;
    per entry the files its ``emitted.json`` lists, and the listing."""
    if not isinstance(entries, list):
        raise GenError("gen.block", f"a block list is a JSON list, not {type(entries).__name__}")
    cat = catalogue()
    for e in entries:
        check_entry(e, cat)
    args = [a for e in entries for a in ("--block", e["datapath"], "--params",
            json.dumps(e["params"]), "--out", str(Path(e["out"]).resolve()))]  # fmt: skip
    cmd = ["java", "-cp", emit_classpath(), "snax.forge.Emit", *args]
    done = subprocess.run(cmd, cwd=CHISEL, check=False)
    return [
        listed(Path(e["out"]), f"Emit --block {e['datapath']}", done.returncode) for e in entries
    ]


def main(argv: list[str] | None = None) -> int:
    """``python -m snax_forge.gen.block LIST.json`` (pixi ``gen-block``): every block of the list
    into its folder (``emit_blocks``), each top printed. Exit 0 when written, 1 when refused."""
    ap = argparse.ArgumentParser(prog="python -m snax_forge.gen.block")
    ap.add_argument("blocks", type=Path, help="a JSON list of {datapath, params, out}")
    args = ap.parse_args(argv)

    def write() -> int:
        entries = json.loads(args.blocks.read_text())
        for e, files in zip(entries, emit_blocks(entries), strict=True):
            top = json.loads((Path(e["out"]) / EMITTED_JSON).read_text())["top"]
            print(f"wrote {cwd_relative(Path(e['out']))}: {top}, {len(files) - 1} file(s) in sv/")
        return 0

    return cli(write)


if __name__ == "__main__":
    sys.exit(main())
