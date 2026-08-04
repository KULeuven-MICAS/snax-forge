"""Emit hardware descriptors from raised SDFGs.

The Python half of the boundary whose Scala half is `snax.forge.HwGen`. Reads
the descriptor a `SnaxVectorOp` carries, reshapes it to the schema in
`hw/descriptors/schema/hw_schema.json`, validates, and writes to
`out/descriptors/<recipe>.json`.

Validation happens BEFORE the write, so a malformed descriptor never reaches
disk. That matters more than it sounds: the alternative is a file that the
generator rejects with a decode error three steps later, at which point the
SDFG that produced it is long gone.

Why this is a reshape and not a dump
------------------------------------

`ElementwisePattern.describe()` answers "what does this SDFG say", in DaCe's
vocabulary. The schema answers "what must a generator know", in the hardware
vocabulary. They overlap but are not the same document, and the differences
are all deliberate:

    describe()                     schema              why
    shape.variant                  unit.variant        family/variant split
    -                              unit.family         a reduction is not a
                                                       fourth elementwise
                                                       variant
    datapath.code + ops            datapath.expr       a histogram works only
                                                       while a tasklet holds
                                                       one operation
    streams.in[] / streams.out[]   ports[] + direction one item schema, one
                                                       loop
    bytes_per_element              bits + signed       widths are what an RTL
                                                       port is measured in
    -                              cluster wrapper     one SDFG state may hold
                                                       several accelerators
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dace

from snax_forge.libnodes.libnodes import SnaxVectorOp
from snax_forge.sdfg.paths import _repo_root
from snax_forge.sdfg.recipes import OUT as TRANSFORM_OUT

OUT = _repo_root() / "out" / "descriptors"
SCHEMA = _repo_root() / "hw" / "descriptors" / "schema" / "hw_schema.json"

SCHEMA_VERSION = "1.0"
TOOL_VERSION = "0.1.0"

__all__ = ["OUT", "SCHEMA", "emit", "emit_recipe", "expr_of", "validate"]


# ---------------------------------------------------------------------------
# Tasklet code -> expression tree
# ---------------------------------------------------------------------------

#: Python AST operator to schema operation name. The schema's `op` enum and
#: ElementwiseOp.scala hold the same list; this is the third copy and the one
#: that decides what is expressible from an SDFG.
_BINOPS: dict[type, str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.BitAnd: "and",
    ast.BitOr: "or",
    ast.BitXor: "xor",
}

#: Two-argument builtins that map to an ALU operation.
_CALLS = {"min": "min", "max": "max"}


def expr_of(code: str) -> dict[str, Any]:
    """Parse a tasklet body into the schema's expression tree.

    Parsed rather than pattern-matched on strings, and general rather than
    limited to what the hardware currently supports: `c = a + b * 2` produces
    a correct nested tree here and is refused by the Chisel generator, which
    is the right place for that refusal. The descriptor records what the SDFG
    computes; whether hardware exists for it is a separate question.

    Only integer literals are accepted, matching the project's integer-only
    scope -- a float constant would silently become an int otherwise.
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError as exc:  # pragma: no cover - malformed tasklet
        raise ValueError(f"tasklet body is not parseable Python: {code!r}") from exc

    stmts = [s for s in tree.body if not isinstance(s, ast.Pass)]
    if len(stmts) != 1 or not isinstance(stmts[0], ast.Assign):
        raise ValueError(
            f"expected exactly one assignment in the tasklet body, got {len(stmts)} "
            f"statement(s): {code!r}"
        )
    return _node(stmts[0].value, code)


def _node(node: ast.AST, code: str) -> dict[str, Any]:
    if isinstance(node, ast.Name):
        return {"port": node.id}

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError(f"only integer literals are supported, got {node.value!r} in {code!r}")
        return {"const": node.value}

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"no hardware operation for {type(node.op).__name__} in {code!r}")
        return {"op": op, "args": [_node(node.left, code), _node(node.right, code)]}

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = _CALLS.get(node.func.id)
        if name is None:
            raise ValueError(f"no hardware operation for call to {node.func.id}() in {code!r}")
        if len(node.args) != 2:
            raise ValueError(f"{node.func.id}() needs exactly two arguments in {code!r}")
        return {"op": name, "args": [_node(a, code) for a in node.args]}

    # Parenthesised expressions do not appear in the AST, so this really is an
    # unsupported construct rather than a formatting artefact.
    raise ValueError(f"unsupported expression node {type(node).__name__} in {code!r}")


# ---------------------------------------------------------------------------
# Library node -> unit
# ---------------------------------------------------------------------------


def _ports(descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    """streams.in/out -> a flat list carrying its own direction.

    Flat because the schema validates one item shape and the generator runs
    one loop; the in/out split is recoverable from the field and costs nothing.
    """
    ports = []
    for direction in ("in", "out"):
        for stream in descriptor["streams"][direction]:
            dtype = stream["dtype"]
            ports.append(
                {
                    "name": stream["name"],
                    "direction": direction,
                    "bind": stream["port"],
                    "subset": stream["subset"],
                    "dtype": dtype,
                    # An RTL port is measured in bits, and signedness decides
                    # whether min/max compare as two's complement. Both are
                    # derivable from the DaCe dtype, so neither is guessed.
                    "bits": stream["bytes_per_element"] * 8,
                    "signed": not dtype.startswith("u"),
                    "shape": list(stream["shape"]),
                }
            )
    return ports


def _unit(node: SnaxVectorOp, module_name: str) -> dict[str, Any]:
    d = node.descriptor
    shape = d["shape"]

    if not shape["bounded"]:
        raise ValueError(
            f"{node.label}: lanes is symbolic, so there is no width to elaborate. "
            f"Add a specialize step to the recipe before the unroll."
        )

    return {
        "module_name": module_name,
        # A reduction is not a fourth elementwise variant -- it has an
        # accumulator, a different port shape, and it is a chaining barrier.
        # Splitting family from variant keeps the generator's match exhaustive
        # per family rather than one flat enum that grows without structure.
        "family": "elementwise",
        "variant": shape["variant"],
        "shape": {
            # Elaboration-time: how much hardware exists.
            "lanes": int(shape["lanes"]),
            # Runtime: a CSR value the driver writes, so it stays symbolic.
            "trips": str(shape["trips"]),
            "elements": str(shape["elements"]),
            "bounded": True,
            "ragged": bool(d["spatial"]["ragged"]) if d.get("spatial") else False,
        },
        "datapath": {
            "label": d["datapath"]["label"],
            # Kept for traceability only; the generator reads `expr`. A
            # descriptor that cannot be checked against its source is hard to
            # trust when the generated RTL looks wrong.
            "source": d["datapath"]["code"],
            "expr": expr_of(d["datapath"]["code"]),
        },
        "ports": _ports(d),
    }


def _module_names(nodes: list[SnaxVectorOp]) -> list[str]:
    """Unique names within the cluster.

    Two identical operations in one state raise to two nodes with the same
    label, which DaCe permits and SystemVerilog's flat module namespace does
    not. Suffixes are added only where they are needed, so the common case
    keeps a clean name.
    """
    counts: dict[str, int] = {}
    for n in nodes:
        counts[n.label] = counts.get(n.label, 0) + 1

    seen: dict[str, int] = {}
    names = []
    for n in nodes:
        if counts[n.label] == 1:
            names.append(n.label)
        else:
            seen[n.label] = seen.get(n.label, 0) + 1
            names.append(f"{n.label}_{seen[n.label]}")
    return names


# ---------------------------------------------------------------------------
# SDFG -> document
# ---------------------------------------------------------------------------


def emit(sdfg: dace.SDFG, recipe: str | None = None) -> dict[str, Any]:
    """Build a descriptor document from a raised SDFG. Does not write."""
    states = [s for s in sdfg.states() if any(isinstance(n, SnaxVectorOp) for n in s.nodes())]
    if not states:
        raise ValueError(f"{sdfg.name}: no SnaxVectorOp found -- was raise_vector_ops applied?")
    if len(states) > 1:
        # One SDFG state is one cluster configuration, so more than one state
        # means more than one document. Refused rather than silently emitting
        # the first, which would drop work with no indication.
        raise ValueError(
            f"{sdfg.name}: raised nodes in {len(states)} states; one descriptor "
            f"describes one cluster configuration"
        )

    state = states[0]
    nodes = [n for n in state.nodes() if isinstance(n, SnaxVectorOp)]
    names = _module_names(nodes)

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "tool": "snax-forge",
            "version": TOOL_VERSION,
            "recipe": recipe or sdfg.name,
            "sdfg": f"{TRANSFORM_OUT.relative_to(_repo_root())}/{recipe or sdfg.name}.sdfg",
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "cluster": {
            "name": state.label,
            "units": [_unit(n, name) for n, name in zip(nodes, names, strict=True)],
        },
    }


def validate(document: dict[str, Any]) -> dict[str, Any]:
    """Check against the schema. Raises before anything is written.

    The same schema the Chisel side checks against, so a descriptor that
    passes here and fails there means the two validators disagree -- which is
    worth knowing loudly rather than debugging as a decode error.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jsonschema is required to emit descriptors") from exc

    schema = json.loads(SCHEMA.read_text(encoding="utf-8-sig"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    if errors:
        lines = [
            f"  at {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message} "
            f"({e.validator}={json.dumps(e.validator_value)[:60]})"
            for e in errors
        ]
        raise ValueError(f"descriptor does not satisfy {SCHEMA.name}:\n" + "\n".join(lines))
    return document


def emit_recipe(recipe: str, out_dir: Path | None = None) -> Path:
    """Read out/transforms/<recipe>.sdfg, emit, validate, write. Returns the path."""
    source = TRANSFORM_OUT / f"{recipe}.sdfg"
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found -- run: python -m snax_forge.sdfg --recipe {recipe}"
        )

    document = validate(emit(dace.SDFG.from_file(str(source)), recipe=recipe))

    out_dir = out_dir or OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{recipe}.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path