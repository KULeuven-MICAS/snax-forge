"""Explore hw_schema.json: structure, vocabulary, examples, and failure explanations.

A JSON Schema is hard to read straight because two different kinds of key are
interleaved. This prints them apart.

    pixi run python util/schema_explain.py --tree
    pixi run python util/schema_explain.py --vocab
    pixi run python util/schema_explain.py --defs
    pixi run python util/schema_explain.py --path '#/$defs/expr'
    pixi run python util/schema_explain.py --example
    pixi run python util/schema_explain.py --validate out/descriptors/vecadd_loop.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:
    sys.exit("jsonschema not installed: pixi add jsonschema")

SCHEMA = Path(__file__).resolve().parent.parent / "hw" / "descriptors" / "schema" / "hw_schema.json"

# The complete set of keywords this schema uses, and what each one asserts.
# Everything NOT in this set that appears as a key is a name we chose.
VOCABULARY = {
    # -- structural: where the document goes -----------------------------
    "$schema": "which JSON Schema dialect this document is written in",
    "$id": "canonical URI for this schema, so $ref can resolve against it",
    "$defs": "named subschemas; keys under it are OURS, values are schemas",
    "$ref": "substitute another subschema here, addressed by JSON Pointer",
    # -- type and shape ---------------------------------------------------
    "type": "the JSON type: object, array, string, integer, number, boolean, null",
    "properties": "per-key schemas for an object; the keys under it are OURS",
    "required": "which keys must be present (does NOT imply they are non-empty)",
    "items": "schema every element of an array must satisfy",
    "additionalProperties": "false forbids keys not listed in properties",
    # -- value constraints -------------------------------------------------
    "enum": "value must be one of this list",
    "const": "value must equal exactly this",
    "pattern": "string must match this regular expression",
    "minimum": "numeric lower bound, inclusive",
    "minLength": "string length lower bound",
    "minItems": "array length lower bound",
    "maxItems": "array length upper bound",
    # -- combinators ------------------------------------------------------
    "oneOf": "exactly one of these subschemas must match",
    "allOf": "every one of these subschemas must match",
    "if": "condition for a conditional block; pairs with then",
    "then": "applies only when the matching if succeeded",
    # -- annotations: never affect validity --------------------------------
    "title": "human label, ignored by the validator",
    "description": "human prose, ignored by the validator",
}

# Keywords whose immediate children are names we invented, not vocabulary.
NAME_MAPS = {"properties", "$defs"}


def load() -> dict:
    """Read the schema, and say something useful when it is not readable.

    A bare json.loads failure reports "Expecting value: line 1 column 1", which
    is the same message for an empty file, a UTF-8 BOM, and a saved HTML page.
    Those need different fixes, so they are separated here.
    """
    if not SCHEMA.exists():
        sys.exit(f"{SCHEMA} not found")

    raw = SCHEMA.read_bytes()
    if not raw.strip():
        sys.exit(f"{SCHEMA} is empty ({len(raw)} bytes) -- the paste did not land")

    # utf-8-sig strips a byte-order mark if present. Editors on Windows and
    # some "save as" paths add one; it is invisible in an editor and fatal to
    # json.loads, which sees it as a stray character before the opening brace.
    if raw.startswith(b"\xef\xbb\xbf"):
        print(f"note: {SCHEMA.name} has a UTF-8 BOM; stripping it", file=sys.stderr)

    try:
        return json.loads(raw.decode("utf-8-sig"))
    except json.JSONDecodeError as err:
        head = raw[:120].decode("utf-8", errors="replace").replace("\n", " ")
        sys.exit(
            f"{SCHEMA} is not valid JSON: {err}\n"
            f"  size:  {len(raw)} bytes\n"
            f"  head:  {head!r}\n"
            f"  (a '<' here means an HTML page was saved instead of the file)"
        )


# ---------------------------------------------------------------------------
# --tree
# ---------------------------------------------------------------------------


def constraints(node: dict) -> str:
    """One-line summary of everything this subschema asserts."""
    if "$ref" in node:
        return f"-> {node['$ref']}"
    bits = []
    if "type" in node:
        bits.append(str(node["type"]))
    if "enum" in node:
        vals = node["enum"]
        shown = ", ".join(map(str, vals[:6])) + (" ..." if len(vals) > 6 else "")
        bits.append(f"one of [{shown}]")
    if "const" in node:
        bits.append(f"== {node['const']!r}")
    if "pattern" in node:
        bits.append(f"matches {node['pattern']}")
    for key, label in (("minimum", ">="), ("minLength", "len >="), ("minItems", "len >=")):
        if key in node:
            bits.append(f"{label} {node[key]}")
    if "maxItems" in node:
        bits.append(f"len <= {node['maxItems']}")
    if "oneOf" in node:
        bits.append(f"exactly one of {len(node['oneOf'])} shapes")
    if node.get("additionalProperties") is False:
        bits.append("no extra keys")
    return "  ".join(bits) or "any"


def tree(node: dict, name: str = "<root>", indent: int = 0, required: set | None = None) -> None:
    mark = "*" if required and name in required else " "
    pad = "  " * indent
    print(f"{pad}{mark} {name:<22} {constraints(node)}")

    req = set(node.get("required", []))
    for key, sub in node.get("properties", {}).items():
        tree(sub, key, indent + 1, req)
    if "items" in node:
        tree(node["items"], "[each item]", indent + 1)
    for i, sub in enumerate(node.get("oneOf", [])):
        tree(sub, f"<shape {i + 1}: {sub.get('title', '?')}>", indent + 1)
    for block in node.get("allOf", []):
        cond = block.get("if", {}).get("properties", {})
        label = ", ".join(f"{k}={v.get('const')}" for k, v in cond.items())
        print(f"{pad}    when {label}:")
        for key, sub in block.get("then", {}).get("properties", {}).items():
            tree(sub, key, indent + 2)


# ---------------------------------------------------------------------------
# --vocab
# ---------------------------------------------------------------------------


def walk_keys(node, in_name_map=False, keywords=None, names=None):
    """Separate JSON Schema keywords from names we invented."""
    keywords = keywords if keywords is not None else {}
    names = names if names is not None else {}
    if isinstance(node, dict):
        for key, sub in node.items():
            if in_name_map:
                names[key] = names.get(key, 0) + 1
                walk_keys(sub, False, keywords, names)
            else:
                keywords[key] = keywords.get(key, 0) + 1
                walk_keys(sub, key in NAME_MAPS, keywords, names)
    elif isinstance(node, list):
        for item in node:
            walk_keys(item, in_name_map, keywords, names)
    return keywords, names


def vocab(schema: dict) -> None:
    keywords, names = walk_keys(schema)

    print("FIXED -- JSON Schema keywords. Defined by the draft 2020-12 spec, not by us.")
    print("Any validator in any language understands exactly these and no others.\n")
    for key in sorted(keywords):
        if key in VOCABULARY:
            print(f"  {key:<22} x{keywords[key]:<3} {VOCABULARY[key]}")

    unknown = [k for k in keywords if k not in VOCABULARY]
    if unknown:
        print("\n  not in this tool's vocabulary table (check spelling -- an unrecognised")
        print("  keyword is SILENTLY IGNORED by validators, it does not error):")
        for key in sorted(unknown):
            print(f"    {key}")

    print("\n\nOURS -- names we chose. Rename any of these and only our two")
    print("consumers care; the validator has no opinion about them.\n")
    for name in sorted(names):
        print(f"  {name:<22} x{names[name]}")


# ---------------------------------------------------------------------------
# --defs / --path
# ---------------------------------------------------------------------------


def resolve(schema: dict, pointer: str):
    """Resolve a JSON Pointer such as '#/$defs/expr'."""
    node = schema
    for part in pointer.lstrip("#/").split("/"):
        if not part:
            continue
        node = node[part] if isinstance(node, dict) else node[int(part)]
    return node


def defs(schema: dict) -> None:
    print("Reusable subschemas. $ref points at these; expr refers to ITSELF, which is")
    print("what makes nested expressions like a + b*2 expressible.\n")
    for name, node in schema.get("$defs", {}).items():
        used = json.dumps(schema).count(f'"#/$defs/{name}"')
        refs = sorted(set(find_refs(node)))
        print(f"  $defs/{name:<12} referenced {used}x", end="")
        print(f"   refers to: {', '.join(refs)}" if refs else "")


def find_refs(node):
    if isinstance(node, dict):
        for key, sub in node.items():
            if key == "$ref":
                yield sub.split("/")[-1]
            else:
                yield from find_refs(sub)
    elif isinstance(node, list):
        for item in node:
            yield from find_refs(item)


# ---------------------------------------------------------------------------
# --example
# ---------------------------------------------------------------------------

EXAMPLE = {
    "schema_version": "1.0",
    "generator": {"tool": "snax-forge", "version": "0.1.0", "recipe": "vecadd_tiled_spatial"},
    "cluster": {
        "name": "MapState",
        "units": [
            {
                "module_name": "snax_vector_tiled_spatial_Add",
                "family": "elementwise",
                "variant": "tiled_spatial",
                "shape": {
                    "lanes": 64,
                    "trips": "ceiling(N/64)",
                    "elements": "64*ceiling(N/64)",
                    "bounded": True,
                    "ragged": False,
                },
                "datapath": {
                    "label": "_Add_",
                    "source": "__out = (__in1 + __in2)",
                    "expr": {"op": "add", "args": [{"port": "__in1"}, {"port": "__in2"}]},
                },
                "ports": [
                    {
                        "name": "A",
                        "direction": "in",
                        "bind": "__in1",
                        "subset": "__i0",
                        "dtype": "int32",
                        "bits": 32,
                        "signed": True,
                        "shape": ["N"],
                    },
                    {
                        "name": "B",
                        "direction": "in",
                        "bind": "__in2",
                        "subset": "__i0",
                        "dtype": "int32",
                        "bits": 32,
                        "signed": True,
                        "shape": ["N"],
                    },
                    {
                        "name": "C",
                        "direction": "out",
                        "bind": "__out",
                        "subset": "__i0",
                        "dtype": "int32",
                        "bits": 32,
                        "signed": True,
                        "shape": ["N"],
                    },
                ],
            }
        ],
    },
}


# ---------------------------------------------------------------------------
# --validate
# ---------------------------------------------------------------------------


def validate(schema: dict, doc: dict, label: str) -> int:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    if not errors:
        print(f"  VALID   {label}")
        return 0
    print(f"  INVALID {label}  ({len(errors)} problem(s))\n")
    for err in errors:
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        print(f"    at   {where}")
        print(f"    why  {err.message}")
        # validator is the keyword that failed -- the fixed-vocabulary word
        # that rejected the document, which is the useful part when the
        # message alone is cryptic.
        print(f"    rule {err.validator} = {json.dumps(err.validator_value)[:70]}\n")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tree", action="store_true", help="structure, * marks required")
    ap.add_argument("--vocab", action="store_true", help="fixed keywords vs names we chose")
    ap.add_argument("--defs", action="store_true", help="reusable subschemas and their references")
    ap.add_argument("--path", metavar="PTR", help="print one subschema, e.g. '#/$defs/expr'")
    ap.add_argument("--example", action="store_true", help="a minimal valid descriptor")
    ap.add_argument("--validate", metavar="FILE", help="validate a descriptor and explain failures")
    args = ap.parse_args()

    schema = load()
    Draft202012Validator.check_schema(schema)

    if args.tree:
        print(f"{SCHEMA.name}: structure  (* = required)\n")
        tree(schema)
    elif args.vocab:
        vocab(schema)
    elif args.defs:
        defs(schema)
    elif args.path:
        print(json.dumps(resolve(schema, args.path), indent=2))
    elif args.example:
        print(json.dumps(EXAMPLE, indent=2))
    elif args.validate:
        return validate(schema, json.loads(Path(args.validate).read_text()), args.validate)
    else:
        # No flag: prove the schema is self-consistent and that the worked
        # example satisfies it, which is the fastest useful thing to run.
        print(f"schema: {SCHEMA}")
        print("  valid draft 2020-12 schema\n")
        validate(schema, EXAMPLE, "built-in example")
        print("\ntry --tree, --vocab, --defs, --path, --example, --validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())