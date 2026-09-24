"""Tests that docs/CONTRACTS.md and the code cannot drift apart (MOD10c).

Two things are checked:

* every fenced block in the document that names a source is really in that
  source, either a checked-in file or a file produced by running
  ``scenarios/reduce``;
* the register names the document gives per block kind are the names the
  adapters produce, for every block of both checked-in clusters.

A failure here means one of the two moved: fix whichever is wrong.

Sections:
  1. helpers
  2. snippets are verbatim
  3. register names match the adapters
  4. the document still has its sections
"""

import re
from pathlib import Path

import pytest

from snax_forge.snax_model.scenario import (
    ClusterConfig,
    Scenario,
    register_map_of,
    run,
    write_outputs,
)

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "CONTRACTS.md"
SCEN = REPO / "scenarios"
CLUSTERS = ("alu4", "red4")

# <!-- snippet: path --> on its own line, immediately before a fenced block.
SNIPPET = re.compile(r"<!-- snippet: ([^\n]+?) -->\n```[a-z]*\n(.*?)```", re.DOTALL)


# =============================================================================
# 1. Helpers
# =============================================================================


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    """The output files of ``scenarios/reduce`` at beat level, by name.

    The profile and the trace have no checked-in file (a scenario holds
    inputs only, D41), so the document quotes a run and this test makes it.
    """
    sc = Scenario.load(SCEN / "reduce" / "scenario.json")
    out = write_outputs(run(sc, trace_level="beat"), tmp_path_factory.mktemp("reduce"))
    return {f"reduce/{f.name}": f.read_text() for f in out.iterdir() if f.suffix != ".npy"}


def snippets():
    return SNIPPET.findall(DOC.read_text())


def cluster(name):
    return ClusterConfig.load(SCEN / "clusters" / f"{name}.json")


# =============================================================================
# 2. Snippets are verbatim
# =============================================================================


def test_the_document_has_snippets():
    """A document with no snippets would make every check below vacuous."""
    sources = {src for src, _ in snippets()}
    assert len(snippets()) >= 10
    assert any(s.startswith("run:") for s in sources)
    assert {"scenarios/clusters/alu4.json", "scenarios/clusters/red4.json"} <= sources


@pytest.mark.parametrize("i", range(len(snippets())))
def test_snippet_is_verbatim(i, produced):
    """Every fenced block is copied out of the file it names.

    A ``.jsonl`` source is one event per line, so a snippet may pick lines
    that are not next to each other; every line must be in the file. Any
    other source must contain the block as it stands.
    """
    source, block = snippets()[i]
    if source.startswith("run:"):
        text = produced[source[len("run:") :]]
    else:
        path = REPO / source
        assert path.is_file(), f"{source}: no such file"
        text = path.read_text()
    if source.endswith(".jsonl"):
        lines = set(text.splitlines())
        for line in block.splitlines():
            assert line in lines, f"{source}: not in the file: {line}"
    else:
        assert block in text, f"{source}: the snippet is not in the file:\n{block}"


# =============================================================================
# 3. Register names match the adapters
# =============================================================================

# The layout of section 3 (streamer), section 4 (accelerator) and section 5
# (the DMA's registers and the three status registers), written out so that a
# change on either side shows up here.
STATUS = ["start", "busy", "busy_cycles"]


def documented_registers(block, spec, regmap):
    """The configuration registers section 3 / 4 / 5 gives for this block kind."""
    kind = regmap[block].kind
    if kind == "streamer":
        d = spec.config.get("temporal_dims", 1)
        s = len(regmap[block].adapter.spatial_bounds)
        return (
            ["base"]
            + [f"tbound[{i}]" for i in range(d)]
            + [f"tstride[{i}]" for i in range(d)]
            + [f"sstride[{i}]" for i in range(s)]
        )
    if kind == "accel":
        rates = []
        for p in regmap[block].comp.cfg.ports:
            if isinstance(p.rate, str) and p.rate not in rates:
                rates.append(p.rate)
        return ["n"] + rates
    if kind == "dma":
        d = spec.config.get("dims", 2)
        out = ["direction"]
        for side in ("src", "dst"):
            out += [f"{side}_base"]
            out += [f"{side}_bound[{i}]" for i in range(d)]
            out += [f"{side}_stride[{i}]" for i in range(d)]
        return out
    raise AssertionError(f"{block}: undocumented block kind {kind}")


@pytest.mark.parametrize("name", CLUSTERS)
def test_register_names_are_the_documented_ones(name):
    """Per block: status registers at 0..2, then the documented configuration ones."""
    cfg = cluster(name)
    regmap = register_map_of(cfg)
    specs = {c.name: c for c in cfg.components}
    for block, blk in regmap.blocks.items():
        expected = documented_registers(block, specs[block], regmap)
        assert blk.config_names == expected, block
        assert [blk.addr(r) - blk.base for r in STATUS] == [0, 1, 2], block
        assert blk.addr(expected[0]) == blk.base + 3, block


@pytest.mark.parametrize("name", CLUSTERS)
def test_every_register_named_in_the_document_exists(name):
    """Every ``block.register`` the document mentions resolves in a cluster."""
    text = DOC.read_text()
    regmap = register_map_of(cluster(name))
    # Registers are written either as a program's "reg" or in backticks;
    # anything else with a dot (a trace source, a file name) is not one.
    pattern = r"[a-z][a-z0-9_]*\.[a-z_]+(?:\[[^\]]*\])?"
    named = set(re.findall(rf'"reg": "({pattern})"', text))
    named |= set(re.findall(rf"`({pattern})`", text))
    named = {tuple(n.split(".", 1)) for n in named}
    seen = 0
    for block, reg in named:
        if block not in regmap.blocks:
            continue  # a block of the other cluster, or a file name
        seen += 1
        names = [*STATUS, *regmap[block].config_names]
        # A loop register may be written as a family (``tbound`` for
        # ``tbound[0]``, ``tbound[1]``, ...) in the prose.
        assert reg in names or any(n.startswith(reg + "[") for n in names), f"{block}.{reg}"
    assert seen, f"{name}: the document names no register of this cluster"


# =============================================================================
# 4. The document still has its sections
# =============================================================================


@pytest.mark.parametrize(
    "heading",
    [
        "## 1. Conventions",
        "## 2. Cluster configuration",
        "## 3. Streamer register layout",
        "## 4. Accelerator interface",
        "## 5. Control program",
        "## 6. Scenario and memory",
        "## 7. Profile and trace",
        "## 8. Rules for a new block kind",
        "## 9. Task list",
        "## 10. Block runtime model",
        "## 11. SNAX-DFG",
    ],
)
def test_section_exists(heading):
    assert heading in DOC.read_text()
