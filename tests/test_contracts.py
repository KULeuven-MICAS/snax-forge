"""Tests that docs/CONTRACTS.md and the code cannot drift apart (D46).

Every fenced block in the document that names a source is really in that
source, a checked-in file (C§7's are the flow's ``run/profile.py`` and
``run/trace.py``), and the document still has its sections, the retired ones
(C§2–C§6, C§8) included, so their numbers stay. A failure here means one of
the two moved: fix whichever is wrong.

Sections:
  1. helpers
  2. snippets are verbatim
  3. the document still has its sections
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "CONTRACTS.md"

# <!-- snippet: path --> on its own line, immediately before a fenced block.
SNIPPET = re.compile(r"<!-- snippet: ([^\n]+?) -->\n```[a-z]*\n(.*?)```", re.DOTALL)


# =============================================================================
# 1. Helpers
# =============================================================================


def snippets():
    return SNIPPET.findall(DOC.read_text())


# =============================================================================
# 2. Snippets are verbatim
# =============================================================================


def test_the_document_has_snippets():
    """A document with no snippets would make every check below vacuous."""
    sources = {src for src, _ in snippets()}
    assert len(snippets()) >= 10
    assert {"snax_forge/run/profile.py", "snax_forge/run/trace.py"} <= sources


@pytest.mark.parametrize("i", range(len(snippets())))
def test_snippet_is_verbatim(i):
    """Every fenced block is copied out of the file it names.

    A ``.jsonl`` source is one event per line, so a snippet may pick lines
    that are not next to each other; every line must be in the file. Any
    other source must contain the block as it stands.
    """
    source, block = snippets()[i]
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
# 3. The document still has its sections
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
        "## 12. Recipe",
        "## 13. Platform",
        "## 14. Design checks",
        "## 15. Memory plan and design point",
        "## 16. Generated hjson files",
    ],
)
def test_section_exists(heading):
    assert heading in DOC.read_text()
