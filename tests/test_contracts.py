"""Tests that docs/CONTRACTS.md and the code cannot drift apart.

Every fenced block in the document that names a source is really in that
source, a checked-in file (a file of the snax-gvsoc submodule is checked where
the submodule is checked out, as tests/gen's fixture test), and every section
the repository cites, as ``C§n`` or ``CONTRACTS, "<title>"``, exists. A failure
here means one of the two moved: fix whichever is wrong.

Sections:
  1. helpers
  2. snippets are verbatim
  3. the sections and their cites
"""

import re
import subprocess

import pytest

from tests.helpers import REPO, SUBMODULE_MISSING

DOC = REPO / "docs" / "CONTRACTS.md"
HERE = "tests/test_contracts.py"  # its own examples are no cites
# The snax-gvsoc submodule: CI's python, chisel and gvsoc jobs fetch it
SUBMODULE = "gvsoc/snax-gvsoc/"

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
    if source.startswith(SUBMODULE) and not (REPO / SUBMODULE / "snax").is_dir():
        pytest.skip(f"{source}: {SUBMODULE_MISSING}")
    assert path.is_file(), f"{source}: no such file"
    text = path.read_text()
    if source.endswith(".jsonl"):
        lines = set(text.splitlines())
        for line in block.splitlines():
            assert line in lines, f"{source}: not in the file: {line}"
    else:
        assert block in text, f"{source}: the snippet is not in the file:\n{block}"


# =============================================================================
# 3. The sections and their cites
# =============================================================================


def sections() -> dict[int, str]:
    """Number -> title of every ``## n. Title`` heading."""
    return {int(n): t for n, t in re.findall(r"^## (\d+)\. (.+)$", DOC.read_text(), re.MULTILINE)}


def cites() -> list[tuple[str, str]]:
    """Every (file, cite) in the repository's tracked text files but the document itself."""
    files = subprocess.run(
        [
            "git",
            "grep",
            "-lE",
            r'C§[0-9]|CONTRACTS, "',
            "--",
            ".",
            ":!docs/CONTRACTS.md",
            ":!" + HERE,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    out = []
    for f in files:
        text = (REPO / f).read_text()
        for c in re.findall(r'C§\d+|CONTRACTS, \\?"[^"\\]+\\?"', text):
            # a title wrapped over comment or docstring lines reads as one
            out.append((f, re.sub(r"\s*\n\s*(?:#|\*|//)?\s*", " ", c)))
    return out


def test_the_sections_are_numbered_in_order():
    assert list(sections()) == list(range(1, len(sections()) + 1))


def test_every_cited_section_exists():
    """``C§n`` names a section by number, ``CONTRACTS, "<title>"`` by its title."""
    numbers, titles = set(sections()), set(sections().values())
    assert cites(), "no cite found: the search is broken"
    bad = []
    for f, c in cites():
        if c.startswith("C§"):
            ok = int(c[2:]) in numbers
        else:
            ok = c.split(", ", 1)[1].strip('\\"') in titles
        if not ok:
            bad.append(f"{f}: {c}")
    assert not bad, "cites of no section:\n" + "\n".join(bad)
