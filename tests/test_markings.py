"""The code's markings (CLAUDE.md, Conventions): an ``NB (Dn)`` comment cites a decision that has a
heading in docs/DECISIONS.md; no other decision or experiment tag stands in the code or the tests
(snax_forge/, tests/, hw/chisel/src/, and the submodule's snax/ and tests/snax/); and every
``CONTRACTS, "…"`` citation names a section of docs/CONTRACTS.md. Run by ``pixi run test-docs``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.helpers import REPO, SUB

TREES = [REPO / "snax_forge", REPO / "tests", REPO / "hw" / "chisel" / "src", SUB / "snax",
         SUB / "tests" / "snax"]  # fmt: skip
SUFFIXES = {".py", ".scala", ".cpp", ".hpp", ".h", ".js", ".txt", ".json", ".hjson"}
SKIP = ("vendor", "__pycache__")
# A technology's tables, copied verbatim from snax-forge-experiments (their sha256 in its SOURCE)
IMPORTED = REPO / "snax_forge" / "char"
NB = re.compile(r"NB \((D\d+)\)")
# A decision (D) or experiment (X) number standing as a tag: letter, digits, not part of a name
TAG = re.compile(r"(?<![\w.$/-])[DX]\d{1,3}(?![\w.$/-])")
CITE = re.compile(r'CONTRACTS, "([^"]+)"')


def sources() -> list[Path]:
    out = []
    for tree in TREES:
        if tree.is_dir():
            out += [p for p in tree.rglob("*") if p.is_file() and p.suffix in SUFFIXES
                    and not any(s in p.parts for s in SKIP) and p != Path(__file__)
                    and not (p.parent.parent == IMPORTED and (p.parent / "SOURCE").is_file())]  # fmt: skip
    return sorted(out)


def headings() -> set[str]:
    """Every D-number of a docs/DECISIONS.md heading (``### D51 · D52 — ...``)."""
    text = (REPO / "docs" / "DECISIONS.md").read_text()
    return {d for line in text.splitlines() if line.startswith("### ")
            for d in re.findall(r"D\d+", line.split(" — ")[0])}  # fmt: skip


def sections() -> set[str]:
    """docs/CONTRACTS.md's section titles, without their numbers."""
    text = (REPO / "docs" / "CONTRACTS.md").read_text()
    return {re.sub(r"^#+ (\d+\. )?", "", line).strip() for line in text.splitlines()
            if line.startswith(("## ", "### "))}  # fmt: skip


@pytest.fixture(scope="module")
def lines() -> list[tuple[str, int, str]]:
    out = []
    for p in sources():
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        rel = p.relative_to(REPO).as_posix()
        out += [(rel, i, line) for i, line in enumerate(text.splitlines(), 1)]
    return out


def test_every_nb_cites_a_decision_with_a_heading(lines):
    have = headings()
    bad = [f"{f}:{i}: {d}" for f, i, line in lines for d in NB.findall(line) if d not in have]
    assert not bad, "NB cites a decision with no heading in docs/DECISIONS.md: " + "; ".join(bad)


def test_no_other_tag_stands_in_the_code(lines):
    """Outside an ``NB (Dn)`` marker, no D- or X-number: the sentence says what the tag stood for."""
    bad = [f"{f}:{i}: {line.strip()[:80]}" for f, i, line in lines
           if TAG.search(NB.sub("", line))]  # fmt: skip
    assert not bad, f"{len(bad)} tags left (write the sentence without it):\n" + "\n".join(bad[:40])


def test_every_contracts_citation_names_a_section(lines):
    have = sections()
    bad = [
        f"{f}:{i}: {t}"
        for f, i, line in lines
        for t in CITE.findall(line)
        if t not in have and not t.startswith("<")
    ]  # a placeholder, as in test_contracts.py
    assert not bad, "CONTRACTS cites a section it does not have: " + "; ".join(bad)
