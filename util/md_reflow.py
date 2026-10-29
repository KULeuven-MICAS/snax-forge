"""Markdown prose to one line per paragraph, list item and table row (D117).

    python util/md_reflow.py [FILE ...]           reflow the files in place
    python util/md_reflow.py --check [FILE ...]   list the hard-wrapped paragraphs; exit 1 if any

Without files, snax-forge's own: ``git ls-files '*.md' ':!gvsoc/'`` (``pixi run lint`` runs the
check on them).

**Joined**, with one space, each continuation line's indent dropped and nothing else: the lines of
a paragraph, of a list item or a footnote (its marker and indent kept) and of a paragraph inside
one, and of a blockquote's paragraphs (the ``>`` kept once, on the first line).

**Never touched**: fenced code blocks (``` and ~~~, any info string), indented code blocks
outside lists, tables (rows starting with ``|``), headings (ATX, and setext: no line of a
paragraph an underline follows is joined), thematic breaks, display maths (``$$`` lines and
blocks), HTML comments and blocks (``<!-- snippet: ... -->`` stays on its own line, right before
its fence), link reference definitions, front matter, and the break after a line ending in two
spaces or a backslash. A line indented by four or more past its container is never joined onto a
paragraph, so a code block written without its blank line stays as it was.

The whitespace-separated words of a file are the same before and after, but for the ``>`` a
blockquote's joined lines drop (``words``): ``reflow_text`` asserts it.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})")
HEADING = re.compile(r"^ {0,3}#{1,6}(\s|$)")
SETEXT = re.compile(r"^ {0,3}(=+|-+)\s*$")
BREAK = re.compile(r"^ {0,3}([-*_])(\s*\1){2,}\s*$")
ITEM = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)]|\[\^[^\]]+\]:)( +|$)")
LINKREF = re.compile(r"^ {0,3}\[[^\]]+\]:")
QUOTE = re.compile(r"^ {0,3}> ?")
MARKERS = re.compile(r"^( {0,3}> ?)+")
BLOCK_TAGS = (
    "address|article|aside|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|"
    "dl|dt|fieldset|figcaption|figure|footer|form|h[1-6]|head|header|hr|html|iframe|legend|li|"
    "link|main|menu|nav|ol|optgroup|option|p|param|pre|script|section|style|summary|table|tbody|"
    "td|textarea|tfoot|th|thead|title|tr|ul"
)
HTML = re.compile(
    rf"^ {{0,3}}(<!--|<\?|<![A-Za-z]|<!\[CDATA\[|</?({BLOCK_TAGS})(\s|/?>|$)"
    r"|</?[A-Za-z][A-Za-z0-9-]*(\s[^<>]*)?/?>\s*$)",
    re.IGNORECASE,
)


def indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def protected(lines: list[str]) -> set[int]:
    """The lines of every paragraph a setext underline follows: a heading, never joined."""
    out: set[int] = set()
    for k, line in enumerate(lines):
        if k and SETEXT.match(line) and lines[k - 1].strip():
            j = k - 1
            while j >= 0 and lines[j].strip():
                out.add(j)
                j -= 1
    return out


def reflow(lines: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """``lines`` reflowed (module doc), and each joined paragraph's first line (0-based) and how
    many lines it had."""
    out: list[str] = []
    src: list[int] = []  # the input line each output line starts at
    joined: dict[int, int] = {}  # output index -> lines joined into it
    keep = protected(lines)
    stack: list[int] = []  # the content indents of the open list items
    para: int | None = None  # the output index of the open paragraph's line

    def emit(k: int, text: str) -> None:
        out.append(text)
        src.append(k)

    i, n = 0, len(lines)
    if n and lines[0].strip() == "---":  # front matter
        end = next((k for k in range(1, n) if lines[k].strip() in ("---", "...")), None)
        if end is not None:
            for k in range(end + 1):
                emit(k, lines[k])
            i = end + 1
    while i < n:
        line, s, ind = lines[i], lines[i].strip(), indent(lines[i])
        if not s:
            emit(i, line)
            para, i = None, i + 1
            continue
        block = bool(
            FENCE.match(line) or HEADING.match(line) or BREAK.match(line) or ITEM.match(line)
            or s.startswith(("|", "$$")) or HTML.match(line) or LINKREF.match(line)
            or QUOTE.match(line)
        )  # fmt: skip
        if para is None or block:
            while stack and ind < stack[-1]:
                stack.pop()
        base = stack[-1] if stack else 0
        if m := FENCE.match(line):
            mark = m[2]
            close = re.compile(rf"^\s*{re.escape(mark[0])}{{{len(mark)},}}\s*$")
            emit(i, line)
            i += 1
            while i < n:
                emit(i, lines[i])
                i += 1
                if close.match(lines[i - 1]):
                    break
            para = None
            continue
        if para is None and ind >= base + 4:  # an indented code block
            while i < n and (not lines[i].strip() or indent(lines[i]) >= base + 4):
                emit(i, lines[i])
                i += 1
            continue
        if s.startswith("$$") and (s == "$$" or not s.endswith("$$")):  # a display maths block
            emit(i, line)
            i += 1
            while i < n:
                emit(i, lines[i])
                i += 1
                if lines[i - 1].strip().endswith("$$"):
                    break
            para = None
            continue
        if s.startswith("<!--"):
            while True:
                emit(i, lines[i])
                i += 1
                if "-->" in lines[i - 1] or i == n:
                    break
            para = None
            continue
        if HTML.match(line):  # an HTML block, up to a blank line
            while i < n and lines[i].strip():
                emit(i, lines[i])
                i += 1
            para = None
            continue
        if q := QUOTE.match(line):
            group = []
            while i < n and QUOTE.match(lines[i]):
                group.append(i)
                i += 1
            inner, inner_joined = reflow([QUOTE.sub("", lines[k], count=1) for k in group])
            prefix = q[0]
            for text in inner:
                emit(group[0], prefix + text if text else prefix.rstrip())
            # the inner lines' sources are not tracked one by one: a quote is short
            for start, count in inner_joined:
                joined[len(out) - len(inner) + start] = count
            para = None
            continue
        if (HEADING.match(line) or BREAK.match(line) or s.startswith(("|", "$$"))
                or (LINKREF.match(line) and not ITEM.match(line))):  # fmt: skip
            emit(i, line)
            para, i = None, i + 1
            continue
        hard = line.endswith(("  ", "\\"))
        if m := ITEM.match(line):
            content = len(m[1]) + len(m[2]) + (len(m[3]) if 0 < len(m[3]) <= 4 else 1)
            stack.append(content)
            emit(i, line)
            para = None if hard or not line[m.end() :].strip() or i in keep else len(out) - 1
            i += 1
            continue
        if para is not None and i not in keep and ind < base + 4:
            out[para] = out[para].rstrip(" ") + " " + s
            joined[para] = joined.get(para, 1) + 1
        else:
            emit(i, line)
            para = len(out) - 1
        if hard or i in keep:
            para = None
        i += 1
    return out, sorted((src[k], c) for k, c in joined.items())


def words(text: str) -> list[str]:
    """The whitespace-separated words of ``text``, each line's blockquote markers left out."""
    return [w for line in text.split("\n") for w in MARKERS.sub("", line).split()]


def reflow_text(text: str) -> tuple[str, list[tuple[int, int]]]:
    """``text`` reflowed, and its joined paragraphs (``reflow``); the words are unchanged."""
    lines, joined = reflow(text.split("\n"))
    new = "\n".join(lines)
    assert words(new) == words(text), "the reflow changed a word"
    return new, joined


def own_files() -> list[Path]:
    """snax-forge's tracked Markdown files, the submodule's left out."""
    root = Path(__file__).resolve().parents[1]
    names = subprocess.run(["git", "ls-files", "*.md", ":!gvsoc/"], cwd=root, check=True,
                           capture_output=True, text=True).stdout.split()  # fmt: skip
    return [root / x for x in names]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", type=Path, help="Markdown files (default: own_files)")
    ap.add_argument("--check", action="store_true", help="list hard-wrapped paragraphs, exit 1")
    args = ap.parse_args(argv)
    bad = 0
    for path in args.files or own_files():
        text, name = path.read_text(), os.path.relpath(path)
        new, joined = reflow_text(text)
        if args.check:
            for start, count in joined:
                print(f"{name}:{start + 1}: a paragraph hard-wrapped over {count} lines")
            bad += len(joined)
        elif new != text:
            path.write_text(new)
            print(f"{name}: {text.count(chr(10))} -> {new.count(chr(10))} lines")
    if args.check and bad:
        print(f"{bad} hard-wrapped paragraphs: python util/md_reflow.py FILE joins them (D117)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
