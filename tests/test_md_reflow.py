"""util/md_reflow.py (D117): prose joined to one line per paragraph, every other block as it is."""

import importlib.util
import subprocess
import sys

import pytest

from tests.helpers import REPO

_spec = importlib.util.spec_from_file_location("md_reflow", REPO / "util" / "md_reflow.py")
md_reflow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(md_reflow)


def reflowed(text: str) -> str:
    return md_reflow.reflow_text(text)[0]


@pytest.mark.parametrize(
    "before, after",
    [
        ("one\ntwo\nthree\n\nnext\npara\n", "one two three\n\nnext para\n"),
        ("- item\n  more\n- next\n  1. sub\n     more\n", "- item more\n- next\n  1. sub more\n"),
        ("1. a\n   b\n\n   in the item\n   too\n", "1. a b\n\n   in the item too\n"),
        ("lazy item\n- x\ny\n", "lazy item\n- x y\n"),
        ("> quoted\n> lines\n>\n> again\n", "> quoted lines\n>\n> again\n"),
        ("[^1]: a note\n    goes on\n", "[^1]: a note goes on\n"),
    ],
)  # fmt: skip
def test_prose_is_joined(before, after):
    assert reflowed(before) == after


@pytest.mark.parametrize(
    "text",
    [
        "```python\na\nb\n```\n",
        "~~~mermaid\ngraph TD\n  A --> B\n~~~\n",
        "text\n\n    indented\n    code\n",
        "| a | b |\n|---|---|\n| 1 | 2 |\n",
        "# Head\n## Two\n",
        "Setext\nheading\n===\n",
        "$$\nx = 1\ny\n$$\n",
        "$$y = x$$\n$$z$$\n",
        "<!-- snippet: a.json -->\n```json\n{}\n```\n",
        "<!-- a\ncomment -->\n",
        "<div>\nraw\nhtml\n</div>\n",
        "[ref]: https://example.org\n[two]: https://example.org/2\n",
        "---\ntitle: x\nmore: y\n---\n",
        "hard  \nbreak\\\nhere\n",
        "***\n",
    ],
)  # fmt: skip
def test_other_blocks_stay(text):
    assert reflowed(text) == text


def test_a_list_item_keeps_its_code_block():
    text = "- item\n\n      code\n      more\n"
    assert reflowed(text) == text


def test_the_check_lists_wrapped_paragraphs(tmp_path):
    (tmp_path / "a.md").write_text("one\ntwo\n\n- ok\n")
    (tmp_path / "b.md").write_text("one line\n")
    run = subprocess.run([sys.executable, str(REPO / "util" / "md_reflow.py"), "--check",
                          str(tmp_path / "a.md"), str(tmp_path / "b.md")],
                         capture_output=True, text=True, check=False)  # fmt: skip
    assert run.returncode == 1
    assert f"{tmp_path / 'a.md'}:1: a paragraph hard-wrapped over 2 lines" in run.stdout
    assert "b.md" not in run.stdout
