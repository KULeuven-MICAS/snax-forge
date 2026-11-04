"""gen/rtl.py's cached classpath (C§17): one file per environment, and an entry that is gone or
cannot be checked (another environment's ``/root/...``) leads to a new sbt export. sbt is mocked;
hw/chisel is a temporary folder."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from snax_forge.gen import rtl


@pytest.fixture
def chisel(tmp_path, monkeypatch):
    """A hw/chisel with a build.sbt and one source, an sbt that exports a classpath of one jar
    that exists, and the export calls counted."""
    root = tmp_path / "chisel"
    (root / "src" / "main").mkdir(parents=True)
    (root / "build.sbt").write_text("")
    (root / "src" / "main" / "A.scala").write_text("")
    jar = tmp_path / "jars" / "a.jar"
    jar.parent.mkdir()
    jar.write_text("")
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=f"[info] ok\n{jar}\n", stderr="")

    monkeypatch.setattr(rtl, "CHISEL", root)
    monkeypatch.setattr(rtl.subprocess, "run", run)
    return root, str(jar), calls


def _cache(path: Path, entries: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(entries + "\n")
    os.utime(path, (2**31, 2**31))  # newer than every source


def test_a_cached_classpath_is_used_while_its_entries_are_there(chisel):
    _, jar, calls = chisel
    _cache(rtl.classpath_file(), jar)
    assert rtl.emit_classpath() == jar and calls == []


@pytest.mark.parametrize("entry", ["/no/such/dir/a.jar", "/x/" + "a" * 5000 + ".jar"])
def test_an_entry_gone_or_that_cannot_be_checked_exports_again(chisel, entry):
    """A missing jar; a path whose stat raises OSError (ENAMETOOLONG here, a PermissionError on
    another environment's /root/... on the host): both are gone, so sbt exports again."""
    _, jar, calls = chisel
    _cache(rtl.classpath_file(), f"{jar}:{entry}")
    assert rtl.emit_classpath() == jar and len(calls) == 1
    assert rtl.classpath_file().read_text().strip() == jar


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any directory")
def test_an_unreadable_entry_exports_again(chisel, tmp_path):
    """The host's case: the image's /root/.cache/... jars, a PermissionError from Path.exists."""
    _, jar, calls = chisel
    locked = tmp_path / "locked"
    (locked / "inner").mkdir(parents=True)
    locked.chmod(0)
    try:
        _cache(rtl.classpath_file(), str(locked / "inner" / "x.jar"))
        assert rtl.emit_classpath() == jar and len(calls) == 1
    finally:
        locked.chmod(0o755)


def test_two_environments_keep_their_own_files(chisel, monkeypatch):
    """The image's and the host's classpaths, named after each environment's hash, coexist: each
    reads its own, an old emit-classpath.txt is ignored."""
    root, jar, calls = chisel
    image, host = "coursier=/root/.cache/coursier\njava=/usr/bin/java\n", rtl.classpath_env()
    a, b = rtl.classpath_file(image), rtl.classpath_file(host)
    assert a != b and a.parent == b.parent == root / "target"
    assert a.name.startswith("emit-classpath-") and len(a.stem) == len("emit-classpath-") + 8
    _cache(a, "/root/.cache/coursier/v1/scala-library.jar")
    _cache(root / "target" / "emit-classpath.txt", "/root/.cache/coursier/v1/old.jar")
    assert rtl.emit_classpath() == jar and len(calls) == 1  # the host's: none yet, exported
    assert b.read_text().strip() == jar
    assert a.read_text().strip() == "/root/.cache/coursier/v1/scala-library.jar"  # untouched
    assert rtl.emit_classpath() == jar and len(calls) == 1  # the host's again, from its file
    monkeypatch.setattr(rtl, "classpath_env", lambda: image)
    assert rtl.classpath_file() == a
