"""Test session setup: the markers ``gvsoc`` (the full flow on SNAX-GVSoC: skipped where
out/snax-gvsoc/install is missing, run by CI's gvsoc job with ``-m gvsoc``) and ``slow`` (DaCe
builds and compiles, and the largest generated design: run by default, left out by
``pytest -m "not slow"`` for a quick local run); and the fixtures more than one package takes:
``roots``, an editor session's three folders under the test's tmp_path."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gvsoc: the full flow on SNAX-GVSoC, needs the GVSoC install in out/snax-gvsoc"
    )
    config.addinivalue_line(
        "markers", "slow: DaCe builds and compiles, the largest design; -m 'not slow' leaves them"
    )


@pytest.fixture
def roots(tmp_path):
    from snax_forge.sandbox.session import Roots

    return Roots(tmp_path / "sessions", tmp_path / "recipes", tmp_path / "sandbox")
