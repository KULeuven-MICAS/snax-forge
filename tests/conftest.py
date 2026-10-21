"""Test session setup: the marker ``gvsoc`` (the full flow on SNAX-GVSoC, D136: skipped where
out/snax-gvsoc/install is missing, run by CI's gvsoc job with ``-m gvsoc``)."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gvsoc: the full flow on SNAX-GVSoC, needs the GVSoC install in out/snax-gvsoc"
    )
