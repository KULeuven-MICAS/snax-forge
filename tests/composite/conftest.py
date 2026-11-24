"""A composite package's tests run on one xdist worker, so its design is walked once."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        parts = Path(item.path).parts
        if "composite" in parts and parts.index("composite") + 2 < len(parts):
            design = parts[parts.index("composite") + 1]
            item.add_marker(pytest.mark.xdist_group(f"composite-{design}"))
