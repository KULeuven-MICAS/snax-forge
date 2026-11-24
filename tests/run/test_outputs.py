"""run/outputs.py: a run directory written and read back."""

import pytest

from snax_forge.run import RunError, read_run


def test_not_a_run_directory(tmp_path):
    with pytest.raises(RunError) as e:
        read_run(tmp_path)
    assert e.value.code == "outputs.missing"
