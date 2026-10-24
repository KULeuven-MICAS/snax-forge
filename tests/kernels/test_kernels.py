"""Every kernel under kernels/: its inputs, its NumPy reference and its SDFG.

For each kernel, ``make_inputs`` runs at its defaults, the kernel's reference
(``ref``, else the function itself) runs on them, and its simplified SDFG
builds, as ``pixi run sdfg`` builds it. Nothing is compiled, and nothing is
compared: the staged kernels are not linked to the flow.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.sdfg.build import build
from snax_forge.sdfg.loader import kernel_paths, load


@pytest.mark.parametrize("name", sorted(kernel_paths()))
def test_kernel_inputs_reference_sdfg(name: str) -> None:
    spec = load(name)
    args = spec.make_inputs(np.random.default_rng(0))
    assert set(spec.descriptors) == set(args)
    spec.reference(**args)
    build(spec, simplify=True)
