"""Every kernel under kernels/: its inputs, its NumPy reference and its SDFG.

For each kernel, ``make_inputs`` runs at its defaults, the kernel's reference
(``ref``, else the function itself) runs on them, and its simplified SDFG
builds, as ``pixi run sdfg`` builds it. Nothing is compiled. A kernel with type
presets is also imported at each, and its function in NumPy compared with the
reference executor on the graph.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.sdfg.build import build
from snax_forge.sdfg.loader import kernel_paths, load


@pytest.mark.slow  # each SDFG builds in DaCe
@pytest.mark.parametrize("name", sorted(kernel_paths()))
def test_kernel_inputs_reference_sdfg(name: str) -> None:
    spec = load(name)
    args = spec.make_inputs(np.random.default_rng(0))
    assert set(spec.descriptors) == set(args)
    spec.reference(**args)
    build(spec, simplify=True)


def _presets() -> list[tuple[str, str, dict[str, str]]]:
    out = [(name, p, {}) for name in sorted(kernel_paths()) for p in load(name).presets]
    return [*out, ("dot", "i8", {"ACC": "int8"}), ("vecmul", "i8", {"C": "int16"})]


@pytest.mark.slow  # each import builds its SDFG in DaCe
@pytest.mark.parametrize(("name", "preset", "dtypes"), _presets())
def test_a_preset_computes_what_numpy_does(name: str, preset: str, dtypes: dict) -> None:
    """The kernel's own function in NumPy, at the preset's types (its type parameters rebound),
    equals the reference executor on the imported graph, exactly (D143 · D175); dot at i8 with
    ACC int8 is NumPy's wrapped product, vecmul at i8 into int16 a wrapped output."""
    from snax_forge.dfg.import_sdfg import import_kernel
    from snax_forge.dfg.reference import reference_check

    graph = import_kernel(name, dtypes, preset)
    want = {**load(name).presets[preset], **dtypes}
    assert {c: graph.containers[c].dtype for c in want if c in graph.containers} == {
        c: d for c, d in want.items() if c in graph.containers}  # fmt: skip
    goldens = reference_check(load(name), graph, seed=0)  # raises on any difference
    assert all(np.asarray(v).size for v in goldens.outputs.values())
