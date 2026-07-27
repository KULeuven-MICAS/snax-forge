"""Fail loudly if the environment can't do the one thing it exists to do."""

import sys

import networkx as nx
import numpy as np

FAILURES = []


def require(cond, msg):
    if not cond:
        FAILURES.append(msg)


nx_major, nx_minor = (int(x) for x in nx.__version__.split(".")[:2])
require(
    (nx_major, nx_minor) < (3, 6),
    f"networkx {nx.__version__} >= 3.6 will break DaCe codegen (block_parent_tree). "
    "Check the pin in pixi.toml.",
)

import dace

require(dace.__version__ == "1.0.2", f"expected DaCe 1.0.2, got {dace.__version__}")


@dace.program
def _smoke(a: dace.float64[64], b: dace.float64[64], c: dace.float64[64]):
    c[:] = a * 2.0 + b


try:
    a, b = np.random.rand(64), np.random.rand(64)
    c = np.zeros(64)
    _smoke(a, b, c)
    require(np.allclose(c, a * 2.0 + b), "smoke kernel compiled but produced wrong output")
except Exception as exc:  # noqa: BLE001
    FAILURES.append(f"end-to-end SDFG build/compile/run failed: {exc!r}")

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    FAILURES.append("matplotlib missing")

if FAILURES:
    print("ENVIRONMENT CHECK FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)

print(
    f"OK  python {sys.version.split()[0]}  dace {dace.__version__}  "
    f"networkx {nx.__version__}  numpy {np.__version__}"
)
