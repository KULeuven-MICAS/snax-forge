"""A kernel's inputs and its reference against a run of a graph or an SDFG.

    kernel_spec(kernel, why) -> KernelSpec
    make_inputs(spec, seed=0, **sizes) -> {container: array}
    kernel_inputs(spec, graph, seed) -> {container: array}
    mismatches(spec, inputs, run) -> {output: (elements that differ, elements)}
    reference_check(spec, graph, seed)

``kernel_inputs`` calls the kernel's ``make_inputs`` at every size the graph binds, each array
cast to its container's dtype; data.h's inputs and the flow's check use it. ``mismatches`` is
the one comparison of a run with the kernel's reference (``execute.against_reference``, each
on its own copy of the inputs): the reference executor's in ``reference_check`` (the flow's
guard that the goldens are the kernel's, D108) and ``dfg check``'s, and a compiled SDFG's in
``sdfg.build.verify``.

A kernel name that is not one, or inputs that do not fit the graph, raise KernelError naming
it; the flow raises each as its FlowError, with the same text. Not imported by
``snax_forge.dfg`` (the kernels' specs import DaCe).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from snax_forge.sdfg.loader import load as load_kernel

from .execute import against_reference, execute
from .graph import Graph


class KernelError(ValueError):
    """A kernel or its inputs that do not fit (the module doc)."""


def kernel_spec(kernel: str, why: str) -> Any:
    """The kernel's spec; an unknown name is a KernelError that says where the name came
    from."""
    try:
        return load_kernel(kernel)
    except KeyError as e:
        raise KernelError(f"kernel ({why}): {e.args[0]}") from None


def make_inputs(spec: Any, seed: int = 0, **sizes: int) -> dict[str, Any]:
    """The kernel's ``make_inputs`` at ``sizes``, from a generator seeded with ``seed``."""
    return spec.make_inputs(np.random.default_rng(seed), **sizes)


def kernel_inputs(spec: Any, graph: Graph, seed: int) -> dict[str, np.ndarray]:
    """The kernel's ``make_inputs`` at every size the graph binds: each bound symbol by its
    lower-case name (``M`` gives ``m=``, ``N`` gives ``n=``); then the arrays' shapes must give
    the graph's values (``spec.bind_symbols``). Each array is cast to its container's dtype
    (``astype``, wrapping: a recipe's ``dtypes``, D113)."""
    params = inspect.signature(spec.make_inputs).parameters
    kw = {}
    for sym, v in graph.bound_symbols.items():
        if sym.lower() not in params:
            raise KernelError(
                f"inputs: the graph binds {sym} = {v}, but {spec.name}'s make_inputs has no "
                f"parameter {sym.lower()!r} (it takes {list(params)})"
            )
        kw[sym.lower()] = v
    inputs = make_inputs(spec, seed, **kw)
    got = spec.bind_symbols(inputs)
    for sym, v in graph.bound_symbols.items():
        if got.get(sym) != v:
            raise KernelError(
                f"inputs: {spec.name}'s make_inputs({kw}) gives {sym} = {got.get(sym)}, "
                f"the graph binds {v}"
            )
    for c, arr in inputs.items():
        cont = graph.containers.get(c)
        if cont is None:
            raise KernelError(
                f"inputs: {spec.name}'s make_inputs gives {c!r}, the graph has no such container"
            )
        if str(np.asarray(arr).dtype) != cont.dtype:
            inputs[c] = np.asarray(arr).astype(cont.dtype)
    return inputs


def mismatches(
    spec: Any, inputs: Mapping[str, Any], run: Callable[[dict[str, Any]], Mapping[str, Any]]
) -> dict[str, tuple[int, int]]:
    """``run`` against the kernel's reference on ``inputs``: each output container (the spec's
    ``inout``, in its order) that differs, with how many of its elements do and its size."""
    return {
        c: (int(np.count_nonzero(got != ref)), int(np.asarray(ref).size))
        for c, (got, ref) in against_reference(spec, inputs, run).items()
        if not np.array_equal(got, ref)
    }


def reference_check(spec: Any, graph: Graph, seed: int) -> None:
    """The reference executor on ``graph`` against the kernel's reference, on the kernel's
    inputs: data.h's goldens are the first, so a graph that computes something else (one
    edited by hand, D108) would pass its own check; a KernelError naming each output that
    differs."""
    wrong = mismatches(spec, kernel_inputs(spec, graph, seed), lambda x: execute(graph, x))
    if wrong:
        text = ", ".join(f"{c} in {n} of {size} elements" for c, (n, size) in wrong.items())
        raise KernelError(f"gen: the graph's reference executor differs from the {spec.name} "
                          f"reference: {text}; the goldens would not be the kernel's "
                          "(D108)")  # fmt: skip
