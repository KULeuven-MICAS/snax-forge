# Kernels

Every kernel SNAX-FORGE targets, one file each with a `KernelSpec`. The flow uses `vecadd`, `vecmul`, `vsum`, `dot` and `relu` (`recipes/`). The kernels tagged `staged` are here to be read and their graphs looked at (`pixi run sdfg K`, `pixi run import-dfg K`, `pixi run view-dfg`); they are not linked to the flow. `pixi run kernels` prints each kernel's directory, tier, dtype, whether its SDFG builds and whether `import-dfg` takes it, or why not.

The PolyBench and DL kernels are NPBench's DaCe sources (BSD-3, `NPBENCH_LICENSE`) with their structure kept (D113):

- a `return X` becomes `out[:] = X`;
- a body that only runs under DaCe (it loops over symbols or calls DaCe programs) has NPBench's NumPy version as its reference (`ref`);
- they are `int64` where the kernel is integer arithmetic, at PolyBench's MINI sizes; `softmax`, `mlp`, `lenet` and `resnet` keep NPBench's `float32` (`float-staged`) until their tier decides the integer form;
- the tiers are PAPER §6.1's.

The kernels that run at narrow types (`vecadd`, `vecmul`, `matadd`, `relu`, `axpy`, `vsum`, `rowsum`, `dot`, `matvec`, `mvt_types`, `bicg_types`, `atax_types`, `gesummv_types`) list type presets in their `SPEC` (`presets`), chosen with `--set types=` (D113 · D173):

| Preset | Inputs | Outputs and `OP_TYPE` |
|---|---|---|
| `i8` | `int8` | `int32` |
| `i16` | `int16` | `int32` |
| `i32` | `int32` | `int64` |

- `OP_TYPE`, a type parameter (`SPEC.types`), is the type a product or sum runs at, written as a cast in the kernel: dot is `np.sum(A.astype(OP_TYPE) * B.astype(OP_TYPE))`. At its default (`int64`, or the containers' own type) the cast changes nothing.
- NumPy's own form, products that wrap at the inputs' type, is `--set dtypes.OP_TYPE=int8` on top of `i8`; `tests/kernels/` checks each preset's graph against the kernel's function in NumPy.

## Regularity

The kernels on two axes of regularity: access, what the streamers do, across; compute, what the unit does, down. The argument is PAPER §6.2's.

| Compute ↓ / Access → | Affine, no reuse | Affine with reuse | Non-rectangular or data-dependent |
|---|---|---|---|
| II 1, fixed latency (elementwise) | `vecadd`, `vecmul`, `matadd`, `relu`, `axpy`, scale by α/β | `gemm` (lanes), `gemver`'s rank-1 update (stride 0) | triangular kernels (left out) |
| Fold, loop-carried (reduction) | `vsum`, `dot`, `rowsum`, mat-vecs (`atax`, `bicg`, `mvt`, `gesummv`) | `gemm`, `k2mm`, `k3mm`, `doitgen`, `conv2d_bias` | `covariance`'s triangular product |
| Multi-cycle, fixed latency or II > 1 (div, sqrt, exp by polynomial, CORDIC or Newton-Raphson; FP adders) | `vdiv`, `covariance`'s divide | `jacobi1d`, `jacobi_2d` (integer divide now, fixed point or FP later) | |
| Multi-phase | `softmax` (max, then exp and sum, then divide), `covariance` (mean, center, product) | attention, `mlp`, `lenet`, `resnet` (batchnorm, skip) | |
| Data-dependent latency or control | early-exit dividers, LUT methods | | `durbin`, `nussinov` (left out) |

## blas/

### axpy

`blas/axpy.py` · no tier · int32 (two per 64-bit word, D142)

$$y \leftarrow y + \alpha x$$

- `alpha` is a scalar: the importer makes it a one-element container with attr `sdfg.scalar`; in hardware it would live in a CSR or a streamer with stride 0.
- `y` is updated in place: read and written by the same task (open item 47).

### matvec

`blas/matvec.py` · no tier · int64 · M13 B1x's row form of dot

$$q = A p, \quad A \in \mathbb{Z}^{M \times N}$$

- Written as `q[:] = np.sum(A * p, axis=1)`, so the import mirrors dot's: a multiply map into the transient `tmp0[M, N]`, then a sum map folding into `q[i]`, each inside its own row map `i` (`mult_map` ⊃ `mult_map_1` ⊃ `mult`, `sum_map` ⊃ `sum_map_1` ⊃ `sum`).
- `recipes/matvec/matvec_<k>.json` bind it as dot's variants with the row loop tagged temporal, the jobs; `p` is read again for every row (stride 0). matvec 1 runs at 8 × 512: at 16 × 512 its `tmp0` does not fit the L1 (open item 97).

### matvec_acc

`blas/matvec_acc.py` · no tier · int64 · a fold with a seed (D163)

$$q \mathrel{+}= A p, \quad A \in \mathbb{Z}^{M \times N}$$

- Written as `q += np.sum(A * p, axis=1)`: matvec's two nests, the sum folding into the transient `tmp1[m]`, then the elementwise `q[m] = q[m] + tmp1[m]` (`augassign_19_4_map`), which `fuse_seed` takes into the fold as its seed, read once per row.
- `recipes/matvec/matvec_acc_v1.json` (adder tree) and `matvec_acc_v2.json` (`chisel_serial`, SerialAccumulate) bind it at 8 × 512, W = 4.

## elementwise/

### matadd

`elementwise/matadd.py` · no tier · int32 (two per 64-bit word, D142)

$$C = A + B, \quad A, B, C \in \mathbb{Z}^{M \times N}$$

- `vecadd` in 2-D: one map over two parameters, which the importer writes as two nested maps, one per parameter (D71).
- Contiguous rows: it can flatten into `vecadd`, or keep a spatial and a temporal loop in the streamer.

### relu

`elementwise/relu.py` · no tier · int32 (two per 64-bit word, D142)

$$B = \max(A, 0)$$

- `np.maximum` becomes `max(in1, 0)`, in the expression grammar; at int32 DaCe writes the 0 as `dace.int32(0)`, a constant cast the importer folds.
- A compare-and-select unit at II 1; `mlp`, `lenet` and `resnet` use it between layers.
- It runs: `recipes/elementwise/relu_v1.json` splits the map by W = 4 and binds `simd` at one stream, op max, its slot k = 0 read off the tasklet's literal and baked into the SimdLanes (C§3).

### rowsum

`elementwise/rowsum.py` · no tier · int64

$$y_i = \sum_j A_{ij}$$

- A partial reduction: the reduced axis is not the output's, so the accumulator restarts per row (T < N).
- It runs: `recipes/reduce/rowsum_v1.json` (adder tree, W = 4) and `recipes/reduce/rowsum_v3.json` (accumulator, W = 1) tag the row map temporal and split the column map, so T = N / W beats per sum and M sums per run (the restart, M13 B0).
- The result goes through a view, which the importer folds: the Reduce writes `y` directly.
- `softmax`'s sum pass has this shape.

### vdiv

`elementwise/vdiv.py` · tier B · int64

$$C_i = \left\lfloor A_i / B_i \right\rfloor, \quad B_i \neq 0$$

- The divider unit: multi-cycle, with II > 1 or a long fixed latency, so the streamer waits on the unit rather than the other way round.
- NumPy floors and C truncates; they differ when the signs differ, and the inputs take both signs on purpose. The divider's BRM states which it does.
- The importer already takes it (`//` is in the grammar); the unit is what is missing.

## polybench/

### vecadd, vecmul

`polybench/vecadd.py`, `polybench/vecmul.py` · tier done · int64

$$C_i = A_i + B_i, \qquad C_i = A_i B_i$$

- In the flow (`recipes/`), and generated and checked on SNAX-GVSoC (GEN2a).
- Lanes are the only knob: W elements per beat.

### vsum

`polybench/vsum.py` · tier done · int64

$$\mathit{out} = \sum_i A_i$$

- The Reduce becomes a sum map folding into `out[0]` (D102), bound to the Accumulator; at W lanes, an adder tree in front of it.

### dot

`polybench/dot.py` · tier done · int64

$$\mathit{out} = \sum_i A_i B_i$$

- A multiply chained into an accumulate: two units through L1 today, one composed accelerator in M13 (D112).
- The serial accumulator against the adder tree (`examples/dot/`).

### atax

`polybench/atax.py` · tier A · int64

$$y = A^T (A x)$$

- No maps at all: a Transpose and two MatMuls, which `sdfg/expand.py` expands before import into dot's shape, the Transpose folded into the second product's indices (D71).
- A chaining barrier: `A x` must be complete before the second product starts, so it lives in L1 or in the unit's state.
- `A` is read twice, the second time transposed: a strided streamer, or one pass if the two products fuse.

### atax_types

`polybench/atax_types.py` · tier A · int64, presets `i8`, `i16`, `i32`

atax with its first product at `OP_TYPE`: `y[:] = A.T @ (A @ x.astype(OP_TYPE))`.

- The transient t (`tmp1`) takes `A @ x`'s type, which `A.T @ t` keeps: at `i8` A is int8, t and y int32, so half 2 multiplies int8 by int32.
- The cast renames half 1's nests (`tmp2_mult_map` where atax has `tmp1_mult_map`); half 2 keeps `y_mult_map`.
- At the full int8 range y wraps at int32 ($|y| \le M N \cdot 2^{21}$), exact modulo $2^{32}$ on every side.

### bicg

`polybench/bicg.py` · tier A · int64

$$q = A p, \qquad s = A^T r$$

- `atax`'s control: the same two products, independent, so no barrier.
- One pass over `A` could feed both, each element used twice.

### bicg_types

`polybench/bicg_types.py` · tier A · int64, presets `i8`, `i16`, `i32`

bicg with each product at `OP_TYPE`: `q[:] = A @ p.astype(OP_TYPE)`, `s[:] = A.T @ r.astype(OP_TYPE)`.

- The products keep bicg's names (`q_mult_map`, `s_mult_map`, ...): they come from the assigned outputs, so bicg's recipes run on it as they are, with `types` given.
- p and r are cast, not `A.T`, which would make a Transpose the importer refuses; one cast per product is enough, as in `mvt_types`.

### mvt

`polybench/mvt.py` · tier A · int64

$$x_1 \leftarrow x_1 + A y_1, \qquad x_2 \leftarrow x_2 + A^T y_2$$

- Accumulates in place into `x1` and `x2` (open item 47, LOW4).
- Two independent mat-vecs over `A`, the second transposed.

### mvt_types

`polybench/mvt_types.py` · tier A · int64, presets `i8`, `i16`, `i32`

mvt with each product at `OP_TYPE`: `x1 += A @ y1.astype(OP_TYPE)`, `x2 += A.T @ y2.astype(OP_TYPE)`.

- A separate kernel, so mvt and its recipes keep their names: the cast's temporaries rename the products `tmp1_*` and `tmp4_*` (mvt's `tmp0_*`, `tmp2_*`); the `+=` maps keep mvt's names, as the function sits on mvt's lines.
- One cast per product: NumPy runs `@` at its operands' result type (int8 with int32 is int32), so A is widened inside the op; without the cast `A @ y` of two int8 arrays wraps at int8. y is cast rather than `A.T`, which would make a Transpose the importer refuses.

### gesummv

`polybench/gesummv.py` · tier A · int64

$$y = \alpha A x + \beta B x$$

- `x` is reused by both products and across every row: stride 0, or held in the unit.
- `alpha * A @ x` is `(alpha * A) @ x`: the import scales both matrices (N² multiplies each, αA and βB through L1) before two MatMuls and an add. `factor_out` (D198) moves each scale past its sum, as a recipe step.

### gesummv_types

`polybench/gesummv_types.py` · tier A · int64, presets `i8`, `i16`, `i32`

gesummv with x cast to `OP_TYPE`: `out[:] = alpha * A @ x.astype(OP_TYPE) + beta * B @ x.astype(OP_TYPE)`.

- At `i8` A, B and x are int8, alpha, beta and out int32 (`i32`: int64), as axpy's presets.
- The cast keeps `factor_out` exact: the product stays `in1 * int32(in2)` when it reads A instead of αA; without it A x of two int8 arrays would wrap at int8.
- The cast renames the products (`tmp2_mult_map`, `tmp5_mult_map` where gesummv has `tmp1_mult_map`, `tmp3_mult_map`); the scales, sums and add keep gesummv's names.
- y does not wrap at int32 at the full int8 range: $|y_i| \le (\alpha + \beta)\thinspace N \cdot 2^{14}$, $5 \cdot 2^{20}$ at N = 64 with α = 3, β = 2.

### covariance

`polybench/covariance.py` · tier C (its mean is tier B) · int64

$$\mu_j = \left\lfloor \tfrac{1}{n} \sum_i d_{ij} \right\rfloor, \quad
d_{ij} \leftarrow d_{ij} - \mu_j, \quad
\mathrm{cov}_{jk} = \left\lfloor \tfrac{1}{n-1} \sum_i d_{ij} d_{ik} \right\rfloor$$

- Three phases, each gating the next: a column sum, the centring (in place on `data`), the product.
- Two integer divides on values of both signs: floor against truncation, as `vdiv`.
- The product loop covers the upper triangle (`i:M`) and mirrors it: a non-rectangular access and control flow (5 states, open item 36).

### gemver

`polybench/gemver.py` · tier D · int64

$$A \leftarrow A + u_1 v_1^T + u_2 v_2^T, \quad
x \leftarrow x + \beta A^T y + z, \quad
w \leftarrow w + \alpha A x$$

- The rank-1 updates reuse `u` along a row and `v` down a column: a streamer with stride 0 (D70), or the unit's state.
- Three phases chained through `A` and `x`, both updated in place.
- `np.multiply.outer` and two MatMuls to expand.

### gemm

`polybench/gemm.py` · tier D · int64

$$C \leftarrow \alpha A B + \beta C$$

- Reuse on every operand: a row of `A` across `j`, a column of `B` across `i`, `C` across `k` (the accumulator). SNAX's GeMM is spatial over all three (8 × 8 × 8), the rest temporal.
- `int8` in and `int32` accumulate from tier P on (PACK1).
- `C` in place; two scalars.

### k2mm

`polybench/k2mm.py` · tier D · int64

$$D \leftarrow \alpha A B C + \beta D$$

- Two GeMMs through a transient `αAB` (NI × NJ): a chaining barrier, or a pipeline per tile once tiling exists (M15).

### k3mm

`polybench/k3mm.py` · tier D · int64

$$\mathit{out} = ((A B) C) D$$

- Three GeMMs, two transients. Python evaluates left to right, so this is not PolyBench's `(AB)(CD)`, whose two first products are independent.

### doitgen

`polybench/doitgen.py` · tier D · int64

$$A_{rq\cdot} \leftarrow A_{rq\cdot}\, C_4 \quad \text{for all } r, q$$

- `C4` is reused by all NR × NQ rows: it can stay in the unit (weights stationary) while the rows stream.
- In place: each slab `A[r]` is read whole before it is written.
- The loop over `r` is control flow (4 states); the reshapes are views.

### jacobi1d

`polybench/jacobi1d.py` · tier D · int32 (until M9)

$$B_i = \left\lfloor \tfrac{1}{3}(A_{i-1} + A_i + A_{i+1}) \right\rfloor, \quad
A_i = \left\lfloor \tfrac{1}{3}(B_{i-1} + B_i + B_{i+1}) \right\rfloor$$

- Each element is read three times: a line buffer in the streamer, the unit or L1 (open item 76, J1).
- The time loop is control flow (open item 36); A and B ping-pong.
- Positive inputs, so floor and truncation agree.

### jacobi_2d

`polybench/jacobi_2d.py` · tier D · int64

$$B_{ij} = \left\lfloor \tfrac{1}{5}(A_{ij} + A_{i,j-1} + A_{i,j+1} + A_{i+1,j} + A_{i-1,j}) \right\rfloor,
\text{ then } A \text{ from } B$$

- A 5-point stencil: two rows of reuse, the same three placements as `jacobi1d`.
- NPBench's `0.2 *` is `// 5` here, a divide, until the tier's fixed-point type; inputs are non-negative, so floor and truncation agree.

## dl/ and image/

Each has its own README: [`dl/README.md`](dl/README.md) (the MLP, softmax, conv and the small models) and [`image/README.md`](image/README.md) (the image pipelines).
