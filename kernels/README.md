# Kernels

Every kernel SNAX-FORGE targets, one file each with a `KernelSpec`. The flow
uses `vecadd`, `vecmul`, `vsum` and `dot` (`recipes/`). The kernels tagged
`staged` are here to be read and their graphs looked at (`pixi run forge K`,
`pixi run import-dfg K`, `pixi run view-dfg`); they are not linked to the
flow. `pixi run kernels` prints each kernel's directory, tier, dtype, whether
its SDFG builds and whether `import-dfg` takes it, or why not.

The PolyBench and DL kernels are NPBench's DaCe sources (BSD-3,
`NPBENCH_LICENSE`) with their structure kept (D113): a `return X` becomes
`out[:] = X`, and a body that only runs under DaCe (it loops over symbols or
calls DaCe programs) has NPBench's NumPy version as its reference (`ref`).
They are `int64` where the kernel is integer arithmetic, at PolyBench's MINI
sizes; `softmax`, `mlp`, `lenet` and `resnet` keep NPBench's `float32`
(`float-staged`) until their tier decides the integer form. The tiers are
PAPER §6.1's.

## Regularity

The kernels on two axes of regularity: access, what the streamers do, across;
compute, what the unit does, down. The argument is PAPER §6.2's.

| Compute ↓ / Access → | Affine, no reuse | Affine with reuse | Non-rectangular or data-dependent |
|---|---|---|---|
| II 1, fixed latency (elementwise) | `vecadd`, `vecmul`, `matadd`, `relu`, `axpy`, scale by α/β | `gemm` (lanes), `gemver`'s rank-1 update (stride 0) | triangular kernels (left out) |
| Fold, loop-carried (reduction) | `vsum`, `dot`, `rowsum`, mat-vecs (`atax`, `bicg`, `mvt`, `gesummv`) | `gemm`, `k2mm`, `k3mm`, `doitgen`, `conv2d_bias` | `covariance`'s triangular product |
| Multi-cycle, fixed latency or II > 1 (div, sqrt, exp by polynomial, CORDIC or Newton-Raphson; FP adders) | `vdiv`, `covariance`'s divide | `jacobi1d`, `jacobi_2d` (integer divide now, fixed point or FP later) | |
| Multi-phase | `softmax` (max, then exp and sum, then divide), `covariance` (mean, center, product) | attention, `mlp`, `lenet`, `resnet` (batchnorm, skip) | |
| Data-dependent latency or control | early-exit dividers, LUT methods | | `durbin`, `nussinov` (left out) |

## blas/

### axpy

`blas/axpy.py` · no tier · int32

$$y \leftarrow y + \alpha x$$

- `alpha` is a scalar; the importer takes arrays only, so it would live in a
  CSR or a streamer with stride 0.
- `y` is updated in place: read and written by the same task (open item 47).

## elementwise/

### matadd

`elementwise/matadd.py` · no tier · int32

$$C = A + B, \quad A, B, C \in \mathbb{Z}^{M \times N}$$

- `vecadd` in 2-D: one map over two parameters, which the importer takes
  once maps nest one per parameter (D77, ND1).
- Contiguous rows: it can flatten into `vecadd`, or keep a spatial and a
  temporal loop in the streamer.

### relu

`elementwise/relu.py` · no tier · int32

$$B = \max(A, 0)$$

- `np.maximum` becomes a `max(...)` call, outside the expression grammar
  (open item 37).
- A compare-and-select unit at II 1; `mlp`, `lenet` and `resnet` use it
  between layers.

### rowsum

`elementwise/rowsum.py` · no tier · int64

$$y_i = \sum_j A_{ij}$$

- A partial reduction: the reduced axis is not the output's, so the
  accumulator restarts per row (T < N).
- The result goes through a view, which the importer does not take yet (FE1).
- `softmax`'s sum pass has this shape.

### vdiv

`elementwise/vdiv.py` · tier B · int64

$$C_i = \left\lfloor A_i / B_i \right\rfloor, \quad B_i \neq 0$$

- The divider unit: multi-cycle, with II > 1 or a long fixed latency, so the
  streamer waits on the unit rather than the other way round.
- NumPy floors and C truncates; they differ when the signs differ, and the
  inputs take both signs on purpose. The divider's BRM states which it does.
- The importer already takes it (`//` is in the grammar); the unit is what is
  missing.

## polybench/

### vecadd, vecmul

`polybench/vecadd.py`, `polybench/vecmul.py` · tier done · int64

$$C_i = A_i + B_i, \qquad C_i = A_i B_i$$

- In the flow (`recipes/`), and generated and checked on SNAX-GVSoC (GEN2a).
- Lanes are the only knob: W elements per beat.

### vsum

`polybench/vsum.py` · tier done · int64

$$\mathit{out} = \sum_i A_i$$

- The Reduce becomes a sum map folding into `out[0]` (D102), bound to the
  Accumulator; at W lanes, an adder tree in front of it (D133).

### dot

`polybench/dot.py` · tier done · int64

$$\mathit{out} = \sum_i A_i B_i$$

- A multiply chained into an accumulate: two units through L1 today, one
  composed accelerator in M13 (D112).
- The serial accumulator against the adder tree (`examples/dot/`).

### atax

`polybench/atax.py` · tier A · int64

$$y = A^T (A x)$$

- No maps at all: a Transpose and two MatMuls, library nodes the importer
  must expand.
- A chaining barrier: `A x` must be complete before the second product
  starts, so it lives in L1 or in the unit's state.
- `A` is read twice, the second time transposed: a strided streamer, or one
  pass if the two products fuse.

### bicg

`polybench/bicg.py` · tier A · int64

$$q = A p, \qquad s = A^T r$$

- `atax`'s control: the same two products, independent, so no barrier.
- One pass over `A` could feed both, each element used twice.

### mvt

`polybench/mvt.py` · tier A · int64

$$x_1 \leftarrow x_1 + A y_1, \qquad x_2 \leftarrow x_2 + A^T y_2$$

- Accumulates in place into `x1` and `x2` (open item 47, LOW4).
- Two independent mat-vecs over `A`, the second transposed.

### gesummv

`polybench/gesummv.py` · tier A · int64

$$y = \alpha A x + \beta B x$$

- `x` is reused by both products and across every row: stride 0, or held in
  the unit.
- Two MatMuls, two scalar scales and an add: a chain that could be one
  composed accelerator (D112); the scalars come with LOW4.

### covariance

`polybench/covariance.py` · tier C (its mean is tier B) · int64

$$\mu_j = \left\lfloor \tfrac{1}{n} \sum_i d_{ij} \right\rfloor, \quad
d_{ij} \leftarrow d_{ij} - \mu_j, \quad
\mathrm{cov}_{jk} = \left\lfloor \tfrac{1}{n-1} \sum_i d_{ij} d_{ik} \right\rfloor$$

- Three phases, each gating the next: a column sum, the centring (in place
  on `data`), the product.
- Two integer divides on values of both signs: floor against truncation, as
  `vdiv`.
- The product loop covers the upper triangle (`i:M`) and mirrors it: a
  non-rectangular access and control flow (5 states, open item 36).

### gemver

`polybench/gemver.py` · tier D · int64

$$A \leftarrow A + u_1 v_1^T + u_2 v_2^T, \quad
x \leftarrow x + \beta A^T y + z, \quad
w \leftarrow w + \alpha A x$$

- The rank-1 updates reuse `u` along a row and `v` down a column: a streamer
  with stride 0 (D69), or the unit's state.
- Three phases chained through `A` and `x`, both updated in place.
- `np.multiply.outer` and two MatMuls to expand.

### gemm

`polybench/gemm.py` · tier D · int64

$$C \leftarrow \alpha A B + \beta C$$

- Reuse on every operand: a row of `A` across `j`, a column of `B` across
  `i`, `C` across `k` (the accumulator). SNAX's GeMM is spatial over all
  three (8 × 8 × 8), the rest temporal.
- `int8` in and `int32` accumulate from tier P on (PACK1).
- `C` in place; two scalars.

### k2mm

`polybench/k2mm.py` · tier D · int64

$$D \leftarrow \alpha A B C + \beta D$$

- Two GeMMs through a transient `αAB` (NI × NJ): a chaining barrier, or a
  pipeline per tile once tiling exists (M15).

### k3mm

`polybench/k3mm.py` · tier D · int64

$$\mathit{out} = ((A B) C) D$$

- Three GeMMs, two transients. Python evaluates left to right, so this is
  not PolyBench's `(AB)(CD)`, whose two first products are independent.

### doitgen

`polybench/doitgen.py` · tier D · int64

$$A_{rq\cdot} \leftarrow A_{rq\cdot}\, C_4 \quad \text{for all } r, q$$

- `C4` is reused by all NR × NQ rows: it can stay in the unit (weights
  stationary) while the rows stream.
- In place: each slab `A[r]` is read whole before it is written.
- The loop over `r` is control flow (4 states); the reshapes are views.

### jacobi1d

`polybench/jacobi1d.py` · tier D · int32 (until M9)

$$B_i = \left\lfloor \tfrac{1}{3}(A_{i-1} + A_i + A_{i+1}) \right\rfloor, \quad
A_i = \left\lfloor \tfrac{1}{3}(B_{i-1} + B_i + B_{i+1}) \right\rfloor$$

- Each element is read three times: a line buffer in the streamer, the unit
  or L1 (open item 76, J1).
- The time loop is control flow (open item 36); A and B ping-pong.
- Positive inputs, so floor and truncation agree.

### jacobi_2d

`polybench/jacobi_2d.py` · tier D · int64

$$B_{ij} = \left\lfloor \tfrac{1}{5}(A_{ij} + A_{i,j-1} + A_{i,j+1} + A_{i+1,j} + A_{i-1,j}) \right\rfloor,
\text{ then } A \text{ from } B$$

- A 5-point stencil: two rows of reuse, the same three placements as
  `jacobi1d`.
- NPBench's `0.2 *` is `// 5` here, a divide, until the tier's fixed-point
  type; inputs are non-negative, so floor and truncation agree.

## dl/

### softmax

`dl/softmax.py` · tier C · float32

$$m_r = \max_j x_{rj}, \quad e_{rj} = \exp(x_{rj} - m_r), \quad
\mathit{out}_{rj} = e_{rj} \Big/ \sum_k e_{rk}$$

- Three phases per row (max, exp and sum, divide), each a reduction that
  gates the next; a row of SM elements fits in the unit.
- Integer form in tier C: an integer max, exp by polynomial or table, an
  integer divide (I-BERT).
- Its 4-D strides (`H*SM**2`) are outside the expression grammar (`**`).

### mlp

`dl/mlp.py` · tier E · float32

$$x_1 = \mathrm{relu}(x W_1 + b_1), \quad x_2 = \mathrm{relu}(x_1 W_2 + b_2), \quad
\mathit{out} = \mathrm{softmax}(x_2 W_3 + b_3)$$

- Three GeMM, bias and activation layers, each a barrier; requantization
  between them in tier E.
- Each weight matrix is reused over the N batch rows.
- Nested DaCe programs (`relu`, `softmax`) that DaCe inlines.

### conv2d_bias

`dl/conv2d_bias.py` · tier E · int64

$$\mathit{out}_{n,i,j,c} = b_c + \sum_{p,q,k} x_{n,\,i+p,\,j+q,\,k}\; w_{p,q,k,c}$$

- Each input element is used up to K² times: im2col (conv as GeMM, E6) or a
  line buffer.
- One reduction over three axes (p, q, k); the output channel `c` is the
  natural lane axis.
- The `i`, `j` loops are Python loops, so control flow (7 states), not maps.

### lenet

`dl/lenet.py` · tier E · float32

$$\mathit{out} = \mathrm{fc}_3 \circ \mathrm{relu} \circ \mathrm{fc}_2 \circ \mathrm{relu} \circ \mathrm{fc}_1
\circ \mathrm{pool} \circ \mathrm{relu} \circ \mathrm{conv}_2 \circ \mathrm{pool} \circ \mathrm{relu} \circ \mathrm{conv}_1 (x)$$

- Eight layers, each a barrier; the conv layers carry a bias.
- 2 × 2 maxpool: a max over a window with stride 2, a non-unit stride in the
  streamer.
- At 16 × 16 the weights are about 60 KiB of the 70 KiB, `fc2w` alone 40 KiB.

### resnet

`dl/resnet.py` · tier E · float32

$$\mathit{out} = \mathrm{relu}\big(\mathrm{bn}(\mathrm{conv}_{1\times1}(\mathrm{relu}(\mathrm{bn}(\mathrm{conv}_{3\times3}(
\mathrm{relu}(\mathrm{bn}(\mathrm{pad}(\mathrm{conv}_{1\times1}(x))))))))) + x\big), \quad
\mathrm{bn}(y) = \frac{y - \mu}{\sqrt{\sigma + \epsilon}}$$

- Batchnorm is multi-phase: the mean over the batch, then σ, then the
  normalisation, with a square root and a divide (multi-cycle units).
- The skip connection keeps `x` in L1 to the end, or reads it again.
- The 1 × 1 convs are GeMMs; the 3 × 3 runs on a zero-padded copy.
