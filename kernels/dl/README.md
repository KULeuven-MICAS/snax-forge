# Deep-learning kernels

Neural-network layers and small models. NPBench's sources keep their `float32` (`float-staged`) until their integer form is written; the integer forms are the ones the flow runs: int8 data, int32 sums, a shift and a clamp between layers, no float. Later layers (conv and the like) come here too.

| Kernel | What | Types | Runs on the flow |
|---|---|---|---|
| `mlp_int` | three dense layers at batch 1, requantized between them | int8 in, int32 sums and out | yes |
| `mlp_int_col3` | `mlp_int` with layer 3 stored for the column form | int8 in, int32 sums and out | yes |
| `mlp` | NPBench's MLP, with softmax | float32 | no (staged) |
| `softmax` | row-wise softmax | float32 | no (staged) |
| `conv2d_bias` | 2-D convolution plus a bias | int64 | no (staged) |
| `lenet` | LeNet-5 | float32 | no (staged) |
| `resnet` | a ResNet basic block | float32 | no (staged) |

## mlp_int

`mlp_int.py`, `mlp_int_col3.py` · int8 data, int32 sums · NPBench's `mlp` at batch 1 in integers

$$z_l = W_l a_{l-1} + b_l, \qquad a_l = \min\Big(\max\Big(\Big\lfloor \frac{z_l}{2^{s_l}} \Big\rfloor, 0\Big), 127\Big) \quad (l = 1, 2), \qquad \mathit{out} = z_3$$

with $a_0 = x$, $W_l$ of shape $(D_l, D_{l-1})$, the sums $z_l$ and biases $b_l$ int32, the activations $a_l$ int8.

Per layer, the passes are:

1. the product $W_l a_{l-1}$, a matvec: $D_{l-1} D_l$ multiply-adds into $D_l$ sums;
2. the bias added to each sum (the fold's seed once bound);
3. the requantization of layers 1 and 2: a right shift by $s_l$, then a clamp to $[0, 127]$, whose lower bound is the relu.

- Sizes: 64-64-64-32 and 256-256-128-10. Both fit L1 whole (about 11 KiB and 101 KiB), so no tiling through L2.
- The output is the logits $z_3$, int32. NPBench's softmax is dropped: it does not change which class is largest.
- The shift is 10 for both layers at both sizes, so one requant unit serves both. The rule $s_l = \mathrm{round}(\log_2 \sigma(z_l)) - 6$ on full-range data gives (10, 9) for the small size and (10, 10) for the large; the small size's layer 2 takes 10 too, since a shared unit needs the same shift (D165). About half the outputs are 0 after the relu, and at most 8 % clamp at 127.
- The bias is drawn from $\pm 2^{15}$, the same order as the sums, on full-range data too (the spec's `ranges`, D209); over the whole int32 range it would swamp them.
- Each weight matrix is stored in the layout its form reads along rows: $(D_{\text{out}}, D_{\text{in}})$ with `w @ a` (a dot per output, the row form) or $(D_{\text{in}}, D_{\text{out}})$ with `w.T @ a` (one input broadcast to the output lanes, the column form), chosen per layer (D208): `mlp_int` stores all three in the row form's layout, `mlp_int_col3` layer 3 in the column form's, for the chained set. Both take the same inputs, w3 transposed, and give the same logits. A vector first (`a @ w`) does not import.
- The casts are written out (`a.astype(np.int32)`) and the literals typed (`np.int32(1024)`): NumPy wraps int8 arithmetic, and DaCe widens an untyped literal's result to 64 bits.

Example, one layer with $D_{l-1} = 4$, $D_l = 3$ and $s = 2$ ($\lfloor z/4 \rfloor$), $W_l$ in the row form's layout:

$$a_{l-1} = \begin{pmatrix}3 \cr -2 \cr 5 \cr 1\end{pmatrix}, \quad W_l = \begin{pmatrix}20 & 5 & 10 & -4 \cr -7 & 30 & -12 & 9 \cr 40 & -60 & 50 & 30\end{pmatrix}, \quad b_l = \begin{pmatrix}10 \cr -20 \cr 5\end{pmatrix}$$

$$z_l = \begin{pmatrix}60 - 10 + 50 - 4 + 10 \cr -21 - 60 - 60 + 9 - 20 \cr 120 + 120 + 250 + 30 + 5\end{pmatrix} = \begin{pmatrix}106 \cr -152 \cr 525\end{pmatrix}, \quad \Big\lfloor \frac{z_l}{4} \Big\rfloor = \begin{pmatrix}26 \cr -38 \cr 131\end{pmatrix}, \quad a_l = \begin{pmatrix}26 \cr 0 \cr 127\end{pmatrix}$$

The second output is cut to 0 by the relu, the third clamped to 127.

## softmax

`softmax.py` · tier C · float32

$$m_r = \max_j x_{rj}, \quad e_{rj} = \exp(x_{rj} - m_r), \quad
\mathit{out}_{rj} = e_{rj} \Big/ \sum_k e_{rk}$$

- Three phases per row (max, exp and sum, divide), each a reduction that gates the next; a row of SM elements fits in the unit.
- Integer form in tier C: an integer max, exp by polynomial or table, an integer divide (I-BERT).
- Its 4-D strides (`H*SM**2`) are outside the expression grammar (`**`).

## mlp

`mlp.py` · tier E · float32

$$x_1 = \mathrm{relu}(x W_1 + b_1), \quad x_2 = \mathrm{relu}(x_1 W_2 + b_2), \quad
\mathit{out} = \mathrm{softmax}(x_2 W_3 + b_3)$$

- Three GeMM, bias and activation layers, each a barrier; requantization between them in tier E.
- Each weight matrix is reused over the N batch rows.
- Nested DaCe programs (`relu`, `softmax`) that DaCe inlines.

## conv2d_bias

`conv2d_bias.py` · tier E · int64

$$\mathit{out}_{n,i,j,c} = b_c + \sum_{p,q,k} x_{n,\,i+p,\,j+q,\,k}\; w_{p,q,k,c}$$

- Each input element is used up to K² times: im2col (conv as GeMM, E6) or a line buffer.
- One reduction over three axes (p, q, k); the output channel `c` is the natural lane axis.
- The `i`, `j` loops are Python loops, so control flow (7 states), not maps.

## lenet

`lenet.py` · tier E · float32

$$\mathit{out} = \mathrm{fc}_3 \circ \mathrm{relu} \circ \mathrm{fc}_2 \circ \mathrm{relu} \circ \mathrm{fc}_1
\circ \mathrm{pool} \circ \mathrm{relu} \circ \mathrm{conv}_2 \circ \mathrm{pool} \circ \mathrm{relu} \circ \mathrm{conv}_1 (x)$$

- Eight layers, each a barrier; the conv layers carry a bias.
- 2 × 2 maxpool: a max over a window with stride 2, a non-unit stride in the streamer.
- At 16 × 16 the weights are about 60 KiB of the 70 KiB, `fc2w` alone 40 KiB.

## resnet

`resnet.py` · tier E · float32

$$\mathit{out} = \mathrm{relu}\big(\mathrm{bn}(\mathrm{conv}_{1\times1}(\mathrm{relu}(\mathrm{bn}(\mathrm{conv}_{3\times3}(
\mathrm{relu}(\mathrm{bn}(\mathrm{pad}(\mathrm{conv}_{1\times1}(x))))))))) + x\big), \quad
\mathrm{bn}(y) = \frac{y - \mu}{\sqrt{\sigma + \epsilon}}$$

- Batchnorm is multi-phase: the mean over the batch, then σ, then the normalisation, with a square root and a divide (multi-cycle units).
- The skip connection keeps `x` in L1 to the end, or reads it again.
- The 1 × 1 convs are GeMMs; the 3 × 3 runs on a zero-padded copy.
