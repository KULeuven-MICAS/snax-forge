# Image kernels

Image-processing pipelines on uint8 pixels: a few passes over the whole image, each pass a map over the pixels that reads the previous pass's array. Integer only.

| Kernel | What | Types | Runs on the flow |
|---|---|---|---|
| `gray_blur_threshold` | grayscale, a 1-2-1 blur along the rows then down the columns, a threshold | uint8 in and out, uint16 and int16 inside | yes |

## gray_blur_threshold

`gray_blur_threshold.py` · uint8 · an RGB image of H × W pixels to a black-and-white one

The four passes, with $i$ the row and $j$ the column:

$$Y_{i,j} = \Big\lfloor \frac{77 R_{i,j} + 150 G_{i,j} + 29 B_{i,j}}{256} \Big\rfloor \qquad \text{(grayscale)}$$

$$X_{i,j} = \Big\lfloor \frac{Y_{i,j-1} + 2 Y_{i,j} + Y_{i,j+1}}{4} \Big\rfloor \qquad \text{(blur along the rows, } 1 \le j \le W - 2\text{)}$$

$$V_{i,j} = \Big\lfloor \frac{X_{i-1,j} + 2 X_{i,j} + X_{i+1,j}}{4} \Big\rfloor \qquad \text{(blur down the columns, } 1 \le i \le H - 2\text{)}$$

$$T_{i,j} = \begin{cases} 255 & V_{i,j} \ge t \cr 0 & \text{else} \end{cases} = 255 \cdot \min\big(\max(V_{i,j} - t + 1, 0), 1\big) \qquad \text{(threshold)}$$

- Sizes: 64 × 64 and 128 × 128. Both fit L1 whole (28 KiB and 112 KiB with every pass through L1).
- R, G and B are three planes, so grayscale reads three streams of the same shape; interleaved RGB would need a stride-3 read.
- 77, 150 and 29 are BT.601's luma weights (0.299, 0.587, 0.114) times 256, rounded to add up to 256, so white stays 255. The divides are shifts (by 8, by 2).
- The border is skipped: the blur writes the interior only, so X is H × (W − 2) and V is (H − 2) × (W − 2), and T's border stays 0.
- The blur's neighbours are shifted reads of one array: `Y[:, :-2]`, `Y[:, 1:-1]`, `Y[:, 2:]` along the rows, the same by rows down the columns.
- Two 1-D passes are not one 3 × 3 pass with $\lfloor \cdot / 16 \rfloor$: the floor is taken twice (28 % of pixels differ by one on random data). The reference is the two passes.
- The threshold $t$ is a scalar argument (a CSR, so one program serves any $t$), default 128; the compare is written with `max` and `min` at int16.
- The casts are written out and the literals typed (`np.uint16(77)`): NumPy wraps uint8 arithmetic (the grayscale sum reaches 65,280), and DaCe widens an untyped literal's result to 64 bits.

Example, 4 × 4 with $t = 132$:

$$R = \begin{pmatrix}200&180&60&40\cr 190&170&50&30\cr 210&160&70&20\cr 220&150&80&10\end{pmatrix}, \quad G = \begin{pmatrix}100&120&140&160\cr 110&130&150&170\cr 90&140&160&180\cr 80&150&170&190\end{pmatrix}, \quad B = \begin{pmatrix}50&60&200&220\cr 40&70&210&230\cr 30&80&190&240\cr 20&90&180&250\end{pmatrix}$$

Pixel (1, 1) is $\lfloor (77 \cdot 170 + 150 \cdot 130 + 29 \cdot 70)/256 \rfloor = \lfloor 34620/256 \rfloor = 135$; the whole image and the two blur passes (X's columns are j = 1, 2; V's rows and columns are 1, 2):

$$Y = \begin{pmatrix}124&131&122&130\cr 126&135&126&134\cr 119&139&136&138\cr 115&143&144&142\end{pmatrix}, \quad X = \begin{pmatrix}127&126\cr 130&130\cr 133&137\cr 136&143\end{pmatrix}, \quad V = \begin{pmatrix}130&130\cr 133&136\end{pmatrix}$$

with $X_{1,1} = \lfloor (126 + 2 \cdot 135 + 126)/4 \rfloor = 130$ and $V_{2,2} = \lfloor (130 + 2 \cdot 137 + 143)/4 \rfloor = 136$. The threshold keeps the bottom row of V:

$$T = \begin{pmatrix}0&0&0&0\cr 0&0&0&0\cr 0&255&255&0\cr 0&0&0&0\end{pmatrix}$$
