# Recipes

A recipe is SNAX-SANDBOX's ordered list of transforms (CONTRACTS, "Recipe"), in a folder by category; `custom/` is where the editor saves. Every other recipe carries a versioned `name`, an `id` that says what it builds and a `description` (D188). The table below lists them all; `tests/sandbox/test_recipe_names.py` keeps it equal to the files.

## Names

A name is `<kernel>_v<N>`, then `_<k>` for each change from its parent: `mvt_v5_1` is `mvt_v5` built another way. A kernel's type presets count as the kernel (`mvt_types` is `mvt`).

- `v<N>` is a new structure: other units, or the same units joined another way.
- `_<k>` is one change from the recipe it extends: built by reuse, shared, another type, layout or size.
- A number is never reused or renumbered; a recipe dropped leaves a gap.

## Ids

An id is the kernel, then words in this order, each from the lists below; a word left out is the default. Recipes that build the same units share an id (a reuse twin and its hand-written recipe), and the description says how each is built.

`<kernel>_<form>[_<form of half 2>][_<sharing>][_<type>][_<size>][_<data>]`

| Part | Word | Means |
|---|---|---|
| Form (a dot, or a matvec's rows) | `dot1` | multiply and fold as two units, the products through L1 (dot 1) |
| | `dot2` | multiply → link of depth 1 → fold, one group (dot 2) |
| | `dot3` | the same with a link of depth 0, a wire (dot 3) |
| | `dot4` | a `Mac` of W partial sums wired to a tree (dot 4) |
| | `dot5` | a `Mac` wired to a serial fold (dot 5) |
| | `dot6` | a `Mac` linked at depth 1 to a serial fold (dot 6) |
| | `dotser` | multiply and a serial fold of one lane, the products through L1 |
| | `col` | the column form: one `Mac` with its lanes on the output, the input broadcast (D186) |
| | `colbank` | the column form rows outer: the `Mac`'s sums of a strip of R·W outputs in a bank of R rows (D191), the broadcast read once a row and held (D190) |
| How x is added (q += A p) or the sum carried, after the form | (none) | q = A p |
| | `add` | a separate `vadd` reading q through L1 |
| | `chain` | the `vadd` linked into the same group |
| | `seed` | the fold starts from x (D163, D187) |
| | `strip` | the columns in strips outermost, each row's sum carried to the next strip through L1 as its own seed, the first strip's 0 (D192) |
| Half 2 | a second form word | mvt's or bicg's half 2 when it differs from half 1 |
| Sharing | (none) | each half or unit its own accelerator |
| | `shared` | both halves on one set of units (D165) |
| | `ports` | a container on ports its writer and reader share (`share_ports`) |
| | `enc` | a group's units encapsulated as one accelerator |
| | `fork` | both halves read A through one reader, in one group of two chains (`fork_read`, D189) |
| Type | (none) | int64 |
| | `i<b>` | inputs of b bits |
| | `o<b>` | the output of b bits, when it differs from the inputs |
| Size | `n<N>`, `m<M>n<N>` | a default size chosen for its tail |
| Data | `dmaT` | A transposed by a DMA copy in L1 (D184) |

The one-unit kernels take a word for their fold or scalar instead of a form: `tree`, `lane` and `serial` (rowsum, vsum), `csr` and `baked` (axpy's alpha); `undo` names `vecadd_v1_4`, a check of undo rather than a design.

## The recipes

| Recipe | Folder | id | What it builds |
|---|---|---|---|
| `bicg_v1` | bicg | `bicg_dot2` | Each half a `dot_v2` group per row, by reuse: `lift matvec` at each half, then `matvec/matvec_v2` on both. Half 2 reads A down its columns, so its banks follow the layout (`padded_rows`). |
| `bicg_v2` | bicg | `bicg_dot2_col` | q a `dot_v2` group per row and s the column form, by reuse: `matvec/matvec_v2` on q and `matvec/matvec_v7` on s, r broadcast to the `Mac`'s lanes (D186). Both halves read A along its rows. |
| `bicg_v2_1` | bicg | `bicg_dot2_col_i8_o32` | `bicg_v2` at int8 in, int32 out (`types` i8). |
| `bicg_v3` | bicg | `bicg_dot2_col_fork_i8_o32` | One pass over A at N = W: `bicg_v2_1`'s halves by reuse, then `fork_read` A, so one reader gives each beat of A to q's `dot_v2` group and s's `Mac` in one group of two chains (D189). M = 64, N = W = 32; the bank for N > W is `bicg_v4` (D191). |
| `bicg_v4` | bicg | `bicg_dot2_colbank_fork_i8_o32` | The one pass over A with a bank: q's half `matvec/matvec_v2` (a `dot_v2` group per row), s's half `matvec/matvec_v8`, so s's N sums stay in the `Mac`'s bank of R rows of W (D191) and r is read once a row and held (D190); A is read once by both (`fork_read`, D189). One strip: N = R·W (64 × 64 at W = 4, R = 16). |
| `bicg_v4_1` | bicg | `bicg_dot2strip_colbank_fork_i8_o32` | `bicg_v4` for N beyond one bank, in strips of R·W columns outermost: q's half `matvec/matvec_v9`, each row's sum carried to the next strip through L1 as the tree's own seed, 0 on the first (D192); s's half `matvec/matvec_v8`, its bank one strip deep; A still read once (`fork_read`, D189). Two strips at 64 × 64, W = 4, R = 8. |
| `dot_v1` | dot | `dot1` | A `vmul` and a `vsum` as two accelerators, the products written to L1 and read back. The first of the six dot variants, the one the others are measured against. |
| `dot_v1_1` | dot | `dot1_enc` | `dot_v1` with its two units encapsulated as one accelerator: one streamer set, two tasks. |
| `dot_v1_2` | dot | `dot1_ports` | `dot_v1` with the products' container on shared ports (`share_ports`): `vmul`'s writer and `vsum`'s reader on one streamer set. |
| `dot_v2` | dot | `dot2` | `mul → link → sum` in one group: the products go through a link of depth 1, never to L1. |
| `dot_v3` | dot | `dot3` | `dot_v2` with a link of depth 0, a wire: the fold takes each product in the cycle it is made. |
| `dot_v4` | dot | `dot4` | The fold split in two (`split_reduce`): a `Mac` keeps W partial sums over time, a wire takes them to a tree that adds them at the end. |
| `dot_v5` | dot | `dot5` | `dot_v4` with a serial fold in place of the tree: the W partial sums added one a cycle. |
| `dot_v6` | dot | `dot6` | `dot_v5` with a link of depth 1 between the `Mac` and the serial fold. |
| `dot_v7` | dot | `dotser` | A `mul` of W lanes and a serial fold of one lane as two accelerators, the products through L1. |
| `axpy_v1` | elementwise | `axpy_csr_i32` | y += alpha x in one `simd` unit of W lanes, alpha a CSR scalar the program writes. |
| `axpy_v1_1` | elementwise | `axpy_baked_i32` | `axpy_v1` with alpha baked into the datapath as a constant (3). |
| `matadd_v1` | elementwise | `matadd_i32` | C = A + B on a 2-D grid: the rows in time, W lanes along a row. |
| `relu_v1` | elementwise | `relu_i32` | B = max(A, 0) in one `simd` unit of W lanes. |
| `vecadd_v1` | elementwise | `vecadd` | C = A + B in one `simd` unit of W lanes. |
| `vecadd_v1_1` | elementwise | `vecadd_i32` | `vecadd_v1` at int32. |
| `vecadd_v1_2` | elementwise | `vecadd_i8` | `vecadd_v1` at int8. |
| `vecadd_v1_3` | elementwise | `vecadd_n7` | `vecadd_v1` at N = 7, a tail the lanes do not fill. |
| `vecadd_v1_4` | elementwise | `vecadd_undo` | Takes `vecadd_v1`'s bound graph back to the import (`unbind`, `join_map`): the check of undo, not a design. |
| `vecmul_v1` | elementwise | `vecmul` | C = A * B in one `simd` unit of W lanes. |
| `vecmul_v1_1` | elementwise | `vecmul_i32` | `vecmul_v1` at int32. |
| `vecmul_v1_1_1` | elementwise | `vecmul_i32_o64` | `vecmul_v1_1` with an int64 output: int32 operands, the full product kept. |
| `matvec_acc_v1` | matvec | `matvec_acc_dot1seed_enc` | `matvec_v1` with the fold starting from q (`fuse_seed`, D163), so no separate add: `mul` and `sum` encapsulated, the products through L1. |
| `matvec_acc_v2` | matvec | `matvec_acc_dotserseed_enc` | `matvec_acc_v1` with a serial fold of one lane in place of the tree. |
| `matvec_acc_v3` | matvec | `matvec_acc_dot2seed` | A seeded `dot_v2` on each row: `mul → link → sum` in one group, the fold starting from q[i]. mvt's half 1 in `mvt_v4` and `mvt_v5`. |
| `matvec_acc_v4` | matvec | `matvec_acc_colseed` | `matvec_v7` seeded: one `Mac` whose W sums start from q's strip (D187). mvt's half 2 in `mvt_v5`. |
| `matvec_v1` | matvec | `matvec_dot1_enc` | A `dot_v1` on each row of A, `mul` and `sum` encapsulated as one accelerator, the products through L1. |
| `matvec_v1_1` | matvec | `matvec_dot1` | `dot/dot_v1` applied to the rows by `apply_recipe`: `matvec_v1` without the encapsulation, so two accelerators. |
| `matvec_v2` | matvec | `matvec_dot2` | A `dot_v2` on each row: `mul → link → sum` in one group, a row a job. |
| `matvec_v2_1` | matvec | `matvec_dot2` | `matvec_v2` by reuse: `dot/dot_v2` applied to the rows. |
| `matvec_v3` | matvec | `matvec_dot3` | A `dot_v3` on each row: the link between `mul` and `sum` a wire. |
| `matvec_v3_1` | matvec | `matvec_dot3` | `matvec_v3` by reuse: `dot/dot_v3` applied to the rows. |
| `matvec_v4` | matvec | `matvec_dot4` | A `dot_v4` on each row: a `Mac` of W partial sums wired to a tree. |
| `matvec_v4_1` | matvec | `matvec_dot4` | `matvec_v4` by reuse: `dot/dot_v4` applied to the rows. |
| `matvec_v5` | matvec | `matvec_dot5` | A `dot_v5` on each row: a `Mac` of W partial sums wired to a serial fold. |
| `matvec_v5_1` | matvec | `matvec_dot5` | `matvec_v5` by reuse: `dot/dot_v5` applied to the rows. |
| `matvec_v6` | matvec | `matvec_dot6` | A `dot_v6` on each row: the `Mac` linked to the serial fold at depth 1. |
| `matvec_v6_1` | matvec | `matvec_dot6` | `matvec_v6` by reuse: `dot/dot_v6` applied to the rows. |
| `matvec_v7` | matvec | `matvec_col` | The column form: one `Mac` with its W lanes on q's index, A read a strip of W columns at a time, p broadcast to the lanes (D186), each sum folded over time. |
| `matvec_v8` | matvec | `matvec_colbank` | The column form rows outer: q in strips of R·W outputs, a strip's sums kept in a `Mac` bank of R rows of W (D191) while A is read along its rows, p read once a row and held (D190). |
| `matvec_v9` | matvec | `matvec_dot2strip` | A `dot_v2` on each row, the columns in strips of S outermost: a row's sum over a strip is a job, carried to the next strip through L1 as the tree's seed, the first strip's seed 0 (D192). The row form that reads A in the order `matvec_v8`'s bank does. |
| `mvt_v1` | mvt | `mvt_dot1add_shared` | Each half a `dot_v1` on each row, the products through L1, and a `vadd` adding x; both halves on the same `vmul`, `vsum` and `vadd`. Half 2 reads A down its columns (`mark_transpose`). |
| `mvt_v1_1` | mvt | `mvt_dot1add_shared` | `mvt_v1` by reuse: `matvec/matvec_v1_1` applied to each half, then the `vadd`s, the halves' units shared. |
| `mvt_v1_2` | mvt | `mvt_dot2add_shared` | `mvt_v1` with each half a `dot_v2` group (`vmul → link → vsum`), one group run for both halves (`share`, D165), and one `vadd`. |
| `mvt_v1_2_1` | mvt | `mvt_dot2add_shared` | `mvt_v1_2` by reuse: `matvec/matvec_v2_1` applied to each half. |
| `mvt_v1_2_2` | mvt | `mvt_dot2add_shared_ports` | `mvt_v1_2` with the group's result on ports it shares with the `vadd` (`share_ports`). |
| `mvt_v2` | mvt | `mvt_dot2add` | Each half its own `dot_v2` group and its own `vadd`, four accelerators; the halves' widths set apart (`W_x1`, `W_x2`). |
| `mvt_v2_1` | mvt | `mvt_dot2add` | `mvt_v2` by reuse: `matvec/matvec_v2_1` applied to each half, then each half's `vadd`. |
| `mvt_v2_2` | mvt | `mvt_dot2add_ports` | `mvt_v2` with each half's result on ports its group shares with its `vadd` (`share_ports`). |
| `mvt_v3` | mvt | `mvt_dot2chain_shared` | Each half one group `vmul → link → vsum → link → vadd`, so only x goes back to L1; both halves on one group (`share`). |
| `mvt_v3_1` | mvt | `mvt_dot2chain` | `mvt_v3` without `share`: the halves are two accelerators, which run under any schedule (D179). |
| `mvt_v4` | mvt | `mvt_dot2seed_shared` | Each half a seeded `dot_v2` group, the fold starting from x (D163), so no `vadd`; both halves on one group (D165). |
| `mvt_v4_1` | mvt | `mvt_dot2seed_shared_dmaT` | `mvt_v4` whose half 2 reads `A_T`, a transposed copy of A the DMA makes in L1 (D184), instead of A down its columns. |
| `mvt_v4_1_1` | mvt | `mvt_dot2seed_shared_i8_o32_dmaT` | `mvt_v4_1` at int8 in, int32 out (`types` i8). |
| `mvt_v5` | mvt | `mvt_dot2seed_colseed_i8_o32` | Half 1 a seeded `dot_v2` group, half 2 the column form: one seeded `Mac` with y2 broadcast to its lanes (D186, D187). Both halves read A along its rows, so A is never copied. Int8 in, int32 out. |
| `mvt_v5_1` | mvt | `mvt_dot2seed_colseed_i8_o32` | `mvt_v5` by reuse: `lift matvec_acc` on each half, then `matvec/matvec_acc_v3` on half 1 and `matvec/matvec_acc_v4` on half 2. |
| `mvt_v5_2` | mvt | `mvt_dot2seed_colchain_i8_o32` | `mvt_v5` with half 2's `Mac` unseeded and its W sums linked into a `vadd` of x2. Costs a little more than `mvt_v5` for the same time. |
| `rowsum_v1` | reduce | `rowsum_tree` | y[i] = sum over j of A[i, j]: the rows in time, a tree over W lanes of a row. |
| `rowsum_v1_1` | reduce | `rowsum_tree_m3n7` | `rowsum_v1` at 3 × 7, a row the lanes do not fill. |
| `rowsum_v2` | reduce | `rowsum_lane` | W rows at once, one a lane, each lane adding along its row over time (`lane_accumulate`). |
| `rowsum_v3` | reduce | `rowsum_serial` | One lane, a serial fold along each row. |
| `vsum_v1` | reduce | `vsum_tree` | out = sum of A with a tree, at W = 1. |
| `vsum_v1_1` | reduce | `vsum_tree` | `vsum_v1` at W = 4. |
| `vsum_v1_1_1` | reduce | `vsum_tree_i32_o64` | `vsum_v1_1` on int32 elements, the sum int64. |
| `vsum_v1_1_2` | reduce | `vsum_tree_i8_o64` | `vsum_v1_1` on int8 elements, the sum int64. |
| `vsum_v1_1_3` | reduce | `vsum_tree_n21` | `vsum_v1_1` at N = 21, a tail the lanes do not fill. |
