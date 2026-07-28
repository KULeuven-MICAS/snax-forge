# :hammer: :fire: SNAX-FORGE :fire: :hammer:

This repository is a work-in-progress (WIP) where we develop a program that uses the DaCe IR to generate and multi-accelerator architecture for the SNAX compute cluster.
This tool would help HW-oriented engineers bring the SW side closer to them rather than the typical otherway around were tools like DaCe make it easier for SW designers and optimization engineers to match the HW-SW combinations.
This work's motivation is to enable a HW-SW co-design but with the perspective on the HW-side.

# Anticipated Features
1. First is to break a program into an IR and use that IR to make accelerator(s) that fit within the SNAX/PULP compute clusters. In here we use the DaCe tool and use it as a model to make our HW accelerators.
2. We will offer some block primitives that enable an efficient yet modular designs that help constrain the design space a bit more unlike classic HLS that maps every operation on every kernel. These accelerators will be generated with Chisel.
3. We offer also a kernel library generation, where for the given designed accelerator we automatically generate the designated library kernels that are light function calls.
4. Finally, we have compute cluster model that simulates the flow of the accelerator for fast investigations rather than relying entirely on RTL simulations. Those can happen afterwards.

# Setup
You need [pixi](https://pixi.prefix.dev/v0.28.1/) shell to install the environment:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Clone the repo:

```bash
git clone git@github.com:KULeuven-MICAS/snax-forge.git
```

# Getting Started

All commands run through pixi:

```bash
pixi run forge [KERNELS...] [OPTIONS]
```

Kernels are discovered automatically from `kernels/`; recipes from `transforms/`.
Omitting kernel names means "all of them".

---

## 1. Ingest — build and verify SDFGs

**What it does.** Builds each kernel twice (raw and simplified), saves both to
`out/sdfg/`, and checks the compiled SDFG bit-exactly against a plain NumPy
reference. Prints a structural summary: state count, map entries, top-level map
scopes, arrays, transients, free symbols.

```bash
pixi run forge              # every kernel
pixi run forge vecadd dot   # named kernels
pixi run forge --list       # just list available kernel names
```

Outputs: `out/sdfg/<kernel>.raw.sdfg`, `out/sdfg/<kernel>.simplified.sdfg`,
`out/sdfg/<kernel>.json`

---

## 2. Profile — timing sweep

**What it does.** Compiles once and times across problem sizes. Reports minimum
wall-clock time, standard deviation, achieved bandwidth, and arithmetic
intensity. Because descriptors give symbolic bounds, a single compile serves the
whole sweep.

```bash
pixi run forge vecadd --profile
pixi run forge vecadd --profile --sizes 4096 65536 --reps 200
pixi run forge atax   --profile --sizes m=256,n=512 m=512,n=128
```

Sizes default to the kernel's own `sweep_sizes`, then to `DEFAULT_SIZES`.
Multi-symbol kernels take `m=...,n=...` tokens.

```
vecadd_simp  (threads=1)
  n=4096                 77.86 us  +/-  33.98    0.63 GB/s  AI=0.0833
```

Outputs: `out/profile/<label>.json`

---

## 3. Instrument — per-state timing inside the generated C++

**What it does.** Compiles a second, timered copy of the SDFG and reports how
long each state takes *inside* the kernel, excluding Python and ctypes overhead.
The gap between this and the wall-clock figure above is the host offload cost.

```bash
pixi run forge jacobi1d --profile --instrument
```

```
  n=65536              1864.81 us  +/-  81.61    4.47 GB/s  AI=0.3750
      State for_17_BinOp_18                1832.00 us  (x80)
      = states sum                         1832.10 us   sdfg total 1762.00 us
```

The `(xN)` count is how many timer events were captured. DaCe names report files
by timestamp, so fast calls collide and N is usually lower than the number of
invocations — a low count means low confidence in that line.

---

## 4. Transforms — what DaCe could apply

**What it does.** Static analysis only; nothing is applied and the graph is not
modified. Lists every pattern transformation DaCe can currently match, filtered
to those meaningful for a SNAX target (GPU, FPGA, MPI and internal-storage
transformations are excluded and reported separately).

```bash
pixi run forge --transforms            # every kernel
pixi run forge jacobi1d --transforms
```

```
jacobi1d  (4 states)  55 relevant / 84 total
    MapTiling                    x6   ['_Add__map[__i0=0:N - 2]']
    MapFusion                    x4   ['_Add__map[__i0=0:N - 2]', '__tmp0']
    StencilOperation             x2   ['_Add__map[__i0=0:N - 2]']
```

Outputs: `out/transforms/<kernel>.transforms.json`

---

## 5. Recipes — apply a named transformation sequence

**What it does.** Loads a recipe from `transforms/`, applies each step in order,
and re-verifies bit-exactness after every step. Saves the transformed SDFG plus
a log of what was applied.

```bash
pixi run forge --list-recipes
pixi run forge --recipe jacobi1d_fused
pixi run forge --recipe                    # bare flag = every recipe
pixi run forge --recipe jacobi1d_fused --no-verify   # skip the per-step check
```

```
jacobi1d_fused  (kernel: jacobi1d)  start {'states': 4, 'maps': 6, 'transients': 4}
  [0] MapFusion            x4  -> {'states': 4, 'maps': 2, 'transients': 8}  bitexact
```

Outputs: `out/transforms/<recipe>.sdfg`, `out/transforms/<recipe>.log.json`

---

## 6. Stored SDFGs — profile or compare `.sdfg` files directly

**What it does.** Profiles a saved `.sdfg` with no `KernelSpec` involved — shapes
and dtypes are recovered from the SDFG's own descriptors. Two or more paths are
compared side by side, with speedups relative to the first.

Symbol values are not stored in the file and must be supplied. Running without
`--symbols` prints which ones the SDFG needs.

```bash
# one file: straight profile
pixi run forge --sdfg out/sdfg/jacobi1d.simplified.sdfg --symbols 4096 65536

# two files: comparison
pixi run forge --sdfg out/sdfg/jacobi1d.simplified.sdfg \
                     out/transforms/jacobi1d_fused.sdfg \
               --symbols N=4096 N=65536
```

```
=== stored SDFG comparison (min us, vs jacobi1d.simplified) ===
variant                         N=4096           N=65536
jacobi1d.simplified       220.1 (1.00x)     2453.7 (1.00x)
jacobi1d_fused            139.0 (1.58x)     1296.7 (1.89x)
```

Bare integers work when the SDFG has exactly one free symbol; otherwise use
`N=...` form. **Correctness is not checked in this mode** — there is no golden
reference without a spec. Produce transformed SDFGs via `--recipe`, which does
verify. Absolute bandwidth here is not comparable to `--profile`, since without a
`bytes_moved` model it assumes each array is touched once.

---

## Options

| Option | Applies to | Meaning |
|---|---|---|
| `--list` | – | List kernel names and exit |
| `--profile` | kernels | Timing sweep instead of ingest |
| `--sizes` | `--profile` | Sizes: `4096` or `m=256,n=512` |
| `--reps` | timing | Repetitions per size (default 50) |
| `--instrument` | timing | Add per-state DaCe timers |
| `--transforms` | kernels | List applicable transformations |
| `--list-recipes` | – | List recipe names and exit |
| `--recipe [NAME...]` | recipes | Apply recipes; bare flag = all |
| `--no-verify` | `--recipe` | Skip the per-step bit-exact check |
| `--sdfg PATH...` | files | Profile stored SDFGs; 2+ are compared |
| `--symbols SPEC...` | `--sdfg` | Symbol bindings, e.g. `N=4096` |

Modes are checked in order: `--list`, recipes, stored SDFGs, then kernels. Only
the first matching mode runs.

---

## Reproducibility

Thread count and CPU affinity are pinned in `pixi.toml` and recorded in every
profile report. On a shared machine, pin to a quiet core:

```bash
taskset -c 4 pixi run forge jacobi1d --profile
```

Check `lscpu -e=CPU,CORE,SOCKET` first — rows sharing a `CORE` value are
hyperthread siblings, and CPU 0 usually handles interrupts.


