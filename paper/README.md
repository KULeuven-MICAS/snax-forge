# paper

The data behind the paper's figures, made again by one command. Each set is a JSON list of design points (a recipe and its `--set`s); `paper/run.py` puts every point through the design step and the ideal estimate, and with `--run` through the flow on SNAX-GVSoC with `--energy`, into one CSV per set in `paper/data/`. The figures are drawn elsewhere from those CSVs. This folder serves the submission and is kept or dropped after it.

## The sets

| Set | Points | What they vary |
|---|---|---|
| `forms.json` | 21 | each tier A kernel's forms at int8, 64 × 64, W = 8: a tree from a shared base, branches by technique (`branch`, `label`) |
| `limits.json` | 36 | at int64, 64 × 64: `mvt_v4` contiguous and padded (one engine against the layout) and `bicg_v2` sequential and pipelined, contiguous and padded (two engines at once); W = 4, 8, 16, 32 on 32 banks, and 16 and 64 banks at W = 8 with L1 kept at 128 KiB |

Every point takes `--set design.data=full`: inputs uniform over each dtype (D199).

## Running them

```sh
pixi run paper forms                    # the estimate only: seconds a point, no SNAX-GVSoC
pixi run paper limits --run -j 8        # also each flow with --energy, in the image
pixi run paper forms --run --only mvt_column atax_read_once   # those rows replaced in the CSV
```

`paper/data/<set>.csv` has one row per point, in the set's order:

| Columns | From |
|---|---|
| `set`, `id`, `kernel`, `branch`, `label`, `row`, `layout`, `W`, `banks`, `schedule`, `recipe`, `sets` | the set |
| `est_cycles`, `est_accelerator_cycles`, `est_copy_cycles`, `est_bank_bound` | the ideal estimate (CONTRACTS, "Estimate"); `est_bank_bound` the largest group's fewest cycles of L1 time |
| `static_<part>_nj`, `static_nj`, `area_<part>_um2`, `area_um2`, `area_without_banks_um2` | the cost model's five parts (CONTRACTS, "Cost model"), area with the banks |
| `cycles`, `dynamic_<part>_nj`, `dynamic_nj`, `passed` | with `--run`: the program's window and the energy run's totals (CONTRACTS, "The energy run") |
| `status` | `ok`, or why the point was refused or failed |

## What the estimate says

From `pixi run paper forms` and `pixi run paper limits` (checked by `tests/paper/`):

- `mvt_column` (`mvt_v5`, sequential): 1028 cycles, 26.44 nJ static, 561 514 µm² with the banks; `mvt_dma_copy`: 11 951 cycles, 373.35 nJ, of which 10 923 cycles are the copy.
- `mvt_contiguous_w8` and `mvt_padded_w8`: the same 1028 cycles and 243.60 nJ, as the estimate does not see bank conflicts (D155); at W = 32, 260 cycles against a bank bound of 520.
- `bicg_pipeline_padded_w32` is refused: 132 crossbar inputs against the platform's 128.
