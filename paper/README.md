# paper

The data behind the paper's figures, made again by one command. Each set is a JSON list of design points (a recipe and its `--set`s); `paper/run.py` puts every point through the design step and the ideal estimate, and with `--run` through the flow on SNAX-GVSoC with `--energy`, into one CSV per set in `paper/data/`. The figures are drawn elsewhere from those CSVs. This folder serves the submission and is kept or dropped after it.

## The sets

| Set | Points | What they vary |
|---|---|---|
| `forms.json` | 21 | each tier A kernel's forms at int8, 64 × 64, W = 8: a tree from a shared base, branches by technique (`branch`, `label`) |
| `limits.json` | 36 | at int64, 64 × 64: `mvt_v4` contiguous and padded (one engine against the layout) and `bicg_v2` sequential and pipelined, contiguous and padded (two engines at once); W = 4, 8, 16, 32 on 32 banks, and 16 and 64 banks at W = 8 with L1 kept at 128 KiB; bicg at W = 32 needs 132 crossbar inputs, so its points raise `l1.max_xbar_inputs` to 256 |

Every point takes `--set design.data=full`: inputs uniform over each dtype (D199).

## Running them

```sh
pixi run paper forms                    # the estimate only: seconds a point, no SNAX-GVSoC
pixi run paper limits --run -j 8        # also each flow with --energy, in the image
pixi run paper forms --run --only mvt_column atax_read_once   # those rows replaced in the CSV
```

Without `--run`, a row keeps its run columns from the CSV while its recipe and sets are unchanged, so the estimate can be refreshed alone. A point that fails keeps its row with the reason in `status`, and the command exits 1.

`paper/data/<set>.csv` has one row per point, in the set's order:

| Columns | From |
|---|---|
| `set`, `id`, `kernel`, `branch`, `label`, `row`, `layout`, `W`, `banks`, `schedule`, `recipe`, `sets` | the set |
| `est_cycles`, `est_accelerator_cycles`, `est_copy_cycles`, `est_bank_bound`, `xbar_inputs` | the ideal estimate (CONTRACTS, "Estimate"); `est_bank_bound` the largest group's fewest cycles of L1 time; `xbar_inputs` the TCDM crossbar's inputs (streamer channels, cores and AXI) |
| `static_<part>_nj`, `static_nj`, `area_<part>_um2`, `area_um2`, `area_design_um2`, `leak_uw`, `static_leak_nj` | the cost model's six parts (datapath, control, streamers, crossbar, memory, DMA) and, in area only, the core and the rest of the cluster (CONTRACTS, "Cost model"); `area_um2` all of them, `area_design_um2` the datapath, control, streamers and crossbar; the six parts' leakage power and its energy over `est_cycles` |
| `cycles`, `dynamic_<part>_nj`, `dynamic_nj`, `leak_design_nj`, `leak_core_nj`, `leak_rest_nj`, `idle_nj`, `passed` | with `--run`: the program's window, the energy run's dynamic totals by part and its background lines over the windows (CONTRACTS, "The energy run") |
| `status` | `ok`, or why the point was refused or failed |

## What the estimate says

From `pixi run paper forms` and `pixi run paper limits` (checked by `tests/paper/`):

- `mvt_column` (`mvt_v5`, sequential): 1028 cycles, 26.18 nJ static, 951 435 µm² with the banks, the DMA, the core and the rest; `mvt_dma_copy`: 11 951 cycles, 803.39 nJ, of which 10 923 cycles and 435.83 nJ (the DMA's bursts) are the copy.
- `mvt_contiguous_w8` and `mvt_padded_w8`: the same 1028 cycles and 241.08 nJ, as the estimate does not see bank conflicts (D155); at W = 32, 260 cycles against a bank bound of 520.
- `bicg_pipeline_padded_w32` needs 132 crossbar inputs, past the 128 of SNAX's crossbar (`l1.max_xbar_inputs`): its point raises the limit to 256, SNAX-GVSoC's most (D85), and the estimate prices its crossbar extrapolated. Without the raise it is refused at design.
