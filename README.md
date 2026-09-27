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

Clone the repo and install the environment:

```bash
git clone git@github.com:KULeuven-MICAS/snax-forge.git
cd snax-forge
pixi install
```

# Getting Started

All commands run through pixi. The whole path, from a kernel to a checked
model run, is one command:

```bash
pixi run flow recipes/vecadd.json --platform platforms/small16.json   # writes out/flow/vecadd/
```

`CLAUDE.md` lists the command of every step on the way (import, sandbox,
design, lower); `docs/ARCHITECTURE.md` says what each one does.

---

## 1. Ingest — build and verify SDFGs

**What it does.** Builds each kernel twice (raw and simplified), saves both to
`out/sdfg/`, and checks the compiled SDFG bit-exactly against the kernel's
NumPy reference. Prints a structural summary: state count, map entries,
top-level map scopes, arrays, transients, free symbols. The simplified SDFG is
what the SNAX-DFG importer reads (`pixi run import-dfg <kernel>`).

```bash
pixi run forge              # every kernel
pixi run forge vecadd dot   # named kernels
pixi run forge --list       # just list available kernel names
```

Outputs: `out/sdfg/<kernel>.raw.sdfg`, `out/sdfg/<kernel>.simplified.sdfg`,
`out/sdfg/<kernel>.json`

---

## 2. Model — generate, run and view a scenario

**What it does.** Runs a cycle-level model of the SNAX cluster (banks,
interconnect, streamers, accelerators, DMA/L2, register interface and
controller) on a scenario, writes a profile, a trace and the final memory, and
shows them in a local web viewer. No CPU and no RTL are involved; see
`docs/ARCHITECTURE.md` sections 5.5–5.7.

Quick start, from the repo root:

```bash
pixi run scenarios                                                   # 1. generate the scenario files
pixi run model-run scenarios/vecadd/scenario.json --out out/vecadd --trace beat   # 2. run one
pixi run view out/vecadd                                             # 3. open http://127.0.0.1:8765/
```

### 2.1 Generate the scenarios

Each scenario is a folder under `scenarios/`. Two files in it are the sources
you edit; the rest is generated and ignored by git:

```
scenarios/
  make.py              writes every generated file below
  common.py            shared helpers
  clusters/
    clusters.py        source: the clusters (alu4, red4, mul1)
    alu4.json ...      generated: one cluster file each
  vecadd/
    tasks.json         source: the task list (what runs, in which order, what it waits for)
    scenario.py        source: data, memory layout, cluster, cycle limit
    scenario.json      generated: the scenario the model runs, with the lowered program
    a.npy, b.npy       generated: input data
```

```bash
pixi run scenarios           # write the generated files
pixi run scenarios --check   # "stale: nothing" when they match the sources
```

```
written: clusters/alu4.json, clusters/red4.json, clusters/mul1.json, dma/scenario.json, ...
```

`pixi run model-run` and `pixi run test` write them first on their own, so on a
fresh clone step 1 is optional. The scenarios are `vecadd`, `vecadd_conflict`
(a and b in the same banks), `vecadd_tiled` (3 tiles, 471 cycles), `fmul`
(double buffered multiply, 525 cycles), `reduce` and `dma`.

### 2.2 Run a scenario

```bash
pixi run model-run scenarios/vecadd/scenario.json          --out out/vecadd --trace beat
pixi run model-run scenarios/vecadd_conflict/scenario.json --out out/vecadd_conflict --trace beat
pixi run model-run scenarios/fmul/scenario.json            --out out/fmul --trace task
```

```
vecadd: 77 cycles (skip on, trace beat) -> out/vecadd
```

Outputs in the `--out` directory: `run.json` (register map, total cycles,
`csr_read` values), `profile.json`, `trace.jsonl` with `trace_meta.json` when
tracing is on, and `l1.npy` / `l2.npy` (flat words in address order). The
files are byte-identical on every run and with `--no-skip`, so two runs can be
compared with `diff`:

```bash
diff out/vecadd/profile.json out/vecadd_conflict/profile.json
grep '"k": "stall"' out/vecadd_conflict/trace.jsonl | head
python -c "import numpy as np; print(np.load('out/vecadd/l2.npy')[256:264, 0])"   # first words of c
```

| Option | Meaning |
|---|---|
| `--out DIR` | Output directory, created if missing (required) |
| `--trace off\|task\|beat` | Trace level: off (default), tasks and commands, or per-beat detail |
| `--trace-source NAME` | Beat events of this component only (repeatable) |
| `--trace-window A:B` | Beat events of cycles A to B only |
| `--no-skip` | Tick every cycle; same results, slower |
| `--max-cycles N` | Overrides the scenario's own limit |

Exit codes: 0 done, 1 the run failed or did not finish, 2 the scenario is
invalid. `--no-skip` only changes the speed, never the results; if the two ever
disagree, that is a bug in the model.

### 2.3 View in the browser

```bash
pixi run view out/vecadd out/vecadd_conflict
```

```
serving vecadd, vecadd_conflict at http://127.0.0.1:8765/ (Ctrl-C to stop)
```

Open that address in a browser. With several runs, pick one in the Run menu
at the top. The tabs switch between the profile report and the schedule
(every component's activity per cycle); under the schedule, the cluster view
shows the selected cycle (banks, interconnect, streamers, accelerator, DMA).
Click a cycle in the schedule or use the arrow keys to step. The schedule needs at least `--trace task`; the per-beat rows
and the cluster view's traffic need `--trace beat`. After rerunning a
scenario into the same directory, press Reload.

The server listens on 127.0.0.1 only. When it runs on a remote machine,
forward the port from your laptop and open http://127.0.0.1:8765/ there:

```bash
ssh -L 8765:127.0.0.1:8765 you@server   # then run `pixi run view ...` on the server
```

Use `--port N` for another port.

### 2.4 Change a scenario

A task list is a readable list of steps that SNAX-LOWER turns into the
controller's program of `csr_write`, `csr_read` and `wait` commands:

| `op` | Fields | Becomes |
|---|---|---|
| `configure` | `task_name`, `type`, `component`, `after`, `wait_mode`, `values` | the component's register writes, at this point |
| `start` | `tasks` | the waits these tasks need, then their starts |
| `sync` | `task`, `mode` | a wait on that task's component, here |
| `read` | `reg` | a register read, e.g. `acc.busy_cycles` |

`after` names the tasks that must finish before this one starts; the lowering
adds only the waits that needs. Edit a `tasks.json`, then regenerate and run:

```bash
pixi run scenarios
pixi run model-run scenarios/vecadd_tiled/scenario.json --out out/tiled_try --trace task
pixi run view out/tiled_try
```

For example, deleting the two `sync` steps in the middle of
`vecadd_tiled/tasks.json` overlaps each tile's first load with the previous
store: 447 cycles instead of 471, same result. Every field of a task list is
described in `docs/CONTRACTS.md` section 9. A new scenario is a new folder with
a `tasks.json` and a `scenario.py` whose `make()` returns the scenario and its
input arrays; `scenarios/make.py` finds it on its own.

### 2.5 Tests

```bash
pixi run test          # everything; writes the generated scenario files first
pixi run test-model    # SNAX-MODEL only
pixi run test-lower    # SNAX-LOWER only (task lists)
```
