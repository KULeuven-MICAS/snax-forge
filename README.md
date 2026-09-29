# :hammer: :fire: SNAX-FORGE :fire: :hammer:

SNAX-FORGE is a platform for exploring how domain-specific accelerators perform
when plugged into a SNAX compute cluster, before committing to RTL. You bring a
kernel and a model of your accelerator (a BRM); SNAX-FORGE imports the kernel
into its own dataflow graph, maps it onto the cluster by a recipe you edit,
runs it in a cycle-level Python model of the cluster, and shows where the
cycles go. Humans and LLMs close the loop by editing the recipe or the
platform and running again.

Where to read more:

| File | What it holds |
|---|---|
| `docs/ARCHITECTURE.md` | what each component does and how they fit together |
| `docs/CONTRACTS.md` | every artefact between the components, field by field |
| `docs/DECISIONS.md` | the decisions behind it, D1 onwards |
| `docs/STATUS.md` | milestones, tasks and open items |
| `hw/chisel/README.md` | the Chisel accelerator blocks |

# Setup

You need [pixi](https://pixi.sh):

```bash
curl -fsSL https://pixi.sh/install.sh | sh
git clone git@github.com:KULeuven-MICAS/snax-forge.git
cd snax-forge
pixi install
pixi run check      # OK  python 3.11…  dace 1.0.2  networkx …  numpy …
```

Every command below runs through pixi, from the repository root.

# 1. The whole path

One command takes a kernel to a checked model run:

```bash
pixi run flow recipes/vecadd.json --platform platforms/small16.json
```

```
flow vecadd (vecadd: W=4; N=64) on small16 -> out/flow/vecadd/
  sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
  design    20 checks passed; platform base small16, 0 changes; memory contiguous, 0 changes -> design/
  lower     cluster.json (7 components), tasks.json (12 steps, 57 commands)
  scenario  scenario.json, inputs A.npy, B.npy, C.npy (make_inputs, seed 0)
  run       85 cycles -> run/  (pixi run view out/flow/vecadd/run)
  check     C (64 elements, from l2) equals the vecadd reference and REF1
  report    report/design.md, report/run.md, flow.log
```

It imports the kernel, applies the recipe, pairs the result with the
platform, derives the cluster file and the task list, runs the model and
checks the output against the kernel's own reference and the reference
executor, then writes a design report and a run report (`report/`) and
`flow.log`, which holds what it printed and the design checks that ran.
Everything lands in `out/flow/<name>/`, traced at `task` level
(`--trace beat` for the data movement, `--trace off` for none). Change a
design point with `--set`; each one is added to the name, so the runs sit side
by side:

```bash
pixi run flow recipes/vecadd.json --platform platforms/small16.json --set W=8                  # out/flow/vecadd_W8: 73 cycles
pixi run flow recipes/vecadd.json --platform platforms/small16.json --set platform.l1.n_banks=32 # out/flow/vecadd_l1.n_banks32
pixi run flow recipes/vecadd.json --platform platforms/small16.json --set memory.B.l1.base=576  # out/flow/vecadd_B.l1.base576: b in other banks, 77 cycles
pixi run view out/flow/vecadd/run out/flow/vecadd_B.l1.base576/run                            # then open http://127.0.0.1:8765/
pixi run report out/flow/vecadd                                                                # writes report/ again, e.g. after a code change
```

A `--set` without a dot is a recipe param; `platform.` and `memory.` ones go to
the design step. `examples/loop1/README.md` walks through one turn of the loop
this way: reading why the default run takes 85 cycles, predicting what moving
B does, and checking it.

# 2. Step by step

The flow runs these tools in turn; each also runs on its own.

| Step | Command | Writes |
|---|---|---|
| SDFG of a kernel | `pixi run forge vecadd` | `out/sdfg/vecadd.raw.sdfg`, `.simplified.sdfg` |
| Import | `pixi run import-dfg vecadd` | `out/dfg/vecadd.snaxdfg` |
| Check a graph | `pixi run check-dfg out/dfg/vecadd.snaxdfg --kernel vecadd` | nothing; compares with the kernel's reference |
| Recipe | `pixi run sandbox recipes/vecadd.json [--set W=8]` | `out/sandbox/vecadd/<i>_<transform>.snaxdfg` |
| Undo a recipe | `pixi run sandbox recipes/vecadd_undo.json --graph out/sandbox/vecadd/2_bind.snaxdfg --out out/sandbox/vecadd_undo` | the graph back as imported |
| Design point | `pixi run design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/small16.json` | `out/design/vecadd/` platform, memory plan, design point |
| Only check | `pixi run design check out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/small16.json` | nothing |
| Keep a platform | `pixi run design save out/design/vecadd/platform.json NAME` | `platforms/NAME.json` |
| Cluster file | `pixi run lower cluster out/design/vecadd/design_point.json` | `cluster.json` beside the point |
| Task list | `pixi run lower tasks out/design/vecadd/design_point.json` | `tasks.json` beside the point |

The design step names every problem with its fix, and writes working copies
of the platform and the memory plan (`platform.json`, `memory.json`) that
later runs continue from with `--platform` and `--memory`.

To look at graphs, e.g. every step of a recipe side by side:

```bash
pixi run view-dfg out/sandbox/vecadd/     # then open http://127.0.0.1:8766/
```

# 3. The model on its own: scenarios

The model also runs hand-written scenarios, which is how SNAX-MODEL is tested
and how a platform feature is tried before any kernel needs it.

```bash
pixi run scenarios                                                                 # 1. generate the scenario files
pixi run model-run scenarios/vecadd/scenario.json --out out/vecadd --trace beat   # 2. run one
pixi run view out/vecadd                                                           # 3. open http://127.0.0.1:8765/
```

### 3.1 The scenario files

Each scenario is a folder under `scenarios/`. Two files in it are the sources
you edit; the rest is generated and ignored by git:

```
scenarios/
  make.py              writes every generated file below
  clusters/
    clusters.py        source: the clusters (alu4, red4, mul1)
    alu4.json ...      generated: one cluster file each
  vecadd/
    tasks.json         source: the task list (what runs, in which order, what it waits for)
    scenario.py        source: data, memory layout, cluster, cycle limit
    scenario.json      generated: the scenario the model runs, with the lowered program
    a.npy, b.npy       generated: input data
```

`pixi run scenarios --check` prints `stale: nothing` when the generated files
match their sources. `pixi run model-run` and `pixi run test` write them first
on their own. The scenarios are `vecadd`, `vecadd_conflict` (a and b in the
same banks), `vecadd_tiled` (3 tiles, 471 cycles), `fmul` (double buffered
multiply, 525 cycles), `reduce` and `dma`.

### 3.2 Run a scenario

```bash
pixi run model-run scenarios/vecadd/scenario.json          --out out/vecadd --trace beat
pixi run model-run scenarios/vecadd_conflict/scenario.json --out out/vecadd_conflict --trace beat
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
python -c "import numpy as np; print(np.load('out/vecadd/l2.npy')[128:136, 0])"   # first words of c
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

### 3.3 View a run

```bash
pixi run view out/vecadd out/vecadd_conflict
```

```
serving vecadd, vecadd_conflict at http://127.0.0.1:8765/ (Ctrl-C to stop)
```

With several runs, pick one in the Run menu at the top. The tabs switch
between the profile report and the schedule (every component's activity per
cycle); under the schedule, the cluster view shows the selected cycle (banks,
interconnect, streamers, accelerator, DMA). Click a cycle in the schedule or
use the arrow keys to step. The schedule needs at least `--trace task`; the
per-beat rows and the cluster view's traffic need `--trace beat`. After
rerunning into the same directory, press Reload.

The server listens on 127.0.0.1 only. On a remote machine, forward the port
from your laptop (`ssh -L 8765:127.0.0.1:8765 you@server`) and open
http://127.0.0.1:8765/ there. Use `--port N` for another port.

### 3.4 Change a scenario

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
in `docs/CONTRACTS.md` section 9. A new scenario is a new folder with a
`tasks.json` and a `scenario.py` whose `make()` returns the scenario and its
input arrays; `scenarios/make.py` finds it on its own.

# 4. Chisel blocks

The accelerator blocks behind the BRMs are a Chisel project of their own in
`hw/chisel/`, with its own environment (`-e hw`):

```bash
pixi run -e hw chisel-test    # the chiseltest specs
pixi run -e hw chisel-gen     # every catalogue configuration to out/hw/*.sv
```

See `hw/chisel/README.md`.

# 5. Tests and lint

```bash
pixi run test          # everything; writes the generated scenario files first
pixi run test-model    # one block: test-model, test-lower, test-brm, test-dfg, test-sandbox, test-design, test-flow
pixi run lint          # ruff check and format check (CI runs it); pixi run fmt fixes
pixi run clean         # remove out/, caches and Chisel build trees (--dry-run lists them)
```
