# :hammer: :fire: SNAX-FORGE :fire: :hammer:

SNAX-FORGE is a platform for exploring how domain-specific accelerators perform
when plugged into a SNAX compute cluster, before committing to RTL. You bring a
kernel and a model of your accelerator (a BRM); SNAX-FORGE imports the kernel
into its own dataflow graph, maps it onto the cluster by a recipe you edit,
runs it on a model of the cluster, and shows where the cycles go. The model
is SNAX-GVSoC, a GVSoC model in which a Snitch core runs the real program,
checked against snax_cluster's RTL; the flow runs on it since FLOW3, and
SNAX-MODEL, the cycle-level Python model it replaced, was removed in RET1.
Humans and LLMs close the loop by editing the recipe or the platform and running again.

Where to read more:

| File | What it holds |
|---|---|
| `docs/ARCHITECTURE.md` | what each component does and how they fit together |
| `docs/CONTRACTS.md` | every artefact between the components, field by field |
| `docs/DECISIONS.md` | the decisions behind it, D1 onwards |
| `docs/STATUS.md` | milestones, tasks and open items |
| `docs/PAPER.md` | the paper plan |
| `docs/RUN.md` | a design point from recipe to a run on SNAX-GVSoC, each command with its output |
| `kernels/README.md` | the target kernels, staged |
| `examples/` | walk-throughs of the flow (written on SNAX-MODEL, out of date until FLOW3 round 3) |
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

## In the container

SNAX-GVSoC and the programs built from snax_cluster need tools pixi does not
carry: GVSoC's build tools, snax_cluster's RISC-V toolchain (clang 21.1.6,
bender 0.28.2). One image holds them, with pixi 0.41.4 and zsh (D118). Build
and push it from this repository's root:

```bash
docker build -t ghcr.io/kuleuven-micas/snax-forge:main container
docker push ghcr.io/kuleuven-micas/snax-forge:main
```

Run it from the folder that holds `snax-forge`, mounted at its own path.
snax-gvsoc comes with it as a submodule (`gvsoc/snax-gvsoc`); a
`snax_cluster` checkout beside it is needed only to run the RTL:

```bash
podman run --rm -it -v "$PWD":"$PWD" -w "$PWD" ghcr.io/kuleuven-micas/snax-forge:main
cd snax-forge && pixi install && pixi run check
```

- The image holds tools only. `pixi install` writes the environment to
  `.pixi/` in this repository; since the path is the same inside and outside
  the container, the same environment works from the host too, with the same
  pixi version (0.41.4, as CI).
- Adding `-v ~/.cache/rattler:/root/.cache/rattler` keeps pixi's downloads
  across containers.

# 1. The whole path

One command takes a kernel to a checked run on SNAX-GVSoC, in snax-forge's
image (D118), with the snax-gvsoc submodule checked out:

```bash
pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json
```

```
flow vecadd (vecadd: W=4, N=64; N=64) on snax_alu -> out/flow/vecadd/
  sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
  design    21 checks passed; platform base snax_alu, 0 changes; memory contiguous, 0 changes -> design/
  lower     tasks.json (12 steps)
  gen       snax_alu.hjson, snax_alu_streamer.hjson, snax_alu_acc.hjson, main.c, data.h (seed 0) -> gen/
  build     up to date (snax-gvsoc 3f8a3e9d)
  sw        sw/forge.elf (sha256 b6b4b1c70d8c8986)
  gvsoc     snax_hjson, exit 0 -> run/
  check     passed: 226 cycles (main to forge_check); data 0 errors (64 words); counters add_acc 19 (16 beats)
  convert   run/events.log -> profile.json (the summary lines; trace off)
  report    report/design.md, report/run.md
```

It imports the kernel, applies the recipe, pairs the result with the
platform, derives the task list, generates the design point's hjson files,
headers, data (the reference executor's goldens) and program, builds
SNAX-GVSoC when needed (the first time, a few minutes), builds the program
and runs it on `snax_hjson`; the program checks itself and prints its cycles
from main's entry to its check (D136, `docs/RUN.md`). Everything lands in
`out/flow/<name>/`, with `flow.log`, which holds what it printed, the design
checks that ran and how long each stage took. `run/profile.json` holds
what SNAX-GVSoC's components counted, and `report/design.md` and
`report/run.md` say what was built and where the cycles went (D99, D137).
With `--trace task` the run also records every start, done, CSR access,
wait and DMA copy in the program's window (`run/trace.jsonl`, C§7), into
`out/flow/vecadd_task/`; `pixi run view out/flow/vecadd_task` shows its
profile in the browser (section 2, View a run). `--trace beat`, `stall` and
`finegrained` add every grant, firing and push, every stall and queue
count, every instruction, and `--vcd` a gtkwave dump (`docs/RUN.md`). Change a design point with `--set`;
each one is added to the name, so the runs sit side by side:

```bash
pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json --set W=8                                    # out/flow/vecadd_W8: 220 cycles, counter 11
pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json --set platform.streamers.default.fifo_depth=2 # 232 cycles, counter 26
pixi run flow recipes/vsum.json --platform platforms/snax_alu.json                                                # out/flow/vsum: 255 cycles
```

The flow also starts from a bound graph: a file ending in `.snaxdfg` (the
last step of a sandbox run, or one you edited by hand) skips the import and
the recipe and goes straight to the design step. The kernel is the graph's
`name` (`--kernel K` otherwise), and there is no recipe, so only `platform.`
and `memory.` settings apply:

```bash
pixi run sandbox recipes/vecadd.json                                                 # out/sandbox/vecadd/2_bind.snaxdfg
pixi run flow out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_alu.json   # out/flow/vecadd: 226 cycles, as the recipe
```

A `--set` without a dot is a recipe param; `platform.` and `memory.` ones go to
the design step. The platform needs a source hjson (`platforms/snax_alu.json`
has one; `small16` does not, `gen.source`), and the design point one
accelerator (`dot` binds two, `gen.instances`: parked until M13).
`examples/loop1/` walks the flow as it ran on SNAX-MODEL and is out of date
until FLOW3 round 3; `examples/dot/` is parked with dot.

# 2. Step by step

The flow runs these tools in turn; each also runs on its own.

| Step | Command | Writes |
|---|---|---|
| Every kernel | `pixi run kernels` | nothing; one line per kernel: directory, tier, dtype, whether its SDFG builds and `import-dfg` takes it |
| SDFG of a kernel | `pixi run forge vecadd` | `out/sdfg/vecadd.raw.sdfg`, `.simplified.sdfg` |
| Import | `pixi run import-dfg vecadd` | `out/dfg/vecadd.snaxdfg` |
| Check a graph | `pixi run check-dfg out/dfg/vecadd.snaxdfg --kernel vecadd` | nothing; compares with the kernel's reference |
| Recipe | `pixi run sandbox recipes/vecadd.json [--set W=8]` | `out/sandbox/vecadd/<i>_<transform>.snaxdfg` |
| Undo a recipe | `pixi run sandbox recipes/vecadd_undo.json --graph out/sandbox/vecadd/2_bind.snaxdfg --out out/sandbox/vecadd_undo` | the graph back as imported |
| Design point | `pixi run design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/small16.json` | `out/design/vecadd/` platform, memory plan, design point |
| Only check | `pixi run design check out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/small16.json` | nothing |
| Keep a platform | `pixi run design save out/design/vecadd/platform.json NAME` | `platforms/NAME.json` |
| Platform from snax_cluster | `pixi run design from-hjson platforms/snax_alu.hjson snax_alu --force` | `platforms/snax_alu.json` from its source, a copy of snax_cluster's cluster file (LNK1) |
| Task list | `pixi run lower tasks out/design/vecadd/design_point.json` | `tasks.json` beside the point |
| The rest at once | `pixi run flow out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_alu.json` | `out/flow/vecadd/`: design point, task list, generated folder, program, checked SNAX-GVSoC run |

The design step names every problem with its fix, and writes working copies
of the platform and the memory plan (`platform.json`, `memory.json`) that
later runs continue from with `--platform` and `--memory`.

To look at graphs, e.g. every step of a recipe side by side:

```bash
pixi run view-dfg out/sandbox/vecadd/     # then open http://127.0.0.1:8766/
```

## View a run

The viewer and the reports read flow runs (section 1, C§7):

```bash
pixi run view out/flow/vecadd_task out/flow/vsum_task
```

```
serving vecadd_task, vsum_task at http://127.0.0.1:8765/ (Ctrl-C to stop)
```

With several runs, pick one in the Run menu at the top. The profile report
shows where each component's cycles went, the accelerator, the movers and
their FIFOs, the cores' waits, the banks and ports, and the DMA. The schedule
and the cluster view come back in FLOW3 round 3, the memory tab in round 2.
After rerunning into the same folder, press Reload. `pixi run report DIR`
writes a flow folder's reports again (`DIR/report/`); the flow writes them
after every run.

The server listens on 127.0.0.1 only. On a remote machine, forward the port
from your laptop (`ssh -L 8765:127.0.0.1:8765 you@server`) and open
http://127.0.0.1:8765/ there. Use `--port N` for another port.

# 3. Chisel blocks

The accelerator blocks behind the BRMs are a Chisel project of their own in
`hw/chisel/`, with its own environment (`-e hw`):

```bash
pixi run -e hw chisel-test    # the chiseltest specs
pixi run -e hw chisel-gen     # every catalogue configuration to out/hw/*.sv
```

See `hw/chisel/README.md`.

# 4. SNAX-GVSoC from snax-forge

[snax-gvsoc](https://github.com/KULeuven-MICAS/snax-gvsoc) is a submodule at
`gvsoc/snax-gvsoc` (D128); CI fetches it in one job (`gvsoc`). `gvsoc/Makefile` names the image's
Python, CMake and host compilers, so it runs the same from the container's plain shell and through
pixi (`pixi run make -C gvsoc ...`, as the flow does).

```bash
git submodule update --init gvsoc/snax-gvsoc
cd gvsoc
make build                      # GVSoC for TARGETS (default snax-gvsoc's SNAX_TARGETS) into out/snax-gvsoc/
make sw APPS=snax-alu           # sw/ and the named apps, into snax-gvsoc/build/sw/
make run TARGET=snax_alu ELF=snax-gvsoc/build/sw/apps/snax-alu/100/snax-alu.elf
```

`run` prints `Accelerator Done!`, `Accelerator Cycles: 25` and `Number of
errors: 0`. It calls the install's `gvrun`; `GVRUN_ARGS` goes before
`run` (e.g. `GVRUN_ARGS=--trace=snax_acc0`) and the work directory is
`out/snax-gvsoc/work/<target>/<design>` with `DESIGN` set, else
`out/snax-gvsoc/work/<target>/<elf name>`. `SNAX_GVSOC=../../snax-gvsoc` uses a
sibling checkout instead of the submodule; `build` clears `out/snax-gvsoc`
when the checkout changes, since CMake refuses a build folder of another one.
`pixi run gen-gvsoc out/design/<design>/design_point.json` writes a design
point's cluster to `out/gvsoc/<design>/` as three hjson files (the cluster
`snax_alu.hjson`, the streamer and the datapath block), its headers (GEN3,
LNK2a) and its self-checking program, `main.c` and `data.h` (GEN2a); the
installed `snax_hjson` reads the cluster at run time. From `gvsoc/`:

```bash
make sw APPS=forge FORGE=../out/gvsoc/<design> DESIGN=<design>
make run TARGET=snax_hjson DESIGN=<design> \
    GVRUN_ARGS="--param hjson=$PWD/../out/gvsoc/<design>/snax_alu.hjson" \
    ELF=../out/snax-gvsoc/sw/<design>/forge.elf
```

The program of an `out/` design is built into `out/snax-gvsoc/sw/<design>/`
(`FORGE_BUILD`), never where the submodule builds its fixture of the same
name.

## Refreshing a fixture

snax-gvsoc keeps one generated design point per folder in `forge/<design>/`
(vecadd, vecmul, vsum on `accumulate`'s Chisel Accumulator, and vsum_tree on
its adder tree), which its `make regress` runs (D128).
`tests/gen/test_fixture.py` generates each again and compares it byte for
byte with the submodule's copy; it skips when the submodule is not checked
out. When the generator or the design point changes, refresh the folder in a
snax-gvsoc checkout next to this one, the folder cleared first:

```bash
pixi run sandbox recipes/vecadd.json
pixi run design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_alu.json
pixi run gen-gvsoc out/design/vecadd/design_point.json --out ../snax-gvsoc/forge/vecadd
```

(the same with `vecmul`, `vsum` or `vsum_tree`), commit the folder in snax-gvsoc as written, then
bump the submodule here so the test compares against it.

# 5. Tests and lint

```bash
pixi run test          # everything
pixi run test-lower    # one block: test-lower, test-brm, test-dfg, test-sandbox, test-design, test-flow, test-run, test-report, test-gen, test-kernels
pixi run lint          # ruff check and format check (CI runs it); pixi run fmt fixes
pixi run clean         # remove out/ but out/snax-gvsoc/, caches and Chisel build trees (--all: out/snax-gvsoc/ too; --dry-run lists them)
```
