# From a recipe to a run on SNAX-GVSoC

One design point the whole way, with one command: `pixi run flow` takes a
recipe through `sandbox` and `design` on `platforms/snax_alu.json`,
generates the design point's hjson files, headers, data and program, builds
SNAX-GVSoC when needed and the program, runs it on `snax_hjson`, and takes
the program's own check (D136), then turns SNAX-GVSoC's events into the
run's profile and reports (D137). Each block is the command as typed, then
what it printed in a fresh tree with the submodule initialised (long
outputs cut to their last lines, marked `...`). Section 5 is the same path
by hand. snax-gvsoc's own `docs/RUN.md` runs its checked-in fixtures and its
regression.

## 1. A clone and the container

snax-gvsoc is a submodule, so initialise it once. Clone into any folder,
`$ROOT` below (the outputs' absolute paths start there; yours will be your
own):

    git clone https://github.com/KULeuven-MICAS/snax-forge.git
    cd snax-forge && git submodule update --init gvsoc/snax-gvsoc

    Cloning into 'snax-forge'...
    Submodule 'gvsoc/snax-gvsoc' (https://github.com/KULeuven-MICAS/snax-gvsoc) registered for path 'gvsoc/snax-gvsoc'
    Cloning into '$ROOT/snax-forge/gvsoc/snax-gvsoc'...
    Submodule path 'gvsoc/snax-gvsoc': checked out '3f8a3e9d7dbaecc7b977c45c1785cf187fa307ac'

Everything else runs in snax-forge's image (D118), started from the folder
that holds `snax-forge`:

    podman run --rm -it -v "$PWD":"$PWD" -w "$PWD" ghcr.io/kuleuven-micas/snax-forge:main
    cd snax-forge
    pixi install

    The default environment has been installed.

## 2. The flow

`recipes/vecadd.json` binds vecadd's add to `elementwise_add` at W = 4. The
first flow builds SNAX-GVSoC into `out/snax-gvsoc/` (a few minutes; its
output shows as it comes and goes to `out/flow/vecadd/build.log`):

    pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json

    Pixi task (flow in default): python -m snax_forge.flow recipes/vecadd.json --platform platforms/snax_alu.json
      build     building SNAX-GVSoC: no install in out/snax-gvsoc; log in out/flow/vecadd/build.log
    make: Entering directory '$ROOT/snax-forge/gvsoc'
    ...
    make: Leaving directory '$ROOT/snax-forge/gvsoc'
    flow vecadd (vecadd: W=4; N=64) on snax_alu -> out/flow/vecadd/
      sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
      design    21 checks passed; platform base snax_alu, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (12 steps)
      gen       snax_alu.hjson, snax_alu_streamer.hjson, snax_alu_acc.hjson, main.c, data.h (seed 0) -> gen/
      build     built: no install in out/snax-gvsoc (snax-gvsoc 3f8a3e9d)
      sw        sw/forge.elf (sha256 b6b4b1c70d8c8986)
      gvsoc     snax_hjson, exit 0 -> run/
      check     passed: 226 cycles (main to forge_check); data 0 errors (64 words); counters add_acc 19 (16 beats)
      convert   run/events.log -> profile.json (the summary lines; trace off)
      report    report/design.md, report/run.md

    echo "exit $?"

    exit 0

One line per stage:

- `sandbox`, `design`: the recipe's steps, each checked by the reference
  executor, then the design checks, the memory plan and the design point.
- `lower`: the task list, `tasks.json`.
- `gen`: the generator's output in `gen/`: the cluster file `snax_alu.hjson`
  (the platform's source with the slot set for this design), the streamer
  and datapath files it refers to, the headers, `data.h` (the kernel's
  `make_inputs` at seed 0, and the reference executor's goldens, checked
  against the kernel's reference first) and `main.c`. Nothing is compiled
  for a design point: SNAX-GVSoC's installed `snax_hjson` reads the cluster
  file at run time.
- `build`: GVSoC through `gvsoc/Makefile`, here because there was no install;
  later flows say `up to date` until the submodule's commit or path changes
  or its tree is dirty.
- `sw`: the program, built by snax-gvsoc's generic app into `sw/`.
- `gvsoc`: the run, its output in `run/stdout.txt` and `run/stderr.txt`.
- `check`: what the program printed and returned. `226 cycles` is its
  window, from `mcycle` at main's entry to `mcycle` at `forge_check()`'s
  entry; `data 0 errors (64 words)` every output word against its golden;
  `add_acc 19 (16 beats)` the shell's cycle counter for the accelerator task
  `add_acc` beside the task's beats. `passed` is the program's exit code 0,
  and the flow's exit follows it.
- `convert`: the counters SNAX-GVSoC's components wrote to `run/events.log`
  (snax-gvsoc `docs/EVENTS.md`) named into `run/profile.json` (C§7). At the
  default trace level, `off`, these are the `summary` lines alone.
- `report`: `report/design.md` (what was built) and `report/run.md` (where
  the cycles went).

The folder holds every stage's output:

    ls out/flow/vecadd out/flow/vecadd/run out/flow/vecadd/report

    out/flow/vecadd:
    build.log
    design
    flow.log
    gen
    report
    run
    sandbox
    sw
    sw.log
    tasks.json

    out/flow/vecadd/report:
    design.md
    run.md

    out/flow/vecadd/run:
    events.log
    profile.json
    run.json
    stderr.txt
    stdout.txt
    work

What the program printed, after make's own lines:

    cat out/flow/vecadd/run/stdout.txt

    make: Entering directory '$ROOT/snax-forge/gvsoc'
    mkdir -p $ROOT/snax-forge/out/flow/vecadd/run/work
    $ROOT/snax-forge/out/snax-gvsoc/install/bin/gvrun --target snax_hjson \
    	--work-dir $ROOT/snax-forge/out/flow/vecadd/run/work --param chip/soc/binary=$ROOT/snax-forge/out/flow/vecadd/sw/forge.elf --param hjson=$ROOT/snax-forge/out/flow/vecadd/gen/snax_alu.hjson \
    	--trace='/summary$:$ROOT/snax-forge/out/flow/vecadd/run/events.log' run
    forge cycles: 226
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    make: Leaving directory '$ROOT/snax-forge/gvsoc'

`run.json` says what ran, `profile.json` what it measured (C§7), from the
accelerator's classes and firings to the cores' CSR counts, the DMA, the
crossbar's ports and the banks:

    cat out/flow/vecadd/run/run.json; head -n 14 out/flow/vecadd/run/profile.json

    {
     "design_point": "design/design_point.json",
     "hjson": ["gen/snax_alu.hjson", "gen/snax_alu_streamer.hjson", "gen/snax_alu_acc.hjson"],
     "elf": "sw/forge.elf",
     "elf_sha256": "b6b4b1c70d8c8986c478ecdc4fce694a42a49bfd3f4ba3eb0440e8bd932070a5",
     "target": "snax_hjson",
     "snax_gvsoc": {"commit": "3f8a3e9d7dbaecc7b977c45c1785cf187fa307ac", "dirty": false},
     "trace_level": "off"
    }
    {
     "total_cycles": 226,
     "accelerators": {
      "acc": {
       "cycles": {"busy": 16, "stall_in": 3, "stall_out": 0, "flush": 0, "idle": 207},
       "firings": 16,
       "utilisation": 0.07079646017699115
      }
     },
     "streamers": {
      "acc_a": {
       "write": false,
       "cycles": {"busy": 16, "stall_xbar": 0, "stall_fifo": 0, "idle": 210},
       "ports": ["acc_a.lane0", "acc_a.lane1", "acc_a.lane2", "acc_a.lane3"],
    ...

`--trace task` also turns on the components' `events` lines; the folder
gets the level in its name:

    pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json --trace task

    Pixi task (flow in default): python -m snax_forge.flow recipes/vecadd.json --platform platforms/snax_alu.json --trace task
    flow vecadd_task (vecadd: W=4; N=64) on snax_alu -> out/flow/vecadd_task/
      sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
      design    21 checks passed; platform base snax_alu, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (12 steps)
      gen       snax_alu.hjson, snax_alu_streamer.hjson, snax_alu_acc.hjson, main.c, data.h (seed 0) -> gen/
      build     up to date (snax-gvsoc 3f8a3e9d)
      sw        sw/forge.elf (sha256 b6b4b1c70d8c8986)
      gvsoc     snax_hjson, exit 0 -> run/
      check     passed: 226 cycles (main to forge_check); data 0 errors (64 words); counters add_acc 19 (16 beats)
      convert   run/events.log -> profile.json, trace.jsonl (49 events in the window; boot 2812)
      report    report/design.md, report/run.md

The trace is every start, done, SNAX CSR access, poll, barrier, state
change, function entry and DMA copy in the program's window, the one the
profile uses: from the program's `mcycle` read in main (`t = 0`) to its
read in `forge_check()`, which the core logs; `trace_meta.json` holds
`boot`, that first read in SNAX-GVSoC's cycles:

    head -n 4 out/flow/vecadd_task/run/trace.jsonl; cat out/flow/vecadd_task/run/trace_meta.json

    {"t": 33, "k": "func", "src": "core0", "name": "forge_core_0", "pc": 2147484444}
    {"t": 33, "k": "func", "src": "core1", "name": "forge_core_1", "pc": 2147484580}
    {"t": 62, "k": "csr", "src": "core0", "pc": 2147484452, "csr": 960, "block": "acc_streamer", "op": "w", "value": 268435456}
    {"t": 63, "k": "csr", "src": "core0", "pc": 2147484456, "csr": 961, "block": "acc_streamer", "op": "w", "value": 0}
    {
     "level": "task",
     "total_cycles": 226,
     "boot": 2812,
     "sources": ["acc", "acc_a", "acc_b", "acc_out", "dma", "core0", "core1"]
    }

The run report names each task from the task list:

    sed -n 1,22p out/flow/vecadd_task/report/run.md

    # Run report: vecadd_task

    |  |  |
    |---|---|
    | Window (main to forge_check, the program's mcycle) | 226 |
    | Functional check | passed |
    | Data | 0 errors over 64 words |
    | Counters | add_acc 19 |
    | Trace | task; boot 2812 cycles (to the mcycle read in main) |

    ## Tasks

    | Block | # | Task | Start | Done | Cycles | Direction | Bytes |
    |---|---:|---|---:|---:|---:|---|---:|
    | dma | 0 | load_A | 75 | 97 | 22 | L2 → L1 | 512 |
    | dma | 1 | load_B | 93 | 114 | 21 | L2 → L1 | 512 |
    | acc_a | 0 | add_acc_a | 119 | 137 | 18 | – | – |
    | acc_b | 0 | add_acc_b | 119 | 138 | 19 | – | – |
    | acc_out | 0 | add_acc_out | 119 | 141 | 22 | – | – |
    | acc | 0 | add_acc | 120 | 140 | 20 | – | – |
    | dma | 2 | store_C | 162 | 177 | 15 | L1 → L2 | 512 |

and the viewer shows the profile in the browser (open the address it
prints; the schedule comes back in FLOW3 round 3):

    pixi run view out/flow/vecadd_task out/flow/vsum_task

    Pixi task (view in default): python -m snax_forge.viz out/flow/vecadd_task out/flow/vsum_task
    serving vecadd_task, vsum_task at http://127.0.0.1:8765/ (Ctrl-C to stop)

`--trace beat` stops with a named error until FLOW3 round 2.

## 3. A second kernel: vsum

`recipes/vsum.json` binds vsum's reduction to the one-lane Chisel
Accumulator. GVSoC is built now, so the whole flow takes seconds:

    pixi run flow recipes/vsum.json --platform platforms/snax_alu.json

    Pixi task (flow in default): python -m snax_forge.flow recipes/vsum.json --platform platforms/snax_alu.json
    flow vsum (vsum: W=1; N=64) on snax_alu -> out/flow/vsum/
      sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
      design    21 checks passed; platform base snax_alu, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (9 steps)
      gen       snax_alu.hjson, snax_alu_streamer.hjson, snax_alu_acc.hjson, main.c, data.h (seed 0) -> gen/
      build     up to date (snax-gvsoc 3f8a3e9d)
      sw        sw/forge.elf (sha256 26b545a3812000c8)
      gvsoc     snax_hjson, exit 0 -> run/
      check     passed: 255 cycles (main to forge_check); data 0 errors (1 words); counters sum_acc 67 (64 beats)
      convert   run/events.log -> profile.json (the summary lines; trace off)
      report    report/design.md, report/run.md

`flow.log` holds the same lines, the design checks that ran and each
stage's time (this machine's; they change from run to run):

    tail -n 14 out/flow/vsum/flow.log

    stage times (wall clock, this machine):
      import    0.232 s
      sandbox   0.020 s
      design    0.015 s
      lower     0.006 s
      gen       0.052 s
      build     0.342 s
      sw        0.593 s
      gvsoc     1.941 s
      check     0.003 s
      convert   0.013 s
      report    0.046 s
      total     3.263 s

## 4. What the flow refuses

The generator writes a cluster file from the platform's source hjson and
puts one accelerator in its one slot, so two kinds of design point stop at
`gen` with a named rule, before anything is built. `small16` is a
hand-written platform with no source hjson (`gen.source`):

    pixi run flow recipes/vecadd.json --platform platforms/small16.json

    Pixi task (flow in default): python -m snax_forge.flow recipes/vecadd.json --platform platforms/small16.json
    FAILED recipes/vecadd.json: gen: [gen.source] platform small16 has no source hjson: read it from one with `pixi run design from-hjson`

`dot` binds two accelerators, a multiplier and a sum, and the platform has
one slot and the program drives one accelerator (`gen.instances`; several
accelerators on one core are open item 79, a composed one is M13):

    pixi run flow recipes/dot.json --platform platforms/snax_alu.json

    Pixi task (flow in default): python -m snax_forge.flow recipes/dot.json --platform platforms/snax_alu.json
    FAILED recipes/dot.json: gen: [gen.instances] 2 accelerated instances ['mul', 'sum']; the platform has one accelerator slot and the program drives one accelerator (open item 79)

Both exit 1 and write the failure to `flow.log`.

## 5. The same path by hand

The flow's stages are tools of their own. The design point, the generator
into `out/gvsoc/vecadd/`, then `gvsoc/Makefile` (from pixi or the plain
container shell: it names the image's tools) for the program and the run:

    pixi run sandbox recipes/vecadd.json
    pixi run design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_alu.json
    pixi run gen-gvsoc out/design/vecadd/design_point.json

    ...
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/main.c
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/manifest.json
    run: make -C gvsoc run TARGET=snax_hjson GVRUN_ARGS="--param hjson=$ROOT/snax-forge/out/gvsoc/vecadd/snax_alu.hjson" ELF=...

`make -C gvsoc build` builds GVSoC (the flow did already), `make sw` the
program into `out/snax-gvsoc/sw/vecadd/`, and `make run` runs it:

    make -C gvsoc sw APPS=forge FORGE=../out/gvsoc/vecadd DESIGN=vecadd
    make -C gvsoc run TARGET=snax_hjson DESIGN=vecadd \
        GVRUN_ARGS="--param hjson=$PWD/out/gvsoc/vecadd/snax_alu.hjson" \
        ELF=../out/snax-gvsoc/sw/vecadd/forge.elf

    ...
    Warning: installed platform tree $ROOT/snax-forge/out/snax-gvsoc/install/lib/libplatform_tree_snax_hjson.so does not match the live system tree, falling back to the JSON config path (rebuild the target to regenerate the tree).
    forge cycles: 226
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    make: Leaving directory '$ROOT/snax-forge/gvsoc'

The warning (also in the flow's `run/stderr.txt`) is expected: `make build`
made `snax_hjson`'s platform tree from its default cluster file, with
snax_cluster's ALU in the slot; this cluster file puts the generated
accelerator there, so gvrun builds the system from its JSON configuration
instead, on the same installed components. `GVRUN_ARGS` takes gvrun's own
options too, such as `--trace=snax_acc0/unit` for the unit's trace.
