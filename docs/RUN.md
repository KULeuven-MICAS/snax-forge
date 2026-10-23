# From a recipe to a run on SNAX-GVSoC

One design point the whole way, with one command: `pixi run flow` takes a
recipe through `sandbox` and `design` on `platforms/snax_forge.json`,
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

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json

    Pixi task (flow in default): python -m snax_forge.flow recipes/vecadd.json --platform platforms/snax_forge.json
      build     building SNAX-GVSoC: no install in out/snax-gvsoc; log in out/flow/vecadd/build.log
    make: Entering directory '$ROOT/snax-forge/gvsoc'
    ...
    make: Leaving directory '$ROOT/snax-forge/gvsoc'
    flow vecadd (vecadd: W=4, N=64; N=64) on snax_forge -> out/flow/vecadd/
      sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
      design    20 checks passed; platform base snax_forge, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (12 steps)
      gen       snax_forge.hjson, snax_forge_streamer.hjson, snax_forge_acc.hjson, main.c, data.h (seed 0) -> gen/
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
- `gen`: the generator's output in `gen/`: the cluster file `snax_forge.hjson`
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
    	--work-dir $ROOT/snax-forge/out/flow/vecadd/run/work --param chip/soc/binary=$ROOT/snax-forge/out/flow/vecadd/sw/forge.elf --param hjson=$ROOT/snax-forge/out/flow/vecadd/gen/snax_forge.hjson \
    	--trace='/summary$:$ROOT/snax-forge/out/flow/vecadd/run/events.log' run
    forge cycles: 226
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    make: Leaving directory '$ROOT/snax-forge/gvsoc'

`run.json` says what ran, with where the linker put each L2 container's
array (`elf_symbols`, the L2 regions, D86), `profile.json` what it measured (C§7), from the
accelerator's classes and firings to the cores' CSR counts, the DMA, the
crossbar's ports and the banks:

    cat out/flow/vecadd/run/run.json; head -n 14 out/flow/vecadd/run/profile.json

    {
     "design_point": "design/design_point.json",
     "hjson": ["gen/snax_forge.hjson", "gen/snax_forge_streamer.hjson", "gen/snax_forge_acc.hjson"],
     "elf": "sw/forge.elf",
     "elf_sha256": "b6b4b1c70d8c8986c478ecdc4fce694a42a49bfd3f4ba3eb0440e8bd932070a5",
     "elf_symbols": {
      "A": {"addr": 2147494144, "size": 512},
      "B": {"addr": 2147494656, "size": 512},
      "C": {"addr": 2147496704, "size": 512}
     },
     "target": "snax_hjson",
     "snax_gvsoc": {"commit": "c64c8e7b2e3a25a320293649b095f48573c16926", "dirty": false},
     "trace_level": "off",
     "trace_sources": [],
     "vcd": false
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

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --trace task

    Pixi task (flow in default): python -m snax_forge.flow recipes/vecadd.json --platform platforms/snax_forge.json --trace task
    flow vecadd_task (vecadd: W=4, N=64; N=64) on snax_forge -> out/flow/vecadd_task/
      sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
      design    20 checks passed; platform base snax_forge, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (12 steps)
      gen       snax_forge.hjson, snax_forge_streamer.hjson, snax_forge_acc.hjson, main.c, data.h (seed 0) -> gen/
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
     "sources": ["acc", "acc_a", "acc_b", "acc_out", "dma", "core0", "core1"],
     "filter": []
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

and the viewer shows the profile, the schedule with the cluster view and
the memory tab in the browser (open the address it prints, or see "The
viewer" below for the MICAS server):

    pixi run view out/flow/vecadd_task out/flow/vsum_task

    Pixi task (view in default): python -m snax_forge.viz out/flow/vecadd_task out/flow/vsum_task
    serving vecadd_task, vsum_task at http://127.0.0.1:8765/ (Ctrl-C to stop)

### The beat levels

Above `task` the levels add a line per beat (C§7, snax-gvsoc
`docs/EVENTS.md`): `beat` every grant, DMA beat and burst, firing and push,
`stall` every refused request and queue count, `finegrained` every
instruction of the cores. Lines are linked by ids, not by order: a mover's
beat `b`, a firing's `n`, a DMA beat's copy `id` and index `i`. By default
the sources are the accelerator, its movers and the DMA (at `finegrained`
every core too, for its instructions); `--trace stall:acc,acc_a` names
them. The profile is the task level's, byte for byte:

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --trace stall

    ...
      check     passed: 226 cycles (main to forge_check); data 0 errors (64 words); counters add_acc 19 (16 beats)
      convert   run/events.log -> profile.json, trace.jsonl (343 events in the window; boot 2812; sources acc, acc_a, acc_b, acc_out, dma)
      report    report/design.md, report/run.md

Each event carries its raw words and what they mean: the region, the
elements and their values; a stall names the port that won the bank:

    grep -m1 '"k": "grant"' out/flow/vecadd_stall/run/trace.jsonl
    grep -m1 '"k": "stall"' out/flow/vecadd_stall/run/trace.jsonl
    grep -m1 '"k": "push"' out/flow/vecadd_stall/run/trace.jsonl

    {"t": 121, "k": "grant", "src": "acc_a.lane0", "task": "add_acc_a", "op": "r", "mem": "l1", "addr": 268435456, "bank": 0, "row": 0, "value": 701, "back": 122, "b": 0, "region": "A", "elems": [0], "vals": [701]}
    {"t": 121, "k": "stall", "src": "acc_b.lane0", "task": "add_acc_b", "addr": 268435968, "bank": 0, "row": 2, "by": "acc_a.lane0", "by_addr": 268435456, "b": 0, "region": "B", "elems": [0]}
    {"t": 124, "k": "push", "src": "acc", "task": "add_acc", "n": 0, "first": 0, "last": 0, "value": [1451, 18446744073709551159, 179, 18446744073709551597], "vals": [1451, -457, 179, -19]}

`--vcd` writes gtkwave's VCD of the same sources and its view into `run/`
(`gtkwave out/flow/vecadd_stall/run/view.gtkw`, outside the container);
snax-forge never reads them:

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --trace stall --vcd --name vecadd_stall_vcd

    ...
      convert   run/events.log -> profile.json, trace.jsonl (343 events in the window; boot 2812; sources acc, acc_a, acc_b, acc_out, dma); run/all.vcd, run/view.gtkw (gtkwave)
      report    report/design.md, report/run.md

### The viewer

From `task` on, the Schedule tab draws the window as lanes: one per core
(its classes, a band of its functions, a tick per CSR access), the DMA (a
bar per copy), each mover and the accelerator (a bar per task, coloured by
its classes from `beat` on). A quiet stretch of 16 cycles or more, where no
task or copy is open, is folded to a fixed width: hover it for what the
cores did there, click it to open it (the ✕ in the ruler folds it again);
the boot before the window stays folded. The wheel zooms the open cycles,
shift + wheel scrolls. Click a cycle to show it in the cluster view under
the schedule: every L1 bank by superbank, the interconnect, each mover's
lanes and FIFO slots, the DMA with L2, the accelerator, and the cycle's
lines. The right and left arrows step to the next or previous cycle with an
event, shift + arrow to the next or previous task boundary; Play steps as
the right arrow does. From `beat` on, detail rows under the lanes show each
port's grants and refusals, the DMA's beats and bursts, the firings and
pushes and each lane's FIFO count. Everything is in the address bar
(`#run=...&view=schedule&open=...&cycle=...`), so a link shows the same.

The recipe's `N` makes the vectors longer, so the accelerator's work stands
out against the cores' and the DMA's; `examples/loop1/` walks such a run:

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --set N=512 --trace stall

    ...
      check     passed: 486 cycles (main to forge_check); data 0 errors (512 words); counters add_acc 131 (128 beats)
      convert   run/events.log -> profile.json, trace.jsonl (2150 events in the window; boot 7238; sources acc, acc_a, acc_b, acc_out, dma)
      report    report/design.md, report/run.md

From `beat` on, the viewer's memory tab shows where each container lies in
L1 and L2 and how its words move (D96, D97): the marks the run's level has
(`arrival`, `use`, `wait`, and at `stall` `conflicts`), a click on a word
for its element's journey (its DMA beats, grants, firings with their values,
the push and where the result went), and the selected cycle's accesses. On
the MICAS server, start it in the container on the host's network, so its
port 8765 is the server's (from the folder that holds `snax-forge`):

    podman run --rm -it --network host -v "$PWD":"$PWD" -w "$PWD/snax-forge" ghcr.io/kuleuven-micas/snax-forge:main
    pixi run view out/flow/vecadd_stall out/flow/vsum_stall

    Pixi task (view in default): python -m snax_forge.viz out/flow/vecadd_stall out/flow/vsum_stall
    serving vecadd_stall, vsum_stall at http://127.0.0.1:8765/ (Ctrl-C to stop)

It listens on 127.0.0.1 only, so on your own machine forward the port, then
open `http://localhost:8765/`:

    ssh -N -L 8765:127.0.0.1:8765 <user>@micaseb19

## 3. A second kernel: vsum

`recipes/vsum.json` binds vsum's reduction to the one-lane Chisel
Accumulator. GVSoC is built now, so the whole flow takes seconds:

    pixi run flow recipes/vsum.json --platform platforms/snax_forge.json

    Pixi task (flow in default): python -m snax_forge.flow recipes/vsum.json --platform platforms/snax_forge.json
    flow vsum (vsum: W=1; N=64) on snax_forge -> out/flow/vsum/
      sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
      design    20 checks passed; platform base snax_forge, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (9 steps)
      gen       snax_forge.hjson, snax_forge_streamer.hjson, snax_forge_acc.hjson, main.c, data.h (seed 0) -> gen/
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

    pixi run flow recipes/dot.json --platform platforms/snax_forge.json

    Pixi task (flow in default): python -m snax_forge.flow recipes/dot.json --platform platforms/snax_forge.json
    FAILED recipes/dot.json: gen: [gen.instances] 2 accelerated instances ['mul', 'sum']; the platform has one accelerator slot and the program drives one accelerator (open item 79)

Both exit 1 and write the failure to `flow.log`.

## 5. The same path by hand

The flow's stages are tools of their own. The design point, the generator
into `out/gvsoc/vecadd/`, then `gvsoc/Makefile` (from pixi or the plain
container shell: it names the image's tools) for the program and the run:

    pixi run sandbox recipes/vecadd.json
    pixi run design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_forge.json
    pixi run gen-gvsoc out/design/vecadd/design_point.json

    ...
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/main.c
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/manifest.json
    run: make -C gvsoc run TARGET=snax_hjson GVRUN_ARGS="--param hjson=$ROOT/snax-forge/out/gvsoc/vecadd/snax_forge.hjson" ELF=...

`make -C gvsoc build` builds GVSoC (the flow did already), `make sw` the
program into `out/snax-gvsoc/sw/vecadd/`, and `make run` runs it:

    make -C gvsoc sw APPS=forge FORGE=../out/gvsoc/vecadd DESIGN=vecadd
    make -C gvsoc run TARGET=snax_hjson DESIGN=vecadd \
        GVRUN_ARGS="--param hjson=$PWD/out/gvsoc/vecadd/snax_forge.hjson" \
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

## 6. The accelerator's RTL

The same design point gives the accelerator's Chisel RTL (GEN1, D138),
which snax_cluster builds from the bundle (section 7); the flow does not
call it. `gen-rtl`
writes `rtl.json` (C§17), then elaborates it through `hw/chisel/`'s
`Emit --spec` (sbt, so the first run fetches the Chisel jars):

    pixi run gen-rtl out/design/vecadd/design_point.json

    ...
    wrote out/rtl/vecadd/rtl.json: snax_forge_shell_wrapper, ElementwiseTiledSpatial
    wrote out/rtl/vecadd/sv/ElementwiseAlu_w64_add.sv
    wrote out/rtl/vecadd/sv/ElementwiseTiledSpatial_w64_n4_t32_add.sv
    wrote out/rtl/vecadd/sv/snax_forge_csr_manager.sv
    wrote out/rtl/vecadd/sv/snax_forge_ctrl_adapter.sv
    wrote out/rtl/vecadd/sv/snax_forge_shell_wrapper.sv
    wrote out/rtl/vecadd/emitted.json

`snax_forge_shell_wrapper` has the ports snax_cluster's
`snax_acc_wrapper.sv.tpl` connects with `snax_disable_csr_manager`, and
carries its own CSR manager (D130); `emitted.json` lists its modules and
files. `hw/chisel/README.md` has the specs that check it.

## 7. Bundle for snax_cluster

The bundle (C§18, D139) is what snax_cluster builds the design's cluster
from: gen-gvsoc's files at the root, the accelerator's RTL under `rtl/`
with its file list, and `bundle.json`, all made again (sbt, as section 6):

    pixi run bundle out/design/vecadd/design_point.json

    ...
    bundle out/bundle/vecadd: snax_forge_shell_wrapper from out/design/vecadd/design_point.json, snax-forge $COMMIT
    wrote out/bundle/vecadd/snax_forge.hjson
    ...
    wrote out/bundle/vecadd/rtl/rtl.json
    wrote out/bundle/vecadd/rtl/sv/ElementwiseAlu_w64_add.sv
    ...
    wrote out/bundle/vecadd/rtl/emitted.json
    wrote out/bundle/vecadd/rtl/files.f
    wrote out/bundle/vecadd/bundle.json

(`$COMMIT` is the checkout's `git rev-parse HEAD`, followed by `, dirty`
when the work tree has uncommitted changes.) `rtl/files.f` lists the RTL
from `rtl/`, in `emitted.json`'s order:

    cat out/bundle/vecadd/rtl/files.f

    sv/ElementwiseAlu_w64_add.sv
    sv/ElementwiseTiledSpatial_w64_n4_t32_add.sv
    sv/snax_forge_csr_manager.sv
    sv/snax_forge_ctrl_adapter.sv
    sv/snax_forge_shell_wrapper.sv

The cluster file's slot names it (`snax_forge_rtl: "rtl/files.f"`), and
snax_cluster builds the cluster from that file alone (C§18): its snaxgen
writes `generated/snax_forge.f`, which its vsim and Verilator builds read,
with the cluster `snax_forge_cluster` as its Bender target. From the folder
that holds both checkouts, `rtl-gen` and the vsim preparation in
snax_cluster's development image (on barnard3 `c5db84ad7bce`), with
`CFG_OVERRIDE` the bundle's cluster file by its absolute path:

    podman run --rm -it -v "$PWD":"$PWD" -w "$PWD" c5db84ad7bce
    cd snax_cluster
    B=$PWD/../snax-forge/out/bundle/vecadd/snax_forge.hjson
    make -C target/snitch_cluster clean
    make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B
    make -C target/snitch_cluster vsim_preparation CFG_OVERRIDE=$B

then Questa outside podman, from snax_cluster with `bender` on `PATH`:

    make -C target/snitch_cluster bin/snitch_cluster.vsim CFG_OVERRIDE=$PWD/../snax-forge/out/bundle/vecadd/snax_forge.hjson

    ...
    Optimized design name is tb_bin_opt
    Errors: 0, Warnings: 18

`generated/snax_forge.f` lists the bundle's five files and snaxgen's three
for the accelerator, and `work-vsim/compile.vsim.tcl` compiles it after
Bender's list. The 18 warnings are `snax_alu_cluster`'s too (the
testharness's open CSR and `obs_o` ports, the vendored AXI burst splitter,
`+acc`). Section 8 runs an ELF on it.

snax_cluster keeps this bundle as a fixture,
`target/snitch_cluster/cfg/snax_forge/vecadd/`, which its
`snax-forge-ci.yml` builds with Verilator. To refresh it, make the bundle
again as above and copy what the fixture's `README.md` lists from
`out/bundle/vecadd/`: the three hjson files, `bundle.json`, `rtl/files.f`
and the files `rtl/files.f` lists.

## 8. Running a bundle on snax_cluster's RTL

GEN2b's comparison (D123, D140): one ELF, snax-forge's, built from the
bundle, runs on SNAX-GVSoC and on snax_cluster's Verilator and Questa
builds, and the printed lines are compared by hand. snax-forge never runs
RTL; snax_cluster's `target/snitch_cluster/snax_forge.mk` does (C§18).

The bundle is section 7's, `out/bundle/vecadd/`. Its ELF comes from
snax-gvsoc's generic app (section 5), in snax-forge's image:

    make -C gvsoc sw APPS=forge FORGE=../out/bundle/vecadd DESIGN=vecadd
    make -C gvsoc run TARGET=snax_hjson DESIGN=vecadd \
        GVRUN_ARGS="--param hjson=$PWD/out/bundle/vecadd/snax_forge.hjson" \
        ELF=../out/snax-gvsoc/sw/vecadd/forge.elf
    sha256sum out/snax-gvsoc/sw/vecadd/forge.elf

    ...
    forge cycles: 226
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    make: Leaving directory '$ROOT/snax-forge/gvsoc'
    b6b4b1c70d8c8986c478ecdc4fce694a42a49bfd3f4ba3eb0440e8bd932070a5  out/snax-gvsoc/sw/vecadd/forge.elf

It is the ELF the flow built (section 2): the same `elf_sha256`. Verilator
runs in snax_cluster's pixi environment, as its CI does; from snax_cluster,
with `CFG_OVERRIDE` the bundle's cluster file and `ELF` the program, both
by absolute path:

    B=$PWD/../snax-forge/out/bundle/vecadd/snax_forge.hjson
    E=$PWD/../snax-forge/out/snax-gvsoc/sw/vecadd/forge.elf
    pixi run make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B
    pixi run make -C target/snitch_cluster snax-forge-sim SIM=vlt CFG_OVERRIDE=$B
    pixi run make -C target/snitch_cluster snax-forge-run SIM=vlt CFG_OVERRIDE=$B ELF=$E

    ...
    bash -c 'set -o pipefail; bin/snitch_cluster.vlt $ROOT/snax-forge/out/snax-gvsoc/sw/vecadd/forge.elf  2>&1 | tee logs/snax-forge-vlt.txt'
    forge cycles: 239
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    [fesvr] Wrote 36 bytes of bootrom to 0x1000
    ...

The output is also in `target/snitch_cluster/logs/snax-forge-vlt.txt`,
beside the run's traces; `snax-forge-run` fails on the program's nonzero
exit code. Without `ELF` it builds and runs snax_cluster's own copy of the
program from the bundle's `main.c` (`sw/apps/snax-forge`), which its CI
uses for pass/fail only; both print the same lines. Before that build,
`snax-forge-sw` checks the bundle's `include/streamer_csr_addr_map.h`
against the one snaxgen wrote (their `#define` lines).

Questa: `rtl-gen` and `vsim_preparation` in snax_cluster's development image
(section 7), started from the folder that holds both checkouts, then the
build and the run outside it, with `bender` and Questa on `PATH`:

    podman run --rm -it -v "$PWD":"$PWD" -w "$PWD" c5db84ad7bce
    cd snax_cluster
    B=$PWD/../snax-forge/out/bundle/vecadd/snax_forge.hjson
    make -C target/snitch_cluster clean-vsim CFG_OVERRIDE=$B
    make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B
    make -C target/snitch_cluster vsim_preparation CFG_OVERRIDE=$B

    B=$PWD/../snax-forge/out/bundle/vecadd/snax_forge.hjson
    E=$PWD/../snax-forge/out/snax-gvsoc/sw/vecadd/forge.elf
    make -C target/snitch_cluster snax-forge-sim SIM=vsim CFG_OVERRIDE=$B
    make -C target/snitch_cluster snax-forge-run SIM=vsim CFG_OVERRIDE=$B ELF=$E

    ...
    forge cycles: 239
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    ...
    # ** Info: [SUCCESS] Program finished successfully

Questa exits 0 whatever the program returns, so `snax-forge-run` reads the
exit code from that last line. Verilator and Questa print the same lines.
Against SNAX-GVSoC, the counter is equal and the window within 10% of the
RTL's (|w_gvsoc − w_rtl| / w_rtl):

| Design | SNAX-GVSoC window | RTL window | Difference | Counter (both) |
|---|---:|---:|---:|---:|
| vecadd, default N (64) | 226 | 239 | 5.4% | 19 |
| vecadd, `N=512` (`--set N=512`, its own bundle) | 486 | 500 | 2.8% | 131 |

`N` reaches the hardware only as the accelerator's `loopCount` CSR, so the
`N=512` bundle's `rtl/` is the default's (its RTL files byte-identical) and
one RTL build runs both ELFs. snax_cluster keeps the default bundle as its
fixture, `target/snitch_cluster/cfg/snax_forge/vecadd/` (section 7), whose
README has the same commands with `CFG_OVERRIDE=cfg/snax_forge/vecadd/snax_forge.hjson`.
