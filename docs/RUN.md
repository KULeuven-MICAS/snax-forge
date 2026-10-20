# From a recipe to a run on SNAX-GVSoC

One design point the whole way, with one command: `pixi run flow` takes a
recipe through `sandbox` and `design` on `platforms/snax_alu.json`,
generates the design point's hjson files, headers, data and program, builds
SNAX-GVSoC when needed and the program, runs it on `snax_hjson`, and takes
the program's own check (D136). Each block is the command as typed, then
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
    Submodule path 'gvsoc/snax-gvsoc': checked out '6e59daaf67eb9005a3393b21896e89d4c956f488'

Everything else runs in snax-forge's image (D118), started from the folder
that holds `snax-forge`:

    podman run --rm -it -v "$PWD":"$PWD" -w "$PWD" ghcr.io/kuleuven-micas/snax-forge:main
    cd snax-forge
    pixi install

    The default environment has been installed.

## 2. The flow

`recipes/vecadd.json` binds vecadd's add to `elementwise_add` at W = 4. The
first flow builds SNAX-GVSoC into `out/snax-gvsoc/` (a few minutes; its
output goes to `out/flow/vecadd/build.log`):

    pixi run flow recipes/vecadd.json --platform platforms/snax_alu.json

    Pixi task (flow in default): python -m snax_forge.flow recipes/vecadd.json --platform platforms/snax_alu.json
    flow vecadd (vecadd: W=4; N=64) on snax_alu -> out/flow/vecadd/
      sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
      design    21 checks passed; platform base snax_alu, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (12 steps)
      gen       snax_alu.hjson, snax_alu_streamer.hjson, snax_alu_acc.hjson, main.c, data.h (seed 0) -> gen/
      build     built: no install in out/snax-gvsoc (snax-gvsoc 6e59daaf)
      sw        sw/forge.elf (sha256 b6b4b1c70d8c8986)
      gvsoc     snax_hjson, exit 0 -> run/
      check     passed: 226 cycles (main to forge_check); data 0 errors (64 words); counters add_acc 19 (16 beats)
      report    none in FLOW3 round 1a; the reports come back in FLOW3 round 1b

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

The folder holds every stage's output:

    ls out/flow/vecadd out/flow/vecadd/gen out/flow/vecadd/run

    out/flow/vecadd:
    build.log
    design
    flow.log
    gen
    run
    sandbox
    sw
    sw.log
    tasks.json

    out/flow/vecadd/gen:
    data.h
    include
    main.c
    manifest.json
    snax_alu.hjson
    snax_alu_acc.hjson
    snax_alu_streamer.hjson

    out/flow/vecadd/run:
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
    	--work-dir $ROOT/snax-forge/out/flow/vecadd/run/work --param chip/soc/binary=$ROOT/snax-forge/out/flow/vecadd/sw/forge.elf --param hjson=$ROOT/snax-forge/out/flow/vecadd/gen/snax_alu.hjson run
    forge cycles: 226
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    make: Leaving directory '$ROOT/snax-forge/gvsoc'

`run.json` says what ran, `profile.json` what it measured (C§7):

    cat out/flow/vecadd/run/run.json out/flow/vecadd/run/profile.json

    {
     "design_point": "design/design_point.json",
     "hjson": ["gen/snax_alu.hjson", "gen/snax_alu_streamer.hjson", "gen/snax_alu_acc.hjson"],
     "elf": "sw/forge.elf",
     "elf_sha256": "b6b4b1c70d8c8986c478ecdc4fce694a42a49bfd3f4ba3eb0440e8bd932070a5",
     "target": "snax_hjson",
     "snax_gvsoc": {"commit": "6e59daaf67eb9005a3393b21896e89d4c956f488", "dirty": false},
     "trace_level": "off"
    }
    {
     "total_cycles": 226,
     "functional_check": {
      "kernel": "vecadd",
      "seed": 0,
      "symbols": {"N": 64},
      "exit_code": 0,
      "data_errors": 0,
      "words": 64,
      "tasks": {"add_acc": {"counter": 19, "beats": 16}},
      "passed": true
     }
    }

The trace is `off`; `--trace task` and `--trace beat` stop with a named
error until FLOW3 round 1b brings the events (D137), and there are no
reports until then either.

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
      build     up to date (snax-gvsoc 6e59daaf)
      sw        sw/forge.elf (sha256 26b545a3812000c8)
      gvsoc     snax_hjson, exit 0 -> run/
      check     passed: 255 cycles (main to forge_check); data 0 errors (1 words); counters sum_acc 67 (64 beats)
      report    none in FLOW3 round 1a; the reports come back in FLOW3 round 1b

`flow.log` holds the same lines, the design checks that ran and each
stage's time (this machine's; they change from run to run):

    tail -12 out/flow/vsum/flow.log

    stage times (wall clock, this machine):
      import    0.331 s
      sandbox   0.034 s
      design    0.051 s
      lower     0.011 s
      gen       0.124 s
      build     0.462 s
      sw        0.511 s
      gvsoc     2.209 s
      check     0.012 s
      total     3.746 s

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
