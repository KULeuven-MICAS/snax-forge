# From a recipe to a run on SNAX-GVSoC and snax_cluster's RTL

One design point the whole way. `pixi run flow` takes a recipe through
`sandbox` and `design` on `platforms/snax_forge.json`, generates the design
point's hjson files, headers, data and program, builds SNAX-GVSoC when needed
and the program, runs it on `snax_hjson` and takes the program's own check
(D136), then turns SNAX-GVSoC's events into the run's profile and reports
(D137). `pixi run forge` adds the bundle and the commands that run the same
ELF on snax_cluster's RTL (section 8). Each block is the command as typed,
then the lines of its output a reader checks, from a fresh clone with the
submodule initialised; `...` stands for the rest. snax-gvsoc's own
`docs/RUN.md` runs its fixtures and its regression.

## 1. A clone and the container

snax-gvsoc is a submodule, so initialise it once. Clone into any folder,
`$ROOT` below (the outputs' absolute paths start there):

    git clone https://github.com/KULeuven-MICAS/snax-forge.git
    cd snax-forge && git submodule update --init gvsoc/snax-gvsoc

Everything else runs in snax-forge's image (D118), started from the folder
that holds `snax-forge`:

    podman run --rm -it -v "$PWD":"$PWD" -w "$PWD" ghcr.io/kuleuven-micas/snax-forge:main
    cd snax-forge
    pixi install

## 2. The flow

`recipes/vecadd.json` binds vecadd's add to `elementwise_add` at W = 4. The
first flow builds SNAX-GVSoC into `out/snax-gvsoc/` (a few minutes; its
output shows as it comes and goes to `out/flow/vecadd/build.log`):

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json

    ...
    flow vecadd (vecadd: W=4, N=64; N=64) on snax_forge -> out/flow/vecadd/
      sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
      design    20 checks passed; platform base snax_forge, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (12 steps)
      gen       snax_forge.hjson, snax_forge_streamer.hjson, accelerator.json, main.c, data.h (seed 0) -> gen/
      build     built: no install in out/snax-gvsoc (snax-gvsoc 95266990)
      sw        sw/forge.elf (sha256 b6b4b1c70d8c8986)
      gvsoc     snax_hjson, exit 0 -> run/
      check     passed: 226 cycles (main to forge_check); data 0 errors (64 words); counters add_acc 19 (16 beats)
      convert   run/events.log -> profile.json (the summary lines; trace off)
      report    report/design.md, report/run.md

It exits 0. One line per stage:

- `sandbox`, `design`: the recipe's steps, each checked by the reference
  executor, then the design checks, the memory plan and the design point.
- `lower`: the task list, `tasks.json`.
- `gen`: the generator's output in `gen/`: the cluster file, the streamer
  and datapath files it refers to, the headers, `data.h` (the kernel's
  `make_inputs` at seed 0 and the reference executor's goldens) and
  `main.c`. Nothing is compiled for a design point: SNAX-GVSoC's installed
  `snax_hjson` reads the cluster file at run time.
- `build`: GVSoC through `gvsoc/Makefile`; later flows say `up to date`
  until the submodule's commit or path changes or its tree is dirty.
- `sw`: the program, built by snax-gvsoc's generic app into `sw/`.
- `gvsoc`: the run, its output in `run/stdout.txt` and `run/stderr.txt`.
- `check`: what the program printed and returned. `226 cycles` is its
  window, from `mcycle` at main's entry to `mcycle` at `forge_check()`'s
  entry; `data 0 errors (64 words)` every output word against its golden;
  `add_acc 19 (16 beats)` the shell's cycle counter for the task `add_acc`
  beside its beats. `passed` is the program's exit code 0, and the flow's
  exit follows it.
- `convert`: the counters in `run/events.log` (snax-gvsoc `docs/EVENTS.md`)
  named into `run/profile.json` (C§7); at trace level `off`, the `summary`
  lines alone.
- `report`: `report/design.md` (what was built) and `report/run.md` (where
  the cycles went).

`out/flow/vecadd/` holds `flow.log` (the same lines, the checks that ran and
each stage's time), `build.log`, `sw.log`, `tasks.json` and the folders
`sandbox/`, `design/`, `gen/`, `sw/`, `run/` and `report/`. What the program
printed:

    cat out/flow/vecadd/run/stdout.txt

    ...
    forge cycles: 226
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    ...

`run/run.json` says what ran, with where the linker put each L2 container's
array (D86); `run/profile.json` what it measured (C§7):

    cat out/flow/vecadd/run/run.json; head -n 7 out/flow/vecadd/run/profile.json

    {
     "design_point": "design/design_point.json",
     "hjson": ["gen/snax_forge.hjson", "gen/snax_forge_streamer.hjson", "gen/accelerator.json"],
     "elf": "sw/forge.elf",
     "elf_sha256": "b6b4b1c70d8c8986c478ecdc4fce694a42a49bfd3f4ba3eb0440e8bd932070a5",
     "elf_symbols": {
      "A": {"addr": 2147494144, "size": 512},
      "B": {"addr": 2147494656, "size": 512},
      "C": {"addr": 2147496704, "size": 512}
     },
     "target": "snax_hjson",
     "snax_gvsoc": {"commit": "952669901628da3eb8e145eb5cc8070e1c6bffbc", "dirty": false},
     ...
    }
    {
     "total_cycles": 226,
     "accelerators": {
      "acc": {
       "cycles": {"busy": 16, "stall_in": 3, "stall_out": 0, "flush": 0, "idle": 207},
       "firings": 16,
    ...

`--trace task` also turns on the components' `events` lines; the folder
gets the level in its name:

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --trace task

    ...
      check     passed: 226 cycles (main to forge_check); data 0 errors (64 words); counters add_acc 19 (16 beats)
      convert   run/events.log -> profile.json, trace.jsonl (49 events in the window; boot 2812)
      report    report/design.md, report/run.md

The trace is every start, done, SNAX CSR access, poll, barrier, state
change, function entry and DMA copy in the program's window, `t = 0` being
the program's `mcycle` read in main; `run/trace_meta.json` holds `boot`,
that read in SNAX-GVSoC's cycles. The run report names each task from the
task list:

    sed -n 13,22p out/flow/vecadd_task/report/run.md

    | Block | # | Task | Start | Done | Cycles | Direction | Bytes |
    |---|---:|---|---:|---:|---:|---|---:|
    | dma | 0 | load_A | 75 | 97 | 22 | L2 → L1 | 512 |
    | dma | 1 | load_B | 93 | 114 | 21 | L2 → L1 | 512 |
    | acc_a | 0 | add_acc_a | 119 | 137 | 18 | – | – |
    | acc_b | 0 | add_acc_b | 119 | 138 | 19 | – | – |
    | acc_out | 0 | add_acc_out | 119 | 141 | 22 | – | – |
    | acc | 0 | add_acc | 120 | 140 | 20 | – | – |
    | dma | 2 | store_C | 162 | 177 | 15 | L1 → L2 | 512 |

and the viewer shows the profile, the schedule with the cluster view and the
memory tab in the browser (see "The viewer" below for the MICAS server):

    pixi run view out/flow/vecadd_task

    serving vecadd_task at http://127.0.0.1:8765/ (Ctrl-C to stop)

### The beat levels

Above `task` the levels add a line per beat (C§7, snax-gvsoc
`docs/EVENTS.md`): `beat` every grant, DMA beat and burst, firing and push,
`stall` every refused request and queue count, `finegrained` every
instruction of the cores. By default the sources are the accelerator, its
movers and the DMA; `--trace stall:acc,acc_a` names them. The profile is the
task level's, byte for byte:

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --trace stall

    ...
      check     passed: 226 cycles (main to forge_check); data 0 errors (64 words); counters add_acc 19 (16 beats)
      convert   run/events.log -> profile.json, trace.jsonl (356 events in the window; boot 2812; sources acc, acc_a, acc_b, acc_out, dma)

Each event carries its raw words and what they mean (the region, the
elements and their values); a stall names the port that won the bank:

    grep -m1 '"k": "stall"' out/flow/vecadd_stall/run/trace.jsonl

    {"t": 121, "k": "stall", "src": "acc_b.lane0", "task": "add_acc_b", "addr": 268435968, "bank": 0, "row": 2, "by": "acc_a.lane0", "by_addr": 268435456, "b": 0, "region": "B", "elems": [0]}

`--vcd` adds gtkwave's `run/all.vcd` and `run/view.gtkw` of the same
sources (open the view outside the container); snax-forge never reads them:

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --trace stall --vcd --name vecadd_stall_vcd

    ...
      convert   run/events.log -> profile.json, trace.jsonl (356 events in the window; boot 2812; sources acc, acc_a, acc_b, acc_out, dma); run/all.vcd, run/view.gtkw (gtkwave)

The recipe's `N` makes the vectors longer, so the accelerator's work stands
out against the cores' and the DMA's; `examples/loop1/` walks such a run:

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --set N=512 --trace stall

    ...
      check     passed: 486 cycles (main to forge_check); data 0 errors (512 words); counters add_acc 131 (128 beats)
      convert   run/events.log -> profile.json, trace.jsonl (2150 events in the window; boot 7238; sources acc, acc_a, acc_b, acc_out, dma)

### The viewer

From `task` on, the Schedule tab draws the window as lanes: one per core,
the DMA, each mover and the accelerator. A quiet stretch of 16 cycles or
more is folded (click to open it); a click on a cycle shows the cluster view
under the schedule (the banks, the interconnect, each mover's lanes and FIFO
slots, the DMA, the accelerator and the cycle's lines), and the arrows step
by event (shift: by task boundary). From `beat` on, the memory tab shows
where each container lies in L1 and L2 and how its words move: a click on a
word gives its element's journey. Everything is in the address bar, so a
link shows the same.

On the MICAS server, start the image on the host's network, so its port
8765 is the server's (from the folder that holds `snax-forge`):

    podman run --rm -it --network host -v "$PWD":"$PWD" -w "$PWD/snax-forge" ghcr.io/kuleuven-micas/snax-forge:main
    pixi run view out/flow/vecadd_stall out/flow/vecadd_N512_stall

    serving vecadd_stall, vecadd_N512_stall at http://127.0.0.1:8765/ (Ctrl-C to stop)

It listens on 127.0.0.1 only, so on your own machine forward the port, then
open `http://localhost:8765/`:

    ssh -N -L 8765:127.0.0.1:8765 <user>@<server>

## 3. A second kernel: vsum

`recipes/vsum.json` binds vsum's reduction to the Chisel datapath
`TreeAccumulate` at one lane. GVSoC is built now, so the whole flow takes seconds:

    pixi run flow recipes/vsum.json --platform platforms/snax_forge.json

    ...
      lower     tasks.json (9 steps)
      ...
      check     passed: 255 cycles (main to forge_check); data 0 errors (1 words); counters sum_acc 67 (64 beats)

Its `run/stderr.txt` holds gvrun's warning that the installed platform tree
does not match: `snax_hjson`'s tree was built from the vecadd fixture's
cluster file, so for another accelerator gvrun builds the system from its
JSON configuration instead, on the same installed components. `flow.log`
ends with each stage's time on this machine.

## 4. What the flow refuses

The generator writes a cluster file from the platform's source hjson and
puts one accelerator in its one slot, so two kinds of design point stop at
`gen` with a named rule, before anything is built, and exit 1:

    pixi run flow recipes/vecadd.json --platform platforms/small16.json

    FAILED recipes/vecadd.json: gen: [gen.source] platform small16 has no source hjson: read it from one with `pixi run design from-hjson`

    pixi run flow recipes/dot_serial.json --platform platforms/snax_forge.json

    FAILED recipes/dot_serial.json: gen: [gen.instances] 2 accelerators ['mul', 'sum']; the platform has one accelerator slot and the program drives one accelerator: encapsulate puts groups into one (D148)

`recipes/dot_1.json` is dot with `encapsulate` as its last step: one
accelerator `dot` of the groups `mul` and `sum`, a task each (D148), which
runs and checks itself (`forge counter mult_mul`, `forge counter sum_sum`).

## 5. The same path by hand

The flow's stages are tools of their own: the design point, the generator
into `out/gvsoc/vecadd/`, then `gvsoc/Makefile` (from pixi or the plain
container shell) for the program and the run:

    pixi run sandbox recipes/vecadd.json
    pixi run design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_forge.json
    pixi run gen-gvsoc out/design/vecadd/design_point.json

    ...
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/manifest.json
    run: make -C gvsoc run TARGET=snax_hjson GVRUN_ARGS="--param hjson=$ROOT/snax-forge/out/gvsoc/vecadd/snax_forge.hjson" ELF=...

    make -C gvsoc sw APPS=forge FORGE=../out/gvsoc/vecadd DESIGN=vecadd
    make -C gvsoc run TARGET=snax_hjson DESIGN=vecadd \
        GVRUN_ARGS="--param hjson=$PWD/out/gvsoc/vecadd/snax_forge.hjson" \
        ELF=../out/snax-gvsoc/sw/vecadd/forge.elf

    ...
    forge cycles: 226
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    ...

`make sw` builds into `out/snax-gvsoc/sw/vecadd/`; `GVRUN_ARGS` takes
gvrun's own options too, such as `--trace=snax_acc0/unit` for the unit's
trace.

## 6. The accelerator's RTL

The same design point gives the accelerator's Chisel RTL (D130), which the
bundle carries (section 7); the flow does not call it. `gen-rtl` writes
`accelerator.json` (C§17, the same bytes as gen-gvsoc's), checks it against
the datapath catalogue, then elaborates it through `hw/chisel/`'s `Emit
--spec` (sbt, so the first run fetches the Chisel jars):

    pixi run gen-rtl out/design/vecadd/design_point.json

    ...
    wrote out/rtl/vecadd/accelerator.json: snax_forge_shell_wrapper, ElementwiseLanes
    wrote out/rtl/vecadd/sv/ElementwiseAlu_w64_add.sv
    wrote out/rtl/vecadd/sv/ElementwiseLanes_i64xi64_i64_i64_n4_add.sv
    wrote out/rtl/vecadd/sv/snax_forge_acc_ctrl.sv
    wrote out/rtl/vecadd/sv/snax_forge_csr.sv
    wrote out/rtl/vecadd/sv/snax_forge_ctrl.sv
    wrote out/rtl/vecadd/sv/snax_forge_datapath.sv
    wrote out/rtl/vecadd/sv/snax_forge_shell_wrapper.sv
    wrote out/rtl/vecadd/emitted.json

`snax_forge_shell_wrapper` has the ports snax_cluster's
`snax_acc_wrapper.sv.tpl` connects with `snax_disable_csr_manager`, plus the
group `acc`'s `acc_busy_o` and `acc_done_o`, and carries its own CSRs
(`snax_forge_csr`, D130); `snax_forge_acc_ctrl` is the group's control
unit (`GroupCtrl`). `hw/chisel/README.md` has the specs that check it.

## 7. Bundle for snax_cluster

The bundle (C§18) is what snax_cluster builds the design's cluster from:
gen-gvsoc's files at the root, the accelerator's RTL under `rtl/` with its
file list, and `bundle.json`, all made again (sbt, as section 6):

    pixi run bundle out/design/vecadd/design_point.json

    ...
    bundle out/bundle/vecadd: snax_forge_shell_wrapper from out/design/vecadd/design_point.json, snax-forge $COMMIT
    wrote out/bundle/vecadd/snax_forge.hjson
    ...
    wrote out/bundle/vecadd/rtl/files.f
    wrote out/bundle/vecadd/bundle.json

`$COMMIT` is the checkout's `git rev-parse HEAD`, followed by `, dirty` when
the work tree has uncommitted changes. `rtl/files.f` lists the five files of
`rtl/sv/` in `emitted.json`'s order, and the cluster file's slot names it
(`snax_forge_rtl: "rtl/files.f"`). snax_cluster builds the cluster from that
file alone (C§18). snax_cluster keeps the vecadd and vsum bundles as
fixtures, `target/snitch_cluster/cfg/snax_forge/<design>/`; to refresh one,
make the bundle again and copy what the fixture's `README.md` lists.

## 8. Running a bundle on snax_cluster's RTL

One ELF, snax-forge's, built from the bundle, runs on SNAX-GVSoC and on
snax_cluster's Verilator and Questa builds, and the printed lines are
compared by hand (D123). snax-forge never runs RTL; snax_cluster's
`target/snitch_cluster/snax_forge.mk` does (C§18).

`pixi run forge` runs section 2's flow, section 7's bundle of its design
point and the bundle's ELF, checks that ELF's sha256 against the flow's
`elf_sha256`, and prints the Verilator commands with `B` and `E` filled in.
Each step's whole output is in `out/forge/<name>/forge.log`; a failing step
prints the last 30 lines of it. `--platform` (default
`platforms/snax_forge.json`), `--set` and `--trace` go to the flow.

    pixi run forge recipes/vecadd.json

    forge vecadd: recipes/vecadd.json on platforms/snax_forge.json; every step's output in out/forge/vecadd/forge.log
      flow      passed: design point out/flow/vecadd/design/design_point.json, run out/flow/vecadd/run/
      bundle    out/bundle/vecadd/: snax_forge_shell_wrapper, snax-forge $COMMIT
      sw        out/forge/vecadd/sw/forge.elf from the bundle (sha256 b6b4b1c70d8c8986, the flow's)
    SNAX-GVSoC (out/flow/vecadd/run/stdout.txt):
      forge cycles: 226
      forge data: 0 errors (64 words)
      forge counter add_acc: 19
      forge errors: 0
    snax_cluster on Verilator, from snax_cluster in its pixi environment:
      B=$ROOT/snax-forge/out/bundle/vecadd/snax_forge.hjson
      E=$ROOT/snax-forge/out/forge/vecadd/sw/forge.elf
      pixi run make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B
      pixi run make -C target/snitch_cluster snax-forge-sim SIM=vlt CFG_OVERRIDE=$B
      pixi run make -C target/snitch_cluster snax-forge-run SIM=vlt CFG_OVERRIDE=$B ELF=$E
    Questa: rtl-gen and vsim_preparation in snax_cluster's development image, then snax-forge-sim and snax-forge-run with SIM=vsim and the same B and E: docs/RUN.md section 8.

(`$ROOT` is the folder that holds both checkouts.) The same for vsum at
`N=512`, into `out/forge/vsum_N512/`:

    pixi run forge recipes/vsum.json --set N=512

    ...
      sw        out/forge/vsum_N512/sw/forge.elf from the bundle (sha256 b173d5254b69346a, the flow's)
    SNAX-GVSoC (out/flow/vsum_N512/run/stdout.txt):
      forge cycles: 755
      forge data: 0 errors (1 words)
      forge counter sum_acc: 515
      forge errors: 0
    ...

Verilator runs in snax_cluster's pixi environment, as its CI does: from
snax_cluster, the three commands forge printed.

    pixi run make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B
    pixi run make -C target/snitch_cluster snax-forge-sim SIM=vlt CFG_OVERRIDE=$B
    pixi run make -C target/snitch_cluster snax-forge-run SIM=vlt CFG_OVERRIDE=$B ELF=$E

    ...
    forge cycles: 239
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    ...

The output is also in `target/snitch_cluster/logs/snax-forge-vlt.txt`;
`snax-forge-run` fails on the program's nonzero exit code. Without `ELF` it
builds and runs snax_cluster's own copy of the program from the bundle's
`main.c` (`sw/apps/snax-forge`), which its CI uses for pass/fail only, after
checking the bundle's `include/streamer_csr_addr_map.h` against the one
snaxgen wrote.

The ELF by hand, in snax-forge's image, is the flow's (section 5's generic
app on the bundle):

    make -C gvsoc sw APPS=forge FORGE=../out/bundle/vecadd DESIGN=vecadd
    sha256sum out/snax-gvsoc/sw/vecadd/forge.elf

    ...
    b6b4b1c70d8c8986c478ecdc4fce694a42a49bfd3f4ba3eb0440e8bd932070a5  out/snax-gvsoc/sw/vecadd/forge.elf

Questa: `rtl-gen` and `vsim_preparation` in snax_cluster's development image
(on barnard3 `c5db84ad7bce`), started from the folder that holds both
checkouts, then the build and the run outside it, with `bender` and Questa
on `PATH`:

    podman run --rm -it -v "$PWD":"$PWD" -w "$PWD" c5db84ad7bce
    cd snax_cluster
    B=$PWD/../snax-forge/out/bundle/vecadd/snax_forge.hjson
    make -C target/snitch_cluster clean-vsim CFG_OVERRIDE=$B
    make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B
    make -C target/snitch_cluster vsim_preparation CFG_OVERRIDE=$B

    B=$PWD/../snax-forge/out/bundle/vecadd/snax_forge.hjson
    E=$PWD/../snax-forge/out/forge/vecadd/sw/forge.elf
    make -C target/snitch_cluster snax-forge-sim SIM=vsim CFG_OVERRIDE=$B
    make -C target/snitch_cluster snax-forge-run SIM=vsim CFG_OVERRIDE=$B ELF=$E

    ...
    Errors: 0, Warnings: 18
    ...
    forge cycles: 239
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    # ** Info: [SUCCESS] Program finished successfully
    ...

Questa exits 0 whatever the program returns, so `snax-forge-run` reads the
exit code from that last line. Against SNAX-GVSoC, the counter is equal and
the window within 10% of the RTL's (|w_gvsoc − w_rtl| / w_rtl):

| Design | SNAX-GVSoC window | RTL window | Difference | Counter (both) |
|---|---:|---:|---:|---:|
| vecadd, default N (64) | 226 | 239 | 5.4% | 19 |
| vecadd, `N=512` | 486 | 500 | 2.8% | 131 |
| vsum, default N (64) | 255 | 259 | 1.5% | 67 |
| vsum, `N=512` | 755 | 739 | 2.2% | 515 |

`N` reaches the hardware only as CSR values (the streamer's bounds and
addresses, the accelerator's `loopCount` and vsum's `len`), so a design's
`N=512` bundle's `rtl/` is the default's, byte for byte, and one RTL build
runs both ELFs (`CFG_OVERRIDE` the default bundle's, `ELF` the `N=512`
one's). snax_cluster's fixture READMEs have the same commands with
`CFG_OVERRIDE=cfg/snax_forge/<design>/snax_forge.hjson`. The shell's rule
the counter comes from is checked on both sides against one set of cases,
snax-gvsoc's `tests/snax/shell/` (C§17).
