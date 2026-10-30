# From a recipe to a run on SNAX-GVSoC and snax_cluster's RTL

One design point the whole way. `pixi run flow` takes a recipe through `sandbox` and `design` on `platforms/snax_forge.json`, generates the design point's files and program, builds SNAX-GVSoC when needed, runs the program on `snax_hjson` and takes its own check (D136), then turns SNAX-GVSoC's events into the profile and reports (D137). `pixi run forge` adds the bundle for snax_cluster, and `pixi run forge-set` a set of bundles for the local RTL check (section 8). Each block is the command, then the lines to check; `...` stands for the rest. snax-gvsoc's own `docs/RUN.md` runs its fixtures and its regression.

## 1. A clone and the container

Clone into any folder, `$ROOT` below, and initialise the submodule once:

    git clone https://github.com/KULeuven-MICAS/snax-forge.git
    cd snax-forge && git submodule update --init gvsoc/snax-gvsoc

Everything else runs in snax-forge's image (D118), started from the folder that holds `snax-forge`:

    podman run --rm -it -v "$PWD":"$PWD" -w "$PWD" ghcr.io/kuleuven-micas/snax-forge:main
    cd snax-forge
    pixi install

## 2. The flow

`recipes/vecadd.json` binds vecadd's add to `elementwise_add` at W = 4. The first flow builds SNAX-GVSoC into `out/snax-gvsoc/` (a few minutes, its output in `out/flow/vecadd/build.log`); later flows say `up to date` until the submodule's commit changes or its tree is dirty:

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json

    flow vecadd (vecadd: W=4, N=64; N=64) on snax_forge -> out/flow/vecadd/
      sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
      design    23 checks passed; platform base snax_forge, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (12 steps)
      gen       snax_forge.hjson, snax_forge_streamer.hjson, accelerator.json, main.c, data.h (seed 0) -> gen/
      build     up to date (snax-gvsoc e27acce0)
      sw        sw/forge.elf (sha256 3fa93c774886c3b3)
      gvsoc     snax_hjson, exit 0 -> run/
      check     passed: 214 cycles (main to forge_check); data 0 errors (64 words); counters add_acc 19 (16 beats)
      convert   run/events.log -> profile.json (the summary lines; trace off)
      report    report/design.md, report/run.md

It exits 0, as the program does. In `check`:

- `214 cycles`: the program's window, `mcycle` at main's entry to `mcycle` at `forge_check()`'s entry (D135);
- `data 0 errors (64 words)`: every output word against its golden in `data.h`;
- `add_acc 19 (16 beats)`: the group's COUNTER for the task `add_acc`, beside its beats.

Nothing is compiled for a design point: SNAX-GVSoC's `snax_hjson` reads the cluster file in `gen/` at run time. `out/flow/vecadd/` holds `flow.log` (these lines, the checks that ran, each stage's time), the build and program logs, `tasks.json` and the folders `sandbox/`, `design/`, `gen/`, `sw/`, `run/` and `report/`. The program's own lines are in `run/stdout.txt`:

    forge cycles: 214
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0

`run/run.json` says what ran (the ELF's sha256, where the linker put each L2 container, D86, the submodule's commit) and `run/profile.json` what it measured (C§7):

    head -n 7 out/flow/vecadd/run/profile.json

    {
     "total_cycles": 214,
     "accelerators": {
      "acc": {
       "cycles": {"busy": 16, "stall_in": 3, "stall_out": 0, "flush": 0, "idle": 195},
       "firings": 16,
    ...

## 3. The trace levels

`--trace task` adds the components' `events` lines: every start, done, SNAX CSR access, poll, barrier, state change, function entry and DMA copy in the window, `t = 0` the program's `mcycle` read in main. The folder gets the level in its name:

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --trace task

    ...
      convert   run/events.log -> profile.json, trace.jsonl (49 events in the window; boot 2822)

The run report names each task from the task list (`report/run.md`, "Tasks"):

    | Block | # | Task | Start | Done | Cycles | Direction | Bytes |
    |---|---:|---|---:|---:|---:|---|---:|
    | dma | 0 | load_A | 76 | 97 | 21 | L2 → L1 | 512 |
    ...
    | acc | 0 | add_acc | 121 | 141 | 20 | – | – |
    | dma | 2 | store_C | 156 | 171 | 15 | L1 → L2 | 512 |

Above `task` (C§7, snax-gvsoc `docs/EVENTS.md`): `beat` adds every grant, DMA beat, burst, firing and push, `stall` every refused request and queue count, `finegrained` every core instruction. The sources are the accelerator, its movers and the DMA unless named (`--trace stall:acc,acc_a`); the profile stays the task level's:

    pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --trace stall
    grep -m1 '"k": "stall"' out/flow/vecadd_stall/run/trace.jsonl

    ...
      convert   run/events.log -> profile.json, trace.jsonl (356 events in the window; boot 2822; sources acc, acc_a, acc_b, acc_out, dma)
    {"t": 122, "k": "stall", "src": "acc_b.lane0", "task": "add_acc_b", "addr": 268435968, "bank": 0, "row": 2, "by": "acc_a.lane0", "by_addr": 268435456, "b": 0, "region": "B", "elems": [0]}

`--vcd` adds gtkwave's `run/all.vcd` and `run/view.gtkw` of the same sources, which snax-forge never reads. A passing run gzips `run/events.log` and deletes gvrun's `run/work/` (unless `--keep-work` or `--vcd`); a failing run keeps both. `--set N=512` makes the accelerator's work stand out against the cores' and the DMA's (`check passed: 489 cycles ...; counters add_acc 131 (128 beats)`); `examples/loop1/` walks such a run.

## 4. The viewer

    pixi run view out/flow/vecadd_stall

    serving vecadd_stall at http://127.0.0.1:8765/ (Ctrl-C to stop)

From `task` on, the Schedule tab draws the window as lanes (each core, the DMA, each mover, the accelerator), quiet stretches of 16 cycles or more folded; a click on a cycle shows the cluster view under it, and the arrows step by event (shift: by task boundary). From `beat` on, the memory tab shows where each container lies in L1 and L2 and how its words move. The address bar holds the whole view, so a link shows the same.

On the MICAS server, start the image with `--network host` so its port 8765 is the server's, and forward it from your machine, then open `http://localhost:8765/`:

    ssh -N -L 8765:127.0.0.1:8765 <user>@<server>

## 5. A second kernel: vsum

`recipes/vsum.json` binds vsum's reduction to `TreeAccumulate` at one lane:

    pixi run flow recipes/vsum.json --platform platforms/snax_forge.json

    ...
      check     passed: 254 cycles (main to forge_check); data 0 errors (1 words); counters sum_acc 67 (64 beats)

Its `run/stderr.txt` holds gvrun's warning that the installed platform tree does not match: the tree was built from vecadd's cluster file, so for another accelerator gvrun builds the system from its JSON configuration, on the same components. `examples/dot/` runs dot 1 to dot 6 the same way.

## 6. What the flow refuses

A platform without a source hjson stops at `gen`, before anything is built, and exits 1:

    pixi run flow recipes/vecadd.json --platform platforms/small16.json

    FAILED recipes/vecadd.json: gen: [gen.source] platform small16 has no source hjson: read it from one with `pixi run design from-hjson`

Two accelerators for the platform's one slot stop earlier, at the design checks (`gen.instances`, C§14), in the flow as in the design step alone:

    pixi run sandbox recipes/dot_serial.json
    pixi run design out/sandbox/dot_serial/4_bind.snaxdfg --platform platforms/snax_forge.json

    design check of out/sandbox/dot_serial/4_bind.snaxdfg on platforms/snax_forge.json: 1 problem
     [gen.instances] graph: 2 accelerators ['mul', 'sum']; the platform has one accelerator slot (snax_forge) and the program drives one accelerator
         fix: encapsulate them into one: a recipe step encapsulate with units ['mul', 'sum'] (D148)

## 7. By hand: the stages, the RTL and the bundle

The flow's stages are tools of their own. The design point, the generator into `out/gvsoc/vecadd/`, then `gvsoc/Makefile` for the program (into `out/snax-gvsoc/sw/vecadd/`) and the run, which prints section 2's four `forge` lines:

    pixi run sandbox recipes/vecadd.json
    pixi run design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_forge.json
    pixi run gen-gvsoc out/design/vecadd/design_point.json
    make -C gvsoc sw APPS=forge FORGE=../out/gvsoc/vecadd DESIGN=vecadd
    make -C gvsoc run TARGET=snax_hjson DESIGN=vecadd \
        GVRUN_ARGS="--param hjson=$PWD/out/gvsoc/vecadd/snax_forge.hjson" \
        ELF=../out/snax-gvsoc/sw/vecadd/forge.elf

`GVRUN_ARGS` takes gvrun's own options too, such as `--trace=snax_acc0/unit`. The flow does not make RTL; `gen-rtl` writes `accelerator.json` (C§17, gen-gvsoc's bytes), checks it against the datapath catalogue and runs `hw/chisel/`'s `Emit --spec` on a classpath sbt exports into `hw/chisel/target/emit-classpath.txt` (the first run, and again after a change under `hw/chisel/src/main/` or to `build.sbt`):

    pixi run gen-rtl out/design/vecadd/design_point.json

    ...
    wrote out/rtl/vecadd/accelerator.json: snax_forge_shell_wrapper, ElementwiseLanes
    wrote out/rtl/vecadd/sv/ElementwiseAlu_w64_add.sv
    ...
    wrote out/rtl/vecadd/sv/snax_forge_shell_wrapper.sv
    wrote out/rtl/vecadd/emitted.json

`snax_forge_shell_wrapper` is what snax_cluster's `snax_acc_wrapper.sv.tpl` connects with `snax_disable_csr_manager` (C§17); inside it `snax_forge_csr`, `snax_forge_ctrl` (the group's `GroupCtrl`, `snax_forge_acc_ctrl`) and `snax_forge_datapath`. The bundle (C§18) is gen-gvsoc's files, the RTL under `rtl/` with its file list, and `bundle.json`:

    pixi run bundle out/design/vecadd/design_point.json

    ...
    wrote out/bundle/vecadd/bundle.json
    no forge_expected.txt: a bundle alone has no SNAX-GVSoC run (pixi run forge writes it)

## 8. The local RTL check

One ELF, snax-forge's, runs on SNAX-GVSoC and on snax_cluster's RTL, and the RTL run must print the SNAX-GVSoC run's `forge counter` and `forge data` lines (D123). snax-forge never runs RTL; snax_cluster's `snax-forge-check` does (C§18).

`pixi run forge` runs section 2's flow, the bundle of its design point with `forge_expected.txt` (the run's counter and data lines), and the bundle's ELF, checked to be the flow's; each step's output is in `out/forge/<name>/forge.log`:

    pixi run forge recipes/vecadd.json

    forge vecadd: recipes/vecadd.json on platforms/snax_forge.json; every step's output in out/forge/vecadd/forge.log
      flow      passed: design point out/flow/vecadd/design/design_point.json, run out/flow/vecadd/run/
      bundle    out/bundle/vecadd/: snax_forge_shell_wrapper, snax-forge $COMMIT
      expected  out/bundle/vecadd/forge_expected.txt: the run's counter and data lines
      sw        out/forge/vecadd/sw/forge.elf from the bundle (sha256 3fa93c774886c3b3, the flow's)
    ...

then the snax_cluster commands for one design (Verilator, and Questa as below). `pixi run forge-set quick|full [--sim vsim|vlt] [-j N]` runs forge on a set's recipes at once (quick: vecadd_i32, vsum_tree_i8, dot_1, dot_4; full: those and vecadd, vsum, vecmul_i32_i64, dot_2, dot_3, dot_5, dot_6, matvec_5, matvec_6), writes `out/forge-set/<set>.md` (each bundle's expected lines and the commands per design) and prints one line:

    pixi run forge-set quick

    forge-set quick: 4 designs, 32 at a time; each forge's output in out/forge-set/quick/<design>.txt
      vecadd_i32       passed
      vsum_tree_i8     passed
      dot_1            passed
      dot_4            passed
    wrote out/forge-set/quick.md; from snax_cluster's target/snitch_cluster:
      make snax-forge-check BUNDLES="$ROOT/snax-forge/out/bundle/vecadd_i32 ... $ROOT/snax-forge/out/bundle/dot_4" SIM=vsim SNAX_CLUSTER_IMAGE=<image>

Run that line from snax_cluster's `target/snitch_cluster`, outside the image. `<image>` is snax_cluster's development image for podman; on the MICAS servers it is `c5db84ad7bce`. With `SIM=vsim`, `rtl-gen` and `vsim_preparation` run in the image and the build and the run outside it, with Questa and bender on `PATH` (snax_cluster's pixi environment); with `SIM=vlt` every step runs in the image. It prints per bundle `PASS <name>` or `FAIL <name>: <want> / <got>` and fails if any failed; `logs/snax-forge-check.txt` keeps those lines with each run's `forge cycles`, `logs/snax-forge-<name>-<sim>.txt` each run's output:

    PASS vecadd_i32
      vecadd_i32 forge cycles: 210
    ...
    PASS dot_4
      dot_4 forge cycles: 443

Each vsim design takes about 2 minutes, its run 6–13 s against SNAX-GVSoC's 1.5–2.2 s. CLAUDE.md's rule says when each set is due: the quick set on vsim before a PR that changes the emitted RTL or `main.c` merges, the full set at a phase's close, its numbers into STATUS's open item 95.

One design by hand, Questa: in the image, from the folder that holds both checkouts, then the build and the run outside it (`B` and `E` as forge printed them):

    podman run --rm -it -v "$PWD":"$PWD" -w "$PWD/snax_cluster" c5db84ad7bce
    make -C target/snitch_cluster clean-generated clean-vsim
    make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B
    make -C target/snitch_cluster vsim_preparation CFG_OVERRIDE=$B

    make -C target/snitch_cluster snax-forge-sim SIM=vsim CFG_OVERRIDE=$B
    make -C target/snitch_cluster snax-forge-run SIM=vsim CFG_OVERRIDE=$B ELF=$E

    ...
    forge cycles: 210
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    # ** Info: [SUCCESS] Program finished successfully

Questa exits 0 whatever the program returns, so `snax-forge-run` reads the exit code from that last line; the output is also in `logs/snax-forge-vsim.txt`. `N` reaches the hardware only as CSR values, so a design's `N=512` bundle's `rtl/` is the default's, byte for byte, and one RTL build runs both ELFs.

## 9. A recipe in the editor

The editor builds a recipe one transform at a time on the drawn graph (D55, C§12). A session on vecadd's import:

    pixi run edit-dfg --kernel vecadd --set N=64 --set W=4

    serving vecadd at http://127.0.0.1:8767/ (Ctrl-C to stop)

Open it as section 4 does the viewer, on port 8767. Click `add_map`: the sidebar lists the transforms whose site accepts it, each with its form. Apply `split` with factor `W`, `tag_map` `temporal` on `add_map` and `spatial` on `add_map_s`, then select `add` and `bind` it to `elementwise_add / chisel_tiled_spatial`. Each Apply is a step checked against the reference, with a log line (the bind's ends `C 0 → 16 · not mapped 16 → 0`). The Estimate (D151, C§19) shows 64 iterations not mapped as imported, 16 once `add_map_s` is spatial, and after the bind the group `add` at `16 = 16 + 0` cycles with COUNTER 19, the counter section 2's run measures (`add_acc 19`). Undo, redo and the step list move the cursor; Apply below the last step asks to drop the steps after it.

Save recipe, here as `vecadd_mine` (a name no recipe has), runs the design checks, writes `recipes/vecadd_mine.json` and its steps to `out/sandbox/vecadd_mine/`, and prints what comes next:

    pixi run sandbox recipes/vecadd_mine.json
    pixi run flow recipes/vecadd_mine.json --platform platforms/snax_forge.json
    pixi run forge recipes/vecadd_mine.json

The session stays in `out/sessions/vecadd/`: `--session out/sessions/vecadd` resumes it, `--recipe recipes/vecadd.json` starts one from a recipe, and Export step writes one step's graph to its `export/`. `pixi run view-dfg out/sandbox/vecadd_mine/ --platform platforms/snax_forge.json` shows the saved steps read-only.
