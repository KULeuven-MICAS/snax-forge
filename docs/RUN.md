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

`recipes/elementwise/vecadd.json` binds vecadd's add to `simd` at W = 4. The first flow builds SNAX-GVSoC into `out/snax-gvsoc/` (a few minutes, its output in `out/flow/vecadd/build.log`); later flows say `up to date` until the submodule's commit changes or its tree is dirty:

    pixi run flow recipes/elementwise/vecadd.json --platform platforms/snax_forge.json

    flow vecadd (vecadd: W=4, N=64; N=64) on snax_forge -> out/flow/vecadd/
      sandbox   2 steps, each equal to the input graph on the reference check -> sandbox/
      design    24 checks passed; platform base snax_forge, 0 changes; memory contiguous, 0 changes -> design/
      lower     tasks.json (12 steps)
      gen       snax_forge.hjson, snax_forge_vecadd_streamer_acc.hjson, accelerator.json, main.c, data.h (seed 0) -> gen/
      build     up to date (snax-gvsoc 1e988679)
      sw        sw/forge.elf (sha256 3fa93c774886c3b3)
      gvsoc     snax_hjson, exit 0 -> run/
      check     passed: 214 cycles (main to forge_check); data 0 errors (64 words); counters add_acc 19 (16 beats)
      convert   run/events.log -> profile.json (the summary lines; trace off)
      report    report/design.md, report/run.md

It exits 0, as the program does. In `check`:

- `214 cycles`: the program's window, `mcycle` at main's entry to `mcycle` at `forge_check()`'s entry (D135);
- `data 0 errors (64 words)`: every output word against its golden in `data.h`;
- `add_acc 19 (16 beats)`: the group's COUNTER for the task `add_acc`, beside its beats.

Nothing is compiled for a design point: SNAX-GVSoC's `snax_hjson` reads the cluster file in `gen/` at run time. The folder is named after the recipe, each `--set` appended (`--set W=8` gives `out/flow/vecadd_W8/`; a param whose name ends in a digit gets a `-` before its value, `--set W_acc_1=10` giving `…_W_acc_1-10`, C§7). `out/flow/vecadd/` holds `flow.log` (these lines, the checks that ran, each stage's time), the build and program logs, `tasks.json` and the folders `sandbox/`, `design/`, `gen/`, `sw/`, `run/` and `report/`. The program's own lines are in `run/stdout.txt`:

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

    pixi run flow recipes/elementwise/vecadd.json --platform platforms/snax_forge.json --trace task

    ...
      convert   run/events.log -> profile.json, trace.jsonl (49 events in the window; boot 2822)

The run report names each task from the task list (`report/run.md`, "Tasks"):

    | Block | # | Task | Accelerator | Start | Done | Cycles | Direction | Bytes |
    |---|---:|---|---|---:|---:|---:|---|---:|
    | dma | 0 | load_A | – | 76 | 97 | 21 | L2 → L1 | 512 |
    ...
    | acc | 0 | add_acc | acc | 121 | 141 | 20 | – | – |
    | dma | 2 | store_C | – | 156 | 171 | 15 | L1 → L2 | 512 |

and its Accelerators section opens with a row per accelerator, its streamer set (D157): its CSRs, its streamer's busy window (STREAMER_START to busy low) and its group's COUNTER:

    | Accelerator | Groups | CSRs | Streamer busy | COUNTER |
    |---|---|---|---|---|
    | acc | acc | 960–981 | 120–142 | acc 19 |

Above `task` (C§7, snax-gvsoc `docs/EVENTS.md`): `beat` adds every grant, DMA beat, burst, firing and push, `stall` every refused request and queue count, `finegrained` every core instruction. The sources are the accelerator, its movers and the DMA unless named (`--trace stall:acc,acc_a`); the profile stays the task level's:

    pixi run flow recipes/elementwise/vecadd.json --platform platforms/snax_forge.json --trace stall
    grep -m1 '"k": "stall"' out/flow/vecadd_stall/run/trace.jsonl

    ...
      convert   run/events.log -> profile.json, trace.jsonl (356 events in the window; boot 2822; sources acc, acc_a, acc_b, acc_out, dma)
    {"t": 122, "k": "stall", "src": "acc_b.lane0", "task": "add_acc_b", "addr": 268435968, "bank": 0, "row": 2, "by": "acc_a.lane0", "by_addr": 268435456, "b": 0, "region": "B", "elems": [0]}

`--vcd` adds gtkwave's `run/all.vcd` and `run/view.gtkw` of the same sources, which snax-forge never reads. A passing run gzips `run/events.log` and deletes gvrun's `run/work/` (unless `--keep-work` or `--vcd`); a failing run keeps both. `--set N=512` makes the accelerator's work stand out against the cores' and the DMA's (`check passed: 489 cycles ...; counters add_acc 131 (128 beats)`); `examples/loop1/` walks such a run.

## 4. The viewer

    pixi run view out/flow/vecadd_stall

    serving vecadd_stall at http://127.0.0.1:8765/ (Ctrl-C to stop)

From `task` on, the Schedule tab draws the window as lanes (each core, the DMA, then per accelerator ▸ group its movers and its unit, as the editor's Bindings nest them), quiet stretches of 16 cycles or more folded; a mover's task that moves nothing, started only because an encapsulated set starts all its movers at once (`dot_1_enc`'s `vsum_a` during the vmul's task), is a dashed bar, `idle (shared start)` in the legend; dot 1's two accelerators each start their own set only, so it has none. A click on a cycle shows the cluster view under it, which scrolls sideways in its own box when the cluster is wider than the window (the page never does), with one streamer box per set inside the accelerator, lit while its movers run, and the unit whose task is open there (dot 1: `vmul`, then `vsum`); the arrows step by event (shift: by task boundary). The Report tab's Accelerator section lists each unit under its accelerator ▸ group with its group's movers in a collapsible (`vmul ▸ vmul_a, vmul_b, vmul_out`), and `run.md` opens its Accelerators section with a row per accelerator: its groups, CSRs, its streamer's busy windows and each group's COUNTER. dot 1 (`recipes/dot/dot_1.json`, two accelerators) runs in 264 cycles at N = 64 against `dot_1_enc`'s 304 (one accelerator of the same two groups), COUNTERs 19 and 19 in both; at N = 512, 584 against 586, COUNTERs 131 and 131: vsum's own set takes 12 CSR writes before its start where the shared streamer rewrites all five movers (27), and at N = 512 vmul's 131-cycle run hides those writes. From `beat` on, the memory tab shows where each container lies in L1 and L2 and how its words move. The address bar holds the whole view, so a link shows the same.

The graph viewer shows `.snaxdfg` steps on port 8766, a recipe's folder as its steps in order. To compare two designs, give both folders:

    pixi run view-dfg out/sandbox/try/dot4_w8/ out/sandbox/try/dot5_w8/ --platform platforms/snax_forge.json

Each step is then named after its folder and stem (`dot4_w8/6_link`), the step dropdown groups the steps by folder, and Compare with one design's step in each panel lays Details and the Estimate out as two columns headed `dot4_w8 · 6 link` and `dot5_w8 · 6 link`, the cost tables' cells `before → after` (C§12). The left column lists the step's Symbols and Containers (`A: N · L1 · 16 KiB`; hover a symbol to see which containers use it) and its Bindings as a tree, accelerator ▸ group ▸ unit, each level a drop-down open by default (dot 1: `accelerator dot` ▸ `group mul` ▸ `mul · simd · W = 4`; a unit not in an accelerator yet and a split not bound yet at the top), each param with what it configures (dot 4's `W` is `shared with` its `mac` and `sum`). With the recipe's params a unit's box names its split's param (`mul · a * b · W 4`; a session's `W_acc_1 8`) and a split map's Details give its range with the param and its iterations (`0:N // W_acc_1 = 256`). The Estimate gives the accelerator's cycles, the unmapped work's (one cycle per iteration, an assumption, D155) and their total, the cost as a table per accelerator (block and datapath in one column, then µm², nJ and share; the L1 accesses' row muted, a placeholder, D154) and a group's bank note where L1's banks bound it; both side panels resize on their inner edge.

On the MICAS server, start the image with `--network host` so its port 8765 is the server's, and forward it from your machine, then open `http://localhost:8765/`:

    ssh -N -L 8765:127.0.0.1:8765 <user>@<server>

## 5. A second kernel: vsum

`recipes/reduce/vsum.json` binds vsum's reduction to `TreeAccumulate` at one lane:

    pixi run flow recipes/reduce/vsum.json --platform platforms/snax_forge.json

    ...
      check     passed: 254 cycles (main to forge_check); data 0 errors (1 words); counters sum_acc 67 (64 beats)

Its `run/stderr.txt` holds gvrun's warning that the installed platform tree does not match: the tree was built from vecadd's cluster file, so for another accelerator gvrun builds the system from its JSON configuration, on the same components. `examples/dot/` runs dot 1 to dot 6 the same way.

## 6. What the flow refuses

A platform without a source hjson stops at `gen`, before anything is built, and exits 1:

    pixi run flow recipes/elementwise/vecadd.json --platform platforms/small16.json

    FAILED recipes/elementwise/vecadd.json: gen: [gen.source] platform small16 has no source hjson: read it from one with `pixi run design from-hjson`

A design the platform cannot take stops earlier, at the design checks (C§14), in the flow as in the design step alone, and exits 1: vecmul at W = 32 needs more TCDM crossbar inputs than SNAX-GVSoC's crossbar takes (`connect.xbar_inputs`):

    pixi run sandbox recipes/elementwise/vecmul.json --set W=32 --out out/sandbox/vecmul_W32
    pixi run design out/sandbox/vecmul_W32/2_bind.snaxdfg --platform platforms/snax_forge.json

    design check of out/sandbox/vecmul_W32/2_bind.snaxdfg on platforms/snax_forge.json: 1 problem
     [connect.xbar_inputs] acc: 99 TCDM crossbar inputs, the platform takes at most 64 (96 streamer channels, 3 for the cores and AXI; l1.max_xbar_inputs, the limit SNAX-GVSoC's crossbar asserts)
         fix: fewer lanes, or a platform with more inputs (--set platform.l1.max_xbar_inputs=99 on a model that takes them, open item 105)

`pixi run flow recipes/elementwise/vecmul.json --platform platforms/snax_forge.json --set W=32` prints the same problem. Two accelerators are no refusal: `recipes/dot/dot_serial.json` leaves dot's `mul` and `sum` unencapsulated, two streamer sets of the one entry `snax_forge_dot_serial` (D157).

## 7. By hand: the stages, the RTL and the bundle

The flow's stages are tools of their own. The design point, the generator into `out/gvsoc/vecadd/`, then `gvsoc/Makefile` for the program (into `out/snax-gvsoc/sw/vecadd/`) and the run, which prints section 2's four `forge` lines:

    pixi run sandbox recipes/elementwise/vecadd.json
    pixi run design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_forge.json
    pixi run gen-gvsoc out/design/vecadd/design_point.json
    make -C gvsoc sw APPS=forge FORGE=../out/gvsoc/vecadd DESIGN=vecadd
    make -C gvsoc run TARGET=snax_hjson DESIGN=vecadd \
        GVRUN_ARGS="--param hjson=$PWD/out/gvsoc/vecadd/snax_forge.hjson" \
        ELF=../out/snax-gvsoc/sw/vecadd/forge.elf

`GVRUN_ARGS` takes gvrun's own options too, such as `--trace=snax_acc0/unit`. The flow does not make RTL; `gen-rtl` writes `accelerator.json` (C§17, gen-gvsoc's bytes), checks it against the datapath catalogue and runs `hw/chisel/`'s `Emit --spec` on a classpath sbt exports into `hw/chisel/target/emit-classpath-<hash>.txt`, one file per environment so a checkout used from the image and from the host keeps one of each (the first run, and again after a change under `hw/chisel/src/main/` or to `build.sbt`, or when a jar it lists is gone or cannot be read):

    pixi run gen-rtl out/design/vecadd/design_point.json

    ...
    wrote out/rtl/vecadd/accelerator.json: snax_forge_vecadd_shell_wrapper, SimdLanes
    wrote out/rtl/vecadd/sv/ElementwiseAlu_w64_add.sv
    ...
    wrote out/rtl/vecadd/sv/snax_forge_vecadd_shell_wrapper.sv
    wrote out/rtl/vecadd/emitted.json

The modules are named after the entry's tag, `<accelerator_slot>_<design>` (`snax_forge_vecadd`, D157). `snax_forge_vecadd_shell_wrapper` is what snax_cluster's `snax_acc_wrapper.sv.tpl` connects with `snax_disable_csr_manager` (C§17); inside it `snax_forge_vecadd_csr`, `snax_forge_vecadd_ctrl` (the group's `GroupCtrl`, `snax_forge_vecadd_acc_ctrl`) and `snax_forge_vecadd_datapath`. The bundle (C§18) is gen-gvsoc's files, the RTL under `rtl/` with its file list, and `bundle.json`:

    pixi run bundle out/design/vecadd/design_point.json

    ...
    wrote out/bundle/vecadd/bundle.json
    no forge_expected.txt: a bundle alone has no SNAX-GVSoC run (pixi run forge writes it)

One catalogue block alone, as an accelerator holds it (CHAR1, C§17), for snax-forge-experiments to synthesize and simulate on its own: `gen-block` takes a JSON list of `{"datapath", "params", "out"}`, checks each against `hw/chisel/blocks.json` and runs `Emit --block` on the same classpath, all in one call:

    echo '[{"datapath": "TreeAccumulate", "params": {"inType": "int64", "outType": "int64", "lanes": 8, "op": "add"}, "out": "out/block/tree"}]' > blocks.json
    pixi run gen-block blocks.json

    ...
    wrote out/block/tree: TreeAccumulate_i64_i64_n8, 1 file(s) in sv/

## 8. The local RTL check

One ELF, snax-forge's, runs on SNAX-GVSoC and on snax_cluster's RTL, and the RTL run must print the SNAX-GVSoC run's `forge data` line and each of its `forge counter` lines within that group's streamer fill (D123): the COUNTER counts the streamer's head start before START, which the core's timing between STREAMER_START and START sets and SNAX-GVSoC does not model (open items 65, 67). snax-forge never runs RTL; snax_cluster's `snax-forge-check` does (C§18).

`pixi run forge` runs section 2's flow, the bundle of its design point with `forge_expected.txt` (the run's counter and data lines, then each counter's `forge fill`, its group's streamer fill from the estimate), and the bundle's ELF, checked to be the flow's; each step's output is in `out/forge/<name>/forge.log`:

    pixi run forge recipes/elementwise/vecadd.json

    forge vecadd: recipes/elementwise/vecadd.json on platforms/snax_forge.json; every step's output in out/forge/vecadd/forge.log
      flow      passed: design point out/flow/vecadd/design/design_point.json, run out/flow/vecadd/run/
      bundle    out/bundle/vecadd/: snax_forge_vecadd_shell_wrapper, snax-forge $COMMIT
      expected  out/bundle/vecadd/forge_expected.txt: the run's counter and data lines, each counter's fill
      sw        out/forge/vecadd/sw/forge.elf from the bundle (sha256 3fa93c774886c3b3, the flow's)
    ...

then the snax_cluster commands for one design (Verilator, and Questa as below). `pixi run forge-set quick|full [--sim vsim|vlt] [-j N]` runs forge on a set's recipes at once (quick: vecadd_i32, vsum_tree_i8, dot_1, dot_1_enc, dot_4; full: those and vecadd, vsum, vecmul_i32_i64, dot_2, dot_3, dot_5, dot_6, matvec_5, matvec_6), writes `out/forge-set/<set>.md` (each bundle's expected lines and the commands per design) and prints one line:

    pixi run forge-set quick

    forge-set quick: 5 designs, 32 at a time; each forge's output in out/forge-set/quick/<design>.txt
      vecadd_i32       passed
      vsum_tree_i8     passed
      dot_1            passed
      dot_1_enc        passed
      dot_4            passed
    wrote out/forge-set/quick.md; from snax_cluster's target/snitch_cluster:
      make snax-forge-check BUNDLES="$ROOT/snax-forge/out/bundle/vecadd_i32 ... $ROOT/snax-forge/out/bundle/dot_4" SIM=vsim SNAX_CLUSTER_IMAGE=<image>

Run that line from snax_cluster's `target/snitch_cluster`, outside the image. `<image>` is snax_cluster's development image for podman; on the MICAS servers it is `d3d8f5572ff4`. The image has no sbt, Coursier or firtool cache, so every `podman run` downloads them: put a `podman` wrapper first on `PATH` that adds `-v` mounts of persistent folders at `/root/.sbt`, `/root/.cache/sbt`, `/root/.cache/coursier` and `/root/.cache/llvm-firtool` to every `run`. With `SIM=vsim`, `rtl-gen` and `vsim_preparation` run in the image and the build and the run outside it, with Questa and bender on `PATH` (snax_cluster's pixi environment); with `SIM=vlt` every step runs in the image. It prints per bundle `PASS <name>`, with a line for each counter that differs from SNAX-GVSoC's within its fill, or `FAIL <name>: <want> / <got>`, and fails if any failed; `logs/snax-forge-check.txt` keeps those lines with each run's `forge cycles`, `logs/snax-forge-<name>-<sim>.txt` each run's output:

    PASS vecadd_i32
      vecadd_i32 forge cycles: 217
    PASS vsum_tree_i8
      vsum_tree_i8 forge cycles: 196
    PASS dot_1
      dot_1 forge cycles: 282
    PASS dot_1_enc
      dot_1_enc forge cycles: 302
    PASS dot_4
      dot_4 forge cycles: 449

A counter that differs within its fill gets a line under its PASS, as `dot_1_enc` at N = 512 on vsim (the RTL's core starts `vmul` 6 cycles after its streamer, so the COUNTER leaves out the fill):

    PASS dot_1_enc_N512
      mult_vmul: counter 128 against 131, within its fill 3
      dot_1_enc_N512 forge cycles: 609

Each vsim design takes about 2 minutes, its run 6–13 s against SNAX-GVSoC's 1.5–2.2 s. CLAUDE.md's rule says when each set is due: the quick set on vsim before a PR that changes the emitted RTL or `main.c` merges, the full set at a phase's close, its numbers into STATUS's open item 95.

One design by hand, Questa: in the image, from the folder that holds both checkouts, then the build and the run outside it (`B` and `E` as forge printed them; below vecadd_i32's run):

    podman run --rm -it -v "$PWD":"$PWD" -w "$PWD/snax_cluster" d3d8f5572ff4
    make -C target/snitch_cluster clean-generated clean-vsim
    make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B
    make -C target/snitch_cluster vsim_preparation CFG_OVERRIDE=$B

    make -C target/snitch_cluster snax-forge-sim SIM=vsim CFG_OVERRIDE=$B
    make -C target/snitch_cluster snax-forge-run SIM=vsim CFG_OVERRIDE=$B ELF=$E

    ...
    forge cycles: 217
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    # ** Info: [SUCCESS] Program finished successfully

Questa exits 0 whatever the program returns, so `snax-forge-run` reads the exit code from that last line; the output is also in `logs/snax-forge-vsim.txt`. `N` reaches the hardware only as CSR values, so a design's `N=512` bundle's `rtl/` is the default's, byte for byte, and one RTL build runs both ELFs.

## 9. A recipe in the editor

The editor builds a recipe one transform at a time on the drawn graph (D55, C§12). A session on vecadd's import:

    pixi run edit-dfg --kernel vecadd --set N=64

    serving vecadd at http://127.0.0.1:8767/ (Ctrl-C to stop)

The session starts from the kernel's symbols only, each a param of its name, so N needs its value (without it: `vecadd: give N, e.g. --set N=2048`) and there is no W yet. Open it as section 4 does the viewer, on port 8767. Click `add_map`: the sidebar lists the transforms whose site accepts it, each with its form. A click selects one item, a box, a container or a group's or an accelerator's dashed outline; shift-click adds an item to the selection or removes it, Esc clears it, and the sidebar then lists what the selection as a whole offers (D72; `2 selected: …`), a one-item transform under "Not here" with `takes one selection`. Apply `split` with factor `W`: W is not a param yet, so the form shows `new param W = 4` beside it; keep 4, and W = 4 joins the session's params, each split free to have its own (C§12). `split` names its maps neutrally (D73): `add_map` keeps its id over `i_1`, around `add_map_inner` over `i_2` (a second split of `i` would take `i_3` and `i_4`). The Bindings list `add_map split · W = 4 · not bound`. Tag them, `tag_map` `temporal` on `add_map` and `spatial` on `add_map_inner`, then select `add` and `bind` it to `simd / chisel_tiled_spatial`, whose instance the form fills as `acc_1`, the session's first bind (`acc_2` for the second, D156). The bind renames the split's W, the editor's name and this unit's alone, to `W_acc_1` in every step and in the params, so the entry becomes `acc_1 · simd · W_acc_1 = 4`, its W an input as the Symbols' N is (the top bar has no params), the unit's box reads `acc_1 · a + b · W_acc_1 4` and `add_map`'s Details give its range as `0:N // W_acc_1 = 16`; a param several units share (dot 2–6's W) or one a recipe brought keeps its name. Each Apply is a step checked against the reference, with a log line: the bind's ends `C 0 → 16 · unmapped 64 → 0 · renamed W → W_acc_1`, the accelerator cycles before and after and the unmapped cycles where they change. Every Apply and param change also runs the design checks Save runs and lists them under the Estimate, "Design checks (Save's)": the problems first, each with its code, where, message and fix, then the notes, a "before Save" one tagged so; the log line's counts (`1 note`) link to them. Until the bind the note is `[graph.unbound] tasklet add in add_map/add_map_inner`, tagged before Save, `bind it (is not bound to an accelerator: the program drives accelerators only)`, after it none; two accelerators before `encapsulate` are a note too, `before Save: shift-click the outlines of acc_1 and acc_2 and apply encapsulate (…)`, so a problem is always something to fix, and a design note such as `char.missing` shows whatever else is listed. Save still refuses both notes. The Estimate (D155, C§19) shows 64 cycles as imported, all unmapped (64 iterations at one cycle each, an assumption), still 64 after the split and both tags (a tag runs nothing), and after the bind 16 on the accelerator, the group `acc_1` at `16 = 16 + 0` cycles with an expected COUNTER of 19, a guide and not a check (a run's COUNTER counts from START until busy drops), here the counter section 2's run measures (`add_acc 19`). Undo, redo and the step list move the cursor; Apply below the last step asks to drop the steps after it. With several units (dot 1), shift-click the groups' outlines and `encapsulate` ticks their units (one unit's box ticks only that unit); after it the accelerator's outline offers `unencapsulate`, and once encapsulated the Bindings show the tree `accelerator dot` ▸ `group acc_1` ▸ `acc_1 · simd · W_acc_1 = 8`, `group acc_2` ▸ `acc_2 · accumulate · W_acc_2 = 8`. The right panel's cost table fits the panel's default width: block and datapath in one column, the numbers right-aligned, the L1 accesses' row muted as a placeholder. A param changed in Bindings replays the steps up to the cursor only, its log line giving the step's problem and note counts as an Apply's does (`params W_acc_2=10  ok · 9 steps replayed · 1 problem`): the steps after it stay as redo steps, replayed and checked when the cursor reaches them, where a step that no longer applies shows its refusal.

Save recipe, here as `vecadd_mine` (a name no recipe has), runs the design checks, writes `recipes/custom/vecadd_mine.json` (params N = 64 and W_acc_1 = 4) and its steps to `out/sandbox/vecadd_mine/`, logs `saved recipes/custom/vecadd_mine.json` and lists what comes next, each command with what it does:

    pixi run sandbox recipes/custom/vecadd_mine.json
        writes the steps and checks each against the reference
    pixi run flow recipes/custom/vecadd_mine.json --platform platforms/snax_forge.json
        runs the recipe on SNAX-GVSoC and writes the reports
    pixi run forge recipes/custom/vecadd_mine.json
        the flow plus the bundle for snax_cluster's RTL

A design problem stops Save until it is forced: vecmul at W = 32 lists `[connect.xbar_inputs] acc_1: 99 TCDM crossbar inputs, the platform takes at most 64` (C§14), which SNAX-GVSoC's crossbar would assert on.

The session stays in `out/sessions/vecadd/`: `--session out/sessions/vecadd` resumes it, `--recipe recipes/elementwise/vecadd.json` starts one from a recipe, and Export step writes one step's graph to its `export/`. `pixi run view-dfg out/sandbox/vecadd_mine/ --platform platforms/snax_forge.json` shows the saved steps read-only.
