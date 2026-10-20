# From a recipe to a run on SNAX-GVSoC

One design point the whole way: a recipe through `sandbox` and `design` on
`platforms/snax_alu.json`, `gen-gvsoc`, GVSoC built through `gvsoc/Makefile`,
the generated program built and run, and what the run prints. Each block is
the command as typed, then what it printed in a fresh clone with the
submodule initialised (long outputs cut to their last lines, marked `...`).
The flow (`pixi run flow`) still runs on SNAX-MODEL until FLOW3; this is the
SNAX-GVSoC path by hand. snax-gvsoc's own `docs/RUN.md` runs its checked-in
fixtures and its regression.

## 1. A clone and the container

snax-gvsoc is a submodule; CI never fetches it, so initialise it once. Clone
into any folder, `$ROOT` below (the outputs' absolute paths start there;
yours will be your own):

    git clone https://github.com/KULeuven-MICAS/snax-forge.git
    cd snax-forge && git submodule update --init gvsoc/snax-gvsoc

    Cloning into 'snax-forge'...
    Submodule 'gvsoc/snax-gvsoc' (https://github.com/KULeuven-MICAS/snax-gvsoc) registered for path 'gvsoc/snax-gvsoc'
    Cloning into '$ROOT/snax-forge/gvsoc/snax-gvsoc'...
    Submodule path 'gvsoc/snax-gvsoc': checked out '4808f8eeb05c6bb99dd64f9f6246413f889a54c5'

Everything else runs in snax-forge's image (D118), started from the folder
that holds `snax-forge`:

    podman run --rm -it -v "$PWD":"$PWD" -w "$PWD" ghcr.io/kuleuven-micas/snax-forge:main
    cd snax-forge
    pixi install

    The default environment has been installed.

## 2. The recipe and the design point

`recipes/vecadd.json` binds vecadd's add to `elementwise_add` at W = 4:

    pixi run sandbox recipes/vecadd.json

    Pixi task (sandbox in default): python -m snax_forge.sandbox recipes/vecadd.json
    $ROOT/snax-forge/out/sandbox/vecadd/0_input.snaxdfg
    $ROOT/snax-forge/out/sandbox/vecadd/1_split_map.snaxdfg
    $ROOT/snax-forge/out/sandbox/vecadd/2_bind.snaxdfg
    2 steps, each equal to the input graph on the reference check

The design step pairs the bound graph with `platforms/snax_alu.json`, the
platform read from snax_cluster's cluster file (its source is
`platforms/snax_alu.hjson`, its accelerator slot `snax_alu`):

    pixi run design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_alu.json

    Pixi task (design in default): python -m snax_forge.design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/snax_alu.json
    out/sandbox/vecadd/2_bind.snaxdfg on snax_alu (base snax_alu, 0 changes): checks passed
    streamer shell:
      acc_a        acc.a      reader  4 lanes [4]  temporal_dims 1  fifo_depth 8  addr_depth 8  prio 0
      acc_b        acc.b      reader  4 lanes [4]  temporal_dims 1  fifo_depth 8  addr_depth 8  prio 0
      acc_out      acc.out    writer  4 lanes [4]  temporal_dims 1  fifo_depth 8  addr_depth 8  prio 0
    memory plan (residency default, layout contiguous, placement contiguous; 0 changes):
      A            l2 [2147483648, 2147484160)   l1 [268435456, 268435968) bank 0
      B            l2 [2147484160, 2147484672)   l1 [268435968, 268436480) bank 0
      C            l2 [2147484672, 2147485184)   l1 [268436480, 268436992) bank 0
    wrote out/design/vecadd/platform.json
    wrote out/design/vecadd/memory.json
    wrote out/design/vecadd/design_point.json

## 3. Generate the cluster, the headers and the program

    pixi run gen-gvsoc out/design/vecadd/design_point.json

    Pixi task (gen-gvsoc in default): python -m snax_forge.gen out/design/vecadd/design_point.json
    cluster snax_alu.hjson from out/design/vecadd/design_point.json
      task add_acc: 16 beats, {'loopCount': 16}
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/snax_alu.hjson
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/snax_alu_streamer.hjson
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/snax_alu_acc.hjson
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/include/streamer_csr_addr_map.h
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/include/acc_csr.h
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/data.h
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/main.c
    wrote $ROOT/snax-forge/out/gvsoc/vecadd/manifest.json
    run: make -C gvsoc run TARGET=snax_hjson GVRUN_ARGS="--param hjson=$ROOT/snax-forge/out/gvsoc/vecadd/snax_alu.hjson" ELF=...

`out/gvsoc/vecadd/` now holds the cluster file `snax_alu.hjson` (the
platform's source with the slot set for this design), the streamer and
datapath files it refers to, the headers, `main.c` with its `data.h`, and
`manifest.json`. Nothing is compiled for a design point: SNAX-GVSoC's
installed `snax_hjson` target reads the cluster file at run time.

## 4. Build GVSoC, then the program

`gvsoc/Makefile` names the image's Python, CMake and host compilers, so it
runs the same in the container's plain shell and through pixi (`pixi run
make -C gvsoc build`). `make build` builds
snax-gvsoc's `SNAX_TARGETS` into `out/snax-gvsoc/`, a few minutes the first
time:

    cd gvsoc
    make build

    ...
    -- Up-to-date: $ROOT/snax-forge/out/snax-gvsoc/install/python/./requirements.txt
    -- Up-to-date: $ROOT/snax-forge/out/snax-gvsoc/install/python/./.gitignore
    make[1]: Leaving directory '$ROOT/snax-forge/gvsoc/snax-gvsoc'

`make sw` builds snax_cluster's runtime and the generic app's program from
the generated folder, into `out/snax-gvsoc/sw/vecadd/`:

    make sw APPS=forge FORGE=../out/gvsoc/vecadd DESIGN=vecadd

    ...
    clang --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -fno-builtin-memset -DPRINTF_DISABLE_SUPPORT_LONG_LONG -I$ROOT/snax-forge/out/gvsoc/vecadd -I$ROOT/snax-forge/out/gvsoc/vecadd/include -I$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/generated/snax_alu_cluster_old -I$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/runtime/rtl-generic/src -I$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/runtime/rtl-generic/../common -I$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/snRuntime/api -I$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/snRuntime/api/omp -I$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/snRuntime/src -I$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/snRuntime/src/omp -I$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/deps/riscv-opcodes -I$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/math/include -mabi=ilp32d -mcmodel=medany -ffast-math -fno-builtin-printf -fno-builtin-sqrtf -fno-common -fopenmp -ftls-model=local-exec -O3 -D__DEFINED_uint64_t -MMD -MP -MT $ROOT/snax-forge/out/snax-gvsoc/sw/vecadd/forge.elf -MF $ROOT/snax-forge/out/snax-gvsoc/sw/vecadd/forge.d  -fuse-ld=lld -nostdlib -L$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/runtime/rtl-generic -T$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/snRuntime/base.ld -L$ROOT/snax-forge/gvsoc/snax-gvsoc/build/sw/ -L$ROOT/snax-forge/gvsoc/snax-gvsoc/build/sw/snax_alu_cluster_old/ -lmath -lsnRuntime $ROOT/snax-forge/out/gvsoc/vecadd/main.c -o $ROOT/snax-forge/out/snax-gvsoc/sw/vecadd/forge.elf
    llvm-objdump -D $ROOT/snax-forge/out/snax-gvsoc/sw/vecadd/forge.elf > $ROOT/snax-forge/out/snax-gvsoc/sw/vecadd/forge.dump
    make[1]: Leaving directory '$ROOT/snax-forge/gvsoc/snax-gvsoc/sw/apps/forge'

## 5. Run it

`snax_hjson` takes the cluster file with `--param hjson=`;
`--trace=snax_acc0/unit` turns on the unit's trace. The work directory is
`out/snax-gvsoc/work/snax_hjson/vecadd/`:

    make run TARGET=snax_hjson DESIGN=vecadd \
        GVRUN_ARGS="--param hjson=$PWD/../out/gvsoc/vecadd/snax_alu.hjson --trace=snax_acc0/unit" \
        ELF=../out/snax-gvsoc/sw/vecadd/forge.elf

    mkdir -p $ROOT/snax-forge/out/snax-gvsoc/work/snax_hjson/vecadd
    $ROOT/snax-forge/out/snax-gvsoc/install/bin/gvrun --target snax_hjson \
    	--work-dir $ROOT/snax-forge/out/snax-gvsoc/work/snax_hjson/vecadd --param chip/soc/binary=$ROOT/snax-forge/out/snax-gvsoc/sw/vecadd/forge.elf --param hjson=$ROOT/snax-forge/gvsoc/../out/gvsoc/vecadd/snax_alu.hjson --trace=snax_acc0/unit run
    Warning: installed platform tree $ROOT/snax-forge/out/snax-gvsoc/install/lib/libplatform_tree_snax_hjson.so does not match the live system tree, falling back to the JSON config path (rebuild the target to regenerate the tree).
    forge data: 0 errors (64 words)
    forge counter add_acc: 19
    forge errors: 0
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/trace                                 ] New slave port (name: clock, port: 0x5629f29da2c8)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/trace                                 ] New slave port (name: reset, port: 0x5629f29da0d0)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] New slave port (name: power_supply, port: 0x5629f29da178)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] New slave port (name: voltage, port: 0x5629f29da220)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] New master port (name: a, port: 0x5629f29ddc80)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] New master port (name: b, port: 0x5629f29ddd58)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] New master port (name: out, port: 0x5629f29da608)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] New slave port (name: ctrl, port: 0x5629f29da450)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] New master port (name: busy, port: 0x5629f29da508)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/trace                                 ] add, 4 lanes of 64 bits, lockstep, latency 0, ii 1
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] Creating final bindings
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] Creating final binding (/chip/soc/cluster_0/snax_acc0/unit:busy -> /chip/soc/cluster_0/snax_acc0/shell:busy)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] Creating final binding (/chip/soc/cluster_0/snax_acc0/unit:out -> /chip/soc/cluster_0/snax_acc0/streamer:writer_0)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] Creating final binding (/chip/soc/cluster_0/snax_acc0/unit:b -> /chip/soc/cluster_0/snax_acc0/streamer:reader_1)
    0: -1: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] Creating final binding (/chip/soc/cluster_0/snax_acc0/unit:a -> /chip/soc/cluster_0/snax_acc0/streamer:reader_0)
    0: 0: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] Reset (active: 1)
    0: 0: [/chip/soc/cluster_0/snax_acc0/unit/comp                                  ] Reset (active: 0)
    292200000: 2922: [/chip/soc/cluster_0/snax_acc0/unit/trace                                 ] Start at cycle 2922: count 16
    294100000: 2941: [/chip/soc/cluster_0/snax_acc0/unit/trace                                 ] run 0: firings 16 busy 16 stall_in 3 stall_out 0 flush 0

    echo "exit $?"

    exit 0

## 6. Reading the output

- The warning is expected: `make build` made `snax_hjson`'s platform tree
  from its default cluster file, with snax_cluster's ALU in the slot; this
  cluster file puts the generated accelerator there, so gvrun builds the
  system from its JSON configuration instead, on the same installed
  components.
- `forge data: 0 errors (64 words)`: every output word against its golden
  in `data.h` (the reference executor's result).
- `forge counter add_acc: 19`: the shell's cycle counter for the
  accelerator task `add_acc`.
- The unit's run line, `run 0: firings 16 busy 16 stall_in 3 stall_out 0
  flush 0`: the counter is busy + stall_in + stall_out + flush (16 + 3 +
  0 + 0 = 19), and the firings are the task's beats in the manifest:

      cd ..
      grep '"tasks"' out/gvsoc/vecadd/manifest.json

      "tasks": [{"name": "add_acc", "beats": 16, "csrs": {"loopCount": 16}}]

- `forge errors: 0`: the data errors plus one per counter below its beats;
  `main` returns it, and `make run` exits 0.

## 7. What the generator refuses

The generator writes a cluster file from the platform's source hjson and
puts one accelerator in its one slot, so two kinds of design point stop
with a named rule. `small16` is a hand-written platform with no source
hjson (`gen.source`):

    pixi run design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/small16.json --out out/design/vecadd_small16
    pixi run gen-gvsoc out/design/vecadd_small16/design_point.json

    Pixi task (design in default): python -m snax_forge.design out/sandbox/vecadd/2_bind.snaxdfg --platform platforms/small16.json --out out/design/vecadd_small16
    out/sandbox/vecadd/2_bind.snaxdfg on small16 (base small16, 0 changes): checks passed
    streamer shell:
      acc_a        acc.a      reader  4 lanes [4]  temporal_dims 1  fifo_depth 2  addr_depth 8  prio 0
      acc_b        acc.b      reader  4 lanes [4]  temporal_dims 1  fifo_depth 2  addr_depth 8  prio 0
      acc_out      acc.out    writer  4 lanes [4]  temporal_dims 1  fifo_depth 2  addr_depth 8  prio 0
    memory plan (residency default, layout contiguous, placement contiguous; 0 changes):
      A            l2 [0, 512)                   l1 [0, 512) bank 0
      B            l2 [512, 1024)                l1 [512, 1024) bank 0
      C            l2 [1024, 1536)               l1 [1024, 1536) bank 0
    wrote out/design/vecadd_small16/platform.json
    wrote out/design/vecadd_small16/memory.json
    wrote out/design/vecadd_small16/design_point.json

    Pixi task (gen-gvsoc in default): python -m snax_forge.gen out/design/vecadd_small16/design_point.json
    [gen.source] platform small16 has no source hjson: read it from one with `pixi run design from-hjson`

`dot` binds two accelerators, a multiplier and a sum, and the platform has
one slot and the program drives one accelerator (`gen.instances`; several
accelerators on one core are open item 79, a composed one is M13):

    pixi run sandbox recipes/dot.json
    pixi run design out/sandbox/dot/4_bind.snaxdfg --platform platforms/snax_alu.json
    pixi run gen-gvsoc out/design/dot/design_point.json

    Pixi task (sandbox in default): python -m snax_forge.sandbox recipes/dot.json
    $ROOT/snax-forge/out/sandbox/dot/0_input.snaxdfg
    $ROOT/snax-forge/out/sandbox/dot/1_split_map.snaxdfg
    $ROOT/snax-forge/out/sandbox/dot/2_bind.snaxdfg
    $ROOT/snax-forge/out/sandbox/dot/3_split_map.snaxdfg
    $ROOT/snax-forge/out/sandbox/dot/4_bind.snaxdfg
    4 steps, each equal to the input graph on the reference check

    Pixi task (design in default): python -m snax_forge.design out/sandbox/dot/4_bind.snaxdfg --platform platforms/snax_alu.json
    out/sandbox/dot/4_bind.snaxdfg on snax_alu (base snax_alu, 0 changes): checks passed
    streamer shell:
      mul_a        mul.a      reader  4 lanes [4]  temporal_dims 1  fifo_depth 8  addr_depth 8  prio 0
      mul_b        mul.b      reader  4 lanes [4]  temporal_dims 1  fifo_depth 8  addr_depth 8  prio 0
      mul_out      mul.out    writer  4 lanes [4]  temporal_dims 1  fifo_depth 8  addr_depth 8  prio 0
      sum_a        sum.a      reader  4 lanes [4]  temporal_dims 1  fifo_depth 8  addr_depth 8  prio 0
      sum_out      sum.out    writer  1 lanes [1]  temporal_dims 1  fifo_depth 8  addr_depth 8  prio 0
    memory plan (residency default, layout contiguous, placement contiguous; 0 changes):
      A            l2 [2147483648, 2147484160)   l1 [268435456, 268435968) bank 0
      B            l2 [2147484160, 2147484672)   l1 [268435968, 268436480) bank 0
      out          l2 [2147484672, 2147484680)   l1 [268436480, 268436488) bank 0
      tmp0         l1 [268436544, 268437056) bank 8
    wrote out/design/dot/platform.json
    wrote out/design/dot/memory.json
    wrote out/design/dot/design_point.json

    Pixi task (gen-gvsoc in default): python -m snax_forge.gen out/design/dot/design_point.json
    [gen.instances] 2 accelerated instances ['mul', 'sum']; the platform has one accelerator slot and the program drives one accelerator (open item 79)

Both exit 1 and write nothing to `out/gvsoc/`.
