# SNAX-FORGE Chisel datapaths

Hardware side of SNAX-FORGE: a hand-written library of accelerator blocks,
their chiseltest specs, and `Emit`, which elaborates a catalogue of
configurations to SystemVerilog and a design point's accelerator from its
`rtl.json` (GEN1). It is a project of its own: nothing here imports
`snax_forge/`, and the Python side calls it only through `pixi run gen-rtl`.
The blocks are the reference implementations of the BRMs (`elementwise_add`
names `ElementwiseTiledSpatial`).

## Why these versions

Scala 2.13.14, Chisel 6.4.0, chiseltest 6.0.0 — identical to
`snax_cluster/hw/chisel`. Generated datapaths are meant to sit inside a SNAX
shell next to the hand-written streamer, so a Chisel version skew between the
two trees buys nothing and costs a bad afternoon.

The JVM comes from pixi (`sbt` on conda-forge drags in an openjdk). Chisel and
FIRRTL jars are *not* conda packages: sbt resolves them from Maven Central on
first build, and Chisel 6 additionally downloads the `firtool` binary through
`firtool-resolver`. Both land in the Coursier cache under `~/.cache/coursier`,
so the first `chisel-test` takes minutes and later ones take seconds. That is
expected, not a hang.

## Layout

```
hw/chisel/
├── build.sbt
├── project/{build.properties,plugins.sbt}
├── .scalafmt.conf                      # copied verbatim from snax_cluster
├── blocks.json                         # the block catalogue (D138)
└── src/
    ├── main/scala/snax/forge/
    │   ├── Emit.scala                  # elaboration entry point and catalogue; --describe, --spec
    │   ├── Blocks.scala                # the block registry (blocks.json)
    │   ├── shell/                      # rtl.json, CsrManager, CtrlAdapter, the shell wrapper
    │   ├── RepoPaths.scala             # out/hw and repository paths
    │   ├── samples/
    │   │   ├── SimpleAdder.scala       # elementwise archetype  (cf. vecadd)
    │   │   └── Accumulator.scala       # reduction archetype    (cf. dot)
    │   └── elementwise/                # ALU, loop, spatial, tiled-spatial
    └── test/
        ├── scala/snax/forge/           # one spec per module, plus a model; shell/ for GEN1
        └── resources/rtl/<design>/     # GEN1's fixtures
```

## A design point's accelerator (GEN1, D138)

The boundary with the Python side is two files. `blocks.json`, checked in,
is the signature of each block (params with types and limits, data ports,
the `csr` section: read-write pins, start, busy and done, tied pins); the
registry `Blocks` holds the same and builds the blocks, and `BlocksSpec`
checks `Emit --describe` equal to it. `rtl.json`, per design point, is
written and checked by `snax_forge/gen/rtl.py` against `blocks.json`
(C§17); `Emit --spec rtl.json [--out DIR]` builds `<tag>_shell_wrapper`
(`shell/`) into `DIR/sv/`, one file per module, with `DIR/emitted.json`:

- `CsrManager`: snax_cluster's ReqRspManager protocol (the fields, start
  held while busy, busy and the counter read-only, D130), as the core
  reaches it after `snax_csr_mux_demux` with snax_cluster's manager off;
- `CtrlAdapter`: an accepted start to the block's start and configuration
  pins, ready while not busy, the counter, the tied pins;
- the unit, a `Blocks` block.

`pixi run gen-rtl DESIGN_POINT` does both steps (`docs/RUN.md` §6). The
fixtures in `src/test/resources/rtl/<design>/` (`rtl.json`, the streamer
hjson, `vectors.json`) come from the Python side, which checks them byte
for byte (`tests/gen/test_rtl.py`); refresh one with

    pixi run sandbox recipes/vsum_tree.json --set W=8 --out out/sandbox/vsum_tree_w8
    pixi run design out/sandbox/vsum_tree_w8/2_bind.snaxdfg --platform platforms/snax_alu.json --out out/design/vsum_tree_w8
    pixi run gen-rtl out/design/vsum_tree_w8/design_point.json --fixture hw/chisel/src/test/resources/rtl/vsum_tree_w8

(the others from their recipe as it is, named after it). The specs on
them: `ShellPortsSpec` (each wrapper's ports are the template's for its
streamer hjson), `ShellCasesSpec` (the shell's rules, worked by hand from
snax-gvsoc's `snax_shell.cpp`) and `ShellEndToEndSpec` (the CSR writes of
`main.c`, an always-ready streamer: the goldens and the counter of
`vectors.json`).

## The unit cases

`ElementwiseTiledSpatialLockstepSpec` drives `ElementwiseTiledSpatial` through
snax-gvsoc's lockstep `SnaxUnit` cases (GEN3), and `AccumulatorRateSpec` the
Accumulator through its pipelined fold cases (rate, GEN3 step 6b; at the
case's `lanes`, GEN2a round 2), run lines included; both read the cases with `SnaxUnitCases`. Set
`SNAX_GVSOC_UNIT_CASES` to their folder, e.g.
`SNAX_GVSOC_UNIT_CASES=$PWD/gvsoc/snax-gvsoc/tests/snax/unit/cases pixi run -e hw
chisel-test`; without it both are canceled, and the Accumulator's also when
the folder has no fold case (a snax-gvsoc before step 6b). Both also run at
the params of each GEN1 fixture of their block; CI's chisel job sets the
variable to the submodule's cases.

## Tests

Every module has a spec; the random ones drive it with seeded random data,
configurations, Decoupled gaps and stalls (`Source` and `Sink` hold valid
and bits until fire) and compare it every cycle with a Scala model written
from the documented behaviour, never from the Chisel code:

| Spec | Random | Model |
|---|---|---|
| `AccumulatorRandomSpec` | lanes, n, T, data, input gaps, output stalls | lane sums every T inputs (the docstring) |
| `ElementwiseRandomSpec` | the three elementwise blocks: lanes, n, op, data, `a` and `b` gaps each its own, output stalls | `ElementwiseModel` |
| `CsrManagerRandomSpec` | fields and widths, reads, writes and starts in and out of range, `rsp`/`ctrl` ready, `ro` | the protocol in `CsrManager`'s docstring |
| `ShellRandomSpec` | per fixture: n, T, data, streamer gaps and stalls, polled or held starts | the two block models; the counter as `snax_shell.cpp` states it |

The random specs run one test per seed of `Seeds` (scalatest names it
"seed N"); `SNAX_FORGE_SEED=N pixi run -e hw chisel-test` runs only seed N.

Waveforms: `SNAX_FORGE_VCD=1 pixi run -e hw chisel-test` writes each test's
to `hw/chisel/test_run_dir/<test name>/<module>.vcd`, e.g. vecadd's end to
end at `test_run_dir/the_shell_wrapper_end_to_end_should_give_vecadds_goldens_and_its_counter/ShellHarness.vcd`.

Package root is `snax.forge`, matching `snax.streamer` / `snax.xdma` in
snax_cluster.

Elaborated SystemVerilog goes to `out/hw/` (gitignored). RTL is never
committed — the Scala is the source of truth here, as the SDFG is on the
Python side.

## Tasks

| Command | Does |
|---|---|
| `pixi run chisel-compile` | compile Scala only |
| `pixi run chisel-test` | run the chiseltest specs |
| `pixi run chisel-gen` | elaborate everything to `out/hw/*.sv`; `-- --describe` prints the catalogue, `-- --spec rtl.json` builds an accelerator |
| `pixi run chisel-fmt` | scalafmt in place |
| `pixi run chisel-check` | fail if unformatted (CI uses this) |
| `pixi run chisel-clean` | drop build trees, keep the Coursier cache |

## The two archetypes

`SimpleAdder` and `Accumulator` are not arbitrary toys. They are the two
distinct hardware shapes the toolchain has to cover, the shapes of
`vecadd` and `dot` in `kernels/polybench/` among the staged kernels:

**Elementwise (`SimpleAdder`, cf. `vecadd.py`).** N beats in, N beats out,
L = 0, II = 1. Stateless. Chains freely with a downstream consumer, because
each output element is produced in producer order and read exactly once.

**Reduction (`Accumulator`, cf. `dot.py`).** T beats in, 1 beat out; a beat
of `lanes` elements goes through an adder tree in front of the register. Carries
state across beats. It is a **chaining barrier**: no output exists until the
last input is absorbed, so a downstream consumer cannot be fed element by
element and gets control chaining instead of data chaining.

Everything on the W11–W14 path is a composition of these two plus a typed ALU:
dot is elementwise-multiply feeding a reduction, softmax is three passes with
two reductions and a barrier between each.

## Conventions the generated code will have to follow

- **Random tests for every module.** Each Chisel module gets a seeded random
  spec against a Scala model (Tests above): not necessarily comprehensive,
  but its function under random data and random stalls.
- **Elastic everywhere.** Every port is `Decoupled`. Because the SDFG has
  already fixed allocation, schedule and binding, latency balancing is the only
  residual problem, and ready/valid turns it from a correctness obligation into
  a throughput knob. A module may stall; it may never produce a wrong answer
  because it stalled.
- **`valid` never depends on `ready`.** `ready` may depend on other channels'
  `valid` (that is how a join works). The reverse deadlocks. Both specs test
  this explicitly.
- **No internal storage.** Datapaths see data in flight only; address
  generation and buffering belong to the streamer. Reduction accumulators are
  the sanctioned exception — they hold a scalar, not a tile.
- **UInt as a bit container.** Two's-complement addition is signedness-
  agnostic, so `UInt` here is bit-identical to the DaCe int32 reference,
  wraparound included. Signedness starts to matter at multiply, compare and
  shift; the typed ALU set in W6 is where it gets handled properly.
- **Width and lane count come from the caller** (the catalogue today, a BRM's
  design params later), never from a default. The defaults in these
  constructors exist so the specs can run standalone.

## Not here yet

Elastic wrapper, skid buffer, typed integer ALU set and lane replication.
The bundle snax_cluster builds from (`Bender.yml`, the hjson) is LNK2b.
