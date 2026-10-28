# SNAX-FORGE Chisel datapaths

Hardware side of SNAX-FORGE: a hand-written library of datapaths and the
nest controller, their chiseltest specs, and `Emit`, which elaborates a
catalogue of configurations to SystemVerilog and a design point's
accelerator from its `accelerator.json` (C§17, GEN1). It is a project of its
own: nothing here imports `snax_forge/`, and the Python side calls it only
through `pixi run gen-rtl`. The datapaths are the reference implementations
of the BRMs (`elementwise_add` names `ElementwiseLanes`, `accumulate`
`TreeAccumulate`).

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
├── blocks.json                         # the datapath catalogue (D130, D147)
└── src/
    ├── main/scala/snax/forge/
    │   ├── Emit.scala                  # elaboration entry point and catalogue; --describe, --spec
    │   ├── Datapaths.scala             # the datapath registry (blocks.json)
    │   ├── ctrl/                       # NestCtrl, the group controller
    │   ├── datapath/                   # ElementwiseLanes, TreeAccumulate
    │   ├── shell/                      # accelerator.json, <tag>_csr, _ctrl, _datapath, the wrapper
    │   ├── RepoPaths.scala             # out/hw and repository paths
    │   ├── elementwise/                # the typed ALU lane, the ops
    │   └── stream/                     # FifoSlicer, FifoPacker (parked, D142)
    └── test/
        ├── scala/snax/forge/           # one spec per module, plus a model; shell/ for GEN1
        └── resources/
            ├── rtl/<design>/           # GEN1's fixtures
            └── groups/dot1/            # a hand-written two-group accelerator.json
```

## A design point's accelerator (GEN1, D130)

The boundary with the Python side is two files. `blocks.json`, checked in,
is the datapath catalogue: per datapath its params with types and limits,
its ports, latency, pins and the rule that fills the params from the unit's
ports (C§17); the registry `Datapaths` holds the same and builds them, and
`DatapathsSpec` checks `Emit --describe` equal to it. `accelerator.json`,
per design point, is written by `snax_forge/gen/` and checked by
`gen/rtl.py` against `blocks.json`; `Emit --spec accelerator.json [--out
DIR]` builds `<tag>_shell_wrapper` (`shell/`) into `DIR/sv/`, one file per
module, with `DIR/emitted.json`:

- `<tag>_csr` (`AccCsr`): the file's CSRs by offset, snax_cluster's
  ReqRspManager protocol (D130) as the core reaches it after
  `snax_csr_mux_demux` with snax_cluster's manager off; a START write with
  bit 0 set is its group's config (JOBS, JOB_BEATS), held while the group
  is busy; STATUS and COUNTER read from the controllers;
- `<tag>_ctrl` (`AccCtrl`): per group `<tag>_<group>_ctrl` (`GroupCtrl`),
  the start, the empty run, STATUS's sticky done, COUNTER and the busy and
  done pins around one `NestCtrl` (SnaxUnit's rule, D147);
- `<tag>_datapath` (`AccDatapath`): the units' datapaths, data and the
  controller's pins only.

The wrapper splits each outer Decoupled port: its data to the datapath,
its valid and ready to the controller. `pixi run gen-rtl DESIGN_POINT` does
the check and the elaboration (`docs/RUN.md` §6). The fixtures in
`src/test/resources/rtl/<design>/` (`accelerator.json`, the streamer hjson,
`vectors.json`) come from the Python side, which checks them byte for byte
(`tests/gen/test_rtl.py`); refresh one with

    pixi run sandbox recipes/vsum_tree.json --set W=8 --out out/sandbox/vsum_tree_w8
    pixi run design out/sandbox/vsum_tree_w8/2_bind.snaxdfg --platform platforms/snax_forge.json --out out/design/vsum_tree_w8
    pixi run gen-rtl out/design/vsum_tree_w8/design_point.json --fixture hw/chisel/src/test/resources/rtl/vsum_tree_w8

(the others from their recipe as it is, named after it). The specs on
them: `ShellPortsSpec` (each wrapper's ports are the template's for its
streamer hjson, plus the group pins, and Emit writes the three modules),
`ShellCasesSpec` (vecadd's and vsum's, through snax-gvsoc's shell cases,
below), `ShellEndToEndSpec` (the CSR writes of `main.c`, an always-ready
streamer: the goldens and the counter of `vectors.json`) and
`ShellRandomSpec`. `TwoGroupsSpec` elaborates `groups/dot1/` (C§17's dot 1:
groups `mul` and `sum`) and runs each group on its own.

## The unit cases

`UnitCasesSpec` drives a group controller with its datapath through
snax-gvsoc's `SnaxUnit` cases (`tests/snax/unit/cases/`, C§17): the start
line's JOBS and JOB_BEATS, every cycle's input valids and output ready, the
run lines and every result checked. A case whose L and II no catalogue
datapath has runs the controller alone. It reads them with `SnaxUnitCases`
from the folder `SNAX_GVSOC_UNIT_CASES` names, e.g.
`SNAX_GVSOC_UNIT_CASES=$PWD/gvsoc/snax-gvsoc/tests/snax/unit/cases pixi run -e hw
chisel-test`; without it the spec is canceled. CI's chisel job sets it to
the submodule's cases.

## The shell cases

`ShellCasesSpec` drives vecadd's and vsum's `<tag>_shell_wrapper` through
snax-gvsoc's `SnaxShell` cases (`tests/snax/shell/cases/`, C§17), which
snax-gvsoc's `tb.cpp` checks `snax_shell.hpp`'s rule against: every
access's result (a write's cycle, a read's value) and, every cycle, the
group's STATUS from the busy and done pins; a case with JOB_BEATS runs on
the fold. It reads them with `SnaxShellCases` from the folder
`SNAX_GVSOC_SHELL_CASES` names, e.g.
`SNAX_GVSOC_SHELL_CASES=$PWD/gvsoc/snax-gvsoc/tests/snax/shell/cases`; without
it the spec is canceled. CI's chisel job sets it beside the unit's.

## Tests

Every module has a spec; the random ones drive it with seeded random data,
configurations, Decoupled gaps and stalls (`Source` and `Sink` hold valid
and bits until fire) and compare it every cycle with a Scala model written
from the documented behaviour, never from the Chisel code:

| Spec | Random | Model |
|---|---|---|
| `NestValidSpec` | L ∈ {0, 1, 2}, II, J, T, input valids, output stalls; `out.ready` toggled within each cycle | `out.valid` unchanged by it (Conventions) |
| `NestCtrlSpec` | L ∈ {0, 1, 2} (L 2 with a two-register test datapath on the advance), II ∈ {1, 2}, T ∈ {1, 3, 4}, J ∈ {0, 1, 5}, input valids, output stalls | `NestModel`, SnaxUnit's rule (C§17) |
| `ElementwiseLanesSpec` | in, op and out types, lanes, every op, data | `ElementwiseModel`, NumPy's rule (D143) |
| `TreeAccumulateSpec` | in and out types, lanes, data, `fire` and `first` | lane sums from each `first`, each lane cast to the out type (the docstring) |
| `AccCsrRandomSpec` | 1 to 3 groups, folds or not; reads, writes and starts in and out of range, `rsp` and each start's ready, STATUS and COUNTER | the protocol in `AccCsr`'s docstring |
| `ShellRandomSpec` | per fixture: J, T, empty runs, data at its types, streamer gaps and stalls, polled or held starts | the datapath models; STATUS and COUNTER as C§17 states them |
| `FifoSlicerSpec`, `FifoPackerSpec` | types, words, lanes, data, input gaps, output stalls; the slicer into the packer | the element order of D142 (the docstrings) |

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
| `pixi run chisel-gen` | elaborate everything to `out/hw/*.sv`; `-- --describe` prints the catalogue, `-- --spec accelerator.json` builds an accelerator |
| `pixi run chisel-fmt` | scalafmt in place |
| `pixi run chisel-check` | fail if unformatted (CI uses this) |
| `pixi run chisel-clean` | drop build trees, keep the Coursier cache |

## The two archetypes

`ElementwiseLanes` and `TreeAccumulate` are the two distinct hardware
shapes the toolchain has to cover, the shapes of `vecadd` and `dot` in
`kernels/polybench/` among the staged kernels; both are data and enables
only, under the nest controller (D147):

**Elementwise (`ElementwiseLanes`, cf. `vecadd.py`).** A beat in, a beat
out, L = 0, II = 1, no pins. Stateless. Chains freely with a downstream
consumer, because each output element is produced in producer order and
read exactly once.

**Reduction (`TreeAccumulate`, cf. `dot.py`).** T beats in, 1 beat out
(JOB_BEATS = T); a beat of `lanes` elements goes through an adder tree in
front of the register, which `fire` and `first` load or add to; L = 1. It is
a **chaining barrier**: no output exists until the last input is absorbed,
so a downstream consumer cannot be fed element by element and gets control
chaining instead of data chaining.

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
  `valid` (that is how a join works). The reverse deadlocks. `NestValidSpec`
  tests it for the nest controller at every latency.
- **No internal storage, no control in a datapath.** Datapaths see data in
  flight only; address generation and buffering belong to the streamer, and
  valid, ready and counting to the nest controller. Reduction accumulators
  are the sanctioned exception to storage — they hold a scalar, not a tile.
- **Typed lanes, a `UInt` ALU.** A block's ports are lanes of NumPy's types
  (`IntType`: `SInt` signed, `UInt` unsigned), fixed at elaboration. Each
  operand is extended to the op type by its own signedness, the op runs on
  `UInt` bits at that width (two's complement wraps alike; min and max
  compare by the op type's signedness), and the result is extended or
  wrapped into the output type (D143).
- **Width and lane count come from the caller** (`accelerator.json` through
  the catalogue's fill rule), never from a default. The defaults in these
  constructors exist so the specs can run standalone.

## Not here yet

Elastic wrapper, skid buffer, typed integer ALU set and lane replication.
The bundle snax_cluster builds from (`Bender.yml`, the hjson) is LNK2b.
