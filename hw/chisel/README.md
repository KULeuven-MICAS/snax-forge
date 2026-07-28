# SNAX-FORGE Chisel datapaths

Hardware side of SNAX-FORGE. Today this is a hand-written primitive library
with a smoke test; by W8 the modules here are instantiated and parameterised
from an Accelerator Descriptor emitted by the SDFG side.

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
└── src/
    ├── main/scala/snax/forge/
    │   ├── SimpleAdder.scala           # elementwise archetype  (cf. vecadd)
    │   ├── Accumulator.scala           # reduction archetype    (cf. dot)
    │   └── Emit.scala                  # elaboration entry point
    └── test/scala/snax/forge/
        ├── SimpleAdderSpec.scala
        └── AccumulatorSpec.scala
```

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
| `pixi run chisel-gen` | elaborate everything to `out/hw/*.sv` |
| `pixi run chisel-fmt` | scalafmt in place |
| `pixi run chisel-check` | fail if unformatted (CI uses this) |
| `pixi run chisel-clean` | drop build trees, keep the Coursier cache |

## The two archetypes

`SimpleAdder` and `Accumulator` are not arbitrary toys. They are the two
distinct hardware shapes the toolchain has to cover, and the two kernels
already sitting in `kernels/polybench/`:

**Elementwise (`SimpleAdder`, cf. `vecadd.py`).** N beats in, N beats out,
L = 0, II = 1. Stateless. Chains freely with a downstream consumer, because
each output element is produced in producer order and read exactly once.

**Reduction (`Accumulator`, cf. `dot.py`).** N beats in, 1 beat out. Carries
state across beats. It is a **chaining barrier**: no output exists until the
last input is absorbed, so a downstream consumer cannot be fed element by
element and gets control chaining instead of data chaining.

Everything on the W11–W14 path is a composition of these two plus a typed ALU:
dot is elementwise-multiply feeding a reduction, softmax is three passes with
two reductions and a barrier between each.

## Conventions the generated code will have to follow

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
- **Width and lane count come from the SDFG**, never from a default. The
  defaults in these constructors exist so the smoke test can run standalone.

## Not here yet

Elastic wrapper, skid buffer, typed integer ALU set and lane replication are
W6. The descriptor-driven emitter replaces the hand-written body of
`Emit.scala` in W8. Nothing in this directory imports anything from
`snax_forge/` yet, and that is intentional — the two halves are developed
independently until the descriptor schema freezes in W5.