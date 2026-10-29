package snax.forge.ctrl

import chisel3._
import chisel3.util._

/** A start as `<tag>_csr` sends it (C§17): JOBS, then each JOB_BEATS in unit order, `jobBeats` the first (1 when the
  * group has none), `beats` the rest (`more` of them).
  */
class NestConfig(val more: Int = 0) extends Bundle {
  val jobs     = UInt(32.W)
  val jobBeats = UInt(32.W)
  val beats    = if (more > 0) Some(Vec(more, UInt(32.W))) else None
}

/** A Decoupled port split in two: its data goes to the datapath, its valid and ready to the controller. */
class Handshake extends Bundle {
  val valid = Input(Bool())
  val ready = Output(Bool())
}

/** B1's nest controller (D147), SnaxUnitRule (snax-gvsoc `snax/snax_group.hpp`): the one-unit case of `GroupRule`, the
  * control model of a group, with B1's ports, which `NestCtrlSpec` and `NestValidSpec` drive. A start of J jobs of T
  * beats fires J * T times and gives a result on every T-th firing; J or T 0 is the empty run (taken, nothing runs).
  * Busy from the cycle after the start to the run's last cycle E, the J-th push, where `done` is high. Input k is ready
  * in the cycle the unit fires, which needs every input's valid, so a valid never depends on a ready (C§17's per-unit
  * join). At most one firing per II cycles, counted from the last firing whatever stalls. L = 0: a firing that gives
  * pushes in its own cycle, `out.valid` that firing's condition without the output's ready; L >= 1: an L-deep valid
  * register that stalls as a whole, `out.valid` its head. To the datapath: `fire`, `first` (a job's first beat) and
  * `adv` (its pipeline advances; L = 0: always).
  *
  * @param latency
  *   L, the datapath's registers from a firing to its result
  * @param ii
  *   the least cycles between two firings
  * @param inputs
  *   the inputs joined
  */
class NestCtrl(val latency: Int, val ii: Int, val inputs: Int) extends Module {
  require(inputs >= 1, s"inputs must be 1 or more, got $inputs")

  override def desiredName: String = s"NestCtrl_l${latency}_ii${ii}_in${inputs}"

  val io = IO(new Bundle {
    val config = Flipped(Decoupled(new NestConfig))
    val in     = Vec(inputs, new Handshake)
    val out    = Flipped(new Handshake)
    val fire   = Output(Bool())
    val first  = Output(Bool())
    val adv    = Output(Bool())
    val busy   = Output(Bool())
    val done   = Output(Bool())
  })

  private val nest = new GroupRule(GroupDesc("unit", Seq(UnitDesc("unit", latency, ii, 1, inputs, None)), Nil))
  nest.io.config.valid      := io.config.valid
  nest.io.config.bits.jobs  := io.config.bits.jobs
  nest.io.config.bits.ts(0) := io.config.bits.jobBeats
  io.config.ready           := nest.io.config.ready
  for ((r, p) <- nest.io.in(0).zip(io.in)) {
    r.valid := p.valid
    p.ready := r.ready
  }
  nest.io.out.ready := io.out.ready
  io.out.valid := nest.io.out.valid
  io.fire      := nest.io.units(0).fire
  io.first     := nest.io.units(0).first
  io.adv       := (if (latency == 0) true.B else nest.io.units(0).adv)
  io.busy      := nest.io.busy
  io.done      := nest.io.done
}
