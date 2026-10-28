package snax.forge.ctrl

import chisel3._
import chisel3.util._

/** A start of one shell's nest: J jobs of T beats each (D146). */
class NestConfig extends Bundle {
  val jobs     = UInt(32.W)
  val jobBeats = UInt(32.W)
}

/** A Decoupled port split in two: its data goes to the datapath, its valid and ready to the controller. */
class Handshake extends Bundle {
  val valid = Input(Bool())
  val ready = Output(Bool())
}

/** The nest controller of one shell (D147): SnaxUnit's rule (snax-gvsoc `snax/snax_unit.hpp`) in Chisel, over a
  * datapath of data and enables only. In B1 a shell is one unit, so the nest has two digits, JOBS outermost and, for a
  * fold, the job's beat counter (T, JOB_BEATS); an elementwise unit has T = 1.
  *
  *   - `config` is a start, ready when idle. A start of J jobs of T beats fires J * T times and gives a result on every
  *     T-th firing (`gives`, the job's last beat). J or T 0 is the empty run, which the group controller ends without
  *     starting this one; offered here anyway, it is taken and nothing runs.
  *   - Busy from the cycle after the start to the run's last cycle E, the J-th push; `done` is high in E.
  *   - The inputs' valids are joined: input k's ready is the others' valids and whether the controller fires, so a
  *     valid never depends on a ready.
  *   - At most one firing per II cycles, counted from the last firing whatever stalls (it carries over to the next
  *     run).
  *   - L = 0: a firing that gives a result needs the output's ready and pushes it in its own cycle (`out.valid` is
  *     `fire && gives`); L = 0, II 1 is lockstep. L >= 1: an L-deep valid register `v`, which stalls as a whole: it
  *     advances (`adv`) when its head `v(L-1)` is empty or the output is ready, a firing needs `adv`, and `out.valid`
  *     is the head. A firing that gives a result enters `v(0)`; the others enter nothing.
  *
  * To the datapath: `fire` (a beat is taken from every input), `first` (the firing is a job's first beat: a fold starts
  * its sum) and `adv` (its pipeline registers advance; L = 0: always). The datapath has no valid, ready or counter.
  *
  * @param latency
  *   L, the datapath's registers from a firing to its result
  * @param ii
  *   the least cycles between two firings
  * @param inputs
  *   the inputs joined
  */
class NestCtrl(val latency: Int, val ii: Int, val inputs: Int) extends Module {
  require(latency >= 0, s"latency must be 0 or more, got $latency")
  require(ii      >= 1, s"ii must be 1 or more, got $ii")
  require(inputs  >= 1, s"inputs must be 1 or more, got $inputs")

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

  private val busy     = RegInit(false.B)
  private val jobs     = RegInit(0.U(32.W))
  private val jobBeats = RegInit(1.U(32.W))
  private val beat     = RegInit(0.U(32.W))               // the beat of the job being fired
  private val fired    = RegInit(0.U(32.W))               // jobs whose last beat has fired
  private val pushed   = RegInit(0.U(32.W))               // results pushed
  private val gap      = RegInit(0.U(log2Ceil(ii + 1).W)) // cycles until II allows a firing
  private val v        = RegInit(VecInit(Seq.fill(latency max 1)(false.B)))

  private val gives = beat === jobBeats - 1.U
  private val room  =
    if (latency == 0) !gives || io.out.ready
    else !v(latency - 1) || io.out.ready
  private val armed = busy  && fired =/= jobs && gap === 0.U && room
  private val allIn = io.in.map(_.valid).reduce(_ && _)
  private val fire  = armed && allIn

  for ((p, k) <- io.in.zipWithIndex)
    p.ready := armed && io.in.zipWithIndex.filter(_._2 != k).map(_._1.valid).foldLeft(true.B)(_ && _)

  io.out.valid := (if (latency == 0) fire && gives else v(latency - 1))
  private val push = io.out.valid && io.out.ready

  io.fire  := fire
  io.first := beat === 0.U
  io.adv   := (if (latency == 0) true.B else room)

  private val done = busy && push && pushed === jobs - 1.U
  io.done := done
  io.busy := busy

  io.config.ready := !busy

  when(fire) {
    gap := (ii - 1).U
  }.elsewhen(gap =/= 0.U) {
    gap := gap - 1.U
  }

  if (latency > 0) when(room) {
    for (i <- latency - 1 until 0 by -1) v(i) := v(i - 1)
    v(0) := fire && gives
  }

  when(io.config.fire) {
    busy     := io.config.bits.jobs =/= 0.U && io.config.bits.jobBeats =/= 0.U
    jobs     := io.config.bits.jobs
    jobBeats := io.config.bits.jobBeats
    beat     := 0.U
    fired    := 0.U
    pushed   := 0.U
  }.otherwise {
    when(fire) {
      beat := Mux(gives, 0.U, beat + 1.U)
      when(gives) { fired := fired + 1.U }
    }
    when(push) { pushed := pushed + 1.U }
    when(done) { busy := false.B }
  }
}
