package snax.forge.ctrl

import chisel3._
import chisel3.util.Decoupled
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, SnaxUnitCases}
import snax.forge.SnaxUnitCases.{Case, RunLine}
import snax.forge.datapath.{ElementwiseLanes, TreeAccumulate}
import snax.forge.elementwise.ElementwiseOp

/** A unit case's hardware: a group controller (`GroupCtrl`, its shell's `NestCtrl`) and the datapath of the case's
  * shape: two inputs at L 0 `ElementwiseLanes`, one input at L 1 `TreeAccumulate`, at the case's `types` (int64 without
  * them), lanes and op; any other shape (L 1 or 2 with two inputs) the controller alone, no data.
  */
class UnitHarness(val c: Case) extends Module {
  private val ctrl = Module(new GroupCtrl(c.latency, c.ii, c.inputs, "unit", "case"))

  private def t(i: Int): IntType = IntType(if (c.types.isEmpty) "int64" else c.types(i))

  val datapath: Option[String] =
    if (c.inputs == 2 && c.latency == 0) Some("ElementwiseLanes")
    else if (c.inputs == 1 && c.latency == 1) Some("TreeAccumulate")
    else None

  private val outLanes = if (c.inputs == 1) 1 else c.lanes

  val io = IO(new Bundle {
    val start = Flipped(Decoupled(new NestConfig))
    val in    = Vec(c.inputs, new Handshake)
    val out   = Flipped(new Handshake)
    val data  = Input(Vec(c.inputs, Vec(c.lanes, UInt(64.W))))
    val res   = Output(Vec(outLanes, UInt(64.W)))
    val fire  = Output(Bool())
    val busy  = Output(Bool())
  })

  ctrl.io.start <> io.start
  ctrl.io.in <> io.in
  ctrl.io.out <> io.out
  io.fire := ctrl.io.fire
  io.busy := ctrl.io.busy
  io.res  := DontCare

  private def lane(x: UInt, ty: IntType): Bits = x(ty.width - 1, 0).asTypeOf(ty.lane)

  datapath match {
    case Some("ElementwiseLanes") =>
      val op     = ElementwiseOp.names.collectFirst { case (k, n) if n == c.op => k }.get
      val opType = snax.forge.elementwise.ElementwiseModel.resultType(t(0), t(1))
      val dp     = Module(new ElementwiseLanes(Seq(t(0).name, t(1).name), opType.name, t(2).name, c.lanes, op))
      for (l <- 0 until c.lanes) {
        dp.io.a(l) := lane(io.data(0)(l), t(0))
        dp.io.b(l) := lane(io.data(1)(l), t(1))
        io.res(l)  := dp.io.out(l).asUInt
      }
    case Some("TreeAccumulate")   =>
      val dp = Module(new TreeAccumulate(t(0).name, t(1).name, c.lanes))
      for (l <- 0 until c.lanes) dp.io.in(l) := lane(io.data(0)(l), t(0))
      dp.io.fire  := ctrl.io.fire
      dp.io.first := ctrl.io.first
      io.res(0)   := dp.io.out.asUInt
    case _                        =>
  }
}

/** The unit cross-check: every case of snax-gvsoc's `SnaxUnit` cases (`SnaxUnitCases`, read from
  * `SNAX_GVSOC_UNIT_CASES`) driven through `UnitHarness`, a group controller with its datapath.
  *
  * Cycle by cycle: each input's valid and the output's ready as the case line gives them, a start offered in the cycle
  * the case offers it, and fire, push (`out.valid && out.ready`) and busy compared every cycle. Each start's answer:
  * `refused` is the controller not ready (busy), `ignored` a start it takes as the empty run (JOBS or JOB_BEATS 0),
  * which the group ends without its shell (busy stays 0), `taken` the rest. With `types`, the N-th firing's input is
  * the case's `beat N` and the N-th push must be its `result N` (each lane at its type's width). The `expect run` lines
  * too: each busy cycle classed by `SnaxUnit`'s rule (D59) from what the hardware shows: a result waiting with no room
  * (`out.valid && !out.ready` at L >= 1; at L = 0 a firing that would give one) stall_out, a firing or the II gap after
  * one busy, no input with firings left stall_in, the rest flush. Without the variable the test is canceled.
  */
class UnitCasesSpec extends AnyFlatSpec with ForgeTester {

  private def high(b: Bool): Boolean = b.peek().litToBoolean

  behavior of "a group controller with its datapath against SnaxUnit's cases"

  private def run(c: Case): Unit =
    test(new UnitHarness(c)) { dut =>
      var runs     = Vector.empty[RunLine]
      var count    = 0L                // the run's firings
      var t        = 1L
      var firings  = 0L                // the run's firings so far
      var hasFired = false
      var lastFire = 0
      var classes  = Array(0, 0, 0, 0) // busy, stall_in, stall_out, flush
      var fired    = 0                 // the case's firings, for `beat N`
      var pushed   = 0                 // the case's pushes, for `result N`
      var wasBusy  = false
      def endRun(): Unit = runs :+= RunLine(firings.toInt, classes(0), classes(1), classes(2), classes(3))
      for ((cyc, i) <- c.cycles.zipWithIndex) {
        val clue = s"${c.name}, cycle $i"
        val busy = high(dut.io.busy)
        if (wasBusy && !busy) endRun()
        wasBusy = busy
        c.starts.get(i) match {
          case Some(s) =>
            dut.io.start.valid.poke(true.B)
            dut.io.start.bits.jobs.poke(s.jobs.U)
            dut.io.start.bits.jobBeats.poke(s.jobBeats.U)
            val got =
              if (!high(dut.io.start.ready)) "refused" else if (s.jobs == 0 || s.jobBeats == 0) "ignored" else "taken"
            assert(got == s.answer, s"$clue: start ${s.jobs} x ${s.jobBeats} $got, the case ${s.answer}")
            if (got == "taken") {
              count = s.jobs.toLong * s.jobBeats; t = s.jobBeats; firings = 0; classes = Array(0, 0, 0, 0)
            }
          case None    => dut.io.start.valid.poke(false.B)
        }
        cyc.valids.zipWithIndex.foreach { case (v, k) => dut.io.in(k).valid.poke(v.B) }
        dut.io.out.ready.poke(cyc.ready.B)
        for {
          k <- 0 until c.inputs
          l <- 0 until c.lanes
        } {
          val v = c.beats.get(fired).map(_(k * c.lanes + l)).getOrElse(BigInt(0))
          dut.io.data(k)(l).poke((v & LaneBits.mask(64)).U)
        }
        val fire = high(dut.io.fire)
        val outValid = high(dut.io.out.valid)
        val push     = outValid && cyc.ready
        assert(
          (fire, push, busy) == (cyc.fire, cyc.push, cyc.busy),
          s"$clue: fire $fire push $push busy $busy, the case ${cyc.fire} ${cyc.push} ${cyc.busy}"
        )
        if (busy) {
          val allIn    = cyc.valids.forall(identity)
          val iiOk     = !hasFired || i >= lastFire + c.ii
          val gives    = (firings + 1)   % t == 0
          val stallOut =
            if (c.latency > 0) outValid && !cyc.ready else firings < count && iiOk && allIn && gives && !cyc.ready
          val k        =
            if (stallOut) 2
            else if (fire || (hasFired && i < lastFire + c.ii)) 0
            else if (firings < count && iiOk && !allIn) 1
            else 3
          classes(k) += 1
        }
        if (push && c.types.nonEmpty) {
          val out  = IntType(c.types.last)
          val want = c.results.getOrElse(pushed, fail(s"$clue: push $pushed has no `result` line"))
          for ((w, l) <- want.zipWithIndex) {
            val got = dut.io.res(l).peek().litValue & LaneBits.mask(out.width)
            assert(got == (w & LaneBits.mask(out.width)), s"$clue: push $pushed lane $l is $got, the case $w")
          }
        }
        if (push) pushed += 1
        if (fire) { firings += 1; fired += 1; hasFired = true; lastFire = i }
        dut.clock.step()
      }
      if (wasBusy) endRun()
      assert(runs == c.runs, s"${c.name}: the runs ${runs.mkString(", ")}, the case ${c.runs.mkString(", ")}")
      if (c.types.nonEmpty) assert(pushed == c.results.size, s"${c.name}: $pushed pushes, ${c.results.size} results")
    }

  SnaxUnitCases.all match {
    case None      =>
      it should "give every SnaxUnit case's cycles, runs and results" in
        cancel(s"${SnaxUnitCases.Env} is not set (snax-gvsoc's tests/snax/unit/cases)")
    case Some(all) =>
      it should "find SnaxUnit's cases" in assert(all.nonEmpty, s"no case in ${sys.env(SnaxUnitCases.Env)}")
      for (c <- all) it should s"give ${c.name}'s cycles, runs and results" in run(c)
  }
}
