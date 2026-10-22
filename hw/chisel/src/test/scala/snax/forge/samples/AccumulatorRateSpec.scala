package snax.forge.samples

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{Blocks, SnaxUnitCases}
import snax.forge.SnaxUnitCases.{RunLine, Start}
import snax.forge.shell.RtlFixtures

/** The fold half of the unit cross-check: the Accumulator driven through snax-gvsoc's `SnaxUnit` rate cases.
  *
  * Every case that is a pipelined fold with a rate (`inputs 1`, `config` holding `rate`; `SnaxUnitCases`, read from
  * `SNAX_GVSOC_UNIT_CASES`) is the Accumulator at L = 1, II 1, `lanes` the case's lanes (1 without the line),
  * `loopCount` the start's count and `len` its rate. Cycle by cycle: `in.valid` and `out.ready` as the case line gives
  * them, a start poked in the cycle the case offers it, and fire (`in.fire`), push (`out.fire`) and busy compared every
  * cycle; each start's answer against the block's state (`refused` busy, `ignored` idle with count or rate 0, `taken`
  * otherwise). Each pushed sum is checked against the inputs it folds (lane j of a case's k-th input, from 0, is k *
  * lanes + j + 1, at the block's lanes). A cases folder with no such case (an older snax-gvsoc checkout) cancels the
  * test too.
  *
  * The `expect run` lines are compared too, each busy cycle classed by `SnaxUnit`'s rule (D59) from what the block
  * shows: a result waiting with no room (`out.valid && !out.ready`) is stall_out, an input taken busy, no input with
  * inputs left stall_in, and the rest flush. Without the variable the test is canceled. It runs at 64 bits and the
  * case's lanes, and at each `RtlFixtures` fixture's Accumulator (vsum's, vsum_tree's at W = 4 and 8): the cycles do
  * not depend on the lanes.
  */
class AccumulatorRateSpec extends AnyFlatSpec with ChiselScalatestTester {

  private val Env = SnaxUnitCases.Env

  private def cases =
    SnaxUnitCases.all.map(_.filter(c => c.waitKind == "pipelined" && c.inputs == 1 && c.config.contains("rate")))

  private def high(b: Bool): Boolean = b.peek().litToBoolean

  behavior of "Accumulator against SnaxUnit's rate cases"

  /** Each fixture's Accumulator (GEN1, `RtlFixtures`): its width, accWidth and lanes. */
  private val fixtures: Seq[(String, Int, Int, Int)] =
    RtlFixtures.all.map(f => f -> f.spec.units.head).collect {
      case (f, u) if u.block == "Accumulator" =>
        val p = Blocks.Params(u.params)
        (f.name, p.int("width"), p.int("accWidth"), p.int("lanes"))
    }

  /** The configurations: 64 bits at the case's lanes, then each fixture's. */
  private val configs: Seq[(String, Option[(Int, Int, Int)])] =
    ("w64, the case's lanes", None) +: fixtures.map { case (n, w, a, l) => n -> Some((w, a, l)) }

  for ((config, params) <- configs)
    it should s"fire, push, be busy and end its runs as every pipelined fold case gives ($config)" in {
      runCases(params)
    }

  private def runCases(params: Option[(Int, Int, Int)]): Unit = {
    val all = cases.getOrElse(cancel(s"$Env is not set (snax-gvsoc's tests/snax/unit/cases)"))
    if (all.isEmpty) cancel(s"no pipelined fold case in ${sys.env(Env)}: the snax-gvsoc checkout predates GEN3 step 6b")
    for (c <- all) {
      require((c.latency, c.ii) == (1, 1), s"${c.name}: the Accumulator is L = 1, II 1")
      val (width, accWidth, lanes) = params.getOrElse((64, 64, c.lanes))
      test(new Accumulator(width = width, accWidth = accWidth, lanes = lanes)) { dut =>
        var next    = BigInt(0)         // the index of the next input
        var count   = 0                 // the run's count
        var rate    = 1
        var taken   = 0                 // inputs taken this run
        var group   = Seq.empty[BigInt] // inputs of the sum in progress
        var sums    = Seq.empty[BigInt] // sums given, not yet pushed
        var classes = RunLine(0, 0, 0, 0, 0)
        var runs    = Seq.empty[RunLine]
        for ((cyc, i) <- c.cycles.zipWithIndex) {
          val clue   = s"${c.name}, cycle $i"
          val busy   = high(dut.io.busy)
          c.starts.get(i) match {
            case Some(st @ Start(_, answer)) =>
              val (n, t) = (c.word(st, "count", 0), c.word(st, "rate", 1))
              val got    = if (busy) "refused" else if (n == 0 || t == 0) "ignored" else "taken"
              assert(got == answer, s"$clue: start $n $t $got, the case says $answer")
              if (got == "taken") { count = n; rate = t; taken = 0 }
              dut.io.start.poke(true.B)
              dut.io.loopCount.poke(n.U)
              dut.io.len.poke(t.U)
            case None                        =>
              dut.io.start.poke(false.B)
          }
          dut.io.in.valid.poke(cyc.valids(0).B)
          val inputs = (0 until lanes).map(j => next * lanes + j + 1)
          for ((v, j) <- inputs.zipWithIndex) dut.io.in.bits(j).poke(v.U)
          dut.io.out.ready.poke(cyc.ready.B)
          val fire = high(dut.io.in.valid)  && high(dut.io.in.ready)
          val push = high(dut.io.out.valid) && cyc.ready
          assert(fire == cyc.fire, s"$clue: fire $fire, the case says ${cyc.fire}")
          assert(push == cyc.push, s"$clue: push $push, the case says ${cyc.push}")
          assert(busy == cyc.busy, s"$clue: busy $busy, the case says ${cyc.busy}")
          if (push) {
            assert(sums.nonEmpty, s"$clue: a push with no sum given")
            dut.io.out.bits.expect(sums.head.U, s"$clue: the sum")
            sums = sums.tail
          }
          if (busy) {
            val frozen = high(dut.io.out.valid) && !cyc.ready
            classes =
              if (frozen) classes.copy(stallOut = classes.stallOut + 1)
              else if (fire) classes.copy(firings = classes.firings + 1, busy = classes.busy + 1)
              else if (taken < count) classes.copy(stallIn = classes.stallIn + 1)
              else classes.copy(flush                      = classes.flush + 1)
          }
          if (fire) {
            group :+= inputs.sum
            next += 1
            taken += 1
            if (group.size == rate || taken == count) { sums :+= group.sum; group = Seq.empty }
          }
          if (high(dut.io.done)) { runs :+= classes; classes = RunLine(0, 0, 0, 0, 0) }
          dut.clock.step()
        }
        assert(sums.isEmpty, s"${c.name}: sums given but never pushed: $sums")
        assert(runs == c.runs, s"${c.name}: runs $runs, the case says ${c.runs}")
      }
    }
  }
}
