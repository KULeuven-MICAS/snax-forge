package snax.forge.ctrl

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Seeds}

/** `NestCtrl`'s `out.valid` never depends on `out.ready` (hw/chisel/README, Conventions), at L 0 (where a firing that
  * gives pushes in its own cycle) and at L 1 and 2. Per seed and configuration, runs of random J and T back to back
  * under random input valids and output stalls; in every cycle, in whatever state the run is in, `out.ready` is poked
  * low and then high before the cycle's own value, and `out.valid` must read the same both times.
  */
class NestValidSpec extends AnyFlatSpec with ForgeTester {

  private val Cycles = 400

  behavior of "NestCtrl's out.valid"

  for {
    latency <- Seq(0, 1, 2)
    ii      <- Seq(1, 2)
    seed    <- Seeds()
  }
    it should s"not change with out.ready within a cycle (L $latency, II $ii, seed $seed)" in Seeds.run(seed) { r =>
      val inputs = r.pick(Seq(1, 2))
      test(new NestUnderTest(latency, ii, inputs)) { dut =>
        val gap   = r.probability()
        val stall = r.probability()
        var seen  = 0 // cycles in which out.valid was high
        for (c <- 0 until Cycles) {
          val idle = dut.io.config.ready.peek().litToBoolean
          dut.io.config.valid.poke((idle && r.chance(0.5)).B)
          dut.io.config.bits.jobs.poke(r.pick(Seq(1, 2, 5)).U)
          dut.io.config.bits.jobBeats.poke(r.pick(Seq(1, 3, 4)).U)
          for (k <- 0 until inputs) dut.io.in(k).valid.poke((!r.chance(gap)).B)
          val clue = s"cycle $c, inputs $inputs, gap $gap, stall $stall"
          dut.io.out.ready.poke(false.B)
          val low  = dut.io.out.valid.peek().litToBoolean
          dut.io.out.ready.poke(true.B)
          val high = dut.io.out.valid.peek().litToBoolean
          assert(low == high, s"$clue: out.valid $low with out.ready low, $high with it high")
          if (high) seen += 1
          dut.io.out.ready.poke((!r.chance(stall)).B)
          dut.clock.step()
        }
        assert(seen > 0, "out.valid never high: the runs did not reach a result")
      }
    }
}
