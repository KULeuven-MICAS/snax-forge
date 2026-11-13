package snax.forge.datapath

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds}
import snax.forge.elementwise.ElementwiseModel

/** `LaneAccumulate` against a Scala model written from its documented behaviour (CONTRACTS, "Accelerator RTL spec"):
  * per lane x = cast_out(a), s ← first ? x : s + x on fire, wrapping at outType, held otherwise; `out` is s. rowsum 2's
  * worked example, then seeded random jobs at 1, 4 and 8 lanes, int32 and int64 in (an int64 sum from int32 too): each
  * job T beats, T random, its fires spread over random gaps (the control unit's stalls), every cycle's `out` the
  * model's and each job's sums its beats' after its last fire.
  */
class LaneAccumulateSpec extends AnyFlatSpec with ForgeTester {

  behavior of "LaneAccumulate"

  it should "sum each lane down its row: rows (1, 2, 3), (4, 5, 6) then (7, 8, 9), (10, 11, 12) give 6, 15, 24, 33" in {
    test(new LaneAccumulate("int64", "int64", 2)) { dut =>
      val cols = Seq(Seq(1, 4), Seq(2, 5), Seq(3, 6), Seq(7, 10), Seq(8, 11), Seq(9, 12))
      for ((col, k) <- cols.zipWithIndex) {
        dut.io.fire.poke(true.B)
        dut.io.first.poke((k % 3 == 0).B)
        for ((v, l) <- col.zipWithIndex) LaneBits.poke(dut.io.a(l), v)
        dut.clock.step()
        if (k % 3 == 2) {
          val want = if (k == 2) Seq(6, 15) else Seq(24, 33)
          assert(dut.io.out.map(x => LaneBits.signed(LaneBits.peek(x), 64)) == want.map(BigInt(_)))
        }
      }
    }
  }

  it should "sign-extend int32 lanes into an int64 sum and wrap an int32 one" in {
    for ((out, want) <- Seq("int64" -> BigInt(2147483648L), "int32" -> BigInt(Int.MinValue))) {
      test(new LaneAccumulate("int32", out, 1)) { dut =>
        for ((v, k) <- Seq(Int.MaxValue, 2, -1).zipWithIndex) {
          dut.io.fire.poke(true.B); dut.io.first.poke((k == 0).B)
          LaneBits.poke(dut.io.a(0), v)
          dut.clock.step()
        }
        dut.io.fire.poke(false.B)
        assert(LaneBits.signed(LaneBits.peek(dut.io.out(0)), IntType(out).width) == want, out)
      }
    }
  }

  it should "refuse an outType narrower than inType" in {
    val e = intercept[Exception](circt.stage.ChiselStage.emitCHIRRTL(new LaneAccumulate("int64", "int32", 4)))
    assert(e.getMessage.contains("outType int32 must be at least as wide as inType int64"), e.getMessage)
  }

  behavior of "LaneAccumulate under random jobs and stalls"

  for (seed <- Seeds())
    it should s"give the model's sums (seed $seed)" in Seeds.run(seed) { r =>
      for {
        lanes       <- Seq(1, 4, 8)
        (tin, tout) <- Seq("int32" -> "int32", "int32" -> "int64", "int64" -> "int64")
      } {
        val (ti, to) = (IntType(tin), IntType(tout))
        val mask     = LaneBits.mask(to.width)
        val stall    = r.probability()
        val clue     = s"$tin -> $tout, $lanes lanes, stalls $stall"
        test(new LaneAccumulate(tin, tout, lanes)) { dut =>
          var s = Seq.fill(lanes)(BigInt(0))
          var c = 0
          def check(what: String): Unit =
            for (l <- 0 until lanes) assert(LaneBits.peek(dut.io.out(l)) == s(l), s"$clue, cycle $c, lane $l: $what")
          for (job <- 0 until 4) {
            val t   = r.between(1, 9)
            var sum = Seq.fill(lanes)(BigInt(0))
            var k   = 0
            while (k < t) {
              val a    = Seq.fill(lanes)(r.word(ti.width))
              val fire = !r.chance(stall)
              a.zipWithIndex.foreach { case (v, l) => LaneBits.poke(dut.io.a(l), v) }
              dut.io.fire.poke(fire.B)
              dut.io.first.poke((fire && k == 0).B)
              if (c > 0) check("held or taken")
              if (fire) {
                val x = a.map(ElementwiseModel.cast(_, ti, to))
                s   = if (k == 0) x else s.zip(x).map { case (p, q) => (p + q) & mask }
                sum = if (k == 0) x else sum.zip(x).map { case (p, q) => (p + q) & mask }
                k += 1
              }
              dut.clock.step(); c += 1
            }
            dut.io.fire.poke(false.B)
            for (l <- 0 until lanes)
              assert(LaneBits.peek(dut.io.out(l)) == sum(l), s"$clue, job $job (T $t), lane $l: the job's sum")
          }
        }
      }
    }
}
