package snax.forge.datapath

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds}
import snax.forge.elementwise.ElementwiseModel

/** `SerialAccumulate` (D149) against a Scala model written from C§17: on read, acc ← (first ∧ lane = 0) ?
  * cast_out(x_lane) : acc + cast_out(x_lane), `out` acc. Random beats, lanes and reads at three sizes (2, 4, 8 lanes; 1
  * too, whose lane pin is 0 bits), int64 and int32 into themselves or int64, signed and unsigned; the partials (-13, 6)
  * of snax-gvsoc's vectors: -13, then -7; and jobs of a run-time T (1 to 4 beats) of random int64 at 1, 2, 4 and 8
  * lanes, read lane by lane as its control unit reads them (`first` high on a job's first beat, so high and low at lane
  * 0, idle cycles between reads), each job's sum against the sum of its elements.
  */
class SerialAccumulateSpec extends AnyFlatSpec with ForgeTester {

  behavior of "SerialAccumulate"

  it should "sum the beat (-13, 6) a lane a cycle: -13, then -7" in {
    test(new SerialAccumulate("int64", "int64", 2)) { dut =>
      LaneBits.poke(dut.io.in(0), -13); LaneBits.poke(dut.io.in(1), 6)
      dut.io.read.poke(true.B); dut.io.first.poke(true.B)
      for ((want, lane) <- Seq(-13, -7).zipWithIndex) {
        dut.io.lane.poke(lane.U)
        dut.clock.step()
        assert(LaneBits.signed(LaneBits.peek(dut.io.out), 64) == want)
      }
    }
  }

  for (seed <- Seeds())
    it should s"sum jobs of a run-time T over several beats (seed $seed)" in Seeds.run(seed) { r =>
      for (lanes <- Seq(1, 2, 4, 8)) {
        test(new SerialAccumulate("int64", "int64", lanes)) { dut =>
          for (job <- 0 until 6) {
            val t     = r.between(1, 4)
            val beats = Seq.fill(t)(Seq.fill(lanes)(r.word(64)))
            for {
              (beat, b) <- beats.zipWithIndex
              lane      <- 0 until lanes
            } {
              while (r.chance(0.2)) { dut.io.read.poke(false.B); dut.clock.step() }
              beat.zipWithIndex.foreach { case (v, l) => LaneBits.poke(dut.io.in(l), v) }
              dut.io.read.poke(true.B); dut.io.lane.poke(lane.U); dut.io.first.poke((b == 0).B)
              dut.clock.step()
            }
            dut.io.read.poke(false.B)
            val want = beats.flatten.map(LaneBits.signed(_, 64)).sum
            assert(
              LaneBits.signed(LaneBits.peek(dut.io.out), 64) == LaneBits.signed(want & LaneBits.mask(64), 64),
              s"int64, $lanes lanes, job $job of $t beats"
            )
          }
        }
      }
    }

  for (seed <- Seeds())
    it should s"give the model's sums under random reads (seed $seed)" in Seeds.run(seed) { r =>
      for (lanes <- Seq(1, 2, 4, 8)) {
        val ti   = IntType(r.pick(Seq("int64", "uint64", "int32", "uint32")))
        val to   = IntType(if (ti.width == 64) ti.name else r.pick(Seq(ti.name, "int64")))
        val clue = s"${ti.name} -> ${to.name}, $lanes lanes"
        test(new SerialAccumulate(ti.name, to.name, lanes)) { dut =>
          var acc = BigInt(0)
          for (c <- 0 until 32) {
            val beat  = Seq.fill(lanes)(r.word(ti.width))
            val read  = r.chance(0.8)
            val lane  = r.between(0, lanes - 1)
            val first = r.chance(0.3)
            beat.zipWithIndex.foreach { case (v, l) => LaneBits.poke(dut.io.in(l), v) }
            dut.io.read.poke(read.B); dut.io.lane.poke(lane.U); dut.io.first.poke(first.B)
            assert(LaneBits.peek(dut.io.out) == acc, s"$clue, cycle $c")
            val x     = ElementwiseModel.cast(beat(lane), ti, to)
            if (read) acc = if (first && lane == 0) x else (acc + x) & LaneBits.mask(to.width)
            dut.clock.step()
          }
        }
      }
    }
}
