package snax.forge.stream

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Rand, Seeds, Sink, Source}

/** `FifoSlicer` against a model written from its docstring: an in-beat of `words` words is the bits of its elements,
  * element e at bits `[w(e+1)-1 : we]`, and pop u of it is elements u * outLanes until (u + 1) * outLanes. Timing with
  * the input always valid and the output always ready, the worked values of E1 and E2, and random configurations under
  * random input gaps and output stalls.
  */
class FifoSlicerSpec extends AnyFlatSpec with ForgeTester {

  /** The model: an in-beat's pops, each its lanes' bits packed (lane 0 at the low end). */
  def pops(beat: Seq[BigInt], laneType: String, outLanes: Int): Seq[BigInt] = {
    val w    = IntType(laneType).width
    val flat = LaneBits.pack(beat, 64)
    val r    = beat.size * (64 / w) / outLanes
    (0 until r).map(u => (flat >> (u * outLanes * w)) & LaneBits.mask(outLanes * w))
  }

  /** Feed `beats` through the slicer with `in` valid unless `gap` says not and `out` ready unless `stall` says not; the
    * pops in order, and the cycles from the first in-beat taken to the last pop, both included.
    */
  def drive(dut: FifoSlicer, beats: Seq[Seq[BigInt]], gap: Double, stall: Double, r: Rand): (Seq[BigInt], Int) = {
    val src     = Source.random(beats.map(LaneBits.pack(_, 64)), gap, r)
    val snk     = Sink.random(stall, r)
    val want    = beats.size * dut.r
    var cycle   = 0
    var first   = -1
    var lastOut = -1
    while (snk.got.size < want) {
      assert(cycle < 100 * want + 100, s"no end after $cycle cycles (${snk.got.size} pops)")
      src.cycle(cycle)
      dut.io.in.valid.poke(src.valid.B)
      LaneBits.pokeBeat(dut.io.in.bits, src.bits)
      dut.io.out.ready.poke(snk.cycle(cycle).B)
      if (src.fired(dut.io.in.ready.peek().litToBoolean) && first < 0) first = cycle
      if (snk.fired(dut.io.out.valid.peek().litToBoolean, LaneBits.peekBeat(dut.io.out.bits))) lastOut = cycle
      dut.clock.step()
      cycle += 1
    }
    (snk.got, lastOut - first + 1)
  }

  behavior of "FifoSlicer"

  for ((laneType, r) <- Seq("int32" -> 2, "int8" -> 8))
    it should s"pop 4 in-beats at r = $r in ${4 * r + 1} cycles from the first input to the last output" in {
      test(new FifoSlicer(1, laneType, 1)) { dut =>
        assert(dut.r == r)
        val beats         = Seq.tabulate(4)(i => Seq(BigInt(0x0123456789abcdefL) + i))
        val (got, cycles) = drive(dut, beats, 0.0, 0.0, new Rand(1))
        assert(got == beats.flatMap(pops(_, laneType, 1)))
        assert(cycles == 4 * r + 1, s"$cycles cycles")
      }
    }

  it should "pop E1's word 0 (int32) at one lane as -3, then 5" in {
    test(new FifoSlicer(1, "int32", 1)) { dut =>
      val (got, _) = drive(dut, Seq(Seq(BigInt("00000005fffffffd", 16))), 0.0, 0.0, new Rand(1))
      assert(got.map(LaneBits.signed(_, 32)) == Seq(BigInt(-3), BigInt(5)))
    }
  }

  it should "pop E2's words 0 and 1 (int8, 9e - 128 wrapped) at four lanes" in {
    // Element e is 9e - 128 wrapped to int8: word 0 is 0xbfb6ada49b928980.
    val elems = (0 until 16).map(e => BigInt(9 * e - 128) & 0xff)
    val words = Seq(LaneBits.pack(elems.take(8), 8), LaneBits.pack(elems.drop(8), 8))
    assert(words(0) == BigInt("bfb6ada49b928980", 16))
    test(new FifoSlicer(2, "int8", 4)) { dut =>
      val (got, _) = drive(dut, Seq(words), 0.0, 0.0, new Rand(1))
      val lanes    = got.map(p => (0 until 4).map(l => LaneBits.signed(p >> (8 * l), 8).toInt))
      assert(
        lanes == Seq(
          Seq(-128, -119, -110, -101),
          Seq(-92, -83, -74, -65),
          Seq(-56, -47, -38, -29),
          Seq(-20, -11, -2, 7)
        )
      )
    }
  }

  behavior of "FifoSlicer under random data and stalls"

  for (seed <- Seeds())
    it should s"give the model's pops in order (seed $seed)" in
      Seeds.run(seed) { r =>
        for (_ <- 0 until 4) {
          val laneType = r.pick(IntType.names)
          val words    = r.between(1, 4)
          val total    = words * 64 / IntType(laneType).width
          val outLanes = r.pick((1 to total).filter(total % _ == 0))
          val beats    = Seq.fill(r.between(1, 12))(Seq.fill(words)(r.word(64)))
          val (g, s)   = (r.probability(), r.probability())
          test(new FifoSlicer(words, laneType, outLanes)) { dut =>
            val (got, _) = drive(dut, beats, g, s, r)
            assert(
              got == beats.flatMap(pops(_, laneType, outLanes)),
              s"$words words of $laneType, $outLanes lanes, gap $g, stall $s"
            )
          }
        }
      }
}
