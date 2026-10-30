package snax.forge.stream

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Rand, Seeds, Sink, Source}

/** A slicer into a packer of the same type and words: in-beats of words in, the same words out. */
class SliceAndPack(words: Int, laneType: String, lanes: Int) extends Module {
  val io             = IO(new Bundle {
    val in  = Flipped(Decoupled(Vec(words, UInt(64.W))))
    val out = Decoupled(Vec(words, UInt(64.W)))
  })
  private val slicer = Module(new FifoSlicer(words, laneType, lanes))
  private val packer = Module(new FifoPacker(lanes, laneType, words))
  slicer.io.in <> io.in
  packer.io.in <> slicer.io.out
  io.out <> packer.io.out
}

/** `FifoPacker` against a model written from its docstring: r in-beats of `inLanes` lanes, in-beat u's lanes at bits
  * `[c(u+1)-1 : cu]` of the out-beat (c the in-beat's bits), the out-beat as `words` words. Timing with the input
  * always valid and the output always ready, random configurations under random input gaps and output stalls, and
  * random streams through a slicer into a packer, which must give back the same words.
  */
class FifoPackerSpec extends AnyFlatSpec with ForgeTester {

  /** The model: the out-beats of `chunks` (each an in-beat's lanes packed), r at a time. */
  def beats(chunks: Seq[BigInt], chunkBits: Int, r: Int): Seq[BigInt] =
    chunks.grouped(r).map(g => g.zipWithIndex.map { case (c, u) => c << (u * chunkBits) }.sum).toSeq

  /** Feed `ins` (packed beats) into `in`, take `want` out-beats from `out`, with `in` valid unless `gap` says not and
    * `out` ready unless `stall` says not; the out-beats, and the cycles from the first input taken to the last output.
    */
  def drive[T <: Data](
    clock: Clock,
    in:    DecoupledIO[T],
    out:   DecoupledIO[Vec[UInt]],
    ins:   Seq[BigInt],
    want:  Int,
    gap:   Double,
    stall: Double,
    r:     Rand
  ): (Seq[BigInt], Int) = {
    val src     = Source.random(ins, gap, r)
    val snk     = Sink.random(stall, r)
    var cycle   = 0
    var first   = -1
    var lastOut = -1
    while (snk.got.size < want) {
      assert(cycle < 100 * ins.size + 100, s"no end after $cycle cycles (${snk.got.size} out-beats)")
      src.cycle(cycle)
      in.valid.poke(src.valid.B)
      LaneBits.pokeBeat(in.bits, src.bits)
      out.ready.poke(snk.cycle(cycle).B)
      if (src.fired(in.ready.peek().litToBoolean) && first < 0) first = cycle
      if (snk.fired(out.valid.peek().litToBoolean, LaneBits.peekBeat(out.bits))) lastOut = cycle
      clock.step()
      cycle += 1
    }
    (snk.got, lastOut - first + 1)
  }

  behavior of "FifoPacker"

  it should "take one in-beat a cycle: 4 out-beats at r = 2 in 8 cycles from the first input to the last output" in {
    test(new FifoPacker(1, "int32", 1)) { dut =>
      assert(dut.r == 2)
      val chunks        = Seq.tabulate(8)(i => BigInt(i + 1))
      val (got, cycles) = drive(dut.clock, dut.io.in, dut.io.out, chunks, 4, 0.0, 0.0, new Rand(1))
      assert(got == beats(chunks, 32, 2))
      assert(cycles == 8, s"$cycles cycles")
    }
  }

  it should "pack the int32 lanes -3, then 5 into the word 0x00000005fffffffd" in {
    test(new FifoPacker(1, "int32", 1)) { dut =>
      val chunks   = Seq(BigInt(-3), BigInt(5)).map(_ & LaneBits.mask(32))
      val (got, _) = drive(dut.clock, dut.io.in, dut.io.out, chunks, 1, 0.0, 0.0, new Rand(1))
      assert(got == Seq(BigInt("00000005fffffffd", 16)))
    }
  }

  behavior of "FifoPacker under random data and stalls"

  for (seed <- Seeds())
    it should s"give the model's out-beats (seed $seed)" in
      Seeds.run(seed) { r =>
        for (_ <- 0 until 4) {
          val laneType = r.pick(IntType.names)
          val w        = IntType(laneType).width
          val words    = r.between(1, 4)
          val total    = words * 64 / w
          val inLanes  = r.pick((1 to total).filter(total % _ == 0))
          val per      = total / inLanes
          val outs     = r.between(1, 6)
          val chunks   = Seq.fill(outs * per)(LaneBits.pack(Seq.fill(inLanes)(r.word(w)), w))
          val (g, s)   = (r.probability(), r.probability())
          test(new FifoPacker(inLanes, laneType, words)) { dut =>
            val (got, _) = drive(dut.clock, dut.io.in, dut.io.out, chunks, outs, g, s, r)
            assert(
              got == beats(chunks, inLanes * w, per),
              s"$inLanes lanes of $laneType, $words words, gap $g, stall $s"
            )
          }
        }
      }

  behavior of "FifoSlicer into FifoPacker"

  for (seed <- Seeds())
    it should s"give back the same words under random valid and ready (seed $seed)" in
      Seeds.run(seed) { r =>
        for (_ <- 0 until 4) {
          val laneType = r.pick(IntType.names)
          val words    = r.between(1, 4)
          val total    = words * 64 / IntType(laneType).width
          val lanes    = r.pick((1 to total).filter(total % _ == 0))
          val ins      = Seq.fill(r.between(1, 12))(LaneBits.pack(Seq.fill(words)(r.word(64)), 64))
          val (g, s)   = (r.probability(), r.probability())
          test(new SliceAndPack(words, laneType, lanes)) { dut =>
            val (got, _) = drive(dut.clock, dut.io.in, dut.io.out, ins, ins.size, g, s, r)
            assert(got == ins, s"$words words of $laneType through $lanes lanes, gap $g, stall $s")
          }
        }
      }
}
