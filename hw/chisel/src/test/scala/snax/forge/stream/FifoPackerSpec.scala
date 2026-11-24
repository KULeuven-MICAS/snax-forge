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
  slicer.io.enable := true.B
  slicer.io.clear  := false.B
  packer.io.clear  := false.B
  packer.io.last   := false.B
  io.out <> packer.io.out
}

/** `FifoPacker` against a model written from its docstring: r in-beats of `inLanes` lanes, in-beat u's lanes at bits
  * `[c(u+1)-1 : cu]` of the out-beat (c the in-beat's bits), the out-beat as `words` words. Timing with the input
  * always valid and the output always ready, random configurations under random input gaps and output stalls, and
  * random streams through a slicer into a packer, which must give back the same words; `clear` and `last` low. Then the
  * packed outer output against a model of snax-gvsoc's `SnaxPacker`, cycle by cycle: runs ended by `last` (a short last
  * word, zero above its slices), `clear` between runs.
  */
class FifoPackerSpec extends AnyFlatSpec with ForgeTester {

  /** The model: the out-beats of `chunks` (each an in-beat's lanes packed), r at a time. */
  def beats(chunks: Seq[BigInt], chunkBits: Int, r: Int): Seq[BigInt] =
    chunks.grouped(r).map(g => g.zipWithIndex.map { case (c, u) => c << (u * chunkBits) }.sum).toSeq

  /** A packer's `clear` and `last` low: a plain stream. */
  def quiet(dut: FifoPacker): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.last.poke(false.B)
  }

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
      quiet(dut)
      val chunks        = Seq.tabulate(8)(i => BigInt(i + 1))
      val (got, cycles) = drive(dut.clock, dut.io.in, dut.io.out, chunks, 4, 0.0, 0.0, new Rand(1))
      assert(got == beats(chunks, 32, 2))
      assert(cycles == 8, s"$cycles cycles")
    }
  }

  it should "pack the int32 lanes -3, then 5 into the word 0x00000005fffffffd" in {
    test(new FifoPacker(1, "int32", 1)) { dut =>
      quiet(dut)
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
            quiet(dut)
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

  /** snax-gvsoc's `SnaxPacker` with the FifoPacker's handshake, a cycle at a time: `count` slices filled, the held
    * in-beats; `sending` with `last`; the out-beat of an in-beat sent, this one at slot count and zeros above.
    */
  class PackerModel(r: Int, chunkBits: Int) {
    var count = 0
    val held  = Array.fill(r)(BigInt(0))
    def sending(last: Boolean): Boolean = count == r - 1 || last
    def word(in: BigInt):                BigInt =
      (0 until count).map(u => held(u) << (u * chunkBits)).sum |
        (in << (count * chunkBits))
    def push(in: BigInt, last: Boolean): Unit   =
      if (sending(last)) count = 0
      else { held(count) = in; count += 1 }
  }

  /** Drive `runs` (each a run's in-beats) through `dut`, a cycle of `clear` before each run, `last` high while the
    * run's next in-beat is its last, `in` valid unless `gap` says not and `out` ready unless `stall` says not; every
    * cycle `in.ready`, `out.valid` and a sent word compared with the model. The words sent.
    */
  def runPacked(dut: FifoPacker, runs: Seq[Seq[BigInt]], gap: Double, stall: Double, r: Rand): Seq[BigInt] = {
    val chunkBits = dut.inLanes * IntType(dut.laneType).width
    val m         = new PackerModel(dut.r, chunkBits)
    val got       = scala.collection.mutable.ArrayBuffer.empty[BigInt]
    var cycle     = 0
    for ((run, i) <- runs.zipWithIndex) {
      dut.io.clear.poke(true.B) // the start accepted: no push in that cycle
      dut.io.last.poke(false.B)
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(r.chance(0.5).B)
      assert(!high(dut.io.out.valid), s"run $i: out.valid in the clear cycle")
      dut.clock.step()
      m.count = 0
      dut.io.clear.poke(false.B)
      var next = 0
      while (next < run.size) {
        assert(cycle < 100 * runs.map(_.size).sum + 100, s"run $i: no end after $cycle cycles")
        val last  = next == run.size - 1
        val valid = !r.chance(gap)
        val ready = !r.chance(stall)
        dut.io.last.poke(last.B)
        dut.io.in.valid.poke(valid.B)
        LaneBits.pokeBeat(dut.io.in.bits, run(next))
        dut.io.out.ready.poke(ready.B)
        val send  = m.sending(last)
        val clue  = s"run $i, in-beat $next, cycle $cycle (count ${m.count}, last $last)"
        assert(high(dut.io.in.ready) == (!send || ready), s"$clue: in.ready")
        assert(high(dut.io.out.valid) == (valid && send), s"$clue: out.valid")
        if (valid                               && (!send || ready)) {
          if (send) {
            val w = LaneBits.peekBeat(dut.io.out.bits)
            assert(w == m.word(run(next)), s"$clue: word ${w.toString(16)}, model ${m.word(run(next)).toString(16)}")
            got += w
          }
          m.push(run(next), last)
          next += 1
        }
        dut.clock.step()
        cycle += 1
      }
    }
    got.toSeq
  }

  behavior of "FifoPacker on a packed outer output"

  it should "send int32 -1, -2, then -3 with last as 0xfffffffeffffffff and 0x00000000fffffffd (d_out 2)" in {
    test(new FifoPacker(1, "int32", 1)) { dut =>
      assert(dut.r == 2)
      val run = Seq(-1, -2, -3).map(v => BigInt(v) & LaneBits.mask(32))
      val got = runPacked(dut, Seq(run), 0.0, 0.0, new Rand(1))
      assert(got == Seq(BigInt("fffffffeffffffff", 16), BigInt("00000000fffffffd", 16)), got.map(_.toString(16)))
    }
  }

  for (seed <- Seeds())
    it should s"give the model's words and handshake over runs ended by last (seed $seed)" in
      Seeds.run(seed) { r =>
        for (_ <- 0 until 4) {
          val laneType = r.pick(IntType.names)
          val w        = IntType(laneType).width
          val total    = 64 / w
          val inLanes  = r.pick((1 to total).filter(total % _ == 0))
          val per      = total / inLanes
          val runs     = Seq.fill(r.between(1, 4)) {
            Seq.fill(r.between(1, 3 * per + 1))(LaneBits.pack(Seq.fill(inLanes)(r.word(w)), w))
          }
          val (g, s)   = (r.probability(), r.probability())
          test(new FifoPacker(inLanes, laneType, 1)) { dut =>
            val got = runPacked(dut, runs, g, s, r)
            assert(
              got == runs.flatMap(beats(_, inLanes * w, per)), // a short last word zero above
              s"$inLanes lanes of $laneType, runs of ${runs.map(_.size)}, gap $g, stall $s"
            )
          }
        }
      }
}
