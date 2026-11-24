package snax.forge.stream

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Rand, Seeds, Sink, Source}

/** `FifoSlicer` against a model written from its docstring: an in-beat of `words` words is the bits of its elements,
  * element e at bits `[w(e+1)-1 : we]`, and pop u of it is elements u * outLanes until (u + 1) * outLanes. Timing with
  * the input always valid and the output always ready, two worked words (two int32, 16 int8), and random configurations
  * under random input gaps and output stalls, `enable` high and `clear` low. Then `enable` and `clear` against a cycle
  * model of snax-gvsoc's `SnaxSlicer`, every cycle under random `enable` and `clear`, and the sliced input's fixed
  * cases: the group's start (`clear`) in cycle 0 and busy (`enable`) from cycle 1, the reader early by 0, 1 and several
  * cycles, a short last word then a clear, a clear after a run that ended on a whole word, and `enable` low mid-run.
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
      dut.io.enable.poke(true.B)
      dut.io.clear.poke(false.B)
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

  it should "pop a word of two int32 at one lane as -3, then 5" in {
    test(new FifoSlicer(1, "int32", 1)) { dut =>
      val (got, _) = drive(dut, Seq(Seq(BigInt("00000005fffffffd", 16))), 0.0, 0.0, new Rand(1))
      assert(got.map(LaneBits.signed(_, 32)) == Seq(BigInt(-3), BigInt(5)))
    }
  }

  it should "pop two words of int8 (9e - 128 wrapped) at four lanes" in {
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

  /** The slicer's state as its docstring gives it: a held beat, `full`, the pop `count`. */
  class Model(val r: Int) {
    var full  = false
    var count = 0
    var beat  = BigInt(0)

    def inReady(outReady: Boolean, enable: Boolean): Boolean = enable && (!full || (outReady && count == r - 1))

    /** One cycle: whether the in-beat and a pop fired. */
    def step(inValid: Boolean, in: BigInt, outReady: Boolean, enable: Boolean, clear: Boolean): (Boolean, Boolean) = {
      val take = inValid && inReady(outReady, enable)
      val pop  = full    && outReady
      if (take) { beat = in; full = true; count = 0 }
      else if (pop) { if (count == r - 1) full = false else count += 1 }
      if (clear) { full = false; count = 0 }
      (take, pop)
    }
  }

  /** One cycle of `dut` and `m` on the same inputs, every output compared: whether the in-beat and a pop fired, and the
    * pop's bits.
    */
  def both(dut: FifoSlicer, m: Model, laneType: String, outLanes: Int, clue: => String)(
    inValid:  Boolean,
    in:       BigInt,
    outReady: Boolean,
    enable:   Boolean,
    clear:    Boolean
  ): (Boolean, Boolean, BigInt) = {
    dut.io.in.valid.poke(inValid.B)
    LaneBits.pokeBeat(dut.io.in.bits, in)
    dut.io.out.ready.poke(outReady.B)
    dut.io.enable.poke(enable.B)
    dut.io.clear.poke(clear.B)
    assert(high(dut.io.out.valid) == m.full, s"$clue: out.valid")
    assert(high(dut.io.in.ready) == m.inReady(outReady, enable), s"$clue: in.ready")
    val bits        = LaneBits.peekBeat(dut.io.out.bits)
    if (m.full) assert(bits == pops(Seq(m.beat), laneType, outLanes)(m.count), s"$clue: the pop's bits")
    val (take, pop) = m.step(inValid, in, outReady, enable, clear)
    dut.clock.step()
    (take, pop, bits)
  }

  behavior of "FifoSlicer with enable and clear (a sliced input)"

  for (seed <- Seeds())
    it should s"match the model every cycle under random enable and clear (seed $seed)" in
      Seeds.run(seed) { r =>
        for (_ <- 0 until 4) {
          val laneType = r.pick(Seq("int32", "int16", "int8"))
          val total    = 64 / IntType(laneType).width
          val outLanes = r.pick((1 until total).filter(total % _ == 0))
          val (g, s)   = (r.probability(), r.probability())
          val (e, c)   = (r.pick(Seq(0.5, 0.9, 1.0)), r.pick(Seq(0.0, 0.05, 0.2)))
          test(new FifoSlicer(1, laneType, outLanes)) { dut =>
            val m = new Model(dut.r)
            for (i <- 0 until 200)
              both(dut, m, laneType, outLanes, s"$laneType, $outLanes lanes, cycle $i")(
                !r.chance(g),
                r.word(64),
                !r.chance(s),
                r.chance(e),
                r.chance(c)
              )
          }
        }
      }

  /** Runs of the sliced input as the wrapper wires it: `clear(cycle)` the group's start accepted, `enable` its busy
    * cycles (`busy`), the unit's firings `out.ready` (`ready`), the reader's next word valid while `reader(cycle)`, for
    * `cycles` cycles; per cycle whether a word was taken and a pop fired, with the pop's bits.
    */
  def run(dut: FifoSlicer, m: Model, words: Seq[BigInt], cycles: Int)(
    clear:  Int => Boolean,
    busy:   Int => Boolean,
    ready:  Int => Boolean = _ => true,
    reader: Int => Boolean = _ => true
  ): Seq[(Boolean, Boolean, BigInt)] = {
    var next = 0
    (0 until cycles).map { i =>
      val valid = next < words.size && reader(i)
      val got   = both(dut, m, "int32", 1, s"cycle $i")(
        valid,
        if (next < words.size) words(next) else BigInt(0),
        ready(i),
        busy(i),
        clear(i)
      )
      if (got._1) next += 1
      got
    }
  }

  /** The cycles in which a word was taken, and those in which a pop fired. */
  def cyclesOf(got: Seq[(Boolean, Boolean, BigInt)]): (Seq[Int], Seq[Int]) =
    (got.indices.filter(got(_)._1), got.indices.filter(got(_)._2))

  def popped(got: Seq[(Boolean, Boolean, BigInt)]): Seq[BigInt] = got.filter(_._2).map(g => LaneBits.signed(g._3, 32))

  private val two = Seq(BigInt("00000002fffffffe", 16), BigInt("00000004fffffffc", 16), BigInt("00000006fffffffa", 16))

  for (early <- Seq(0, 1, 3))
    it should s"take the first word in the first busy cycle and pop it the next, the reader early by $early" in {
      test(new FifoSlicer(1, "int32", 1)) { dut =>
        // the reader valid from cycle 0, the start in cycle `early`, busy from the cycle after
        val m           = new Model(dut.r)
        val s           = early
        val got         = run(dut, m, two, s + 10)(clear = _ == s, busy = _ > s)
        val (take, pop) = cyclesOf(got)
        assert(take == Seq(1, 3, 5).map(_ + s), s"taken in $take")
        assert(pop == (2 to 7).map(_ + s), s"popped in $pop")
        assert(popped(got) == Seq(-2, 2, -4, 4, -6, 6).map(BigInt(_)))
      }
    }

  it should "drop a short last word's second slice at the next start" in {
    test(new FifoSlicer(1, "int32", 1)) { dut =>
      // run 1: three pops from two words (busy 1 to 4); run 2 starts in cycle 6 and pops the third word
      val m           = new Model(dut.r)
      val got         = run(dut, m, two, 12)(
        clear  = i => i == 0 || i == 6,
        busy   = i => (1 to 4).contains(i) || i >= 7,
        ready  = i => !(5 to 7).contains(i),
        reader = i => i < 4 || i >= 6
      )
      val (take, pop) = cyclesOf(got)
      assert(take == Seq(1, 3, 7), s"taken in $take")
      assert(pop == Seq(2, 3, 4, 8, 9), s"popped in $pop")
      assert(popped(got) == Seq(-2, 2, -4, -6, 6).map(BigInt(_)), "the short word's 4 dropped")
    }
  }

  it should "start empty after a run that ended on a whole word, then a clear" in {
    test(new FifoSlicer(1, "int32", 1)) { dut =>
      // run 1: four pops from two words (busy 1 to 5), the reader idle from then to run 2's start in cycle 7
      val m           = new Model(dut.r)
      val got         = run(dut, m, two, 12)(
        clear  = i => i == 0 || i == 7,
        busy   = i => (1 to 5).contains(i) || i >= 8,
        reader = i => i < 4 || i >= 7
      )
      val (take, pop) = cyclesOf(got)
      assert(take == Seq(1, 3, 8), s"taken in $take: none before run 2's first busy cycle")
      assert(pop == Seq(2, 3, 4, 5, 9, 10), s"popped in $pop")
    }
  }

  it should "go on popping with enable low mid-run and take no word until it is high" in {
    test(new FifoSlicer(1, "int32", 1)) { dut =>
      val m           = new Model(dut.r)
      val got         = run(dut, m, two, 12)(clear = _ == 0, busy = i => i >= 1 && !(2 to 5).contains(i))
      val (take, pop) = cyclesOf(got)
      assert(take == Seq(1, 6, 8), s"taken in $take")
      assert(pop == Seq(2, 3, 7, 8, 9, 10), s"popped in $pop: the held word's pops go on, then a gap")
      assert(popped(got) == Seq(-2, 2, -4, 4, -6, 6).map(BigInt(_)))
    }
  }
}
