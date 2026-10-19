package snax.forge.samples

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class AccumulatorSpec extends AnyFlatSpec with ChiselScalatestTester {

  private val W    = 32
  private val Mask = (BigInt(1) << W) - 1

  /** `peek().litToBoolean` rather than `peekBoolean()`: the latter moved between the testableBool and testableData
    * implicit classes across chiseltest versions, and resolving it is not worth a version argument.
    */
  private def isHigh(b: Bool): Boolean = b.peek().litToBoolean

  /** Pulse start for one cycle with the given configuration. */
  private def begin(dut: Accumulator, n: Int, t: Int): Unit = {
    dut.io.loopCount.poke(n.U)
    dut.io.len.poke(t.U)
    dut.io.start.poke(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)
  }

  /** What one cycle did: an input taken, an output given (its value), busy, done. */
  private case class Cycle(fire: Boolean, push: Option[BigInt], busy: Boolean, done: Boolean)

  /** Run `cycles` cycles from now: `in` offers `data` in order (valid while any is left, unless `inValid` says not),
    * `out` is ready when `outReady` says so. Returns what each cycle did, and checks every cycle that `out.valid` holds
    * until it fires and that `in.ready` is low while idle.
    */
  private def run(
    dut:      Accumulator,
    data:     Seq[BigInt],
    cycles:   Int,
    outReady: Int => Boolean = _ => true,
    inValid:  Int => Boolean = _ => true
  ): Seq[Cycle] = {
    var left    = data
    var waiting = false
    (0 until cycles).map { c =>
      val offer = left.nonEmpty && inValid(c)
      dut.io.in.valid.poke(offer.B)
      dut.io.in.bits.poke(left.headOption.getOrElse(BigInt(0)).U)
      dut.io.out.ready.poke(outReady(c).B)
      val busy  = isHigh(dut.io.busy)
      if (!busy) dut.io.in.ready.expect(false.B, s"cycle $c: in.ready while idle")
      if (waiting) dut.io.out.valid.expect(true.B, s"cycle $c: out.valid fell before out fired")
      val fire  = offer         && isHigh(dut.io.in.ready)
      val push  =
        if (isHigh(dut.io.out.valid) && outReady(c)) Some(dut.io.out.bits.peek().litValue)
        else None
      waiting = isHigh(dut.io.out.valid) && !outReady(c)
      val done  = isHigh(dut.io.done)
      if (fire) left = left.tail
      dut.clock.step()
      Cycle(fire, push, busy, done)
    }
  }

  private def sums(data: Seq[BigInt], t: Int, mask: BigInt = Mask): Seq[BigInt] =
    data.grouped(t).map(_.sum & mask).toSeq

  behavior of "Accumulator"

  it should "give each sum one cycle after its T-th input, with no gap between sums (T = 4, n = 8)" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      val data = (1 to 8).map(BigInt(_))
      begin(dut, n = 8, t = 4)
      val trace = run(dut, data, cycles = 11)
      // Cycles 0 to 7 take the inputs, 4 and 8 give the sums, 8 is done; idle after.
      assert(trace.map(_.fire) == Seq.tabulate(11)(_ < 8))
      assert(trace.map(_.push) == Seq.tabulate(11) {
        case 4 => Some(BigInt(10))
        case 8 => Some(BigInt(26))
        case _ => None
      })
      assert(trace.map(_.busy) == Seq.tabulate(11)(_ <= 8))
      assert(trace.map(_.done) == Seq.tabulate(11)(_ == 8))
    }
  }

  it should "hold its result and take no input while out is not ready" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      val data = (1 to 8).map(BigInt(_))
      begin(dut, n = 8, t = 4)
      // Out stalls in cycles 4 to 6 (the first sum waiting) and 11 (the last).
      val trace = run(dut, data, cycles = 14, outReady = c => !(4 to 6).contains(c) && c != 11)
      assert(trace.map(_.fire) == Seq.tabulate(14)(c => c < 4 || (7 to 10).contains(c)))
      assert(trace.map(_.push) == Seq.tabulate(14) {
        case 7  => Some(BigInt(10))
        case 12 => Some(BigInt(26))
        case _  => None
      })
      assert(trace.map(_.busy) == Seq.tabulate(14)(_ <= 12))
      assert(trace.map(_.done) == Seq.tabulate(14)(_ == 12))
    }
  }

  it should "make every input a sum at len = 1" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      val data = Seq[BigInt](5, 6, 7, 8)
      begin(dut, n = 4, t = 1)
      val trace = run(dut, data, cycles = 6)
      assert(trace.map(_.fire) == Seq(true, true, true, true, false, false))
      assert(trace.map(_.push) == Seq(None) ++ data.map(Some(_)) ++ Seq(None))
      assert(trace.map(_.busy) == Seq(true, true, true, true, true, false))
      assert(trace.map(_.done) == Seq(false, false, false, false, true, false))
    }
  }

  it should "ignore a start with loopCount 0 or len 0" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      for ((n, t) <- Seq((0, 4), (8, 0))) {
        begin(dut, n, t)
        val trace = run(dut, Seq.fill(4)(BigInt(1)), cycles = 3)
        assert(trace.forall(c => !c.busy && !c.fire && c.push.isEmpty && !c.done), s"loopCount $n, len $t")
      }
      // Still able to run afterwards.
      begin(dut, n = 2, t = 2)
      val trace = run(dut, Seq[BigInt](3, 4), cycles = 4)
      assert(trace.flatMap(_.push) == Seq(BigInt(7)))
    }
  }

  it should "not take inputs before a start" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      val data  = Seq[BigInt](1, 2, 3, 4)
      val idle  = run(dut, data, cycles = 3)
      assert(idle.forall(c => !c.fire && !c.busy))
      begin(dut, n = 4, t = 4)
      val trace = run(dut, data, cycles = 6)
      assert(trace.map(_.fire) == Seq(true, true, true, true, false, false))
      assert(trace.flatMap(_.push) == Seq(BigInt(10)))
    }
  }

  it should "run twice back to back, the second run's beats waiting through the first's last output" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      // Both runs' beats offered without a break: in the cycle the first run's last sum leaves, in.ready stays low,
      // so the second run's first beat is not taken (n inputs taken, not n + 1).
      val first  = Seq[BigInt](1, 2, 3, 4)
      val second = Seq[BigInt](10, 20, 30, 40, 50, 60)
      begin(dut, n = 4, t = 2)
      val trace1 = run(dut, first ++ second, cycles = 5)
      assert(trace1.count(_.fire) == 4)
      assert(trace1.flatMap(_.push) == Seq(BigInt(3), BigInt(7)))
      assert(trace1.map(_.done) == Seq(false, false, false, false, true))
      // A different len catches a stale latch.
      begin(dut, n = 6, t = 3)
      val trace2 = run(dut, second, cycles = 8)
      assert(trace2.count(_.fire) == 6)
      assert(trace2.flatMap(_.push) == Seq(BigInt(60), BigInt(150)))
      assert(!trace2.last.busy)
    }
  }

  it should "end the last sum at the n-th input when n is not a multiple of len" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      begin(dut, n = 6, t = 4)
      val trace = run(dut, (1 to 8).map(BigInt(_)), cycles = 9)
      assert(trace.count(_.fire) == 6)
      assert(trace.flatMap(_.push) == Seq(BigInt(10), BigInt(11)))
      assert(!trace.last.busy)
    }
  }

  it should "tolerate bubbles in the input stream" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      val data = Seq[BigInt](11, 22, 33, 44, 55, 66)
      begin(dut, n = 6, t = 3)
      // A reduction has no II obligation to meet, so starving it must only slow it down, never corrupt it.
      val trace = run(dut, data, cycles = 20, inValid = c => c % 3 == 2)
      assert(trace.flatMap(_.push) == sums(data, 3))
      assert(!trace.last.busy)
    }
  }

  it should "wrap the accumulator rather than saturate" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      begin(dut, n = 2, t = 2)
      val trace = run(dut, Seq(Mask, BigInt(3)), cycles = 4)
      assert(trace.flatMap(_.push) == Seq((Mask + 3) & Mask))
    }
  }

  it should "use the extra headroom when accWidth exceeds width" in {
    test(new Accumulator(width = W, accWidth = 40)) { dut =>
      // Four near-max 32-bit values overflow 32 bits but fit comfortably in 40, so the widened accumulator must NOT
      // wrap here.
      val data = Seq.fill(4)(Mask)
      begin(dut, n = 4, t = 4)
      val trace = run(dut, data, cycles = 6)
      assert(trace.flatMap(_.push) == Seq(data.sum))
    }
  }
}
