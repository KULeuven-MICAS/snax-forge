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

  /** Drive one element, waiting out any backpressure first. */
  private def push(dut: Accumulator, v: BigInt): Unit = {
    dut.io.in.bits.poke(v.U)
    dut.io.in.valid.poke(true.B)
    while (!isHigh(dut.io.in.ready)) dut.clock.step()
    dut.clock.step()
    dut.io.in.valid.poke(false.B)
  }

  /** Wait for the result, check it, and hand the unit back ready for reuse. */
  private def drain(dut: Accumulator, expected: BigInt): Unit = {
    while (!isHigh(dut.io.out.valid)) dut.clock.step()
    dut.io.out.bits.expect(expected.U)
    dut.io.out.ready.poke(true.B)
    dut.clock.step()
    dut.io.out.ready.poke(false.B)
  }

  behavior of "Accumulator"

  it should "sum a stream of len elements" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      val data = Seq[BigInt](1, 2, 3, 4, 5)
      dut.io.len.poke(data.length.U)
      dut.io.out.ready.poke(false.B)

      data.foreach(push(dut, _))
      drain(dut, data.sum)
    }
  }

  it should "not consume input while a result is waiting" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      dut.io.len.poke(2.U)
      dut.io.out.ready.poke(false.B)

      push(dut, 7)
      push(dut, 8)

      // Result is parked. Until it is collected the unit must refuse new
      // data, otherwise the next run's first element silently joins this
      // run's sum.
      dut.io.out.valid.expect(true.B)
      dut.io.in.ready.expect(false.B)
      dut.clock.step(3)
      dut.io.in.ready.expect(false.B)
      dut.io.out.bits.expect(15.U)

      drain(dut, 15)
      dut.io.in.ready.expect(true.B)
    }
  }

  it should "tolerate bubbles in the input stream" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      val data = Seq[BigInt](11, 22, 33, 44)
      dut.io.len.poke(data.length.U)
      dut.io.out.ready.poke(false.B)

      for ((v, i) <- data.zipWithIndex) {
        // Idle gaps of growing length. A reduction has no II obligation to
        // meet, so starving it must only slow it down, never corrupt it.
        // The guard is because step(0) is not uniformly a no-op across
        // chiseltest backends.
        if (i > 0) {
          dut.io.in.valid.poke(false.B)
          dut.clock.step(i)
        }
        push(dut, v)
      }
      drain(dut, data.sum)
    }
  }

  it should "reset itself for a back-to-back second run" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      dut.io.out.ready.poke(false.B)

      dut.io.len.poke(3.U)
      Seq[BigInt](1, 2, 3).foreach(push(dut, _))
      drain(dut, 6)

      // No external reset between runs: the drain handshake is what clears
      // the accumulator. A different len is used to catch a stale latch.
      dut.io.len.poke(4.U)
      Seq[BigInt](10, 20, 30, 40).foreach(push(dut, _))
      drain(dut, 100)
    }
  }

  it should "wrap the accumulator rather than saturate" in {
    test(new Accumulator(width = W, accWidth = W)) { dut =>
      dut.io.len.poke(2.U)
      dut.io.out.ready.poke(false.B)

      push(dut, Mask)
      push(dut, 3)
      drain(dut, (Mask + 3) & Mask)
    }
  }

  it should "use the extra headroom when accWidth exceeds width" in {
    test(new Accumulator(width = W, accWidth = 40)) { dut =>
      dut.io.len.poke(4.U)
      dut.io.out.ready.poke(false.B)

      // Four near-max 32-bit values overflow 32 bits but fit comfortably in
      // 40, so the widened accumulator must NOT wrap here.
      val data = Seq.fill(4)(Mask)
      data.foreach(push(dut, _))
      drain(dut, data.sum)
    }
  }
}
