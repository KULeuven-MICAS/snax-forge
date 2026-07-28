package snax.forge.samples

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SimpleAdderSpec extends AnyFlatSpec with ChiselScalatestTester {

  private val W     = 32
  private val Lanes = 4
  private val Mask  = (BigInt(1) << W) - 1

  behavior of "SimpleAdder"

  it should "add lane-wise when both inputs are valid" in {
    test(new SimpleAdder(width = W, lanes = Lanes)) { dut =>
      dut.io.out.ready.poke(true.B)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)

      for (i <- 0 until Lanes) {
        dut.io.a.bits(i).poke((i + 1).U)
        dut.io.b.bits(i).poke((10 * (i + 1)).U)
      }

      // Combinational: no step needed, chiseltest settles the logic on peek.
      dut.io.out.valid.expect(true.B)
      for (i <- 0 until Lanes) {
        dut.io.out.bits(i).expect((11 * (i + 1)).U)
      }
    }
  }

  it should "wrap on overflow exactly as two's-complement does" in {
    test(new SimpleAdder(width = W, lanes = Lanes)) { dut =>
      dut.io.out.ready.poke(true.B)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)

      // 0xFFFFFFFF + 1 == 0. If this ever fails, the emitted datapath has
      // stopped matching the DaCe int32 golden reference and every
      // differential test downstream is lying.
      for (i <- 0 until Lanes) {
        dut.io.a.bits(i).poke(Mask.U)
        dut.io.b.bits(i).poke(1.U)
      }
      for (i <- 0 until Lanes) {
        dut.io.out.bits(i).expect(0.U)
      }
    }
  }

  it should "match a software model over random vectors" in {
    test(new SimpleAdder(width = W, lanes = Lanes)) { dut =>
      val rng = new scala.util.Random(0xf0f0)
      dut.io.out.ready.poke(true.B)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)

      for (_ <- 0 until 64) {
        val as = Seq.fill(Lanes)(BigInt(W, rng))
        val bs = Seq.fill(Lanes)(BigInt(W, rng))
        for (i <- 0 until Lanes) {
          dut.io.a.bits(i).poke(as(i).U)
          dut.io.b.bits(i).poke(bs(i).U)
        }
        for (i <- 0 until Lanes) {
          dut.io.out.bits(i).expect(((as(i) + bs(i)) & Mask).U)
        }
        dut.clock.step()
      }
    }
  }

  it should "hold out.valid low until both operand streams are valid" in {
    test(new SimpleAdder(width = W, lanes = Lanes)) { dut =>
      dut.io.out.ready.poke(true.B)

      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(false.B)
      dut.io.out.valid.expect(false.B)
      // Neither side may be consumed: dropping `a` here would desynchronise
      // the two streams permanently.
      dut.io.a.ready.expect(false.B)

      dut.io.a.valid.poke(false.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.valid.expect(false.B)
      dut.io.b.ready.expect(false.B)
    }
  }

  it should "backpressure both operands without dropping out.valid" in {
    test(new SimpleAdder(width = W, lanes = Lanes)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(false.B)

      // valid must not depend on ready -- the classic elastic-interface
      // deadlock. Data is available; it just is not being taken.
      dut.io.out.valid.expect(true.B)
      dut.io.a.ready.expect(false.B)
      dut.io.b.ready.expect(false.B)

      dut.io.out.ready.poke(true.B)
      dut.io.a.ready.expect(true.B)
      dut.io.b.ready.expect(true.B)
    }
  }
}
