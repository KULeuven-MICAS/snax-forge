package snax.forge.elementwise

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.ForgeTester

class ElementwiseLoopSpec extends AnyFlatSpec with ForgeTester {

  /** Random operand pairs per operation, and also the trip count of the run that carries them. */
  private val NumTests = 20

  private val W = 32
  private val T = 16

  private def begin(dut: ElementwiseLoop, trips: Int, op: Int = ElementwiseOp.Add): Unit = {
    dut.io.opSel.poke(op.U)
    dut.io.loopCount.poke(trips.U)
    dut.io.start.poke(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)
  }

  behavior of "ElementwiseLoop"

  // -- arithmetic --------------------------------------------------------

  it should "match the software model across a full run, for every operation" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      val rng   = new scala.util.Random(ElementwiseModel.Seed)
      val pairs = ElementwiseModel.operandPairs(W, NumTests, rng)

      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)

      for (op <- ElementwiseOp.all) {
        // One run per operation, long enough to carry every operand pair. The
        // sweep therefore exercises the trip counter as well as the datapath:
        // the unit must stay busy for exactly `pairs.size` beats.
        begin(dut, pairs.size, op)
        for (((a, b), i) <- pairs.zipWithIndex) {
          dut.io.a.bits.poke(a.U)
          dut.io.b.bits.poke(b.U)
          dut.io.busy.expect(true.B)
          dut.io.out.valid.expect(true.B)
          dut.io.out.bits.expect(ElementwiseModel(op, a, b, W).U, ElementwiseModel.clue(op, a, b))
          dut.io.done.expect((i == pairs.size - 1).B)
          dut.clock.step()
        }
        dut.io.busy.expect(false.B)
      }
    }
  }

  it should "match the signed model for min and max" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T, signed = true)) { dut =>
      val rng   = new scala.util.Random(ElementwiseModel.Seed)
      val pairs = ElementwiseModel.operandPairs(W, NumTests, rng)

      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)

      for (op <- Seq(ElementwiseOp.Min, ElementwiseOp.Max)) {
        begin(dut, pairs.size, op)
        for ((a, b) <- pairs) {
          dut.io.a.bits.poke(a.U)
          dut.io.b.bits.poke(b.U)
          dut.io.out.bits.expect(
            ElementwiseModel(op, a, b, W, signed = true).U,
            ElementwiseModel.clue(op, a, b)
          )
          dut.clock.step()
        }
      }
    }
  }

  it should "run correctly at narrow widths" in {
    val narrow = 8
    test(new ElementwiseLoop(dataWidth = narrow, loopCountWidth = T)) { dut =>
      val rng   = new scala.util.Random(ElementwiseModel.Seed)
      val pairs = ElementwiseModel.operandPairs(narrow, NumTests, rng)

      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      begin(dut, pairs.size, ElementwiseOp.Add)

      for ((a, b) <- pairs) {
        dut.io.a.bits.poke(a.U)
        dut.io.b.bits.poke(b.U)
        dut.io.out.bits.expect(
          ElementwiseModel(ElementwiseOp.Add, a, b, narrow).U,
          ElementwiseModel.clue(ElementwiseOp.Add, a, b)
        )
        dut.clock.step()
      }
    }
  }

  it should "produce the same results with fixed-function hardware" in {
    val op = ElementwiseOp.Sub
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T, supportedOps = Seq(op))) { dut =>
      val rng   = new scala.util.Random(ElementwiseModel.Seed)
      val pairs = ElementwiseModel.operandPairs(W, NumTests, rng)

      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      // opSel deliberately names a different operation; it must be ignored.
      begin(dut, pairs.size, ElementwiseOp.Add)

      for ((a, b) <- pairs) {
        dut.io.a.bits.poke(a.U)
        dut.io.b.bits.poke(b.U)
        dut.io.out.bits.expect(ElementwiseModel(op, a, b, W).U, ElementwiseModel.clue(op, a, b))
        dut.clock.step()
      }
    }
  }

  // -- control -----------------------------------------------------------

  it should "stay idle and refuse operands until started" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)

      // The whole point of owning the trip count: a producer that offers data
      // continuously must not push any through before a run starts.
      dut.io.busy.expect(false.B)
      dut.io.out.valid.expect(false.B)
      dut.io.a.ready.expect(false.B)
      dut.io.b.ready.expect(false.B)
    }
  }

  it should "accept a different loopCount on each run" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      dut.io.a.bits.poke(3.U)
      dut.io.b.bits.poke(4.U)

      // loopCount is latched at start, so a driver may reprogram it freely
      // between runs. Random trip counts check that the counter resets rather
      // than carrying residue from the previous run.
      for (_ <- 0 until NumTests) {
        val trips = 1 + rng.nextInt(12)
        begin(dut, trips)
        for (_ <- 0 until trips) {
          dut.io.busy.expect(true.B, s"trips=$trips")
          dut.io.out.bits.expect(7.U)
          dut.clock.step()
        }
        dut.io.busy.expect(false.B, s"trips=$trips")
      }
    }
  }

  it should "gate each operand's ready on the other's valid and the consumer's ready while busy" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      begin(dut, 4)

      // A join: each ready is the other operand's valid and the consumer's ready.
      dut.io.a.ready.expect(true.B)
      dut.io.b.ready.expect(true.B)
      dut.io.b.valid.poke(false.B)
      dut.io.a.ready.expect(false.B)
      dut.io.b.ready.expect(true.B)
      dut.io.b.valid.poke(true.B)

      dut.io.out.ready.poke(false.B)
      dut.io.a.ready.expect(false.B)
      dut.io.b.ready.expect(false.B)
    }
  }

  it should "raise out.valid whatever the consumer's ready" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.a.bits.poke(1.U)
      dut.io.b.bits.poke(1.U)
      dut.io.out.ready.poke(true.B)
      begin(dut, 2)

      // valid does not depend on ready: both operands valid is enough.
      dut.io.out.ready.poke(false.B)
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.expect(2.U)

      dut.io.out.ready.poke(true.B)
      dut.io.out.valid.expect(true.B)
    }
  }

  it should "not advance the counter while the consumer backpressures" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.a.bits.poke(1.U)
      dut.io.b.bits.poke(1.U)
      dut.io.out.ready.poke(true.B)
      begin(dut, 2)

      dut.io.out.ready.poke(false.B)
      dut.clock.step(4)
      dut.io.busy.expect(true.B)

      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      dut.io.busy.expect(false.B)
    }
  }

  it should "make a lone operand wait for the other" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.a.bits.poke(7.U)
      dut.io.b.bits.poke(7.U)
      dut.io.out.ready.poke(true.B)
      begin(dut, 4)

      // `b` absent: `a` is not taken and no output is offered.
      dut.io.b.valid.poke(false.B)
      dut.io.a.ready.expect(false.B)
      dut.io.out.valid.expect(false.B)
      dut.clock.step(3)
      dut.io.busy.expect(true.B)

      // Four trips were programmed and none has been consumed: four firings end the run.
      dut.io.b.valid.poke(true.B)
      dut.clock.step(4)
      dut.io.busy.expect(false.B)
    }
  }

  it should "ignore a start pulse with loopCount zero rather than hang" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      begin(dut, 0)

      dut.io.busy.expect(false.B)
      dut.io.out.valid.expect(false.B)
    }
  }
}
