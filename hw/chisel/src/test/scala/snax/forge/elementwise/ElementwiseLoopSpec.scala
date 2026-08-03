package snax.forge.elementwise

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ElementwiseLoopSpec extends AnyFlatSpec with ChiselScalatestTester {

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

  it should "wire operand ready straight to the consumer ready while busy" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      begin(dut, 4)

      // Ready is a pass-through, so it does not depend on the operand valids.
      dut.io.a.valid.poke(false.B)
      dut.io.b.valid.poke(false.B)
      dut.io.a.ready.expect(true.B)
      dut.io.b.ready.expect(true.B)

      dut.io.out.ready.poke(false.B)
      dut.io.a.ready.expect(false.B)
      dut.io.b.ready.expect(false.B)
    }
  }

  it should "hold out.valid low unless the consumer is ready" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.a.bits.poke(1.U)
      dut.io.b.bits.poke(1.U)
      dut.io.out.ready.poke(true.B)
      begin(dut, 2)

      // NOTE: out.valid is the conjunction of both operand handshakes, so it
      // depends on out.ready. This is the deliberate trade in this wiring --
      // a consumer whose ready is a function of valid will deadlock here.
      dut.io.out.ready.poke(false.B)
      dut.io.out.valid.expect(false.B)

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

  it should "require operands in lockstep: a lone valid operand is consumed and lost" in {
    test(new ElementwiseLoop(dataWidth = W, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.a.bits.poke(7.U)
      dut.io.b.bits.poke(7.U)
      dut.io.out.ready.poke(true.B)
      begin(dut, 4)

      // This pins the contract rather than endorsing it. With ready wired
      // straight through, `a` is accepted here even though `b` is absent, so
      // that beat of `a` is consumed while no output is produced and the
      // counter does not advance. The producer must present both operands
      // together. Gating each ready on the other operand's valid would make
      // this impossible, at one AND term per operand.
      dut.io.b.valid.poke(false.B)
      dut.io.a.ready.expect(true.B) // a.valid is high too, so a fires
      dut.io.out.valid.expect(false.B)

      dut.clock.step()
      // Four trips were programmed and none has been consumed.
      dut.io.busy.expect(true.B)
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
