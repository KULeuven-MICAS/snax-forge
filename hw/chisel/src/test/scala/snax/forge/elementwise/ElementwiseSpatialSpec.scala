package snax.forge.elementwise

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ElementwiseSpatialSpec extends AnyFlatSpec with ChiselScalatestTester {

  /** Random beats per operation. Each beat carries `Lanes` independent operand pairs. */
  private val NumTests = 20

  private val W     = 32
  private val Lanes = 8

  /** One random beat: every lane gets its own operand pair, so a lane that is wired to the wrong index shows up. */
  private def randomBeat(rng: scala.util.Random, dataWidth: Int): Seq[(BigInt, BigInt)] =
    Seq.fill(Lanes)((BigInt(dataWidth, rng), BigInt(dataWidth, rng)))

  private def drive(dut: ElementwiseSpatial, beat: Seq[(BigInt, BigInt)]): Unit = {
    dut.io.a.valid.poke(true.B)
    dut.io.b.valid.poke(true.B)
    dut.io.out.ready.poke(true.B)
    for (((a, b), i) <- beat.zipWithIndex) {
      dut.io.a.bits(i).poke(a.U)
      dut.io.b.bits(i).poke(b.U)
    }
  }

  private def checkBeat(
    dut:       ElementwiseSpatial,
    beat:      Seq[(BigInt, BigInt)],
    op:        Int,
    dataWidth: Int     = W,
    signed:    Boolean = false
  ): Unit =
    for (((a, b), i) <- beat.zipWithIndex) {
      dut.io.out
        .bits(i)
        .expect(
          ElementwiseModel(op, a, b, dataWidth, signed).U,
          ElementwiseModel.clue(op, a, b, lane = i)
        )
    }

  behavior of "ElementwiseSpatial"

  // -- arithmetic --------------------------------------------------------

  it should "match the software model on every lane, for every operation" in {
    test(new ElementwiseSpatial(dataWidth = W, lanes = Lanes)) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)

      for (op <- ElementwiseOp.all) {
        dut.io.opSel.poke(op.U)
        // Edge cases first, broadcast to every lane, then random beats where
        // each lane carries different operands.
        for ((a, b) <- ElementwiseModel.edgePairs(W)) {
          val beat = Seq.fill(Lanes)((a, b))
          drive(dut, beat)
          checkBeat(dut, beat, op)
          dut.clock.step()
        }
        for (_      <- 0 until NumTests) {
          val beat = randomBeat(rng, W)
          drive(dut, beat)
          checkBeat(dut, beat, op)
          dut.clock.step()
        }
      }
    }
  }

  it should "match the signed model for min and max" in {
    test(new ElementwiseSpatial(dataWidth = W, lanes = Lanes, signed = true)) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)

      for (op <- Seq(ElementwiseOp.Min, ElementwiseOp.Max)) {
        dut.io.opSel.poke(op.U)
        for (_ <- 0 until NumTests) {
          val beat = randomBeat(rng, W)
          drive(dut, beat)
          checkBeat(dut, beat, op, signed = true)
          dut.clock.step()
        }
      }
    }
  }

  it should "keep lanes independent" in {
    test(new ElementwiseSpatial(dataWidth = W, lanes = Lanes)) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)
      dut.io.opSel.poke(ElementwiseOp.Add.U)

      // Distinct random operands per lane, checked per lane. A crossed wire
      // between lanes is invisible when every lane sees the same value, which
      // is why the beats above are not simply broadcast.
      for (_ <- 0 until NumTests) {
        val beat = randomBeat(rng, W)
        drive(dut, beat)
        checkBeat(dut, beat, ElementwiseOp.Add)
        dut.clock.step()
      }
    }
  }

  it should "run correctly at narrow widths" in {
    val narrow = 8
    test(new ElementwiseSpatial(dataWidth = narrow, lanes = Lanes)) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)
      dut.io.opSel.poke(ElementwiseOp.Add.U)

      for (_ <- 0 until NumTests) {
        val beat = randomBeat(rng, narrow)
        drive(dut, beat)
        checkBeat(dut, beat, ElementwiseOp.Add, dataWidth = narrow)
        dut.clock.step()
      }
    }
  }

  it should "produce the same results with fixed-function hardware" in {
    val op = ElementwiseOp.Xor
    test(new ElementwiseSpatial(dataWidth = W, lanes = Lanes, supportedOps = Seq(op))) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)
      dut.io.opSel.poke(ElementwiseOp.Add.U) // deliberately wrong; must be ignored

      for (_ <- 0 until NumTests) {
        val beat = randomBeat(rng, W)
        drive(dut, beat)
        checkBeat(dut, beat, op)
        dut.clock.step()
      }
    }
  }

  it should "share one opSel across all lanes" in {
    test(new ElementwiseSpatial(dataWidth = W, lanes = Lanes)) { dut =>
      // The lanes are replications of a single SDFG tasklet, so they cannot
      // differ. Switching opSel must move all of them together.
      val beat = Seq.fill(Lanes)((BigInt(40), BigInt(7)))
      drive(dut, beat)

      dut.io.opSel.poke(ElementwiseOp.Add.U)
      for (i <- 0 until Lanes) dut.io.out.bits(i).expect(47.U)

      dut.io.opSel.poke(ElementwiseOp.Sub.U)
      for (i <- 0 until Lanes) dut.io.out.bits(i).expect(33.U)
    }
  }

  // -- control -----------------------------------------------------------

  it should "expose one handshake for all lanes of a port" in {
    test(new ElementwiseSpatial(dataWidth = W, lanes = Lanes)) { dut =>
      // Decoupled(Vec(...)) rather than Vec(Decoupled(...)): a single valid
      // covers every lane, because a streamer delivers a tile as one beat and
      // cannot offer lanes independently.
      dut.io.out.ready.poke(true.B)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(false.B)
      dut.io.out.valid.expect(false.B)

      dut.io.b.valid.poke(true.B)
      dut.io.out.valid.expect(true.B)
    }
  }

  it should "wire operand ready straight to the consumer ready" in {
    test(new ElementwiseSpatial(dataWidth = W, lanes = Lanes)) { dut =>
      // Pass-through, so ready does not depend on either operand's valid.
      dut.io.a.valid.poke(false.B)
      dut.io.b.valid.poke(false.B)

      dut.io.out.ready.poke(true.B)
      dut.io.a.ready.expect(true.B)
      dut.io.b.ready.expect(true.B)

      dut.io.out.ready.poke(false.B)
      dut.io.a.ready.expect(false.B)
      dut.io.b.ready.expect(false.B)
    }
  }

  it should "hold out.valid low unless the consumer is ready" in {
    test(new ElementwiseSpatial(dataWidth = W, lanes = Lanes)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)

      // NOTE: out.valid is the conjunction of both operand handshakes and so
      // depends on out.ready -- the deliberate trade in this wiring.
      dut.io.out.ready.poke(false.B)
      dut.io.out.valid.expect(false.B)

      dut.io.out.ready.poke(true.B)
      dut.io.out.valid.expect(true.B)
    }
  }

  it should "require operands in lockstep: a lone valid operand is consumed and lost" in {
    test(new ElementwiseSpatial(dataWidth = W, lanes = Lanes)) { dut =>
      // Pins the contract rather than endorsing it: `a` is accepted while `b`
      // is absent, so that beat is consumed with no output produced.
      dut.io.out.ready.poke(true.B)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(false.B)

      dut.io.a.ready.expect(true.B)
      dut.io.out.valid.expect(false.B)
    }
  }

  it should "elaborate a single-lane instance" in {
    // lanes = 1 is the boundary between this variant and ElementwiseLoop:
    // same arithmetic, no trip counter.
    test(new ElementwiseSpatial(dataWidth = W, lanes = 1)) { dut =>
      dut.io.opSel.poke(ElementwiseOp.Add.U)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      dut.io.a.bits(0).poke(2.U)
      dut.io.b.bits(0).poke(5.U)
      dut.io.out.bits(0).expect(7.U)
    }
  }
}
