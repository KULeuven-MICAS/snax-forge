package snax.forge.elementwise

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ElementwiseTiledSpatialSpec extends AnyFlatSpec with ChiselScalatestTester {

  /** Random tiles per run. Each tile carries `Lanes` independent operand pairs, so a sweep covers `NumTests * Lanes`
    * elements and doubles as a trip-counter exercise.
    */
  private val NumTests = 20

  private val W     = 32
  private val Lanes = 4
  private val T     = 16

  private def begin(dut: ElementwiseTiledSpatial, trips: Int, op: Int = ElementwiseOp.Add): Unit = {
    dut.io.opSel.poke(op.U)
    dut.io.loopCount.poke(trips.U)
    dut.io.start.poke(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)
  }

  private def randomBeat(rng: scala.util.Random, dataWidth: Int): Seq[(BigInt, BigInt)] =
    Seq.fill(Lanes)((BigInt(dataWidth, rng), BigInt(dataWidth, rng)))

  private def drive(dut: ElementwiseTiledSpatial, beat: Seq[(BigInt, BigInt)]): Unit =
    for (((a, b), i) <- beat.zipWithIndex) {
      dut.io.a.bits(i).poke(a.U)
      dut.io.b.bits(i).poke(b.U)
    }

  private def checkBeat(
    dut:       ElementwiseTiledSpatial,
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

  behavior of "ElementwiseTiledSpatial"

  // -- arithmetic --------------------------------------------------------

  it should "match the software model across a full run, for every operation" in {
    test(new ElementwiseTiledSpatial(dataWidth = W, lanes = Lanes, loopCountWidth = T)) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)

      for (op <- ElementwiseOp.all) {
        // One run of NumTests tiles per operation. Fresh random operands every
        // tile, so the sweep also proves nothing carries between tiles.
        begin(dut, NumTests, op)
        for (t <- 0 until NumTests) {
          val beat = randomBeat(rng, W)
          drive(dut, beat)
          dut.io.busy.expect(true.B)
          dut.io.out.valid.expect(true.B)
          checkBeat(dut, beat, op)
          dut.io.done.expect((t == NumTests - 1).B)
          dut.clock.step()
        }
        dut.io.busy.expect(false.B)
      }
    }
  }

  it should "cover the edge operand pairs on every lane" in {
    test(new ElementwiseTiledSpatial(dataWidth = W, lanes = Lanes, loopCountWidth = T)) { dut =>
      val edges = ElementwiseModel.edgePairs(W)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)

      // Uniform random draws essentially never produce 0, 1 or all-ones, and
      // those are exactly where wrapping changes behaviour.
      for (op <- ElementwiseOp.all) {
        begin(dut, edges.size, op)
        for ((a, b) <- edges) {
          val beat = Seq.fill(Lanes)((a, b))
          drive(dut, beat)
          checkBeat(dut, beat, op)
          dut.clock.step()
        }
      }
    }
  }

  it should "match the signed model for min and max" in {
    test(new ElementwiseTiledSpatial(dataWidth = W, lanes = Lanes, loopCountWidth = T, signed = true)) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)

      for (op <- Seq(ElementwiseOp.Min, ElementwiseOp.Max)) {
        begin(dut, NumTests, op)
        for (_ <- 0 until NumTests) {
          val beat = randomBeat(rng, W)
          drive(dut, beat)
          checkBeat(dut, beat, op, signed = true)
          dut.clock.step()
        }
      }
    }
  }

  it should "run correctly at narrow widths" in {
    val narrow = 8
    test(new ElementwiseTiledSpatial(dataWidth = narrow, lanes = Lanes, loopCountWidth = T)) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      begin(dut, NumTests, ElementwiseOp.Add)

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
    test(new ElementwiseTiledSpatial(dataWidth = W, lanes = Lanes, loopCountWidth = T, supportedOps = Seq(op))) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      // opSel deliberately names a different operation; it must be ignored.
      begin(dut, NumTests, ElementwiseOp.Add)

      for (_ <- 0 until NumTests) {
        val beat = randomBeat(rng, W)
        drive(dut, beat)
        checkBeat(dut, beat, op)
        dut.clock.step()
      }
    }
  }

  it should "carry no state between tiles" in {
    test(new ElementwiseTiledSpatial(dataWidth = W, lanes = Lanes, loopCountWidth = T)) { dut =>
      // A tile boundary is not a reduction. Feeding the same operands every
      // pass must give the same result every pass -- if this drifts, an
      // accumulator has appeared where an elementwise unit was intended.
      val beat = Seq.fill(Lanes)((BigInt(5), BigInt(9)))
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      drive(dut, beat)
      begin(dut, NumTests)

      for (_ <- 0 until NumTests) {
        checkBeat(dut, beat, ElementwiseOp.Add)
        dut.clock.step()
      }
    }
  }

  // -- control -----------------------------------------------------------

  it should "stay idle and refuse operands until started" in {
    test(new ElementwiseTiledSpatial(dataWidth = W, lanes = Lanes, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)

      dut.io.busy.expect(false.B)
      dut.io.out.valid.expect(false.B)
      dut.io.a.ready.expect(false.B)
    }
  }

  it should "accept a different loopCount on each run" in {
    test(new ElementwiseTiledSpatial(dataWidth = W, lanes = Lanes, loopCountWidth = T)) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      drive(dut, Seq.fill(Lanes)((BigInt(1), BigInt(1))))

      // loopCount is latched at start; ceil(N/lanes) changes with N and
      // nothing here recomputes it. Random trip counts check that the counter
      // resets rather than carrying residue from the previous run.
      for (_ <- 0 until NumTests) {
        val trips = 1 + rng.nextInt(12)
        begin(dut, trips)
        for (_ <- 0 until trips) {
          dut.io.busy.expect(true.B, s"trips=$trips")
          dut.clock.step()
        }
        dut.io.busy.expect(false.B, s"trips=$trips")
      }
    }
  }

  it should "advance the counter only when both operand transfers complete" in {
    test(new ElementwiseTiledSpatial(dataWidth = W, lanes = Lanes, loopCountWidth = T)) { dut =>
      dut.io.out.ready.poke(true.B)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      drive(dut, Seq.fill(Lanes)((BigInt(1), BigInt(1))))
      begin(dut, 3)

      // One operand missing: no tile is counted, even though `a` is accepted.
      dut.io.b.valid.poke(false.B)
      dut.clock.step(3)
      dut.io.busy.expect(true.B)

      dut.io.b.valid.poke(true.B)
      dut.clock.step(3)
      dut.io.busy.expect(false.B)
    }
  }

  it should "not advance the counter while the consumer backpressures" in {
    test(new ElementwiseTiledSpatial(dataWidth = W, lanes = Lanes, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      drive(dut, Seq.fill(Lanes)((BigInt(1), BigInt(1))))
      begin(dut, 3)

      dut.io.out.ready.poke(false.B)
      dut.io.out.valid.expect(false.B)
      dut.clock.step(5)
      // Five idle cycles must not have consumed any of the three tiles.
      dut.io.busy.expect(true.B)

      dut.io.out.ready.poke(true.B)
      dut.clock.step(3)
      dut.io.busy.expect(false.B)
    }
  }

  it should "ignore a start pulse with loopCount zero rather than hang" in {
    test(new ElementwiseTiledSpatial(dataWidth = W, lanes = Lanes, loopCountWidth = T)) { dut =>
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      begin(dut, 0)

      dut.io.busy.expect(false.B)
      dut.io.out.valid.expect(false.B)
    }
  }
}
