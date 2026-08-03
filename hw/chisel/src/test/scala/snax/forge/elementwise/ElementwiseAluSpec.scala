package snax.forge.elementwise

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ElementwiseAluSpec extends AnyFlatSpec with ChiselScalatestTester {

  /** Random operand pairs per operation, on top of the edge cases in `ElementwiseModel.edgePairs`. */
  private val NumTests = 20

  private val W = 32

  behavior of "ElementwiseAlu"

  it should "match the software model for every supported operation" in {
    test(new ElementwiseAlu(dataWidth = W)) { dut =>
      val rng   = new scala.util.Random(ElementwiseModel.Seed)
      val pairs = ElementwiseModel.operandPairs(W, NumTests, rng)

      for (op <- ElementwiseOp.all) {
        dut.io.opSel.poke(op.U)
        for ((a, b) <- pairs) {
          dut.io.a.poke(a.U)
          dut.io.b.poke(b.U)
          dut.io.out.expect(
            ElementwiseModel(op, a, b, W).U,
            ElementwiseModel.clue(op, a, b)
          )
        }
      }
    }
  }

  it should "match the signed model for min and max" in {
    test(new ElementwiseAlu(dataWidth = W, signed = true)) { dut =>
      val rng   = new scala.util.Random(ElementwiseModel.Seed)
      val pairs = ElementwiseModel.operandPairs(W, NumTests, rng)

      // Min and Max are the only operations where signedness is observable,
      // so this is the only place the signed model differs from the unsigned
      // one. Random 32-bit draws are negative half the time, which is what
      // makes a sweep worth more here than any single hand-picked pair.
      for (op <- Seq(ElementwiseOp.Min, ElementwiseOp.Max)) {
        dut.io.opSel.poke(op.U)
        for ((a, b) <- pairs) {
          dut.io.a.poke(a.U)
          dut.io.b.poke(b.U)
          dut.io.out.expect(
            ElementwiseModel(op, a, b, W, signed = true).U,
            ElementwiseModel.clue(op, a, b)
          )
        }
      }
    }
  }

  it should "wrap at narrow widths as the model does" in {
    // int8 is the SNAX GEMM native width. Wrapping is far more frequent at 8
    // bits than at 32, so this catches a width-handling slip that a 32-bit
    // sweep might not provoke.
    val narrow = 8
    test(new ElementwiseAlu(dataWidth = narrow)) { dut =>
      val rng   = new scala.util.Random(ElementwiseModel.Seed)
      val pairs = ElementwiseModel.operandPairs(narrow, NumTests, rng)

      for (op <- ElementwiseOp.all) {
        dut.io.opSel.poke(op.U)
        for ((a, b) <- pairs) {
          dut.io.a.poke(a.U)
          dut.io.b.poke(b.U)
          dut.io.out.expect(
            ElementwiseModel(op, a, b, narrow).U,
            ElementwiseModel.clue(op, a, b)
          )
        }
      }
    }
  }

  it should "ignore opSel when fixed-function" in {
    test(new ElementwiseAlu(dataWidth = W, supportedOps = Seq(ElementwiseOp.Sub))) { dut =>
      val rng = new scala.util.Random(ElementwiseModel.Seed)

      for ((a, b) <- ElementwiseModel.operandPairs(W, NumTests, rng)) {
        dut.io.a.poke(a.U)
        dut.io.b.poke(b.U)
        // Sweeping every encoding must not change the result: a fixed-function
        // instance has no decode logic for opSel to reach.
        for (encoding <- 0 until (1 << ElementwiseOp.width)) {
          dut.io.opSel.poke(encoding.U)
          dut.io.out.expect(
            ElementwiseModel(ElementwiseOp.Sub, a, b, W).U,
            s"${ElementwiseModel.clue(ElementwiseOp.Sub, a, b)}, opSel=$encoding"
          )
        }
      }
    }
  }

  it should "decode only the operations it was given" in {
    val subset = Seq(ElementwiseOp.And, ElementwiseOp.Or)
    test(new ElementwiseAlu(dataWidth = W, supportedOps = subset)) { dut =>
      val rng   = new scala.util.Random(ElementwiseModel.Seed)
      val pairs = ElementwiseModel.operandPairs(W, NumTests, rng)

      for ((a, b) <- pairs) {
        dut.io.a.poke(a.U)
        dut.io.b.poke(b.U)
        for (op <- subset) {
          dut.io.opSel.poke(op.U)
          dut.io.out.expect(ElementwiseModel(op, a, b, W).U, ElementwiseModel.clue(op, a, b))
        }

        // An unsupported encoding falls through to the MuxLookup default,
        // which is the first supported op. Documented rather than asserted as
        // ideal: the generator is expected never to program one.
        dut.io.opSel.poke(ElementwiseOp.Add.U)
        dut.io.out.expect(ElementwiseModel(subset.head, a, b, W).U)
      }
    }
  }
}
