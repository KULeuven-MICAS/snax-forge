package snax.forge.datapath

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds}
import snax.forge.elementwise.{ElementwiseModel, ElementwiseOp}

/** `ElementwiseLanes`: its types on fixed vectors (D143: each input extended to the op type by its own signedness, the
  * op at the op type, the result extended or wrapped into the output type; each expected value NumPy's, and
  * `ElementwiseModel.numpy`'s too); every op on the edge operands (`ElementwiseModel.edgePairs`: 0, 1, all ones, the
  * sign bit, ...) at int32, uint32, int8 and uint8, where min and max compare by the type's signedness and int8 wraps
  * far more often than int32; then random data against `ElementwiseModel.numpy` at random types, ops and lanes.
  */
class ElementwiseLanesSpec extends AnyFlatSpec with ForgeTester {

  /** The lanes of `out` for `a` and `b` (numbers of their types), as numbers of `outType`. */
  private def run(
    inTypes: Seq[String],
    opType:  String,
    outType: String,
    op:      Int,
    a:       Seq[BigInt],
    b:       Seq[BigInt]
  ): Seq[BigInt] = {
    val tr  = IntType(outType)
    var got = Seq.empty[BigInt]
    test(new ElementwiseLanes(inTypes, opType, outType, a.size, op)) { dut =>
      for (l <- a.indices) {
        LaneBits.poke(dut.io.a(l), a(l))
        LaneBits.poke(dut.io.b(l), b(l))
      }
      got = a.indices.map { l =>
        val bits = LaneBits.peek(dut.io.out(l))
        if (tr.signed) LaneBits.signed(bits, tr.width) else bits
      }
    }
    got
  }

  private def fixed(
    inTypes: Seq[String],
    opType:  String,
    outType: String,
    op:      Int,
    a:       Seq[BigInt],
    b:       Seq[BigInt],
    want:    Seq[BigInt]
  ): Unit = {
    val (ta, tb, to, tr) = (IntType(inTypes(0)), IntType(inTypes(1)), IntType(opType), IntType(outType))
    val clue             = s"${inTypes.mkString(" ")} -> $opType -> $outType, ${ElementwiseOp.name(op)}"
    assert(ElementwiseModel.resultType(ta, tb) == to, s"$clue: the op type is not NumPy's")
    val model            = a.zip(b).map { case (x, y) =>
      LaneBits.signed(ElementwiseModel.numpy(op, x, y, ta, tb, to, tr), tr.width)
    }
    val wantBits         = want.map(w => if (tr.signed) w else w & LaneBits.mask(tr.width))
    assert(model == wantBits, s"$clue: the model")
    val got              = run(inTypes, opType, outType, op, a, b)
    assert(got == want, s"$clue: lanes $got, NumPy $want")
  }

  private val Max32 = Int.MaxValue // 2^31 - 1

  behavior of "ElementwiseLanes' types"

  it should "multiply int32 by int32 at int32 and sign-extend into int64" in {
    // Lane 2 wraps at int32 to -2 and is sign-extended, not 4294967294.
    fixed(
      Seq("int32", "int32"),
      "int32",
      "int64",
      ElementwiseOp.Mul,
      Seq(3, -2, Max32, -5).map(BigInt(_)),
      Seq(4, 7, 2, -1).map(BigInt(_)),
      Seq(12, -14, -2, 5).map(BigInt(_))
    )
  }

  it should "add int8 and uint8 at int16, each extended by its own signedness" in {
    // 0xff is -1 as int8 and 255 as uint8.
    fixed(
      Seq("int8", "uint8"),
      "int16",
      "int16",
      ElementwiseOp.Add,
      Seq.fill(8)(BigInt(-1)),
      Seq.fill(8)(BigInt(255)),
      Seq.fill(8)(BigInt(254))
    )
  }

  it should "compare by the op type's signedness in max, extending into int32" in {
    // 0x90 is -112 as int8 and 144 as uint8.
    fixed(Seq("int8", "int8"), "int8", "int32", ElementwiseOp.Max, Seq(BigInt(-112)), Seq(BigInt(0)), Seq(BigInt(0)))
    fixed(
      Seq("uint8", "uint8"),
      "uint8",
      "int32",
      ElementwiseOp.Max,
      Seq(BigInt(144)),
      Seq(BigInt(0)),
      Seq(BigInt(144))
    )
  }

  it should "add at int32 and wrap into int16" in {
    fixed(
      Seq("int32", "int32"),
      "int32",
      "int16",
      ElementwiseOp.Add,
      Seq(70000, -1, 5, Max32).map(BigInt(_)),
      Seq.fill(4)(BigInt(1)),
      Seq(4465, 0, 6, 0).map(BigInt(_))
    )
  }

  it should "refuse an op type narrower than an input" in {
    val e = intercept[Exception] {
      circt.stage.ChiselStage.emitCHIRRTL(new ElementwiseLanes(Seq("int32", "int32"), "int16", "int32", 1))
    }
    assert(e.getMessage.contains("opType int16 must be at least as wide as each input"), e.getMessage)
  }

  it should "give the model's lanes on the edge operands, every op, at int32, uint32, int8 and uint8" in {
    for {
      t  <- Seq("int32", "uint32", "int8", "uint8").map(IntType(_))
      op <- ElementwiseOp.all
    } {
      val pairs = ElementwiseModel.edgePairs(t.width)
      test(new ElementwiseLanes(Seq(t.name, t.name), t.name, t.name, pairs.size, op)) { dut =>
        for (((a, b), l) <- pairs.zipWithIndex) {
          LaneBits.poke(dut.io.a(l), a)
          LaneBits.poke(dut.io.b(l), b)
        }
        for (((a, b), l) <- pairs.zipWithIndex) {
          val want = ElementwiseModel.numpy(op, a, b, t, t, t, t)
          val got  = LaneBits.peek(dut.io.out(l))
          assert(got == want, s"${t.name}, ${ElementwiseModel.clue(op, a, b, l)}: $got, the model $want")
        }
      }
    }
  }

  behavior of "ElementwiseLanes under random data"

  /** The input types with an integer result type: not a signed type with uint64 (NumPy gives a float). */
  private val Pairs = for {
    a <- IntType.names.map(IntType(_))
    b <- IntType.names.map(IntType(_))
    if !((a.signed && b.name == "uint64") || (b.signed && a.name == "uint64"))
  } yield (a, b)

  for (seed <- Seeds())
    it should s"give NumPy's lanes at random types, ops and lanes (seed $seed)" in Seeds.run(seed) { r =>
      for (_ <- 0 until 4) {
        val (ta, tb) = r.pick(Pairs)
        val to       = ElementwiseModel.resultType(ta, tb)
        val tr       = IntType(r.pick(IntType.names))
        val op       = r.pick(ElementwiseOp.all)
        val lanes    = r.pick(Seq(1, 2, 4, 8))
        val clue     = s"${ta.name} ${tb.name} -> ${to.name} -> ${tr.name}, ${ElementwiseOp.name(op)}, $lanes lanes"
        test(new ElementwiseLanes(Seq(ta.name, tb.name), to.name, tr.name, lanes, op)) { dut =>
          for (_ <- 0 until 16) {
            val a = Seq.fill(lanes)(r.word(ta.width))
            val b = Seq.fill(lanes)(r.word(tb.width))
            for (l <- 0 until lanes) {
              LaneBits.poke(dut.io.a(l), a(l))
              LaneBits.poke(dut.io.b(l), b(l))
            }
            for (l <- 0 until lanes) {
              val want = ElementwiseModel.numpy(op, a(l), b(l), ta, tb, to, tr)
              val got  = LaneBits.peek(dut.io.out(l))
              assert(got == want, s"$clue, lane $l: a ${a(l)}, b ${b(l)}: $got, the model $want")
            }
            dut.clock.step()
          }
        }
      }
    }
}
