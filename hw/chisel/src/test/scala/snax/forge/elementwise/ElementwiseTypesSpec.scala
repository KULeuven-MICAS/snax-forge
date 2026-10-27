package snax.forge.elementwise

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits}

/** `ElementwiseTiledSpatial`'s types on fixed vectors (D143): each input extended to the op type by its own signedness,
  * the op at the op type, the result extended or wrapped into the output type. Each expected value is NumPy's, and
  * `ElementwiseModel.numpy` must give it too.
  */
class ElementwiseTypesSpec extends AnyFlatSpec with ForgeTester {

  /** One firing of the block of these types and ops: lanes `a` and `b` (numbers of their types) in, `out`'s lanes as
    * numbers of `outType`, checked against `want` and the model.
    */
  private def fire(
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
    test(new ElementwiseTiledSpatial(inTypes, opType, outType, a.size, 8, Seq(op))) { dut =>
      dut.io.loopCount.poke(1.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.io.a.valid.poke(true.B)
      dut.io.b.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      for (l <- a.indices) {
        LaneBits.poke(dut.io.a.bits(l), a(l))
        LaneBits.poke(dut.io.b.bits(l), b(l))
      }
      dut.io.out.valid.expect(true.B)
      val got = a.indices.map { l =>
        val bits = LaneBits.peek(dut.io.out.bits(l))
        if (tr.signed) LaneBits.signed(bits, tr.width) else bits
      }
      assert(got == want, s"$clue: lanes $got, NumPy $want")
    }
  }

  private val Max32 = Int.MaxValue // 2^31 - 1

  behavior of "ElementwiseTiledSpatial's types"

  it should "multiply int32 by int32 at int32 and sign-extend into int64" in {
    // Lane 2 wraps at int32 to -2 and is sign-extended, not 4294967294.
    fire(
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
    fire(
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
    fire(Seq("int8", "int8"), "int8", "int32", ElementwiseOp.Max, Seq(BigInt(-112)), Seq(BigInt(0)), Seq(BigInt(0)))
    fire(
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
    fire(
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
      circt.stage.ChiselStage.emitCHIRRTL(new ElementwiseTiledSpatial(Seq("int32", "int32"), "int16", "int32", 1))
    }
    assert(e.getMessage.contains("opType int16 must be at least as wide as each input"), e.getMessage)
  }
}
