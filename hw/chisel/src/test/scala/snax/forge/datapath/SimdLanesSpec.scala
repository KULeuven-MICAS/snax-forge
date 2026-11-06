package snax.forge.datapath

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds}
import snax.forge.elementwise.{ElementwiseModel, ElementwiseOp}

/** `SimdLanes`: its types on fixed vectors (D143: each input extended to the op type by its own signedness, the op at
  * the op type, the result extended or wrapped into the output type; each expected value NumPy's, and
  * `ElementwiseModel.numpy`'s too); every op on the edge operands (`ElementwiseModel.edgePairs`: 0, 1, all ones, the
  * sign bit, ...) at int32, uint32, int8 and uint8, where min and max compare by the type's signedness and int8 wraps
  * far more often than int32; then random data against `ElementwiseModel.numpy` at random types, ops and lanes. With
  * one input: its name's `_k`, its refusals, every op at int32 and int64 at the edge constants (0, -1, the type's min
  * and max), then random data and a random or edge k per seed, against `ElementwiseModel.numpy` with b the constant k.
  */
class SimdLanesSpec extends AnyFlatSpec with ForgeTester {

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
    test(new SimdLanes(inTypes, opType, outType, a.size, op)) { dut =>
      for (l <- a.indices) {
        LaneBits.poke(dut.io.a(l), a(l))
        LaneBits.poke(dut.io.b.get(l), b(l))
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

  behavior of "SimdLanes' types"

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
      circt.stage.ChiselStage.emitCHIRRTL(new SimdLanes(Seq("int32", "int32"), "int16", "int32", 1))
    }
    assert(e.getMessage.contains("opType int16 must be at least as wide as each input"), e.getMessage)
  }

  it should "give the model's lanes on the edge operands, every op, at int32, uint32, int8 and uint8" in {
    for {
      t  <- Seq("int32", "uint32", "int8", "uint8").map(IntType(_))
      op <- ElementwiseOp.all
    } {
      val pairs = ElementwiseModel.edgePairs(t.width)
      test(new SimdLanes(Seq(t.name, t.name), t.name, t.name, pairs.size, op)) { dut =>
        for (((a, b), l) <- pairs.zipWithIndex) {
          LaneBits.poke(dut.io.a(l), a)
          LaneBits.poke(dut.io.b.get(l), b)
        }
        for (((a, b), l) <- pairs.zipWithIndex) {
          val want = ElementwiseModel.numpy(op, a, b, t, t, t, t)
          val got  = LaneBits.peek(dut.io.out(l))
          assert(got == want, s"${t.name}, ${ElementwiseModel.clue(op, a, b, l)}: $got, the model $want")
        }
      }
    }
  }

  behavior of "SimdLanes under random data"

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
        test(new SimdLanes(Seq(ta.name, tb.name), to.name, tr.name, lanes, op)) { dut =>
          for (_ <- 0 until 16) {
            val a = Seq.fill(lanes)(r.word(ta.width))
            val b = Seq.fill(lanes)(r.word(tb.width))
            for (l <- 0 until lanes) {
              LaneBits.poke(dut.io.a(l), a(l))
              LaneBits.poke(dut.io.b.get(l), b(l))
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

  behavior of "SimdLanes with one input"

  private val OneOps =
    Seq(ElementwiseOp.Add, ElementwiseOp.Sub, ElementwiseOp.Mul, ElementwiseOp.Min, ElementwiseOp.Max)

  private def edges(t: IntType): Seq[BigInt] = {
    val (lo, hi) =
      if (t.signed) (-(BigInt(1) << (t.width - 1)), (BigInt(1) << (t.width - 1)) - 1)
      else (BigInt(0), LaneBits.mask(t.width))
    Seq(BigInt(0), BigInt(-1), lo, hi).filter(k => SimdLanes.fits(k, t))
  }

  /** `op(a, k)` on 16 beats of random `a`, every lane against the model with b = k at the op type. */
  private def oneInput(r: snax.forge.Rand, t: IntType, tr: IntType, op: Int, k: BigInt, lanes: Int): Unit = {
    val clue = s"${t.name} -> ${tr.name}, ${ElementwiseOp.name(op)}, k $k, $lanes lanes"
    test(new SimdLanes(Seq(t.name), t.name, tr.name, lanes, op, Some(k))) { dut =>
      assert(dut.io.b.isEmpty, s"$clue: a b port")
      for (_ <- 0 until 16) {
        val a = Seq.fill(lanes)(r.word(t.width))
        for (l <- 0 until lanes) LaneBits.poke(dut.io.a(l), a(l))
        for (l <- 0 until lanes) {
          val want = ElementwiseModel.numpy(op, a(l), k & LaneBits.mask(t.width), t, t, t, tr)
          val got  = LaneBits.peek(dut.io.out(l))
          assert(got == want, s"$clue, lane $l: a ${a(l)}: $got, the model $want")
        }
        dut.clock.step()
      }
    }
  }

  it should "end its name in _k and its k, a negative one as _km" in {
    def name(k: BigInt) = {
      val sv = circt.stage.ChiselStage.emitCHIRRTL(
        new SimdLanes(Seq("int32"), "int32", "int32", 4, ElementwiseOp.Max, Some(k))
      )
      raw"module (SimdLanes\w*)".r.findFirstMatchIn(sv).get.group(1)
    }
    assert(name(0) == "SimdLanes_i32_i32_i32_n4_max_k0")
    assert(name(-3) == "SimdLanes_i32_i32_i32_n4_max_km3")
    assert(name(7) == "SimdLanes_i32_i32_i32_n4_max_k7")
  }

  it should "refuse one input without k, two with one, and a k that does not fit the op type" in {
    def refused(body: => SimdLanes, text: String): Unit = {
      val e = intercept[Exception](circt.stage.ChiselStage.emitCHIRRTL(body))
      assert(e.getMessage.contains(text), e.getMessage)
    }
    refused(new SimdLanes(Seq("int32"), "int32", "int32", 4, ElementwiseOp.Max), "one input takes the constant k")
    refused(
      new SimdLanes(Seq("int32", "int32"), "int32", "int32", 4, ElementwiseOp.Max, Some(BigInt(0))),
      "k = 0 is one input's constant; two inputs take none"
    )
    refused(
      new SimdLanes(Seq("int32"), "int32", "int32", 4, ElementwiseOp.Add, Some(BigInt(1) << 31)),
      "k = 2147483648 does not fit opType int32"
    )
    refused(
      new SimdLanes(Seq("uint8"), "uint8", "uint8", 8, ElementwiseOp.Add, Some(BigInt(-1))),
      "k = -1 does not fit opType uint8"
    )
  }

  it should "give the model's lanes at every op at int32 and int64 on the edge constants" in Seeds.run(1) { r =>
    for {
      t  <- Seq("int32", "int64").map(IntType(_))
      k  <- edges(t)
      op <- OneOps
    } oneInput(r, t, t, op, k, 4)
  }

  for (seed <- Seeds())
    it should s"give NumPy's op(a, k) at int32 and int64, random k and data (seed $seed)" in Seeds.run(seed) { r =>
      for {
        t  <- Seq("int32", "int64").map(IntType(_))
        op <- OneOps
      } {
        val k  = if (r.chance(0.3)) r.pick(edges(t)) else LaneBits.signed(r.word(t.width), t.width)
        val tr = IntType(r.pick(IntType.names.filter(IntType(_).width >= t.width)))
        oneInput(r, t, tr, op, k, r.pick(Seq(1, 2, 4, 8)))
      }
    }
}
