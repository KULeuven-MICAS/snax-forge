package snax.forge.datapath

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds}
import snax.forge.elementwise.{ElementwiseModel, ElementwiseOp}

/** `SimdLanes`: its types on fixed vectors (each input extended to the op type by its own signedness, the op at the op
  * type, the result extended or wrapped into the output type; each expected value NumPy's, and
  * `ElementwiseModel.numpy`'s too); every op on the edge operands (`ElementwiseModel.edgePairs`: 0, 1, all ones, the
  * sign bit, ...) at int32, uint32, int8 and uint8, where min and max compare by the type's signedness and int8 wraps
  * far more often than int32; then random data against `ElementwiseModel.numpy` at random types, ops and lanes. With
  * one input: its name's `_k`, its refusals, every op at int32 and int64 at the edge constants (0, -1, the type's min
  * and max), then random data and a random or edge k per seed, against `ElementwiseModel.numpy` with b the constant k;
  * and k from its pin (a CSR), random per beat. axpby (`k1 * a + k2 * b` at the op type, `k2` unset `+ b`): its names
  * and refusals, then per seed at int32 and int64 (and a narrower `b` extended into int64) random data with k1 from its
  * CSR pin, k1 baked, k1 and k2 both from pins, and k1 baked with k2 from its pin, each against `axpbyModel`: NumPy's
  * products and sum at the op type, wrapping, cast into the output's type. requant, axpbypcz and threshold: names,
  * refusals, the worked lanes, the edge slots (k at the type's min and max, lo = hi, d = 1 and the largest power of two
  * the op type holds) at 1, 4 and 8 lanes, then per seed random slots and data, each CSR slot rewritten every few
  * beats, against `NamedModel`, written from the options' definitions in values, not bits. With `narrowMul` (D207): its
  * name's `_nm` where a product narrows and the default's module where none can (int32 x int32, a k from its pin or as
  * wide as opType, an op without a product), then per seed mul (two inputs, and one with k baked at int8's, int16's and
  * int32's range or from its pin), axpby and axpbypcz at int8 and uint8 inputs -> int32, on and off, against the models
  * above.
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

  it should "take k from its pin (a CSR), random per beat, at int32 and int64" in Seeds.run(Seeds().head) { r =>
    for {
      t  <- Seq("int32", "int64").map(IntType(_))
      op <- OneOps
    } {
      val clue = s"${t.name}, ${ElementwiseOp.name(op)}, k from its pin"
      test(new SimdLanes(Seq(t.name), t.name, t.name, 4, op, kCsr = true)) { dut =>
        for (_ <- 0 until 16) {
          val k = r.word(t.width)
          val a = Seq.fill(4)(r.word(t.width))
          LaneBits.poke(dut.io.k.get, k)
          for (l <- 0 until 4) LaneBits.poke(dut.io.a(l), a(l))
          for (l <- 0 until 4) {
            val want = ElementwiseModel.numpy(op, a(l), k, t, t, t, t)
            assert(LaneBits.peek(dut.io.out(l)) == want, s"$clue, lane $l: a ${a(l)}, k $k")
          }
          dut.clock.step()
        }
      }
    }
  }

  behavior of "SimdLanes at axpby"

  /** NumPy's `k1 * a + k2 * b` (`k2` None: `+ b`): a and b cast to the op type, the products and the sum wrapping
    * there, the result cast to `outType`. Bit patterns in and out; k1 and k2 the op type's.
    */
  private def axpbyModel(
    a:  BigInt,
    b:  BigInt,
    k1: BigInt,
    k2: Option[BigInt],
    ta: IntType,
    tb: IntType,
    to: IntType,
    tr: IntType
  ): BigInt = {
    val (x, y) = (ElementwiseModel.cast(a, ta, to), ElementwiseModel.cast(b, tb, to))
    val sum    = (k1 * x + k2.map(_ * y).getOrElse(y)) & LaneBits.mask(to.width)
    ElementwiseModel.cast(sum, to, tr)
  }

  private def axpbyName(k1: Option[BigInt], k1Csr: Boolean, k2: Option[BigInt], k2Csr: Boolean): String = {
    val sv = circt.stage.ChiselStage.emitCHIRRTL(
      new SimdLanes(
        Seq("int32", "int32"),
        "int32",
        "int32",
        4,
        SimdLanes.Axpby,
        k1    = k1,
        k2    = k2,
        k1Csr = k1Csr,
        k2Csr = k2Csr
      )
    )
    raw"module (SimdLanes\w*)".r.findFirstMatchIn(sv).get.group(1)
  }

  it should "end its name in _axpby and each slot's _k1<v>, _k1m<v> or _k1r (k2 likewise)" in {
    assert(axpbyName(Some(3), false, None, false) == "SimdLanes_i32xi32_i32_i32_n4_axpby_k13")
    assert(axpbyName(None, true, None, false) == "SimdLanes_i32xi32_i32_i32_n4_axpby_k1r")
    assert(axpbyName(Some(-3), false, None, true) == "SimdLanes_i32xi32_i32_i32_n4_axpby_k1m3_k2r")
    assert(axpbyName(None, true, Some(7), false) == "SimdLanes_i32xi32_i32_i32_n4_axpby_k1r_k27")
    val one = circt.stage.ChiselStage.emitCHIRRTL(
      new SimdLanes(Seq("int32"), "int32", "int32", 4, ElementwiseOp.Mul, kCsr = true)
    )
    assert(one.contains("module SimdLanes_i32_i32_i32_n4_mul_kr"))
  }

  it should "refuse axpby without k1, with k1 twice, at one input, and k1 on another op" in {
    def refused(body: => SimdLanes, text: String): Unit = {
      val e = intercept[Exception](circt.stage.ChiselStage.emitCHIRRTL(body))
      assert(e.getMessage.contains(text), e.getMessage)
    }
    val two = Seq("int32", "int32")
    refused(new SimdLanes(two, "int32", "int32", 4, SimdLanes.Axpby), "axpby takes k1, baked or from its CSR")
    refused(
      new SimdLanes(two, "int32", "int32", 4, SimdLanes.Axpby, k1 = Some(1), k1Csr = true),
      "axpby takes k1, baked or from its CSR, not both"
    )
    refused(new SimdLanes(Seq("int32"), "int32", "int32", 4, SimdLanes.Axpby, k = Some(1)), "axpby takes two inputs")
    refused(
      new SimdLanes(two, "int32", "int32", 4, ElementwiseOp.Add, k1 = Some(1)),
      "k1 and k2 are axpby's and axpbypcz's, not add's"
    )
    refused(
      new SimdLanes(two, "int32", "int32", 4, SimdLanes.Axpby, k1 = Some(BigInt(1) << 31)),
      "k1 = 2147483648 does not fit opType int32"
    )
    refused(
      new SimdLanes(Seq("int32"), "int32", "int32", 4, ElementwiseOp.Add, k = Some(1), kCsr = true),
      "k = 1 is baked and read from its CSR too"
    )
  }

  /** The four forms: (k1 baked?, k2: None, baked or from its pin). */
  private val Forms = Seq(("csr", "none"), ("baked", "none"), ("csr", "csr"), ("baked", "csr"))

  for (seed <- Seeds())
    it should s"give NumPy's k1 * a + k2 * b in every form at int32 and int64 (seed $seed)" in Seeds.run(seed) { r =>
      for {
        (ins, to) <- Seq(
          (Seq("int32", "int32"), "int32"),
          (Seq("int64", "int64"), "int64"),
          (Seq("int64", "int16"), "int64")
        )
        (f1, f2)  <- Forms
      } {
        val (ta, tb, tOp) = (IntType(ins(0)), IntType(ins(1)), IntType(to))
        val tr            = IntType(r.pick(IntType.names.filter(IntType(_).width >= tOp.width)))
        val lanes         = r.pick(Seq(1, 2, 4))
        val baked         = LaneBits.signed(r.word(tOp.width), tOp.width)
        val clue          = s"${ins.mkString(" ")} -> $to -> ${tr.name}, k1 $f1, k2 $f2, $lanes lanes"
        test(
          new SimdLanes(
            ins,
            to,
            tr.name,
            lanes,
            SimdLanes.Axpby,
            k1    = if (f1 == "baked") Some(baked) else None,
            k1Csr = f1 == "csr",
            k2Csr = f2 == "csr"
          )
        ) { dut =>
          for (_ <- 0 until 16) {
            val k1 = if (f1 == "baked") baked & LaneBits.mask(tOp.width) else r.word(tOp.width)
            val k2 = if (f2 == "csr") Some(r.word(tOp.width)) else None
            dut.io.k1.foreach(LaneBits.poke(_, k1))
            dut.io.k2.foreach(LaneBits.poke(_, k2.get))
            val a  = Seq.fill(lanes)(r.word(ta.width))
            val b  = Seq.fill(lanes)(r.word(tb.width))
            for (l <- 0 until lanes) {
              LaneBits.poke(dut.io.a(l), a(l))
              LaneBits.poke(dut.io.b.get(l), b(l))
            }
            for (l <- 0 until lanes) {
              val want = axpbyModel(a(l), b(l), k1, k2, ta, tb, tOp, tr)
              val got  = LaneBits.peek(dut.io.out(l))
              assert(got == want, s"$clue, lane $l: a ${a(l)}, b ${b(l)}, k1 $k1, k2 $k2: $got, the model $want")
            }
            dut.clock.step()
          }
        }
      }
    }

  behavior of "SimdLanes at requant, axpbypcz and threshold"

  private def refused(body: => SimdLanes, text: String): Unit = {
    val e = intercept[Exception](circt.stage.ChiselStage.emitCHIRRTL(body))
    assert(e.getMessage.contains(text), e.getMessage)
  }

  private def nameOf(body: => SimdLanes): String =
    raw"module (SimdLanes\w*)".r.findFirstMatchIn(circt.stage.ChiselStage.emitCHIRRTL(body)).get.group(1)

  it should "end its name in the op and each slot, _lo, _hi and _d with m for minus" in {
    assert(
      nameOf(
        new SimdLanes(Seq("int32"), "int32", "int8", 8, SimdLanes.Requant, d = Some(1024), lo = Some(0), hi = Some(127))
      ) == "SimdLanes_i32_i32_i8_n8_requant_d1024_lo0_hi127"
    )
    assert(
      nameOf(
        new SimdLanes(Seq("int16"), "int16", "int8", 4, SimdLanes.Requant, d = Some(1), lo = Some(-5), hi = Some(-5))
      ) == "SimdLanes_i16_i16_i8_n4_requant_d1_lom5_him5"
    )
    val u8 = Seq.fill(3)("uint8")
    assert(
      nameOf(
        new SimdLanes(
          u8,
          "uint16",
          "uint8",
          8,
          SimdLanes.Axpbypcz,
          k1 = Some(77),
          k2 = Some(150),
          k3 = Some(29),
          d  = Some(256)
        )
      ) == "SimdLanes_u8xu8xu8_u16_u8_n8_axpbypcz_k177_k2150_k329_d256"
    )
    assert(
      nameOf(new SimdLanes(u8, "uint16", "uint8", 8, SimdLanes.Axpbypcz, k2 = Some(2), d = Some(4))) ==
        "SimdLanes_u8xu8xu8_u16_u8_n8_axpbypcz_k22_d4"
    )
    assert(
      nameOf(new SimdLanes(u8, "uint16", "uint8", 8, SimdLanes.Axpbypcz, k1Csr = true, k3Csr = true, d = Some(1))) ==
        "SimdLanes_u8xu8xu8_u16_u8_n8_axpbypcz_k1r_k3r_d1"
    )
    assert(
      nameOf(new SimdLanes(Seq("uint8"), "int16", "uint8", 8, SimdLanes.Threshold, kCsr = true, hi = Some(255))) ==
        "SimdLanes_u8_i16_u8_n8_threshold_kr_hi255"
    )
    assert(
      nameOf(new SimdLanes(Seq("uint8"), "int16", "uint8", 8, SimdLanes.Threshold, Some(-32768), hi = Some(255))) ==
        "SimdLanes_u8_i16_u8_n8_threshold_km32768_hi255"
    )
  }

  it should "refuse a slot not the op's, a missing one, the wrong stream count, and a d that is no power of two" in {
    val (one, two, three) = (Seq("int32"), Seq("int32", "int32"), Seq("int32", "int32", "int32"))
    def rq(
      ins: Seq[String]    = one,
      d:   Option[BigInt] = Some(4),
      lo:  Option[BigInt] = Some(0),
      hi:  Option[BigInt] = Some(9)
    ) = new SimdLanes(ins, "int32", "int32", 4, SimdLanes.Requant, d = d, lo = lo, hi = hi)
    refused(rq(two), "requant takes one input")
    refused(rq(three), "three inputs are axpbypcz's, not requant's")
    refused(rq(d = None), "requant takes d, baked (1 for no shift)")
    refused(rq(lo = None), "requant takes lo and hi, both baked")
    refused(rq(d = Some(6)), "d = 6 must be a power of two at least 1: the lanes shift by log2(d)")
    refused(rq(d = Some(0)), "d = 0 must be a power of two at least 1")
    refused(rq(lo = Some(BigInt(1) << 31)), "lo = 2147483648 does not fit opType int32")
    refused(
      new SimdLanes(one, "int32", "int32", 4, SimdLanes.Requant, Some(1), d = Some(4), lo = Some(0), hi = Some(9)),
      "k = 1 is not requant's; it takes d, lo and hi"
    )
    // the largest power of two a signed type holds is 2^(w-2): 2^(w-1) is its min's magnitude
    refused(
      new SimdLanes(Seq("int16"), "int16", "int8", 4, SimdLanes.Requant, d = Some(32768), lo = Some(0), hi = Some(9)),
      "d = 32768 does not fit opType int16"
    )
    refused(new SimdLanes(two, "int32", "int32", 4, SimdLanes.Axpbypcz, d = Some(1)), "axpbypcz takes three inputs")
    refused(new SimdLanes(three, "int32", "int32", 4, ElementwiseOp.Add), "three inputs are axpbypcz's, not add's")
    refused(new SimdLanes(three, "int32", "int32", 4, SimdLanes.Axpbypcz), "axpbypcz takes d, baked (1 for no shift)")
    refused(
      new SimdLanes(three, "int32", "int32", 4, SimdLanes.Axpbypcz, k3 = Some(1), k3Csr = true, d = Some(1)),
      "k3 = 1 is baked and read from its CSR too"
    )
    refused(
      new SimdLanes(three, "int32", "int32", 4, SimdLanes.Axpbypcz, k1 = Some(1), k1Csr = true, d = Some(1)),
      "k1 = 1 is baked and read from its CSR too"
    )
    refused(
      new SimdLanes(three, "int32", "int32", 4, SimdLanes.Axpbypcz, d = Some(1), lo = Some(0)),
      "lo is requant's, not axpbypcz's"
    )
    refused(
      new SimdLanes(two, "int32", "int32", 4, SimdLanes.Axpby, k1 = Some(1), k3 = Some(1)),
      "k3 is axpbypcz's, not axpby's"
    )
    refused(
      new SimdLanes(two, "int32", "int32", 4, SimdLanes.Axpby, k1 = Some(1), d = Some(2)),
      "d is axpbypcz's and requant's, not axpby's"
    )
    refused(
      new SimdLanes(one, "int32", "int32", 4, SimdLanes.Threshold, hi = Some(1)),
      "threshold takes k, baked or from its CSR"
    )
    refused(new SimdLanes(one, "int32", "int32", 4, SimdLanes.Threshold, Some(1)), "threshold takes hi, baked")
    refused(
      new SimdLanes(two, "int32", "int32", 4, SimdLanes.Threshold, hi = Some(1)),
      "threshold takes one input"
    )
    refused(
      new SimdLanes(one, "int32", "int32", 4, ElementwiseOp.Max, Some(0), hi = Some(1)),
      "hi is requant's and threshold's, not max's"
    )
  }

  /** One SimdLanes config of a named op: types, lanes and slots (a baked value, or None and its pin). */
  private case class Named(
    op:     Int,
    ins:    Seq[IntType],
    opT:    IntType,
    out:    IntType,
    lanes:  Int,
    k:      Option[BigInt]      = None,
    kCsr:   Boolean             = false,
    ks:     Seq[Option[BigInt]] = Seq(None, None, None),
    kCsrs:  Seq[Boolean]        = Seq(false, false, false),
    d:      Option[BigInt]      = None,
    lo:     Option[BigInt]      = None,
    hi:     Option[BigInt]      = None,
    narrow: Boolean             = false
  ) {
    def build: SimdLanes =
      new SimdLanes(
        ins.map(_.name),
        opT.name,
        out.name,
        lanes,
        op,
        k,
        ks(0),
        ks(1),
        kCsr,
        kCsrs(0),
        kCsrs(1),
        ks(2),
        kCsrs(2),
        d,
        lo,
        hi,
        narrow
      )

    def clue: String =
      s"${SimdLanes.opName(op)} ${ins.map(_.name).mkString(" ")} -> ${opT.name} -> ${out.name}, $lanes lanes, k " +
        s"${if (kCsr) "csr" else k.getOrElse("-")}, ks ${ks
            .zip(kCsrs)
            .map { case (v, c) => if (c) "csr" else v.getOrElse("-") }
            .mkString(" ")}, d ${d.getOrElse("-")}, lo ${lo.getOrElse("-")}, hi ${hi.getOrElse("-")}" +
        (if (narrow) ", narrowMul" else "")
  }

  /** The options' lanes, from their definitions: every value an integer of its type, NumPy's arithmetic at the op type
    * (each step wrapped there as `astype` does), floor division by d, min and max as integers of the op type, and the
    * result cast into the output's type. Bit patterns of the inputs' types in, of the output's out; a CSR slot is given
    * as its bits.
    */
  private object NamedModel {

    /** The integer of type `t` with bits `v` (its low width bits). */
    def value(v: BigInt, t: IntType): BigInt = if (t.signed) LaneBits.signed(v, t.width) else v & LaneBits.mask(t.width)

    /** `v`, any integer, wrapped into `t` (`astype`). */
    def wrap(v: BigInt, t: IntType): BigInt = value(v & LaneBits.mask(t.width), t)

    /** NumPy's `x // d`: rounded towards minus infinity. */
    def floorDiv(x: BigInt, d: BigInt): BigInt = {
      val (q, r) = x /% d
      if (r != 0 && (r < 0) != (d < 0)) q - 1 else q
    }

    private def in(x: BigInt, from: IntType, opT: IntType): BigInt = wrap(value(x, from), opT)

    private def out(v: BigInt, opT: IntType, t: IntType): BigInt = wrap(v, opT) & LaneBits.mask(t.width)

    /** One lane of `c` on the inputs `xs` (bits of `c.ins`), the CSR slots' bits in `csr` (k, k1, k2, k3). */
    def lane(c: Named, xs: Seq[BigInt], csr: Map[String, BigInt]): BigInt = {
      val t = c.opT
      val v = xs.zip(c.ins).map { case (x, ti) => in(x, ti, t) }
      def at(s: String, baked: Option[BigInt], fromCsr: Boolean): Option[BigInt] =
        if (fromCsr) Some(value(csr(s), t)) else baked
      val r = c.op match {
        case SimdLanes.Requant   => floorDiv(v(0), c.d.get).max(c.lo.get).min(c.hi.get)
        case SimdLanes.Axpbypcz  =>
          val terms = Seq("k1", "k2", "k3").indices.map { i =>
            at(s"k${i + 1}", c.ks(i), c.kCsrs(i)).map(kv => wrap(kv * v(i), t)).getOrElse(v(i))
          }
          floorDiv(wrap(terms.sum, t), c.d.get)
        case SimdLanes.Threshold =>
          val delta = wrap(v(0) - at("k", c.k, c.kCsr).get + 1, t)
          if (delta >= 1) c.hi.get else BigInt(0)
      }
      out(r, t, c.out)
    }
  }

  private def range(t: IntType): (BigInt, BigInt) =
    if (t.signed) (-(BigInt(1) << (t.width - 1)), (BigInt(1) << (t.width - 1)) - 1)
    else (BigInt(0), LaneBits.mask(t.width))

  /** The largest power of two `t` holds: 2^(w-1) unsigned, 2^(w-2) signed. */
  private def dMax(t: IntType): BigInt = BigInt(1) << (t.width - (if (t.signed) 2 else 1))

  /** A random value of `t`; one in four at its min or max. */
  private def any(r: snax.forge.Rand, t: IntType): BigInt =
    if (r.chance(0.25)) r.pick(Seq(range(t)._1, range(t)._2)) else value(r.word(t.width), t)

  private def value(bits: BigInt, t: IntType) = NamedModel.value(bits, t)

  /** `c` over `beats` beats of random inputs, each lane against `NamedModel`; every CSR slot written again every
    * `every` beats, at random or (one in four) at its type's min or max.
    */
  private def drive(r: snax.forge.Rand, c: Named, beats: Int = 16, every: Int = 3): Unit =
    test(c.build) { dut =>
      val pins = Seq("k" -> dut.io.k, "k1" -> dut.io.k1, "k2" -> dut.io.k2, "k3" -> dut.io.k3).collect {
        case (s, Some(p)) => s -> p
      }
      assert(dut.io.c.isDefined == (c.ins.size == 3), s"${c.clue}: a c port at ${c.ins.size} inputs")
      var csr  = Map.empty[String, BigInt]
      for (beat <- 0 until beats) {
        if (beat % every == 0)
          for ((s, p) <- pins) {
            val v = any(r, c.opT) & LaneBits.mask(c.opT.width)
            csr += s -> v
            LaneBits.poke(p, v)
          }
        val xs = Seq.fill(c.lanes)(c.ins.map(t => r.word(t.width)))
        val port = Seq(Some(dut.io.a), dut.io.b, dut.io.c)
        for {
          l    <- 0 until c.lanes
          i    <- c.ins.indices
        } LaneBits.poke(port(i).get(l), xs(l)(i))
        for (l <- 0 until c.lanes) {
          val want = laneModel(c, xs(l), csr)
          val got  = LaneBits.peek(dut.io.out(l))
          assert(
            got == want,
            s"${c.clue}, beat $beat, lane $l: in ${xs(l).mkString(" ")}, csr $csr: $got, the model $want"
          )
        }
        dut.clock.step()
      }
    }

  /** The given lanes of `c`, inputs as values of their types, each output a value of `c.out`. */
  private def worked(c: Named, xs: Seq[Seq[BigInt]], want: Seq[BigInt]): Unit = {
    val bits  = xs.map(_.zip(c.ins).map { case (v, t) => v & LaneBits.mask(t.width) })
    val wantB = want.map(_ & LaneBits.mask(c.out.width))
    assert(bits.map(NamedModel.lane(c, _, Map.empty)) == wantB, s"${c.clue}: the model")
    test(c.build) { dut =>
      val port = Seq(Some(dut.io.a), dut.io.b, dut.io.c)
      for {
        l <- bits.indices
        i <- c.ins.indices
      } LaneBits.poke(port(i).get(l), bits(l)(i))
      val got = bits.indices.map(l => value(LaneBits.peek(dut.io.out(l)), c.out))
      assert(got == want, s"${c.clue}: lanes $got, NumPy $want")
    }
  }

  private val (i32, i16, i8, u32, u16, u8) =
    (IntType("int32"), IntType("int16"), IntType("int8"), IntType("uint32"), IntType("uint16"), IntType("uint8"))

  it should "give the worked lanes: requant, gray and blur at axpbypcz, threshold at the type's ends" in {
    def big(xs: Int*) = xs.map(BigInt(_))
    worked(
      Named(SimdLanes.Requant, Seq(i32), i32, i8, 2, d = Some(1024), lo = Some(0), hi = Some(127)),
      Seq(big(150000), big(-5000)),
      big(127, 0)
    )
    worked(
      Named(SimdLanes.Axpbypcz, Seq(u8, u8, u8), u16, u8, 1, ks = big(77, 150, 29).map(Some(_)), d = Some(256)),
      Seq(big(200, 100, 50)),
      big(124)
    )
    worked(
      Named(SimdLanes.Axpbypcz, Seq(u8, u8, u8), u16, u8, 1, ks = Seq(None, Some(2), None), d = Some(4)),
      Seq(big(10, 20, 30)),
      big(20)
    )
    worked(Named(SimdLanes.Threshold, Seq(u8), i16, u8, 1, Some(128), hi = Some(255)), Seq(big(128)), big(255))
    // a - k + 1 = 33024 wraps at int16 to -32512: 0, where a comparator a >= k would give 255
    worked(Named(SimdLanes.Threshold, Seq(u8), i16, u8, 1, Some(-32768), hi = Some(255)), Seq(big(255)), big(0))
  }

  /** requant's, axpbypcz's and threshold's type triples (in, op, out): the worked ones and signed and unsigned others.
    */
  private val RequantTypes   = Seq((i32, i32, i8), (i16, i16, i8), (u32, u32, u8))
  private val AxpbypczTypes  = Seq((u8, u16, u8), (i32, i32, i32), (i32, i32, i8))
  private val ThresholdTypes = Seq((u8, i16, u8), (u16, u16, u16), (i32, i32, i8))

  private val Lanes = Seq(1, 4, 8)

  it should "give the model's lanes at the edge slots, at 1, 4 and 8 lanes" in Seeds.run(1) { r =>
    var at      = 0
    def lanes() = { at += 1; Lanes(at % Lanes.size) }
    for ((ti, t, tr) <- RequantTypes) {
      val (lo, hi) = range(t)
      val edges    = Seq(
        (BigInt(1), lo, hi),
        (dMax(t), lo, hi),
        (BigInt(1), BigInt(0), BigInt(0)),
        (dMax(t), hi, hi),
        (BigInt(4), lo, lo),
        (BigInt(2), hi, lo) // lo above hi: NumPy's min(max(q, lo), hi) is hi
      )
      for ((d, l, h) <- edges)
        drive(r, Named(SimdLanes.Requant, Seq(ti), t, tr, lanes(), d = Some(d), lo = Some(l), hi = Some(h)))
    }
    for ((ti, t, tr) <- AxpbypczTypes) {
      val (lo, hi) = range(t)
      for {
        ks <- Seq(Seq(lo, lo, lo), Seq(hi, hi, hi), Seq(lo, hi, BigInt(1))).map(_.map(Some(_))) :+ Seq(None, None, None)
        d  <- Seq(BigInt(1), dMax(t))
      } drive(r, Named(SimdLanes.Axpbypcz, Seq.fill(3)(ti), t, tr, lanes(), ks = ks, d = Some(d)))
    }
    for ((ti, t, tr) <- ThresholdTypes) {
      val (lo, hi) = range(t)
      for {
        k <- Seq(lo, hi, BigInt(0), BigInt(1))
        h <- Seq(hi, BigInt(1))
      } drive(r, Named(SimdLanes.Threshold, Seq(ti), t, tr, lanes(), Some(k), hi = Some(h)))
    }
  }

  /** A slot of `t`: baked (a random value, a third of them at the type's ends) or, at `csr`, None and its pin. */
  private def slotOf(r: snax.forge.Rand, t: IntType): (Option[BigInt], Boolean) =
    r.between(0, 1) match {
      case 0 => (Some(any(r, t)), false)
      case _ => (None, true)
    }

  for (seed <- Seeds())
    it should s"give the model's requant, axpbypcz and threshold at random slots and data (seed $seed)" in Seeds.run(
      seed
    ) { r =>
      def d(t: IntType) =
        if (r.chance(0.3)) r.pick(Seq(BigInt(1), dMax(t))) else BigInt(1) << r.between(0, dMax(t).bitLength - 1)
      for ((ti, t, tr) <- RequantTypes) {
        val (lo, hi) = (any(r, t), any(r, t))
        val (l, h)   = if (r.chance(0.2)) (lo, lo) else (lo.min(hi), lo.max(hi))
        drive(r, Named(SimdLanes.Requant, Seq(ti), t, tr, r.pick(Lanes), d = Some(d(t)), lo = Some(l), hi = Some(h)))
      }
      for ((ti, t, tr) <- AxpbypczTypes) {
        val slots = Seq.fill(3)(if (r.chance(0.25)) (None, false) else slotOf(r, t))
        drive(
          r,
          Named(
            SimdLanes.Axpbypcz,
            Seq.fill(3)(ti),
            t,
            tr,
            r.pick(Lanes),
            ks    = slots.map(_._1),
            kCsrs = slots.map(_._2),
            d     = Some(d(t))
          )
        )
      }
      for ((ti, t, tr) <- ThresholdTypes) {
        val (k, csr) = slotOf(r, t)
        drive(r, Named(SimdLanes.Threshold, Seq(ti), t, tr, r.pick(Lanes), k, csr, hi = Some(any(r, t))))
      }
    }

  behavior of "SimdLanes with narrowMul"

  /** `c`'s lane by the models above: NumPy's mul (`ElementwiseModel.numpy`, one input's k as b at the op type),
    * `axpbyModel`, else `NamedModel`. A CSR slot's bits in `csr`.
    */
  private def laneModel(c: Named, xs: Seq[BigInt], csr: Map[String, BigInt]): BigInt = {
    def slot(s: String, v: Option[BigInt], pin: Boolean): Option[BigInt] =
      if (pin) Some(csr(s)) else v.map(_ & LaneBits.mask(c.opT.width))
    c.op match {
      case ElementwiseOp.Mul if c.ins.size == 2 =>
        ElementwiseModel.numpy(c.op, xs(0), xs(1), c.ins(0), c.ins(1), c.opT, c.out)
      case ElementwiseOp.Mul                    =>
        ElementwiseModel.numpy(c.op, xs(0), slot("k", c.k, c.kCsr).get, c.ins(0), c.opT, c.opT, c.out)
      case SimdLanes.Axpby                      =>
        val (k1, k2) = (slot("k1", c.ks(0), c.kCsrs(0)).get, slot("k2", c.ks(1), c.kCsrs(1)))
        axpbyModel(xs(0), xs(1), k1, k2, c.ins(0), c.ins(1), c.opT, c.out)
      case _                                    => NamedModel.lane(c, xs, csr)
    }
  }

  private def nameAt(nm: Boolean)(body: Boolean => SimdLanes): String = nameOf(body(nm))

  it should "end its name in _nm where a product narrows, and be the default's module where none can" in {
    val two = Seq("int8", "int8")
    def mul(nm: Boolean) = new SimdLanes(two, "int32", "int32", 8, ElementwiseOp.Mul, narrowMul = nm)
    def axpby(nm: Boolean) =
      new SimdLanes(two, "int32", "int32", 8, SimdLanes.Axpby, k1 = Some(3), k2Csr = true, narrowMul = nm)
    def pcz(nm: Boolean)   =
      new SimdLanes(
        Seq.fill(3)("uint8"),
        "int32",
        "int32",
        8,
        SimdLanes.Axpbypcz,
        k1        = Some(77),
        k2        = Some(150),
        k3        = Some(29),
        d         = Some(256),
        narrowMul = nm
      )
    assert(nameAt(true)(mul) == "SimdLanes_i8xi8_i32_i32_n8_mul_nm")
    assert(nameAt(true)(axpby) == "SimdLanes_i8xi8_i32_i32_n8_axpby_k13_k2r_nm")
    assert(nameAt(true)(pcz) == "SimdLanes_u8xu8xu8_i32_i32_n8_axpbypcz_k177_k2150_k329_d256_nm")
    assert(nameAt(false)(pcz) == "SimdLanes_u8xu8xu8_i32_i32_n8_axpbypcz_k177_k2150_k329_d256")
    val none: Seq[(String, Boolean => SimdLanes)] = Seq(
      "int32 x int32"   -> (nm =>
        new SimdLanes(Seq("int32", "int32"), "int32", "int32", 8, ElementwiseOp.Mul, narrowMul = nm)
      ),
      "k from its pin"  -> (nm =>
        new SimdLanes(Seq("int8"), "int32", "int32", 8, ElementwiseOp.Mul, kCsr = true, narrowMul = nm)
      ),
      "k1 from its pin" -> (nm =>
        new SimdLanes(two, "int32", "int32", 8, SimdLanes.Axpby, k1Csr = true, narrowMul = nm)
      ),
      "k1 an int32"     -> (nm =>
        new SimdLanes(two, "int32", "int32", 8, SimdLanes.Axpby, k1 = Some(BigInt(1) << 20), narrowMul = nm)
      ),
      "add"             -> (nm => new SimdLanes(two, "int32", "int32", 8, ElementwiseOp.Add, narrowMul = nm))
    )
    for ((what, m) <- none) {
      val Seq(wide, narrow) = Seq(false, true).map(nm => circt.stage.ChiselStage.emitCHIRRTL(m(nm)))
      assert(wide == narrow, s"$what: narrowMul changed a module with no product it narrows")
    }
  }

  /** A k for a narrow product at int32: baked at int8's, int16's or int32's range (`w` 0, 1, 2; one in four at the
    * range's ends), or (3) None and its pin.
    */
  private def kAt(r: snax.forge.Rand, w: Int): (Option[BigInt], Boolean) =
    if (w == 3) (None, true) else (Some(any(r, IntType(8 << w, signed = true))), false)

  private val NarrowIns = Seq(Seq(i8, i8), Seq(i8, u8), Seq(u8, u8))

  for (seed <- Seeds())
    it should s"give the models' lanes with narrowMul on and off: mul, axpby and axpbypcz at int8 -> int32 (seed $seed)" in Seeds
      .run(seed) { r =>
        for (narrow <- Seq(false, true)) {
          for (ins   <- NarrowIns) drive(r, Named(ElementwiseOp.Mul, ins, i32, i32, 8, narrow = narrow))
          for (w     <- 0 to 3) {
            val (k, csr) = kAt(r, w)
            drive(r, Named(ElementwiseOp.Mul, Seq(r.pick(Seq(i8, u8))), i32, i32, 8, k, csr, narrow = narrow))
          }
          for {
            ins      <- NarrowIns
            (w1, w2) <- Seq((0, -1), (0, 1), (1, 3), (3, 0), (3, 3), (2, 0))
          } {
            val (k1, c1) = kAt(r, w1)
            val (k2, c2) = if (w2 < 0) (None, false) else kAt(r, w2)
            drive(
              r,
              Named(
                SimdLanes.Axpby,
                ins,
                i32,
                i32,
                8,
                ks     = Seq(k1, k2, None),
                kCsrs  = Seq(c1, c2, false),
                narrow = narrow
              )
            )
          }
          for (ws    <- Seq(Seq(0, 0, 0), Seq(3, 3, 3), Seq(0, 1, 2), Seq.fill(3)(r.between(-1, 3)))) {
            val slots = ws.map(w => if (w < 0) (None, false) else kAt(r, w))
            drive(
              r,
              Named(
                SimdLanes.Axpbypcz,
                Seq.fill(3)(r.pick(Seq(i8, u8))),
                i32,
                i32,
                8,
                ks     = slots.map(_._1),
                kCsrs  = slots.map(_._2),
                d      = Some(BigInt(1) << r.between(0, 8)),
                narrow = narrow
              )
            )
          }
        }
      }
}
