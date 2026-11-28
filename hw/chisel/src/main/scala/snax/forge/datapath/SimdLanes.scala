package snax.forge.datapath

import chisel3._

import snax.forge.IntType
import snax.forge.elementwise.{ElementwiseAlu, ElementwiseOp}

/** `lanes` lanes of one op: the datapath of a SIMD unit, latency 0, no pins but its scalars'.
  *
  * Data only: no valid, ready or counter. Its group's control unit (`ctrl.GroupCtrl`) decides when a beat is taken and
  * the result pushed; in that cycle `out` is the op on `a` and `b`, lane by lane. With one input (one stream) there is
  * no `b`: each lane's second operand is the constant `k` at `opType`, baked in at elaboration, so `out` is `op(a, k)`
  * (relu's `max(a, 0)`), or, with `kCsr`, the input pin `k` (opType's width, the unit's K CSR, written by the program
  * at run time); the module's name ends in `_k<k>` (`_km3` for -3), or `_kr`. Two inputs build the same module, by the
  * same name, as before `k` existed.
  *
  * ==axpby (op `SimdLanes.Axpby`)==
  *
  * Two inputs, per lane `k1 * a + k2 * b` at `opType` (`k2` unset: `k1 * a + b`), wrapped there and then cast into
  * `outType`. Each of `k1` and `k2` is a constant baked in at elaboration (`k1`, `k2`) or an input pin of opType's
  * width (`k1Csr`, `k2Csr`: the unit's K1 or K2 CSR); `k1` is one of the two, `k2` may be neither. The name ends in
  * `_axpby` and, per slot, `_k1<v>` (`_k1m3`) or `_k1r`, then `_k2<v>` or `_k2r`.
  *
  * ==axpbypcz (op `SimdLanes.Axpbypcz`)==
  *
  * Three inputs (a port `c`), per lane `(k1 * a + k2 * b + k3 * c) >> log2(d)` at `opType`: each of `k1`, `k2` and `k3`
  * baked, from its pin (`k1Csr`, `k2Csr`, `k3Csr`) or unset, an unset one no multiplier (the input itself); each
  * product and the sum the low opType bits, as axpby's; `d` a power of two, baked, the shift requant's (`d` = 1: none).
  * The name ends in `_axpbypcz`, then each slot's `_k1<v>` or `_k1r`, `_k2..`, `_k3..`, then `_d<v>`.
  *
  * ==requant (op `SimdLanes.Requant`)==
  *
  * One input, per lane `min(max(a >> log2(d), lo), hi)` at `opType`, `d`, `lo` and `hi` all baked: the shift arithmetic
  * at a signed opType and logical at an unsigned one, NumPy's floor `a // d`; min and max compare by opType's
  * signedness. The name ends in `_requant_d<d>_lo<lo>_hi<hi>` (`_lom5` for -5).
  *
  * ==threshold (op `SimdLanes.Threshold`)==
  *
  * One input, per lane `hi * min(max(a - k + 1, 0), 1)` at `opType`: a subtractor gives `a - k + 1`, wrapped at opType,
  * and the lane is `hi` where that is at least 1 by opType's signedness, else 0. `k` is baked or from its pin (`kCsr`),
  * `hi` baked. The name ends in `_threshold`, then `_k<k>` or `_kr`, then `_hi<hi>`.
  *
  * ==Types==
  *
  * Each lane runs NumPy's arithmetic: `a`, `b` and `c` are lanes of their own types, each extended to `opType` (the
  * operands' result type, chosen by Python) by its own signedness, the op runs at `opType` (min and max comparing by
  * its signedness), and the result is extended or wrapped into `outType`. Each lane of a plain op is an
  * `ElementwiseAlu` of the one op; the casts sit at its boundary. The products and sums of axpby and axpbypcz are the
  * low opType bits of the operands' (two's complement, the same at either signedness). Every type is fixed at
  * elaboration, and every baked slot must fit opType.
  *
  * @param inTypes
  *   the inputs' types, `a`'s, `b`'s and `c`'s, or fewer, NumPy's names ("int32", "uint8"), from the unit's input
  *   ports' dtypes
  * @param opType
  *   the type the op runs at: NumPy's result type of `inTypes`, at least as wide as each (the unit's `opType`; a's type
  *   at one input)
  * @param outType
  *   `out`'s type, from the unit's output port's dtype
  * @param lanes
  *   elements per beat, the unit's input ports' lanes
  * @param op
  *   the op, `ElementwiseOp`'s encoding, or `SimdLanes.Axpby`, `Axpbypcz`, `Requant` or `Threshold` (the unit's
  *   `params.op`)
  * @param k
  *   one input's constant second operand, or threshold's `k`, which must fit `opType` (the unit's `params.k`); at one
  *   input but requant it or `kCsr`, refused otherwise
  * @param k1
  *   axpby's or axpbypcz's baked `k1` (the unit's `params.k1`), fitting `opType`
  * @param k2
  *   axpby's or axpbypcz's baked `k2` (the unit's `params.k2`), fitting `opType`
  * @param kCsr
  *   one input's `k` from the pin `k` (the unit's `scalars.k`)
  * @param k1Csr
  *   axpby's or axpbypcz's `k1` from the pin `k1` (the unit's `scalars.k1`)
  * @param k2Csr
  *   axpby's or axpbypcz's `k2` from the pin `k2` (the unit's `scalars.k2`)
  * @param k3
  *   axpbypcz's baked `k3` (the unit's `params.k3`), fitting `opType`
  * @param k3Csr
  *   axpbypcz's `k3` from the pin `k3` (the unit's `scalars.k3`)
  * @param d
  *   axpbypcz's and requant's divisor, a power of two at least 1 fitting `opType` (the unit's `params.d`)
  * @param lo
  *   requant's lower bound, fitting `opType` (the unit's `params.lo`)
  * @param hi
  *   requant's upper bound and threshold's output above it, fitting `opType` (the unit's `params.hi`)
  */
class SimdLanes(
  val inTypes: Seq[String]    = Seq("int32", "int32"),
  val opType:  String         = "int32",
  val outType: String         = "int32",
  val lanes:   Int            = 4,
  val op:      Int            = ElementwiseOp.Add,
  val k:       Option[BigInt] = None,
  val k1:      Option[BigInt] = None,
  val k2:      Option[BigInt] = None,
  val kCsr:    Boolean        = false,
  val k1Csr:   Boolean        = false,
  val k2Csr:   Boolean        = false,
  val k3:      Option[BigInt] = None,
  val k3Csr:   Boolean        = false,
  val d:       Option[BigInt] = None,
  val lo:      Option[BigInt] = None,
  val hi:      Option[BigInt] = None
) extends DatapathModule {
  require(
    Seq(1, 2, 3).contains(inTypes.size),
    s"inTypes must name a, b and c, a and b, or a alone, got ${inTypes.mkString(", ")}"
  )
  require(lanes > 0, s"lanes must be positive, got $lanes")
  private val n         = inTypes.size
  private val axpby     = op == SimdLanes.Axpby
  private val axpbypcz  = op == SimdLanes.Axpbypcz
  private val requant   = op == SimdLanes.Requant
  private val threshold = op == SimdLanes.Threshold
  private val opn       = SimdLanes.opName(op)
  private val count     = Map(1 -> "one input", 2 -> "two inputs", 3 -> "three inputs")

  require(!axpby || n == 2, s"${SimdLanes.AxpbyName} takes two inputs")
  require(
    axpbypcz == (n == 3),
    if (axpbypcz) s"$opn takes three inputs" else s"three inputs are axpbypcz's, not $opn's"
  )
  require(!(requant || threshold) || n == 1, s"$opn takes one input")

  private val takesK = n == 1 && !requant
  require(!(kCsr && k.nonEmpty), s"k = ${k.getOrElse("")} is baked and read from its CSR too")
  require(
    takesK == (k.nonEmpty || kCsr),
    if (takesK && threshold) s"$opn takes k, baked or from its CSR"
    else if (takesK) "one input takes the constant k"
    else if (requant) s"k = ${k.getOrElse("its CSR")} is not $opn's; it takes d, lo and hi"
    else s"k = ${k.getOrElse("its CSR")} is one input's constant; ${count(n)} take none"
  )
  require(
    axpby || axpbypcz || (k1.isEmpty && k2.isEmpty && !k1Csr && !k2Csr),
    s"k1 and k2 are ${SimdLanes.AxpbyName}'s and axpbypcz's, not $opn's"
  )
  require(!axpby || (k1.nonEmpty != k1Csr), s"${SimdLanes.AxpbyName} takes k1, baked or from its CSR, not both")
  require(!(k1.nonEmpty && k1Csr), s"k1 = ${k1.getOrElse("")} is baked and read from its CSR too")
  require(!(k2.nonEmpty && k2Csr), s"k2 = ${k2.getOrElse("")} is baked and read from its CSR too")
  require(axpbypcz || (k3.isEmpty && !k3Csr), s"k3 is axpbypcz's, not $opn's")
  require(!(k3.nonEmpty && k3Csr), s"k3 = ${k3.getOrElse("")} is baked and read from its CSR too")
  require(axpbypcz || requant || d.isEmpty, s"d is axpbypcz's and requant's, not $opn's")
  require(requant || lo.isEmpty, s"lo is requant's, not $opn's")
  require(requant || threshold || hi.isEmpty, s"hi is requant's and threshold's, not $opn's")
  require(!(axpbypcz || requant) || d.nonEmpty, s"$opn takes d, baked (1 for no shift)")
  require(!requant || (lo.nonEmpty && hi.nonEmpty), s"$opn takes lo and hi, both baked")
  require(!threshold || hi.nonEmpty, s"$opn takes hi, baked")
  for (x <- d)
    require(x >= 1 && (x & (x - 1)) == 0, s"d = $x must be a power of two at least 1: the lanes shift by log2(d)")

  private val ins = inTypes.map(IntType(_))
  private val opT = IntType(opType)
  private val res = IntType(outType)
  require(
    ins.forall(_.width <= opT.width),
    s"opType $opType must be at least as wide as each input (${inTypes.mkString(", ")})"
  )

  for {
    (s, v) <- Seq("k" -> k, "k1" -> k1, "k2" -> k2, "k3" -> k3, "d" -> d, "lo" -> lo, "hi" -> hi)
    x      <- v
  }
    require(SimdLanes.fits(x, opT), s"$s = $x does not fit opType $opType")

  private def suffix(s: String, v: Option[BigInt], csr: Boolean = false): String =
    if (csr) s"_${s}r" else v.map(x => if (x < 0) s"_${s}m${-x}" else s"_$s$x").getOrElse("")

  override def desiredName: String =
    s"SimdLanes_${ins.map(_.tag).mkString("x")}_${opT.tag}_${res.tag}_n${lanes}_$opn" +
      suffix("k", k, kCsr) + suffix("k1", k1, k1Csr) + suffix("k2", k2, k2Csr) + suffix("k3", k3, k3Csr) +
      suffix("d", d) + suffix("lo", lo) + suffix("hi", hi)

  val io = IO(new Bundle {
    val a   = Input(Vec(lanes, ins(0).lane))
    val b   = if (n >= 2) Some(Input(Vec(lanes, ins(1).lane))) else None
    val c   = if (n == 3) Some(Input(Vec(lanes, ins(2).lane))) else None
    val out = Output(Vec(lanes, res.lane))
    val k   = if (kCsr) Some(Input(UInt(opT.width.W))) else None
    val k1  = if (k1Csr) Some(Input(UInt(opT.width.W))) else None
    val k2  = if (k2Csr) Some(Input(UInt(opT.width.W))) else None
    val k3  = if (k3Csr) Some(Input(UInt(opT.width.W))) else None
  })

  private val w = opT.width

  /** `v` at opType, as its two's complement bits. */
  private def bits(v: BigInt): UInt = {
    val m = BigInt(1) << w
    (((v % m) + m) % m).U(w.W)
  }

  private def slot(v: Option[BigInt], pin: Option[UInt]): Option[UInt] = pin.orElse(v.map(bits))

  /** Input `i`'s lane `l`, extended to opType, as bits. */
  private def at(i: Int, l: Int): UInt = {
    val port = Seq(Some(io.a), io.b, io.c)(i).get
    IntType.cast(port(l), ins(i), opT).asUInt
  }

  // NB (D143): NumPy's floor `x // d` for d = 2^s: arithmetic at a signed opType, logical at an unsigned one
  private val shift = d.map(x => x.bitLength - 1).getOrElse(0)
  private def floorDiv(x: UInt): UInt =
    if (shift == 0) x
    else if (opT.signed) (x.asSInt >> shift).pad(w).asUInt
    else (x >> shift).pad(w)

  /** `x < y` at opType's signedness. */
  private def less(x: UInt, y: UInt): Bool = if (opT.signed) x.asSInt < y.asSInt else x < y

  private def out(l: Int, v: UInt): Unit = io.out(l) := IntType.cast(v(w - 1, 0), opT, res)

  /** `kx` at opType, the low bits of `k * x`; with no k, `x` itself (no multiplier). */
  private def term(kx: Option[UInt], x: UInt): UInt = kx.map(c => (c * x)(w - 1, 0)).getOrElse(x)

  if (axpby) {
    val s1 = slot(k1, io.k1)
    val s2 = slot(k2, io.k2)
    for (l <- 0 until lanes)
      out(l, term(s1, at(0, l)) + term(s2, at(1, l)))
  } else if (axpbypcz) {
    val ks = Seq(slot(k1, io.k1), slot(k2, io.k2), slot(k3, io.k3))
    for (l <- 0 until lanes) {
      val sum = (0 until 3).map(i => term(ks(i), at(i, l))).reduce((x, y) => (x + y)(w - 1, 0))
      out(l, floorDiv(sum))
    }
  } else if (requant) {
    val (lb, hb) = (bits(lo.get), bits(hi.get))
    for (l <- 0 until lanes) {
      val q = floorDiv(at(0, l))
      // NB (D143): min(max(q, lo), hi), in that order, as NumPy's: lo > hi gives hi
      val m = Mux(less(q, lb), lb, q)
      out(l, Mux(less(hb, m), hb, m))
    }
  } else if (threshold) {
    val kk = slot(k, io.k).get
    val hb = bits(hi.get)
    for (l <- 0 until lanes) {
      // NB (D143): a - k + 1 wraps at opType, as NumPy's; a comparator a >= k differs at the type's ends
      val delta = (at(0, l) - kk + 1.U)(w - 1, 0)
      val pos   = if (opT.signed) delta.asSInt > 0.S else delta =/= 0.U
      out(l, Mux(pos, hb, 0.U(w.W)))
    }
  } else {
    val kOperand = slot(k, io.k)
    val alus     = Seq.fill(lanes)(Module(new ElementwiseAlu(w, op, opT.signed)))
    alus.zipWithIndex.foreach { case (alu, l) =>
      alu.io.a  := at(0, l)
      alu.io.b  := (if (n == 2) at(1, l) else kOperand.get)
      io.out(l) := IntType.cast(alu.io.out, opT, res)
    }
  }
}

object SimdLanes {

  /** axpby's encoding in `op`, past `ElementwiseOp`'s: `k1 * a + k2 * b`, no `ElementwiseAlu`. */
  val Axpby: Int = 8

  /** axpbypcz's: `(k1 * a + k2 * b + k3 * c) >> log2(d)`. */
  val Axpbypcz: Int = 9

  /** requant's: `min(max(a >> log2(d), lo), hi)`. */
  val Requant: Int = 10

  /** threshold's: `hi * min(max(a - k + 1, 0), 1)`. */
  val Threshold: Int = 11

  val AxpbyName = "axpby"

  /** The ops of their own past `ElementwiseOp`'s, encoding -> name. */
  val Named: Map[Int, String] =
    Map(Axpby -> AxpbyName, Axpbypcz -> "axpbypcz", Requant -> "requant", Threshold -> "threshold")

  /** The op names `params.op` takes past `ElementwiseOp`'s, in encoding order. */
  val NamedNames: Seq[String] = Named.toSeq.sortBy(_._1).map(_._2)

  /** The op's name: `ElementwiseOp`'s, or one of `Named`. */
  def opName(op: Int): String = Named.getOrElse(op, ElementwiseOp.name(op))

  /** The encoding of the op named `name` (`params.op`). */
  def opByName(name: String): Int =
    Named.collectFirst { case (e, n) if n == name => e }.getOrElse(ElementwiseOp.byName(name))

  /** `v` is one of `t`'s values. */
  def fits(v: BigInt, t: IntType): Boolean =
    if (t.signed) v >= -(BigInt(1) << (t.width - 1)) && v < (BigInt(1) << (t.width - 1))
    else v          >= 0                             && v < (BigInt(1) << t.width)
}
