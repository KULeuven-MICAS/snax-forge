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
  * ==Types (D143)==
  *
  * Each lane runs NumPy's arithmetic: `a` and `b` are lanes of their own types, each extended to `opType` (the
  * operands' result type, chosen by Python) by its own signedness, the op runs at `opType` (min and max comparing by
  * its signedness), and the result is extended or wrapped into `outType`. Each lane of a plain op is an
  * `ElementwiseAlu` of the one op; the casts sit at its boundary. axpby's products and sum are the low opType bits of
  * the operands' (two's complement, the same at either signedness). Every type is fixed at elaboration.
  *
  * @param inTypes
  *   `a`'s and `b`'s types, or `a`'s alone, NumPy's names ("int32", "uint8"), from the unit's input ports' dtypes
  * @param opType
  *   the type the op runs at: NumPy's result type of `inTypes`, at least as wide as each (the unit's `opType`; a's type
  *   at one input)
  * @param outType
  *   `out`'s type, from the unit's output port's dtype
  * @param lanes
  *   elements per beat, the unit's input ports' lanes
  * @param op
  *   the op, `ElementwiseOp`'s encoding, or `SimdLanes.Axpby` (the unit's `params.op`)
  * @param k
  *   one input's constant second operand, which must fit `opType` (the unit's `params.k`); at one input it or `kCsr`,
  *   refused at two
  * @param k1
  *   axpby's baked `k1` (the unit's `params.k1`), fitting `opType`
  * @param k2
  *   axpby's baked `k2` (the unit's `params.k2`), fitting `opType`
  * @param kCsr
  *   one input's `k` from the pin `k` (the unit's `scalars.k`)
  * @param k1Csr
  *   axpby's `k1` from the pin `k1` (the unit's `scalars.k1`)
  * @param k2Csr
  *   axpby's `k2` from the pin `k2` (the unit's `scalars.k2`)
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
  val k2Csr:   Boolean        = false
) extends DatapathModule {
  require(Seq(1, 2).contains(inTypes.size), s"inTypes must name a and b, or a alone, got ${inTypes.mkString(", ")}")
  require(lanes > 0, s"lanes must be positive, got $lanes")
  private val two   = inTypes.size == 2
  private val axpby = op == SimdLanes.Axpby
  require(!(kCsr && k.nonEmpty), s"k = ${k.getOrElse("")} is baked and read from its CSR too")
  require(
    two == (k.isEmpty && !kCsr),
    if (two) s"k = ${k.getOrElse("its CSR")} is one input's constant; two inputs take none"
    else "one input takes the constant k"
  )
  require(!axpby || two, s"${SimdLanes.AxpbyName} takes two inputs")
  require(
    axpby || (k1.isEmpty && k2.isEmpty && !k1Csr && !k2Csr),
    s"k1 and k2 are ${SimdLanes.AxpbyName}'s, not ${ElementwiseOp.name(op)}'s"
  )
  require(!axpby || (k1.nonEmpty != k1Csr), s"${SimdLanes.AxpbyName} takes k1, baked or from its CSR, not both")
  require(!(k2.nonEmpty && k2Csr), s"k2 = ${k2.getOrElse("")} is baked and read from its CSR too")

  private val ins = inTypes.map(IntType(_))
  private val opT = IntType(opType)
  private val res = IntType(outType)
  require(
    ins.forall(_.width <= opT.width),
    s"opType $opType must be at least as wide as each input (${inTypes.mkString(", ")})"
  )

  for {
    (n, v) <- Seq("k" -> k, "k1" -> k1, "k2" -> k2)
    x      <- v
  }
    require(SimdLanes.fits(x, opT), s"$n = $x does not fit opType $opType")

  private def suffix(n: String, v: Option[BigInt], csr: Boolean): String =
    if (csr) s"_${n}r" else v.map(x => if (x < 0) s"_${n}m${-x}" else s"_$n$x").getOrElse("")

  override def desiredName: String =
    s"SimdLanes_${ins.map(_.tag).mkString("x")}_${opT.tag}_${res.tag}_n${lanes}_${SimdLanes.opName(op)}" +
      suffix("k", k, kCsr) + suffix("k1", k1, k1Csr) + suffix("k2", k2, k2Csr)

  val io = IO(new Bundle {
    val a   = Input(Vec(lanes, ins(0).lane))
    val b   = if (two) Some(Input(Vec(lanes, ins(1).lane))) else None
    val out = Output(Vec(lanes, res.lane))
    val k   = if (kCsr) Some(Input(UInt(opT.width.W))) else None
    val k1  = if (k1Csr) Some(Input(UInt(opT.width.W))) else None
    val k2  = if (k2Csr) Some(Input(UInt(opT.width.W))) else None
  })

  /** `v` at opType, as its two's complement bits. */
  private def bits(v: BigInt): UInt = {
    val m = BigInt(1) << opT.width
    (((v % m) + m) % m).U(opT.width.W)
  }

  private def slot(v: Option[BigInt], pin: Option[UInt]): Option[UInt] = pin.orElse(v.map(bits))

  if (axpby) {
    val s1 = slot(k1, io.k1).get
    val s2 = slot(k2, io.k2)
    for (i <- 0 until lanes) {
      val a   = IntType.cast(io.a(i), ins(0), opT).asUInt
      val b   = IntType.cast(io.b.get(i), ins(1), opT).asUInt
      val sum = (s1 * a)(opT.width - 1, 0) + s2.map(x => (x * b)(opT.width - 1, 0)).getOrElse(b)
      io.out(i) := IntType.cast(sum(opT.width - 1, 0), opT, res)
    }
  } else {
    val kOperand = slot(k, io.k)
    val alus     = Seq.fill(lanes)(Module(new ElementwiseAlu(opT.width, op, opT.signed)))
    alus.zipWithIndex.foreach { case (alu, i) =>
      alu.io.a  := IntType.cast(io.a(i), ins(0), opT).asUInt
      alu.io.b  := io.b.map(b => IntType.cast(b(i), ins(1), opT).asUInt).getOrElse(kOperand.get)
      io.out(i) := IntType.cast(alu.io.out, opT, res)
    }
  }
}

object SimdLanes {

  /** axpby's encoding in `op`, past `ElementwiseOp`'s: `k1 * a + k2 * b`, no `ElementwiseAlu`. */
  val Axpby: Int = 8

  val AxpbyName = "axpby"

  /** The op's name: `ElementwiseOp`'s, or axpby. */
  def opName(op: Int): String = if (op == Axpby) AxpbyName else ElementwiseOp.name(op)

  /** The encoding of the op named `name` (`params.op`). */
  def opByName(name: String): Int = if (name == AxpbyName) Axpby else ElementwiseOp.byName(name)

  /** `v` is one of `t`'s values. */
  def fits(v: BigInt, t: IntType): Boolean =
    if (t.signed) v >= -(BigInt(1) << (t.width - 1)) && v < (BigInt(1) << (t.width - 1))
    else v          >= 0                             && v < (BigInt(1) << t.width)
}
