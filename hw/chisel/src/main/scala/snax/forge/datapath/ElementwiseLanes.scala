package snax.forge.datapath

import chisel3._

import snax.forge.IntType
import snax.forge.elementwise.{ElementwiseAlu, ElementwiseOp}

/** `lanes` lanes of one op: the datapath of an elementwise unit (D147), latency 0, no pins.
  *
  * Data only: no valid, ready or counter. Its group's control unit (`ctrl.GroupCtrl`) decides when a beat is taken and
  * the result pushed; in that cycle `out` is the op on `a` and `b`, lane by lane.
  *
  * ==Types (D143)==
  *
  * Each lane runs NumPy's arithmetic: `a` and `b` are lanes of their own types, each extended to `opType` (the
  * operands' result type, chosen by Python) by its own signedness, the op runs at `opType` (min and max comparing by
  * its signedness), and the result is extended or wrapped into `outType`. Each lane is an `ElementwiseAlu` of the one
  * op, so there is no `opSel`; the casts sit at its boundary. Every type is fixed at elaboration.
  *
  * @param inTypes
  *   `a`'s and `b`'s types, NumPy's names ("int32", "uint8"), from the unit's input ports' dtypes
  * @param opType
  *   the type the op runs at: NumPy's result type of `inTypes`, at least as wide as each (the unit's `opType`)
  * @param outType
  *   `out`'s type, from the unit's output port's dtype
  * @param lanes
  *   elements per beat, the unit's input ports' lanes
  * @param op
  *   the op, `ElementwiseOp`'s encoding (the unit's `params.op`)
  */
class ElementwiseLanes(
  val inTypes: Seq[String] = Seq("int32", "int32"),
  val opType:  String      = "int32",
  val outType: String      = "int32",
  val lanes:   Int         = 4,
  val op:      Int         = ElementwiseOp.Add
) extends Module {
  require(inTypes.size == 2, s"inTypes must name a and b, got ${inTypes.mkString(", ")}")
  require(lanes > 0, s"lanes must be positive, got $lanes")

  private val ins = inTypes.map(IntType(_))
  private val opT = IntType(opType)
  private val res = IntType(outType)
  require(
    ins.forall(_.width <= opT.width),
    s"opType $opType must be at least as wide as each input (${inTypes.mkString(", ")})"
  )

  override def desiredName: String =
    s"ElementwiseLanes_${ins.map(_.tag).mkString("x")}_${opT.tag}_${res.tag}_n${lanes}_${ElementwiseOp.name(op)}"

  val io = IO(new Bundle {
    val a   = Input(Vec(lanes, ins(0).lane))
    val b   = Input(Vec(lanes, ins(1).lane))
    val out = Output(Vec(lanes, res.lane))
  })

  private val alus = Seq.fill(lanes)(Module(new ElementwiseAlu(opT.width, Seq(op), opT.signed)))
  alus.zipWithIndex.foreach { case (alu, i) =>
    alu.io.opSel := op.U
    alu.io.a     := IntType.cast(io.a(i), ins(0), opT).asUInt
    alu.io.b     := IntType.cast(io.b(i), ins(1), opT).asUInt
    io.out(i)    := IntType.cast(alu.io.out, opT, res)
  }
}
