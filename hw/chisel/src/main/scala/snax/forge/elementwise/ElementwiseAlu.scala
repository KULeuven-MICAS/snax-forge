package snax.forge.elementwise

import chisel3._

/** The elementwise ops: an encoding per op, the key a datapath is built with. A BRM names the op (`op: "add"`), and
  * `accelerator.json`'s unit carries that name (`params.op`), which `Datapaths` resolves here.
  */
object ElementwiseOp {
  val Add: Int = 0
  val Sub: Int = 1
  val Mul: Int = 2
  val And: Int = 3
  val Or:  Int = 4
  val Xor: Int = 5
  val Min: Int = 6
  val Max: Int = 7

  val all: Seq[Int] = Seq(Add, Sub, Mul, And, Or, Xor, Min, Max)

  val names: Map[Int, String] = Map(
    Add -> "add",
    Sub -> "sub",
    Mul -> "mul",
    And -> "and",
    Or  -> "or",
    Xor -> "xor",
    Min -> "min",
    Max -> "max"
  )

  def name(op: Int): String = names.getOrElse(op, s"op$op")
}

/** One lane of integer ALU, as its own SystemVerilog module.
  *
  * Combinational, and therefore does not use the clock or reset that `Module` gives it. It extends `Module` rather than
  * `RawModule` anyway, so that every unit in this package presents the same skeleton -- the datapaths around it need a
  * clock (TreeAccumulate's register), and one convention is worth more than eliminating a pair of dangling ports on
  * this one.
  *
  * Being a module rather than an inlined function costs nothing at the gate level (synthesis flattens the hierarchy)
  * and buys two things. Area reports break down per module, so an HLS comparison can say what the ALU costs separately
  * from the streamer interface. And it is testable on its own instead of only through a datapath wrapper.
  *
  * Handshaking deliberately stops at the datapath boundary: this module has plain Input/Output ports and no ready/valid
  * of its own. A lane has no capacity to fill and nothing to stall on, so a handshake here would be ceremony -- the
  * elastic interface exists where beats are counted, which is one level up.
  *
  * One op, fixed at elaboration: the bare operator, no select port and no decode.
  *
  * Signedness applies to `Min` and `Max` only. Add, Sub and the bitwise operators are signedness-agnostic in two's
  * complement, and so is the truncated low half of a multiply -- which is why `Mul` needs no signed variant here, even
  * though a widening multiply would. All of these wrap exactly as NumPy's fixed-width integers do, which is what keeps
  * the datapath bit-identical to the NumPy golden reference.
  *
  * @param dataWidth
  *   bits per element, from the BRM port's dtype
  * @param op
  *   the op, `ElementwiseOp`'s encoding
  * @param signed
  *   interpret operands as two's complement for Min and Max
  */
class ElementwiseAlu(val dataWidth: Int, val op: Int, val signed: Boolean = false) extends Module {
  require(dataWidth > 0, s"dataWidth must be positive, got $dataWidth")
  require(ElementwiseOp.all.contains(op), s"unknown elementwise op encoding: $op")

  /** `_s` or `_u` when min or max compares: two such ALUs of one width but not one signedness are different modules.
    * The other ops are signedness-agnostic and keep their names.
    */
  private def signedness: String =
    if (op == ElementwiseOp.Min || op == ElementwiseOp.Max) if (signed) "_s" else "_u"
    else ""

  override def desiredName: String = s"ElementwiseAlu_w${dataWidth}${signedness}_${ElementwiseOp.name(op)}"

  val io = IO(new Bundle {
    val a   = Input(UInt(dataWidth.W))
    val b   = Input(UInt(dataWidth.W))
    val out = Output(UInt(dataWidth.W))
  })

  io.out := compute(op, io.a, io.b)

  private def compute(op: Int, a: UInt, b: UInt): UInt =
    op match {
      case ElementwiseOp.Add => a + b
      case ElementwiseOp.Sub => a - b
      // Chisel widens `*` to 2 * dataWidth; the low half is the wrapping product
      // and is identical for signed and unsigned operands.
      case ElementwiseOp.Mul => (a * b)(dataWidth - 1, 0)
      case ElementwiseOp.And => a & b
      case ElementwiseOp.Or  => a | b
      case ElementwiseOp.Xor => a ^ b
      case ElementwiseOp.Min => Mux(lessThan(a, b), a, b)
      case ElementwiseOp.Max => Mux(lessThan(a, b), b, a)
      case other             => throw new IllegalArgumentException(s"unknown elementwise op encoding: $other")
    }

  private def lessThan(a: UInt, b: UInt): Bool = if (signed) a.asSInt < b.asSInt else a < b
}
