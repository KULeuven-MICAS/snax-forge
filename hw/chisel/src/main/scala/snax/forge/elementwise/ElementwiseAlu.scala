package snax.forge.elementwise

import chisel3._
import chisel3.util._

/** Operation encoding shared by every elementwise datapath.
  *
  * Four bits, so sixteen encodings are available and eight are used. The width is fixed rather than derived from
  * `supportedOps.size` on purpose: a fixed-function unit and a fully selectable one then present the same port list,
  * and swapping one for the other is a generator decision that does not ripple into whatever instantiates it.
  *
  * The numbering is part of the descriptor contract once the JSON boundary lands -- the Python side will emit `op:
  * "add"` and the generator will resolve it here, so these values must not be renumbered casually.
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

  /** Width of every `opSel` port. */
  val width: Int = 4

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

  /** Suffix used in every module name, so that a fixed-function instance is distinguishable from a selectable one in
    * the elaborated SystemVerilog without reading its ports.
    */
  def tag(supportedOps: Seq[Int]): String =
    if (supportedOps.size == 1) name(supportedOps.head) else s"sel${supportedOps.size}"
}

/** One lane of integer ALU, as its own SystemVerilog module.
  *
  * Combinational, and therefore does not use the clock or reset that `Module` gives it. It extends `Module` rather than
  * `RawModule` anyway, so that every unit in this package presents the same skeleton -- the two counted variants
  * genuinely need a clock, and one convention across the package is worth more than eliminating a pair of dangling
  * ports on this one.
  *
  * Being a module rather than an inlined function costs nothing at the gate level (synthesis flattens the hierarchy)
  * and buys two things. Area reports break down per module, so the W17 HLS comparison can say what the ALU costs
  * separately from the streamer interface. And it is testable on its own instead of only through a datapath wrapper.
  *
  * Handshaking deliberately stops at the datapath boundary: this module has plain Input/Output ports and no ready/valid
  * of its own. A lane has no capacity to fill and nothing to stall on, so a handshake here would be ceremony -- the
  * elastic interface exists where beats are counted, which is one level up.
  *
  * Fixed-function is the degenerate case of selectable, not a separate code path. `supportedOps` of length one emits
  * the bare operator with no mux and no decode; longer lists emit a MuxLookup over exactly those encodings. That is
  * what makes "fixed or selectable" a generator policy: the same construction produces either, depending on a list it
  * is handed.
  *
  * Signedness applies to `Min` and `Max` only. Add, Sub and the bitwise operators are signedness-agnostic in two's
  * complement, and so is the truncated low half of a multiply -- which is why `Mul` needs no signed variant here, even
  * though a widening multiply would. All of these wrap exactly as NumPy's fixed-width integers do, which is what keeps
  * the datapath bit-identical to the DaCe golden reference.
  *
  * @param dataWidth
  *   bits per element, from the DaCe dtype
  * @param supportedOps
  *   encodings this instance can perform; one entry means fixed-function
  * @param signed
  *   interpret operands as two's complement for Min and Max
  */
class ElementwiseAlu(
  val dataWidth:    Int      = 32,
  val supportedOps: Seq[Int] = ElementwiseOp.all,
  val signed:       Boolean  = false
) extends Module {
  require(dataWidth > 0, s"dataWidth must be positive, got $dataWidth")
  require(supportedOps.nonEmpty, "supportedOps must not be empty: a datapath with no operation is not a datapath")
  require(
    supportedOps.distinct.size == supportedOps.size,
    s"supportedOps contains duplicates: ${supportedOps.mkString(",")}"
  )
  require(
    supportedOps.forall(o => o >= 0 && o < (1 << ElementwiseOp.width)),
    s"supportedOps must fit in ${ElementwiseOp.width} bits, got ${supportedOps.mkString(",")}"
  )

  override def desiredName: String = s"ElementwiseAlu_w${dataWidth}_${ElementwiseOp.tag(supportedOps)}"

  val io = IO(new Bundle {
    val opSel = Input(UInt(ElementwiseOp.width.W))
    val a     = Input(UInt(dataWidth.W))
    val b     = Input(UInt(dataWidth.W))
    val out   = Output(UInt(dataWidth.W))
  })

  private val results = supportedOps.map(op => op -> compute(op, io.a, io.b))

  // One supported op: no decode logic at all. firtool would constant-fold a
  // single-entry MuxLookup anyway, but emitting the bare operator keeps the
  // generated SystemVerilog readable, which matters when it is the artifact
  // being reviewed against the descriptor.
  io.out :=
    (if (results.size == 1) results.head._2
     else MuxLookup(io.opSel, results.head._2)(results.map { case (op, r) => op.U(ElementwiseOp.width.W) -> r }))

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
