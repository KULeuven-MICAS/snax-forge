package snax.forge.elementwise

import chisel3._
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec

/** `ElementwiseAlu`'s module name carries its signedness when min or max compares (`_s`, `_u`), so a signed and an
  * unsigned max ALU of one width are two modules; the signedness-agnostic ops keep their names.
  */
class ElementwiseAluNameSpec extends AnyFlatSpec {

  /** The name of the ALU of these params, elaborated. */
  private def name(width: Int, ops: Seq[Int], signed: Boolean): String = {
    var n = ""
    ChiselStage.emitCHIRRTL(new Module {
      val alu = Module(new ElementwiseAlu(width, ops, signed))
      alu.io := DontCare
      n = alu.name
    })
    n
  }

  behavior of "ElementwiseAlu's name"

  it should "differ between a signed and an unsigned max ALU" in {
    val (s, u) = (name(32, Seq(ElementwiseOp.Max), true), name(32, Seq(ElementwiseOp.Max), false))
    assert((s, u) == ("ElementwiseAlu_w32_s_max", "ElementwiseAlu_w32_u_max"))
    assert(name(8, ElementwiseOp.all, true) == "ElementwiseAlu_w8_s_sel8")
  }

  it should "keep the names of the ops that do not compare" in {
    for ((op, n) <- Seq(ElementwiseOp.Add -> "add", ElementwiseOp.Mul -> "mul", ElementwiseOp.Xor -> "xor"))
      for (signed <- Seq(true, false))
        assert(name(64, Seq(op), signed) == s"ElementwiseAlu_w64_$n")
  }
}
