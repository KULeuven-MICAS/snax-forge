package snax.forge

import chisel3.{Bool, Module, UInt}
import chiseltest._

/** `ChiselScalatestTester` for every spec here, with `SNAX_FORGE_VCD=1` adding `WriteVcdAnnotation` to each `test`: the
  * waveform lands in `hw/chisel/test_run_dir/<test name>/<module>.vcd`. (The task cannot pass chiseltest's
  * `-DwriteVcd=1`: pixi hands sbt the `--` before it as a command of its own.)
  */
trait ForgeTester extends ChiselScalatestTester { this: org.scalatest.TestSuite =>

  override def test[T <: Module](dutGen: => T): TestBuilder[T] =
    if (sys.env.get("SNAX_FORGE_VCD").contains("1")) super.test(dutGen).withAnnotations(Seq(WriteVcdAnnotation))
    else super.test(dutGen)

  /** A Bool's value, an int's (32 bits or fewer), peeked. */
  def high(b: Bool): Boolean = b.peek().litToBoolean
  def int(u:  UInt): Int     = u.peek().litValue.toInt
}
