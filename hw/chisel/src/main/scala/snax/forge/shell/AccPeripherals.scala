package snax.forge.shell

import chisel3._
import chisel3.util.MixedVec

/** `<tag>_peripherals`: what only the core reads and the run never needs (D168), inside `<tag>_csr`, always present.
  * Per group its BUSY_COUNTER (`<group>_busy_counter`, the CSR's width w, busy_counter_width): cleared by the accepted
  * start, + 1 per busy cycle, held after the run and saturating at 2^w - 1 (BlockGemm's performance counter), so after
  * a run started in a and ended in E it is E - a. `<tag>_csr` reads it through its read mux; the reported csr leaves
  * this module out.
  *
  * @param tag
  *   the platform's accelerator slot, in the module's name
  * @param counters
  *   per group, in CSR order, its name and its BUSY_COUNTER's width
  */
class AccPeripherals(tag: String, counters: Seq[(String, Int)]) extends Module {
  override def desiredName: String = s"${tag}_peripherals"

  val io = IO(new Bundle {
    val start       = Input(Vec(counters.size, Bool()))
    val busy        = Input(Vec(counters.size, Bool()))
    val busyCounter = Output(MixedVec(counters.map { case (_, w) => UInt(w.W) }))
  })

  for (((g, w), i) <- counters.zipWithIndex) {
    val c = RegInit(0.U(w.W)).suggestName(s"${g}_busy_counter")
    when(io.start(i)) {
      c := 0.U
    }.elsewhen(io.busy(i) && c =/= ((BigInt(1) << w) - 1).U) {
      c := c + 1.U
    }
    io.busyCounter(i) := c
  }
}

object AccPeripherals {

  /** The design's peripherals: a BUSY_COUNTER per group of `spec`. */
  def apply(spec: AcceleratorSpec): AccPeripherals =
    new AccPeripherals(spec.tag, spec.groups.map(g => g.name -> g.csr("BUSY_COUNTER").get.width))
}
