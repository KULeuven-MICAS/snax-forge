package snax.forge.samples

import chisel3._
import chisel3.util._

/** Serial integer accumulator -- the hardware image of the WCR edge in
  * `kernels/polybench/dot.py`.
  *
  * Consumes `len` elements from an elastic input stream and emits their sum on
  * an elastic output stream. This is the second archetype the toolchain has to
  * cover, and it is qualitatively different from `SimpleAdder`: a reduction
  * collapses N input beats into 1 output beat, so it is a *chaining barrier*.
  * A downstream consumer cannot be fed element-by-element in producer order,
  * because no output element exists until the last input has been absorbed.
  * Chained reductions therefore get control chaining, not data chaining.
  *
  * The `len` port is a placeholder for what will become a CSR field once the
  * descriptor and driver generation land in W9. It is sampled continuously
  * rather than latched, so it must be held stable for the duration of a run --
  * which is what a CSR write followed by a start pulse gives you anyway.
  *
  * Protocol: `in.ready` is high while absorbing and low while a result is
  * waiting to be collected, so absorb and drain are mutually exclusive by
  * construction and the two `when` blocks below can never both fire in the
  * same cycle. After `out` fires the unit resets itself and is immediately
  * ready for the next run; back-to-back runs need no external reset.
  *
  * Not handled, deliberately: `len === 0`. A zero-length run never reaches the
  * drain state and the unit sits waiting forever. Guarding it in RTL costs a
  * comparator on a case the extractor should reject at descriptor-build time,
  * where the error message can actually name the offending map range.
  *
  * @param width    bits per input element, from the DaCe dtype
  * @param accWidth bits of accumulator; widening here is how you buy headroom
  *                 against overflow that the DaCe reference would also wrap on
  */
class Accumulator(val width: Int = 32, val accWidth: Int = 32) extends Module {
  require(width > 0, s"width must be positive, got $width")
  require(
    accWidth >= width,
    s"accWidth ($accWidth) must be at least width ($width), otherwise every " +
      "partial sum is truncated on the way in"
  )

  val io = IO(new Bundle {
    val len = Input(UInt(32.W))
    val in  = Flipped(Decoupled(UInt(width.W)))
    val out = Decoupled(UInt(accWidth.W))
  })

  private val acc      = RegInit(0.U(accWidth.W))
  private val count    = RegInit(0.U(32.W))
  private val draining = RegInit(false.B)

  io.in.ready  := !draining
  io.out.valid := draining
  io.out.bits  := acc

  when(io.in.fire) {
    val next = count + 1.U
    acc      := acc + io.in.bits
    count    := next
    draining := next === io.len
  }

  when(io.out.fire) {
    acc      := 0.U
    count    := 0.U
    draining := false.B
  }
}