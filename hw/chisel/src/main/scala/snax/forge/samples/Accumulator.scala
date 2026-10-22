package snax.forge.samples

import chisel3._
import chisel3.util._

/** The `accumulate` BRM's integer accumulator -- the hardware image of the reduction in `kernels/polybench/vsum.py`.
  *
  * Sums every `len` consecutive beats of an elastic input stream and emits each sum on an elastic output stream,
  * `loopCount` input beats per run. A beat is `lanes` elements; a combinational tree of `lanes` → 1 adders in front of
  * the register sums them, so each beat adds its lanes' sum (D133). `lanes = 1` is the serial accumulator. A reduction
  * collapses T input beats into 1 output beat, so it is a *chaining barrier*: a downstream consumer cannot be fed
  * element-by-element in producer order, because no output element exists until the last input of its sum has been
  * absorbed. Chained reductions therefore get control chaining, not data chaining.
  *
  * ==Control==
  *
  * As `ElementwiseTiledSpatial`'s: `start`, `busy`, `done`, with two configuration inputs, `loopCount` (n, the input
  * beats of the run) and `len` (T, the inputs per output). A start while idle clears the sum and the counts, latches
  * both and raises `busy`; a start while busy is ignored, and so is one with `loopCount === 0` or `len === 0` (the unit
  * stays idle; a zero-length run has no meaningful completion). Both inputs are sampled only at start. `busy` falls in
  * the cycle after the (n / T)-th output fires, and `done` is high in that cycle of the last output.
  *
  * n not a multiple of T is the toolchain's to refuse. The block then ends the last sum at the n-th input: its last
  * output is the sum of the n mod T inputs left, and the run ends as any other.
  *
  * ==The tree==
  *
  * `lanes` → 1 adders (`reduceTree`), each `accWidth` bits and wrapping as the register does, between `in.bits` and
  * `acc`, with no register of its own: latency stays 1 for every `lanes`. Pipeline stages in the tree would make the
  * latency depend on `lanes`; that is the choice of the generator from the hardware binding.
  *
  * ==One register, no drain cycle==
  *
  * `acc` holds the sum in progress and, once its T-th input has fired, the result: `out.valid` rises the cycle after
  * (latency 1) and stays high until `out` fires. In the cycle the result leaves, the next input is taken by loading
  * `acc` with it instead of adding, so with `out` always ready a sum takes T cycles and back-to-back sums have no gap;
  * `len === 1` makes every input a sum, `out.valid` staying high while `in` and `out` both fire. The only stall is
  * `out` not ready. After the n-th input `in.ready` stays low, so a beat of the next run is never taken.
  *
  * ==Handshake==
  *
  * `out.valid` is a register, so `valid` never depends on `ready`. `in.ready` does depend on `out.ready`
  * combinationally (the input is taken only when the result can leave), a path the generated shell wrapper has to keep
  * in mind when it puts the block behind the streamer.
  *
  * @param width
  *   bits per input element, from the BRM port's dtype
  * @param accWidth
  *   bits of accumulator; widening here is how you buy headroom against overflow that the NumPy reference would also
  *   wrap on
  * @param lanes
  *   elements per input beat, the BRM's W (the binding's `args`)
  */
class Accumulator(val width: Int = 32, val accWidth: Int = 32, val lanes: Int = 1) extends Module {
  require(width > 0, s"width must be positive, got $width")
  require(lanes > 0, s"lanes must be positive, got $lanes")
  require(
    accWidth >= width,
    s"accWidth ($accWidth) must be at least width ($width), otherwise every " +
      "partial sum is truncated on the way in"
  )

  override def desiredName: String = s"Accumulator_w${width}_a${accWidth}_n${lanes}"

  val io = IO(new Bundle {
    val loopCount = Input(UInt(32.W))
    val len       = Input(UInt(32.W))
    val start     = Input(Bool())
    val busy      = Output(Bool())
    val done      = Output(Bool())

    val in  = Flipped(Decoupled(Vec(lanes, UInt(width.W))))
    val out = Decoupled(UInt(accWidth.W))
  })

  private val busy   = RegInit(false.B)
  private val target = RegInit(0.U(32.W))
  private val len    = RegInit(0.U(32.W))
  private val taken  = RegInit(0.U(32.W)) // inputs taken this run
  private val count  = RegInit(0.U(32.W)) // inputs in the sum in progress
  private val acc    = RegInit(0.U(accWidth.W))
  private val valid  = RegInit(false.B)

  private val allIn = taken === target

  io.in.ready  := busy && !allIn && (!valid || io.out.ready)
  io.out.valid := valid
  io.out.bits  := acc

  // The tree: the beat's lanes summed, wrapping at accWidth bits.
  private val beat = VecInit(io.in.bits.map(_.pad(accWidth))).reduceTree(_ + _)

  private val lastIn  = io.in.fire  && (taken + 1.U) === target
  private val lastSum = io.in.fire  && ((count + 1.U) === len || lastIn)
  private val last    = io.out.fire && allIn

  when(!busy) {
    when(io.start && io.loopCount =/= 0.U && io.len =/= 0.U) {
      busy   := true.B
      target := io.loopCount
      len    := io.len
      taken  := 0.U
      count  := 0.U
      acc    := 0.U
      valid  := false.B
    }
  }.otherwise {
    when(io.in.fire) {
      // In the cycle the result leaves, the next sum starts from this input.
      acc   := Mux(io.out.fire, beat, acc + beat)
      taken := taken + 1.U
      count := Mux(lastSum, 0.U, count + 1.U)
    }.elsewhen(io.out.fire) {
      acc := 0.U
    }
    valid := lastSum || (valid && !io.out.fire)
    when(last) {
      busy := false.B
    }
  }

  io.busy := busy
  io.done := last
}
