package snax.forge.samples

import chisel3._
import chisel3.util._

/** Streaming interface for a two-input, one-output elementwise operator.
  *
  * The shape mirrors what a SNAX streamer presents: each port is an
  * independent elastic channel carrying `lanes` elements per beat. Nothing
  * here knows about addresses, strides or CSRs -- address generation stays in
  * the streamer, and the datapath only ever sees data in flight. That split is
  * what makes "one operation instance to one hardware instance" a legal
  * binding: the SDFG has already decided the schedule, so the datapath has no
  * scheduling decisions left to make.
  *
  * NB the parameter is `elemWidth`, not `width`. Bundle extends Aggregate,
  * which already declares `width: chisel3.Width`; a `val width: Int` here is a
  * type-incompatible override and will not compile. The same applies to any
  * other Data subclass. Modules are unaffected -- Module is not Data -- which
  * is why SimpleAdder below can still call its parameter `width`.
  */
class ElementwiseBinaryIO(val elemWidth: Int, val lanes: Int) extends Bundle {
  val a   = Flipped(Decoupled(Vec(lanes, UInt(elemWidth.W))))
  val b   = Flipped(Decoupled(Vec(lanes, UInt(elemWidth.W))))
  val out = Decoupled(Vec(lanes, UInt(elemWidth.W)))
}

/** Lane-parallel integer adder -- the hardware image of `kernels/polybench/vecadd.py`.
  *
  * Purely combinational: `out.bits` is a function of the current inputs, and
  * the only state is the absence of state. This is the degenerate case of the
  * latency-balancing problem the thesis is actually about (L = 0, II = 1), and
  * it is deliberately the first thing to work end to end.
  *
  * Handshake: a join. `out` fires only when both inputs are valid, and each
  * input is consumed only when the other input and the consumer are also
  * ready. Note that `a.ready` depends on `b.valid` and vice versa -- that is
  * legal (ready may depend on other channels' valid) whereas making `valid`
  * depend on its own `ready` would deadlock.
  *
  * Arithmetic: Chisel's `+` on UInt truncates to the operand width, which is
  * exactly two's-complement wraparound and therefore bit-identical to NumPy's
  * int32/int8 overflow behaviour. Addition does not care about signedness, so
  * UInt is used as a raw bit container here. Signedness starts to matter at
  * multiply, compare and shift -- those arrive with the typed ALU set in W6.
  *
  * @param width bits per element, taken from the DaCe dtype
  * @param lanes elements per beat, i.e. the unroll factor of the map scope
  */
class SimpleAdder(val width: Int = 32, val lanes: Int = 4) extends Module {
  require(width > 0, s"width must be positive, got $width")
  require(lanes > 0, s"lanes must be positive, got $lanes")

  // Distinguishes hand-written stand-ins from emitter output once both are
  // landing in out/hw. SystemVerilog's module namespace is flat and global,
  // and Chisel derives the module name from the class name alone -- the Scala
  // package is discarded, so it cannot do this job.
  override def desiredName = "sample_SimpleAdder"

  val io = IO(new ElementwiseBinaryIO(width, lanes))

  private val bothValid = io.a.valid && io.b.valid

  io.out.valid := bothValid
  io.a.ready   := io.b.valid && io.out.ready
  io.b.ready   := io.a.valid && io.out.ready

  io.out.bits := VecInit(Seq.tabulate(lanes)(i => io.a.bits(i) + io.b.bits(i)))
}