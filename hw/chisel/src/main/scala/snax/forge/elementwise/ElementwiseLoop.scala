package snax.forge.elementwise

import chisel3._
import chisel3.util._

/** One lane, fed `loopCount` times -- the hardware image of the `loop` variant.
  *
  * W = 1, T = N: the fully temporal end of the elementwise family. A single ALU is cycled once per element, which is
  * almost certainly slower than running the loop on the Snitch core. Its value is as a bring-up target: when the
  * cluster integration is what is being debugged, the smallest possible datapath is the one you want on the other end
  * of the streamer.
  *
  * ==Why the counter is here and not in the streamer==
  *
  * A SNAX streamer already performs address generation and would happily deliver exactly N beats, which makes an
  * internal counter look redundant. It is not, because the module must also work against a host whose "streamer" is a
  * FIFO that pushes data continuously and has no notion of when a run ends. Owning the trip count means the datapath
  * decides when it stops accepting, rather than trusting the producer to stop offering.
  *
  * ==Handshake: a join==
  *
  * A firing takes a beat from `a` and `b` and gives one to `out` in the same cycle (D129's lockstep): `out.valid` is
  * busy and both operands valid, `a.ready` busy, `b.valid` and `out.ready`, `b.ready` the same with `a.valid`. An
  * operand that arrives alone waits for the other, so the two streams cannot desynchronise, and `valid` does not depend
  * on `ready`. `ready` depends on the other operand's `valid`, which is legal (`SimpleAdder` is the same join).
  *
  * `loopCount` is sampled only at start, so it may change freely afterwards -- which is what a CSR write followed by a
  * start pulse gives you anyway. A start pulsed with `loopCount === 0` is ignored and the unit stays idle; a
  * zero-length run has no meaningful completion, and silently hanging is the worse failure (`samples.Accumulator`
  * ignores a start with `len === 0` too).
  *
  * @param dataWidth
  *   bits per element, from the BRM port's dtype
  * @param loopCountWidth
  *   bits of trip counter; must hold the largest T the driver will program
  * @param supportedOps
  *   encodings this instance can perform; a single entry emits fixed-function hardware and leaves `opSel` unused
  * @param signed
  *   interpret operands as two's complement for Min and Max
  */
class ElementwiseLoop(
  val dataWidth:      Int      = 32,
  val loopCountWidth: Int      = 32,
  val supportedOps:   Seq[Int] = ElementwiseOp.all,
  val signed:         Boolean  = false
) extends Module {
  require(dataWidth > 0, s"dataWidth must be positive, got $dataWidth")
  require(loopCountWidth > 0, s"loopCountWidth must be positive, got $loopCountWidth")

  /** Parameters are baked into the module name so that several instantiations coexist in SystemVerilog's flat, global
    * module namespace without Chisel's positional `_1` suffixes, which carry no information. This is also the name a
    * BRM's hardware binding will name.
    */
  override def desiredName: String =
    s"ElementwiseLoop_w${dataWidth}_t${loopCountWidth}_${ElementwiseOp.tag(supportedOps)}"

  val io = IO(new Bundle {
    val opSel     = Input(UInt(ElementwiseOp.width.W))
    val loopCount = Input(UInt(loopCountWidth.W))
    val start     = Input(Bool())
    val busy      = Output(Bool())
    val done      = Output(Bool())

    val a   = Flipped(Decoupled(UInt(dataWidth.W)))
    val b   = Flipped(Decoupled(UInt(dataWidth.W)))
    val out = Decoupled(UInt(dataWidth.W))
  })

  private val busy   = RegInit(false.B)
  private val count  = RegInit(0.U(loopCountWidth.W))
  private val target = RegInit(0.U(loopCountWidth.W))

  private val alu = Module(new ElementwiseAlu(dataWidth, supportedOps, signed))
  alu.io.opSel := io.opSel
  alu.io.a     := io.a.bits
  alu.io.b     := io.b.bits
  io.out.bits  := alu.io.out

  // The join. The busy term is not optional decoration: without it an idle
  // unit would consume and discard whatever a free-running producer offered,
  // which is precisely what owning the trip count exists to prevent.
  io.a.ready   := busy && io.b.valid && io.out.ready
  io.b.ready   := busy && io.a.valid && io.out.ready
  io.out.valid := busy && io.a.valid && io.b.valid

  private val advance = io.a.fire && io.b.fire
  private val last    = advance   && (count + 1.U) === target

  when(!busy) {
    when(io.start && io.loopCount =/= 0.U) {
      busy   := true.B
      target := io.loopCount
      count  := 0.U
    }
  }.otherwise {
    when(advance) {
      count := count + 1.U
      when(last) {
        busy  := false.B
        count := 0.U
      }
    }
  }

  io.busy := busy
  // Asserted on the same cycle as the final transfer, not one after it, so a
  // consumer can treat `done && out.fire` as "this beat was the last one".
  io.done := last
}
