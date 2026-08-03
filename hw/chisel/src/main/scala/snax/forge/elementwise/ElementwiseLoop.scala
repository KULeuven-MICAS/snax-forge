package snax.forge.elementwise

import chisel3._
import chisel3.util._

/** One lane, fed `loopCount` times -- the hardware image of the `loop` variant.
  *
  * W = 1, T = N in the descriptor's shape section: the fully temporal end of the elementwise family. A single ALU is
  * cycled once per element, which is almost certainly slower than running the loop on the CVA6 core. Its value is as a
  * bring-up target: when the cluster integration is what is being debugged, the smallest possible datapath is the one
  * you want on the other end of the streamer.
  *
  * ==Why the counter is here and not in the streamer==
  *
  * A SNAX streamer already performs address generation and would happily deliver exactly N beats, which makes an
  * internal counter look redundant. It is not, because the module must also work against a host whose "streamer" is a
  * FIFO that pushes data continuously and has no notion of when a run ends. Owning the trip count means the datapath
  * decides when it stops accepting, rather than trusting the producer to stop offering.
  *
  * ==Handshake, and the two properties it gives up==
  *
  * `ready` is passed straight through from the consumer: `a.ready` and `b.ready` are `out.ready`, gated only by `busy`.
  * `out.valid` is the conjunction of both operand handshakes, i.e. it asserts on the cycle a transfer actually
  * completes. Simple control, and correct as long as the producer supplies both operands together -- which is what a
  * streamer delivering a tile does.
  *
  * Two consequences follow, and neither is a bug in the wiring so much as a contract on the producer:
  *
  *   1. '''Operands must arrive in lockstep.''' If `a.valid` is high while `b.valid` is low and the consumer is ready,
  *      `a.ready` is still asserted, so that beat of `a` is consumed while no output is produced. The element is lost
  *      and the two streams desynchronise permanently. A join would gate each operand's ready on the other's valid and
  *      make this impossible; that is one extra AND term per operand, and it is the fix if this assumption ever has to
  *      be relaxed. `ElementwiseLoopSpec` pins the behaviour explicitly so it cannot change silently.
  *   1. '''`valid` depends on `ready`.''' Because `out.valid` includes both operand handshakes, and those include
  *      `out.ready`, this unit will not assert `out.valid` into a consumer that is not ready. A consumer whose `ready`
  *      is in turn a function of `valid` deadlocks against it. SNAX's streamer write port does not behave that way, so
  *      this is safe in the intended setting and hazardous outside it.
  *
  * `loopCount` is sampled only at start, so it may change freely afterwards -- which is what a CSR write followed by a
  * start pulse gives you anyway. A start pulsed with `loopCount === 0` is ignored and the unit stays idle; a
  * zero-length run has no meaningful completion, and silently hanging (as `samples.Accumulator` does in the same
  * situation) is the worse failure.
  *
  * @param dataWidth
  *   bits per element, from the DaCe dtype
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
    * module namespace without Chisel's positional `_1` suffixes, which carry no information. This is also the name the
    * descriptor's `module_name` will eventually be resolved against.
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

  // Ready is the consumer's ready, gated by busy. The busy term is not
  // optional decoration: without it an idle unit would consume and discard
  // whatever a free-running producer offered, which is precisely what owning
  // the trip count exists to prevent.
  io.a.ready := busy && io.out.ready
  io.b.ready := busy && io.out.ready

  // A transfer is happening on both operand channels this cycle.
  io.out.valid := io.a.valid && io.a.ready && io.b.valid && io.b.ready

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
