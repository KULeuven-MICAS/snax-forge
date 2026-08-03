package snax.forge.elementwise

import chisel3._
import chisel3.util._

/** `lanes` lanes, fed `loopCount` times -- the hardware image of the `tiled_spatial` variant.
  *
  * W = tile, T = ceil(N/tile): the interior of the elementwise family, and the only member of it that is a plausible
  * design rather than a limiting case. Structurally it is `ElementwiseSpatial` plus the trip counter from
  * `ElementwiseLoop`, and the three are kept as separate modules so each can be brought up and read on its own.
  *
  * ==No accumulator==
  *
  * A tile boundary is not a reduction. Each pass reads its own operands and writes its own results, and nothing is
  * carried between passes, so there is no accumulation register and none is wanted. Cross-beat state appears when a WCR
  * edge does -- see `samples.Accumulator` -- and that is a different pattern with a different hazard profile: a
  * reduction is a chaining barrier, an elementwise tile is not.
  *
  * ==loopCount is given, not derived==
  *
  * The unit counts passes; it does not compute how many passes a problem needs. If N elements are to be processed at
  * `lanes` per beat, the driver programs ceil(N / lanes) and the arithmetic happens where N is known. Putting a divider
  * here would duplicate a calculation the descriptor already carries in its `trips` field, and would do it in the one
  * place with no access to N.
  *
  * A consequence worth stating: a final partial tile is not handled. If `lanes` does not divide N, the last pass still
  * processes `lanes` elements and the surplus is whatever the producer supplied. Masking it is the streamer's job or a
  * future predication port; the descriptor's `ragged` flag is what says which runs need it.
  *
  * ==Handshake==
  *
  * One ready/valid pair per port, covering all `lanes` elements. `ready` passes through from the consumer gated by
  * `busy`, and the trip counter advances on the cycle both operand transfers complete. See `ElementwiseLoop` for the
  * two properties this wiring trades away -- operands must arrive in lockstep, and `valid` depends on `ready`.
  *
  * @param dataWidth
  *   bits per element, from the DaCe dtype
  * @param lanes
  *   elements per beat, i.e. the tile size the SDFG map was tiled by
  * @param loopCountWidth
  *   bits of trip counter; must hold the largest T the driver will program
  * @param supportedOps
  *   encodings this instance can perform; a single entry emits fixed-function hardware and leaves `opSel` unused
  * @param signed
  *   interpret operands as two's complement for Min and Max
  */
class ElementwiseTiledSpatial(
  val dataWidth:      Int      = 32,
  val lanes:          Int      = 4,
  val loopCountWidth: Int      = 32,
  val supportedOps:   Seq[Int] = ElementwiseOp.all,
  val signed:         Boolean  = false
) extends Module {
  require(dataWidth > 0, s"dataWidth must be positive, got $dataWidth")
  require(lanes > 0, s"lanes must be positive, got $lanes")
  require(loopCountWidth > 0, s"loopCountWidth must be positive, got $loopCountWidth")

  override def desiredName: String =
    s"ElementwiseTiledSpatial_w${dataWidth}_n${lanes}_t${loopCountWidth}_${ElementwiseOp.tag(supportedOps)}"

  val io = IO(new Bundle {
    val opSel     = Input(UInt(ElementwiseOp.width.W))
    val loopCount = Input(UInt(loopCountWidth.W))
    val start     = Input(Bool())
    val busy      = Output(Bool())
    val done      = Output(Bool())

    val a   = Flipped(Decoupled(Vec(lanes, UInt(dataWidth.W))))
    val b   = Flipped(Decoupled(Vec(lanes, UInt(dataWidth.W))))
    val out = Decoupled(Vec(lanes, UInt(dataWidth.W)))
  })

  private val busy   = RegInit(false.B)
  private val count  = RegInit(0.U(loopCountWidth.W))
  private val target = RegInit(0.U(loopCountWidth.W))

  private val alus = Seq.fill(lanes)(Module(new ElementwiseAlu(dataWidth, supportedOps, signed)))
  alus.zipWithIndex.foreach { case (alu, i) =>
    alu.io.opSel := io.opSel
    alu.io.a     := io.a.bits(i)
    alu.io.b     := io.b.bits(i)
  }
  io.out.bits := VecInit(alus.map(_.io.out))

  io.a.ready := busy && io.out.ready
  io.b.ready := busy && io.out.ready

  io.out.valid := io.a.valid && io.a.ready && io.b.valid && io.b.ready

  // One tile per completed pair of operand transfers.
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
  io.done := last
}
