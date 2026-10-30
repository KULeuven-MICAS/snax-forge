package snax.forge.stream

import chisel3._
import chisel3.util._

import snax.forge.IntType

/** `inLanes` lanes of `laneType` in per push, a beat of `words` 64-bit words out: the FIFO packer, `FifoSlicer`'s
  * mirror.
  *
  * Built and parked: the flow places only the wire slice (D142), and this block is not in `Datapaths`' catalogue.
  *
  * A beat is `words` * num_slc elements (num_slc = 64 / width), element e at bits `[w(e+1)-1 : we]` of the beat. r =
  * `words` * num_slc / `inLanes` in-beats fill one out-beat, lane l of in-beat u element u * `inLanes` + l. The first r
  * \- 1 in-beats are held in registers with a counter of ceil(log2 r) bits; the out-beat is emitted in the cycle the
  * last lanes arrive (`out.valid` is that in-beat's valid, the out-beat the held lanes and these), and that in-beat is
  * taken in the cycle the out-beat leaves (`in.ready` is then `out.ready`). With the input always valid and the output
  * always ready, it takes one in-beat a cycle.
  *
  * @param inLanes
  *   lanes per in-beat; divides `words` * num_slc
  * @param laneType
  *   the elements' type, NumPy's name ("int32", "uint8")
  * @param words
  *   64-bit words per out-beat
  */
class FifoPacker(val inLanes: Int, val laneType: String, val words: Int) extends Module {
  require(words > 0, s"words must be positive, got $words")
  require(inLanes > 0, s"inLanes must be positive, got $inLanes")

  private val t      = IntType(laneType)
  private val numSlc = 64 / t.width
  private val total  = words * numSlc
  require(total % inLanes == 0, s"inLanes $inLanes must divide the beat's $total elements of $laneType")

  /** In-beats per out-beat. */
  val r: Int = total / inLanes

  override def desiredName: String = s"FifoPacker_n${inLanes}_${t.tag}_w${words}"

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(Vec(inLanes, t.lane)))
    val out = Decoupled(Vec(words, UInt(64.W)))
  })

  private val chunk = inLanes * t.width
  private val held  = Option.when(r > 1)(Reg(Vec(r - 1, UInt(chunk.W))))
  private val count = RegInit(0.U(log2Ceil(r).W))
  private val last  = if (r == 1) true.B else count === (r - 1).U

  io.out.valid := io.in.valid && last
  io.out.bits  := VecInit(held.toSeq.flatten :+ io.in.bits.asUInt).asUInt.asTypeOf(io.out.bits)
  io.in.ready  := !last || io.out.ready

  held.foreach { h =>
    when(io.in.fire) {
      when(last) {
        count := 0.U
      }.otherwise {
        h(count) := io.in.bits.asUInt
        count    := count + 1.U
      }
    }
  }
}
