package snax.forge.stream

import chisel3._
import chisel3.util._

import snax.forge.IntType

/** A beat of `words` 64-bit words in, `outLanes` lanes of `laneType` out per pop: the FIFO slicer (D142).
  *
  * Built and parked: the flow places only the wire slice (D142, a port of W lanes takes W / num_slc whole words), and
  * this block is not in `Blocks`' catalogue. It is here for a block whose lanes are fewer than a beat's elements.
  *
  * A word holds num_slc = 64 / width slices (elements), element j of word i at bits `[w(j+1)-1 : wj]` (D142), so a beat
  * is `words` * num_slc elements, element e at bits `[w(e+1)-1 : we]` of the beat. Each in-beat is popped r = `words` *
  * num_slc / `outLanes` times, lane l of pop u element u * `outLanes` + l.
  *
  * State: the in-beat's words and a counter of ceil(log2 r) bits (none for r = 1), with a bit for whether a beat is
  * held. It takes the next in-beat in the cycle its last pop leaves, so a stream of in-beats has no bubble: with the
  * input always valid and the output always ready, n in-beats give their n * r pops in n * r cycles after the first is
  * taken. `out.valid` is a register's; `in.ready` depends on `out.ready` in the cycle of the last pop.
  *
  * @param words
  *   64-bit words per in-beat
  * @param laneType
  *   the elements' type, NumPy's name ("int32", "uint8")
  * @param outLanes
  *   lanes per pop; divides `words` * num_slc
  */
class FifoSlicer(val words: Int, val laneType: String, val outLanes: Int) extends Module {
  require(words > 0, s"words must be positive, got $words")
  require(outLanes > 0, s"outLanes must be positive, got $outLanes")

  private val t      = IntType(laneType)
  private val numSlc = 64 / t.width
  private val total  = words * numSlc
  require(total % outLanes == 0, s"outLanes $outLanes must divide the beat's $total elements of $laneType")

  /** Pops per in-beat. */
  val r: Int = total / outLanes

  override def desiredName: String = s"FifoSlicer_w${words}_${t.tag}_n${outLanes}"

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(Vec(words, UInt(64.W))))
    val out = Decoupled(Vec(outLanes, t.lane))
  })

  private val data  = Reg(Vec(words, UInt(64.W)))
  private val full  = RegInit(false.B)
  private val count = RegInit(0.U(log2Ceil(r).W))
  private val last  = if (r == 1) true.B else count === (r - 1).U

  // The beat's pops, pop u its elements u * outLanes until (u + 1) * outLanes.
  private val pops = data.asUInt.asTypeOf(Vec(r, UInt((outLanes * t.width).W)))

  io.out.valid := full
  io.out.bits  := (if (r == 1) pops(0) else pops(count)).asTypeOf(io.out.bits)
  io.in.ready  := !full || (io.out.ready && last)

  when(io.in.fire) {
    data  := io.in.bits
    full  := true.B
    count := 0.U
  }.elsewhen(io.out.fire) {
    when(last) {
      full := false.B
    }.otherwise {
      count := count + 1.U
    }
  }
}
