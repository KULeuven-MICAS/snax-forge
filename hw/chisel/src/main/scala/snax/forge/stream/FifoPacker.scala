package snax.forge.stream

import chisel3._
import chisel3.util._

import snax.forge.IntType

/** `inLanes` lanes of `laneType` in per push, a beat of `words` 64-bit words out: the FIFO packer, `FifoSlicer`'s
  * mirror.
  *
  * Placed by `shell.ShellWrapper` on a packed outer output (D174): a one-lane result, `FifoPacker(lanes, dtype, 1)`,
  * between the unit's output and its writer, r = d_out results to a word. snax-gvsoc's `SnaxPacker` is its model, cycle
  * for cycle. Only the slicer stays parked (open item 93).
  *
  * A beat is `words` * num_slc elements (num_slc = 64 / width), element e at bits `[w(e+1)-1 : we]` of the beat. r =
  * `words` * num_slc / `inLanes` in-beats fill one out-beat, lane l of in-beat u element u * `inLanes` + l. The
  * in-beats before the sending one are held in registers with a counter of ceil(log2 r) bits. An in-beat sends when it
  * is the r-th (count = r - 1) or `last` is high (the run's last result, from the group's counters, never from this
  * cycle's push): `out.valid` is that in-beat's valid, the out-beat the held in-beats, this one at slot count and zeros
  * above it (no older held value leaks), and a sending in-beat is taken when the out-beat leaves (`in.ready` is
  * `out.ready`); a filling in-beat needs no room (`in.ready` high). `clear` (the
  * start accepted) resets the count. With the input always valid, the output always ready and `clear` and `last` low,
  * it takes one in-beat a cycle.
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

  /** In-beats per out-beat: d_out on a packed outer output. */
  val r: Int = total / inLanes

  override def desiredName: String = s"FifoPacker_n${inLanes}_${t.tag}_w${words}"

  val io = IO(new Bundle {
    val in    = Flipped(Decoupled(Vec(inLanes, t.lane)))
    val out   = Decoupled(Vec(words, UInt(64.W)))
    val clear = Input(Bool())
    val last  = Input(Bool())
  })

  private val chunk   = inLanes * t.width
  private val held    = Option.when(r > 1)(Reg(Vec(r - 1, UInt(chunk.W))))
  private val count   = RegInit(0.U(log2Ceil(r).W))
  private val sending = if (r == 1) true.B else count === (r - 1).U || io.last

  // Slot u: a held in-beat below count, this one at count, zero above
  private val slots = (0 until r).map { u =>
    val here = if (r == 1) true.B else count === u.U
    val kept = held.filter(_ => u < r - 1).map(h => Mux(count > u.U, h(u), 0.U(chunk.W))).getOrElse(0.U(chunk.W))
    Mux(here, io.in.bits.asUInt, kept)
  }

  io.out.valid := io.in.valid && sending
  io.out.bits  := VecInit(slots).asUInt.asTypeOf(io.out.bits)
  io.in.ready  := !sending || io.out.ready

  held.foreach { h =>
    when(io.in.fire) {
      when(sending) {
        count := 0.U
      }.otherwise {
        h(count) := io.in.bits.asUInt
        count    := count + 1.U
      }
    }
    when(io.clear) {
      count := 0.U
    }
  }
}
