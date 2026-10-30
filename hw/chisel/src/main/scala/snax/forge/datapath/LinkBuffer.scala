package snax.forge.datapath

import chisel3._
import chisel3.util.log2Ceil

import snax.forge.IntType

/** A link's buffer of `depth` beats (D149; C§17's catalogue entry `LinkBuffer`): data and enables only (D147).
  *
  * On `load` slot `wslot` takes the input beat; `out` is slot `rslot`. The occupancy and the ring's slots are its
  * group's control unit's (`ctrl.GroupCtrl`), so a beat written in one cycle is read from the next (latency 1). The
  * slots are ⌈log₂ depth⌉ bits wide, 0 at depth 1. SNAX-GVSoC's `SnaxLinkBuffer` (`snax/snax_datapath.hpp`) is the same
  * block.
  *
  * @param depth
  *   the beats it holds, at least 1
  * @param lanes
  *   elements per beat, the link's ends'
  * @param dtype
  *   the elements' type, the link's ends'
  */
class LinkBuffer(val depth: Int = 1, val lanes: Int = 4, val dtype: String = "int64") extends DatapathModule {
  require(depth > 0, s"depth must be at least 1, got $depth")
  require(lanes > 0, s"lanes must be positive, got $lanes")

  private val t = IntType(dtype)

  override def desiredName: String = s"LinkBuffer_${t.tag}_n${lanes}_d$depth"

  val io = IO(new Bundle {
    val in    = Input(Vec(lanes, t.lane))
    val out   = Output(Vec(lanes, t.lane))
    val load  = Input(Bool())
    val wslot = Input(UInt(log2Ceil(depth).W))
    val rslot = Input(UInt(log2Ceil(depth).W))
  })

  private val slots = RegInit(VecInit(Seq.fill(depth)(VecInit(Seq.fill(lanes)(0.U(t.width.W))))))

  when(io.load) {
    slots(if (depth == 1) 0.U else io.wslot) := VecInit(io.in.map(_.asUInt))
  }

  private val beat = slots(if (depth == 1) 0.U else io.rslot)
  for (l <- 0 until lanes) io.out(l) := IntType.cast(beat(l), t, t)
}
