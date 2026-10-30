package snax.forge.datapath

import chisel3._

import snax.forge.IntType

/** The datapath of a fold (D133): the `accumulate` BRM's sum, latency 1, pins `fire` and `first`.
  *
  * Data and enables only: no valid, ready or counter. Each input beat is `lanes` elements of `inType`; each is cast to
  * `outType` (NumPy's fold: sign-extended from a signed `inType`, zero-extended from an unsigned one, D143) and a
  * combinational tree of `lanes` → 1 adders sums them, wrapping at `outType`. On `fire` the register `acc` takes the
  * beat's sum when `first` (a job's first beat) and adds it otherwise; `out` is `acc`. Its group's control unit
  * (`ctrl.GroupCtrl`, L 1) pushes `out` in the cycle after a job's last beat fires, and may fire the next job's first
  * beat in that same cycle, since `acc` changes only at the cycle's end: no drain cycle.
  *
  * @param inType
  *   the input's type, NumPy's name ("int32", "uint8"), from the unit's input port's dtype
  * @param outType
  *   the output's type, from the unit's output port's dtype, at least as wide as `inType`: the tree and the register
  *   run at it, wrapping as NumPy's sum into it does
  * @param lanes
  *   elements per input beat, the unit's input port's lanes
  */
class TreeAccumulate(val inType: String = "int32", val outType: String = "int32", val lanes: Int = 1)
    extends DatapathModule {
  require(lanes > 0, s"lanes must be positive, got $lanes")

  private val inT  = IntType(inType)
  private val outT = IntType(outType)
  require(
    outT.width >= inT.width,
    s"outType $outType must be at least as wide as inType $inType, otherwise every lane is truncated on the way in"
  )

  override def desiredName: String = s"TreeAccumulate_${inT.tag}_${outT.tag}_n${lanes}"

  val io = IO(new Bundle {
    val in    = Input(Vec(lanes, inT.lane))
    val out   = Output(outT.lane)
    val fire  = Input(Bool())
    val first = Input(Bool())
  })

  // The tree: each lane cast to outType (extended by inType's signedness), then summed, wrapping at outType's width.
  private val beat = VecInit(io.in.map(IntType.cast(_, inT, outT).asUInt)).reduceTree(_ + _)
  private val acc  = RegInit(0.U(outT.width.W))

  when(io.fire) {
    acc := Mux(io.first, beat, acc + beat)
  }

  io.out := IntType.cast(acc, outT, outT)
}
