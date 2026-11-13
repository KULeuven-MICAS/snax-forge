package snax.forge.datapath

import chisel3._

import snax.forge.IntType

/** The lane accumulator (the catalogue entry `LaneAccumulate` of CONTRACTS, "Accelerator RTL spec", the
  * `lane_accumulate` BRM): each lane keeps its own running sum, Mac's accumulating form without the product. Data and
  * enables only.
  *
  * Per lane x = cast_out(a), `a` extended to `outType` by its own signedness; on `fire`, s ← `first` ? x : s + x at
  * `outType`, wrapping; `out` is s. Latency 1, pins `fire` and `first`: its control unit pushes `out` in the cycle
  * after a job's last beat fires (one sum per lane every T beats, T the job's JOB_BEATS), and may fire the next job's
  * first beat in that cycle, since s changes only at the cycle's end. rowsum 2 runs it with W rows on the W lanes, a
  * column a beat.
  *
  * SNAX-GVSoC's `SnaxLaneAccumulate` (snax-gvsoc `snax/snax_datapath.hpp`) is the same block, checked on the same
  * vectors.
  *
  * @param inType
  *   `a`'s type, NumPy's name, from the unit's input port's dtype
  * @param outType
  *   `out`'s type, the sum's, at least as wide as `inType`
  * @param lanes
  *   elements per beat, on both ports
  */
class LaneAccumulate(val inType: String = "int64", val outType: String = "int64", val lanes: Int = 4)
    extends DatapathModule {
  require(lanes > 0, s"lanes must be positive, got $lanes")

  private val inT  = IntType(inType)
  private val outT = IntType(outType)
  require(
    outT.width >= inT.width,
    s"outType $outType must be at least as wide as inType $inType, otherwise every lane is truncated on the way in"
  )

  override def desiredName: String = s"LaneAccumulate_${inT.tag}_${outT.tag}_n$lanes"

  val io = IO(new Bundle {
    val a     = Input(Vec(lanes, inT.lane))
    val out   = Output(Vec(lanes, outT.lane))
    val fire  = Input(Bool())
    val first = Input(Bool())
  })

  private val s = RegInit(VecInit(Seq.fill(lanes)(0.U(outT.width.W))))
  when(io.fire) {
    for (l <- 0 until lanes) {
      val x = IntType.cast(io.a(l), inT, outT).asUInt
      s(l) := Mux(io.first, x, s(l) + x)
    }
  }
  for (l <- 0 until lanes) io.out(l) := IntType.cast(s(l), outT, outT)
}
