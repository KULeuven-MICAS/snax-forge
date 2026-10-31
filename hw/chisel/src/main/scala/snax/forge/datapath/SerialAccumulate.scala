package snax.forge.datapath

import chisel3._
import chisel3.util.log2Ceil

import snax.forge.IntType

/** The one-lane sum (C§17's catalogue entry `SerialAccumulate`): data and enables only, span = `lanes`.
  *
  * It reads its input beat in place, one lane a cycle: on `read`, acc ← (`first` ∧ `lane` = 0) ? cast_out(x_lane) : acc
  * + cast_out(x_lane), each lane cast to `outType` by `inType`'s signedness (D143); `out` is acc. It is written as acc
  * ← ((`first` ∧ `lane` = 0) ? 0 : acc) + x, the adder's acc input cleared on a restart (0 + x = x), so the lane `x` is
  * selected once and feeds the adder only. Written as a choice between x and acc + x, firtool folds a constant `first`
  * and, at 2 lanes (a 1-bit `lane`), leaves `acc + (lane ? in_1 : in_0)` under `if (lane)`: a 1-bit-select mux inside
  * an adder under a branch on the same bit, valid RTL on which Design Compiler crashes (DC7 in snax-forge-experiments'
  * PITFALLS). Its control unit reads lane k in cycle k and takes the beat on the last lane, so `out` holds the beat's
  * sum from the cycle after (latency 1 after the last lane). SNAX-GVSoC's `SnaxSerialAccumulate`
  * (`snax/snax_datapath.hpp`) is the same block.
  *
  * @param inType
  *   the input's lane type, NumPy's name
  * @param outType
  *   the sum's type, at least as wide as `inType`
  * @param lanes
  *   elements per input beat: its span
  */
class SerialAccumulate(val inType: String = "int64", val outType: String = "int64", val lanes: Int = 4)
    extends DatapathModule {
  require(lanes > 0, s"lanes must be positive, got $lanes")

  private val inT  = IntType(inType)
  private val outT = IntType(outType)
  require(outT.width >= inT.width, s"outType $outType must be at least as wide as inType $inType")

  override def desiredName: String = s"SerialAccumulate_${inT.tag}_${outT.tag}_n${lanes}"

  val io = IO(new Bundle {
    val in    = Input(Vec(lanes, inT.lane))
    val out   = Output(outT.lane)
    val read  = Input(Bool())
    val lane  = Input(UInt(log2Ceil(lanes).W))
    val first = Input(Bool())
  })

  private val x   = IntType.cast(if (lanes == 1) io.in(0) else io.in(io.lane), inT, outT).asUInt
  private val acc = RegInit(0.U(outT.width.W))

  when(io.read) {
    acc := Mux(io.first && io.lane === 0.U, 0.U, acc) + x
  }

  io.out := IntType.cast(acc, outT, outT)
}
