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
  * Seeded (`seed`, D163, B2 R5), it has one more input, `seed`, one element of `outType` read once per job, added on
  * the job's first beat's last lane, the read that fires it (`read` ∧ `first` ∧ `lane` = lanes − 1), when its control
  * unit takes the seed: the job's sum is the seed's plus its lanes'. Unseeded it is the block it was, the same RTL.
  *
  * @param inType
  *   the input's lane type, NumPy's name
  * @param outType
  *   the sum's type, at least as wide as `inType`
  * @param lanes
  *   elements per input beat: its span
  * @param seed
  *   whether the unit has a seed port (the BRM's `seed`)
  */
class SerialAccumulate(
  val inType:  String  = "int64",
  val outType: String  = "int64",
  val lanes:   Int     = 4,
  val seed:    Boolean = false
) extends DatapathModule {
  require(lanes > 0, s"lanes must be positive, got $lanes")

  private val inT  = IntType(inType)
  private val outT = IntType(outType)
  require(outT.width >= inT.width, s"outType $outType must be at least as wide as inType $inType")

  override def desiredName: String = s"SerialAccumulate_${inT.tag}_${outT.tag}_n${lanes}${if (seed) "_seed" else ""}"

  val io = IO(new Bundle {
    val in    = Input(Vec(lanes, inT.lane))
    val seed  = Option.when(SerialAccumulate.this.seed)(Input(outT.lane))
    val out   = Output(outT.lane)
    val read  = Input(Bool())
    val lane  = Input(UInt(log2Ceil(lanes).W))
    val first = Input(Bool())
  })

  private val x   = IntType.cast(if (lanes == 1) io.in(0) else io.in(io.lane), inT, outT).asUInt
  private val acc = RegInit(0.U(outT.width.W))

  io.seed match {
    case None    =>
      when(io.read) {
        acc := Mux(io.first && io.lane === 0.U, 0.U, acc) + x
      }
    case Some(q) => // the job's first beat's last lane adds the seed
      val last = if (lanes == 1) true.B else io.lane === (lanes - 1).U
      val add  = Mux(io.first && last, IntType.cast(q, outT, outT).asUInt, 0.U)
      when(io.read) {
        acc := Mux(io.first && io.lane === 0.U, 0.U, acc) + x + add
      }
  }

  io.out := IntType.cast(acc, outT, outT)
}
