package snax.forge.datapath

import chisel3._

import snax.forge.IntType

/** The MAC (C§17's catalogue entry `Mac`): one block with design-time variants, data and enables only.
  *
  * Per lane the product p = cast_out(cast_op(a) · cast_op(b)): `a` and `b` each extended to `opType` by its own
  * signedness, multiplied wrapping at `opType`, then extended or wrapped into `outType` (D143).
  *
  *   - `accumulate`: on `fire`, s ← `first` ? p : s + p at `outType`; `out` is s. Latency 1, pins `fire` and `first`:
  *     its control unit pushes `out` in the cycle after a job's last beat fires, and may fire the next job's first beat
  *     in that cycle, since s changes only at the cycle's end.
  *   - plain: `out` = p + c at `outType` (`c` of `out`'s type and lanes), combinational (latency 0), or with `regOut` a
  *     register that takes it on `en` (latency 1, pin `en`: its firing).
  *
  * SNAX-GVSoC's `SnaxMac` (snax-gvsoc `snax/snax_datapath.hpp`) is the same block, checked on the same vectors.
  *
  * @param inTypes
  *   `a`'s and `b`'s types (and, plain, `c`'s: `outType`), NumPy's names
  * @param opType
  *   the product's type, NumPy's result type of `a` and `b` (the unit's `opType`)
  * @param outType
  *   `out`'s type, the sum's
  * @param lanes
  *   elements per beat
  * @param accumulate
  *   the accumulating form; otherwise the plain a·b + c
  * @param regOut
  *   the plain form's output register (`regIn` is not built)
  */
class Mac(
  val inTypes:    Seq[String] = Seq("int64", "int64"),
  val opType:     String      = "int64",
  val outType:    String      = "int64",
  val lanes:      Int         = 4,
  val accumulate: Boolean     = true,
  val regOut:     Boolean     = false
) extends DatapathModule {
  require(lanes > 0, s"lanes must be positive, got $lanes")
  require(
    !(accumulate && regOut),
    "regOut is the plain form's output register; an accumulating Mac's s is its register"
  )
  require(
    inTypes.size == (if (accumulate) 2 else 3),
    s"inTypes must name a, b${if (accumulate) "" else " and c"}, got ${inTypes.mkString(", ")}"
  )

  private val ins  = inTypes.map(IntType(_))
  private val opT  = IntType(opType)
  private val outT = IntType(outType)
  require(!accumulate == (ins.size == 3) && (accumulate || ins(2) == outT), s"c must be of outType $outType")

  private val form = if (accumulate) "acc" else if (regOut) "reg" else "plain"

  override def desiredName: String =
    s"Mac_${ins.take(2).map(_.tag).mkString("x")}_${opT.tag}_${outT.tag}_n${lanes}_$form"

  val io = IO(new Bundle {
    val a     = Input(Vec(lanes, ins(0).lane))
    val b     = Input(Vec(lanes, ins(1).lane))
    val c     = if (!accumulate) Some(Input(Vec(lanes, outT.lane))) else None
    val out   = Output(Vec(lanes, outT.lane))
    val fire  = if (accumulate) Some(Input(Bool())) else None
    val first = if (accumulate) Some(Input(Bool())) else None
    val en    = if (!accumulate && regOut) Some(Input(Bool())) else None
  })

  // Per lane p = cast_out(cast_op(a) * cast_op(b)): an unsigned product's low opType bits are the signed one's too
  private val p = (0 until lanes).map { l =>
    val x = IntType.cast(io.a(l), ins(0), opT).asUInt
    val y = IntType.cast(io.b(l), ins(1), opT).asUInt
    IntType.cast((x * y)(opT.width - 1, 0), opT, outT).asUInt
  }

  if (accumulate) {
    val s = RegInit(VecInit(Seq.fill(lanes)(0.U(outT.width.W))))
    when(io.fire.get) {
      for (l <- 0 until lanes) s(l) := Mux(io.first.get, p(l), s(l) + p(l))
    }
    for (l <- 0 until lanes) io.out(l) := IntType.cast(s(l), outT, outT)
  } else {
    val y = (0 until lanes).map(l => p(l) + io.c.get(l).asUInt)
    if (regOut) {
      val r = RegInit(VecInit(Seq.fill(lanes)(0.U(outT.width.W))))
      when(io.en.get) {
        for (l <- 0 until lanes) r(l) := y(l)
      }
      for (l <- 0 until lanes) io.out(l) := IntType.cast(r(l), outT, outT)
    } else {
      for (l <- 0 until lanes) io.out(l) := IntType.cast(y(l)(outT.width - 1, 0), outT, outT)
    }
  }
}
