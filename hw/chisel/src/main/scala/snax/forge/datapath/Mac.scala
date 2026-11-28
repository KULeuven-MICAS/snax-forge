package snax.forge.datapath

import chisel3._
import chisel3.util.log2Ceil

import snax.forge.IntType

/** The MAC (the catalogue entry `Mac` of CONTRACTS, "Accelerator RTL spec"): one block with design-time variants, data
  * and enables only.
  *
  * Per lane the product p = cast_out(cast_op(a) · cast_op(b)): `a` and `b` each extended to `opType` by its own
  * signedness, multiplied wrapping at `opType`, then extended or wrapped into `outType`. With `narrowMul` (D207) a·b is
  * built at the operands' own widths and then extended to `opType` (`IntType.narrowProduct`), the same bits, where they
  * fit it; the module name then ends in `_nm`.
  *
  *   - `accumulate`: on `fire`, s ← `first` ? p : s + p at `outType`; `out` is s. Latency 1, pins `fire` and `first`:
  *     its control unit pushes `out` in the cycle after a job's last beat fires, and may fire the next job's first beat
  *     in that cycle, since s changes only at the cycle's end. Seeded (`seed`), it has one more input, `seed`, of
  *     `out`'s type and lanes, read once per job: s ← `first` ? seed + p : s + p, so lane l's sum starts from seed(l).
  *     The control unit holds the first beat until the seed is valid and takes it with that firing (`ctrl.GroupRule`);
  *     unseeded it is the block it was, the same RTL. A bank (`rows` R > 1) keeps R rows of W sums: a firing adds into
  *     row `row` (`first` starting it) and `out` is row `drain`, the row the last firing wrote, so a result pushed a
  *     cycle after its firing is that row while the next firing adds into another. Both pins are the control unit's
  *     (`<unit>_row`, the beat mod ROWS; `drain` the row before it, ROWS − 1 after 0), so the block keeps no row of its
  *     own and its registers are the R rows only. A seeded bank takes a seed beat on each of a job's first R firings,
  *     row ← (`first` ? seed : s[row]) + p: one W-lane mux before the adders, shared by the rows.
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
  * @param seed
  *   the accumulating form's seed port (the BRM's `seed`); refused on the plain form
  * @param rows
  *   the accumulating form's rows of sums, its bank (1: the plain `Mac`, no pin `row` or `drain`)
  * @param narrowMul
  *   a·b built at the operands' own widths where w_a + w_b fits `opType` (D207); elsewhere the default's
  */
class Mac(
  val inTypes:    Seq[String] = Seq("int64", "int64"),
  val opType:     String      = "int64",
  val outType:    String      = "int64",
  val lanes:      Int         = 4,
  val accumulate: Boolean     = true,
  val regOut:     Boolean     = false,
  val seed:       Boolean     = false,
  val rows:       Int         = 1,
  val narrowMul:  Boolean     = false
) extends DatapathModule {
  require(lanes > 0, s"lanes must be positive, got $lanes")
  require(
    !(accumulate && regOut),
    "regOut is the plain form's output register; an accumulating Mac's s is its register"
  )
  require(!seed || accumulate, "seed is the accumulating form's start; a plain Mac adds c instead")
  require(rows >= 1, s"rows must be 1 or more, got $rows")
  require(
    rows == 1 || accumulate,
    s"rows $rows, but accumulate is false; only an accumulating Mac has a bank (set accumulate true, or rows 1)"
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

  /** a·b is built narrow: asked for, and it fits `opType` (D207). */
  val narrow: Boolean = narrowMul && IntType.narrowFits(ins(0), ins(1), opT)

  override def desiredName: String =
    s"Mac_${ins.take(2).map(_.tag).mkString("x")}_${opT.tag}_${outT.tag}_n${lanes}_$form${if (seed) "_seed" else ""}" +
      (if (rows > 1) s"_r$rows" else "") + (if (narrow) "_nm" else "")

  val io = IO(new Bundle {
    val a     = Input(Vec(lanes, ins(0).lane))
    val b     = Input(Vec(lanes, ins(1).lane))
    val c     = if (!accumulate) Some(Input(Vec(lanes, outT.lane))) else None
    val seed  = Option.when(Mac.this.seed)(Input(Vec(lanes, outT.lane)))
    val out   = Output(Vec(lanes, outT.lane))
    val fire  = if (accumulate) Some(Input(Bool())) else None
    val first = if (accumulate) Some(Input(Bool())) else None
    val en    = if (!accumulate && regOut) Some(Input(Bool())) else None
    val row   = Option.when(rows > 1)(Input(UInt(log2Ceil(rows).W)))
    val drain = Option.when(rows > 1)(Input(UInt(log2Ceil(rows).W)))
  })

  // Per lane p = cast_out(cast_op(a) * cast_op(b)): an unsigned product's low opType bits are the signed one's too
  private val p = (0 until lanes).map { l =>
    if (narrow) IntType.cast(IntType.narrowProduct(io.a(l), ins(0), io.b(l), ins(1), opT), opT, outT).asUInt
    else {
      val x = IntType.cast(io.a(l), ins(0), opT).asUInt
      val y = IntType.cast(io.b(l), ins(1), opT).asUInt
      IntType.cast((x * y)(opT.width - 1, 0), opT, outT).asUInt
    }
  }

  if (rows > 1) { // NB (D191): the bank's rows, written at `row` and read at `drain`; no register of a row
    val s = RegInit(VecInit(Seq.fill(rows)(VecInit(Seq.fill(lanes)(0.U(outT.width.W))))))
    io.seed match {
      case None    =>
        when(io.fire.get) {
          for (l <- 0 until lanes) s(io.row.get)(l) := Mux(io.first.get, p(l), s(io.row.get)(l) + p(l))
        }
      case Some(q) => // NB (D194): each row's first firing in a job starts from the seed beat it takes
        when(io.fire.get) {
          for (l <- 0 until lanes)
            s(io.row.get)(l) := Mux(io.first.get, IntType.cast(q(l), outT, outT).asUInt, s(io.row.get)(l)) + p(l)
        }
    }
    for (l <- 0 until lanes) io.out(l) := IntType.cast(s(io.drain.get)(l), outT, outT)
  } else if (accumulate) {
    val s = RegInit(VecInit(Seq.fill(lanes)(0.U(outT.width.W))))
    io.seed match {
      case None    =>
        when(io.fire.get) {
          for (l <- 0 until lanes) s(l) := Mux(io.first.get, p(l), s(l) + p(l))
        }
      case Some(q) => // a job's first beat starts from the seed
        when(io.fire.get) {
          for (l <- 0 until lanes) s(l) := Mux(io.first.get, IntType.cast(q(l), outT, outT).asUInt, s(l)) + p(l)
        }
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
