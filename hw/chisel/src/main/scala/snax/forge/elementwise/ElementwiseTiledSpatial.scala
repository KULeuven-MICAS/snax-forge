package snax.forge.elementwise

import chisel3._
import chisel3.util._

import snax.forge.IntType

/** `lanes` lanes, fed `loopCount` times -- the hardware image of the `tiled_spatial` variant.
  *
  * W = tile, T = ceil(N/tile): the interior of the elementwise family, and the only member of it that is a plausible
  * design rather than a limiting case. Structurally it is `ElementwiseSpatial` plus the trip counter from
  * `ElementwiseLoop`, and the three are kept as separate modules so each can be brought up and read on its own.
  *
  * ==No accumulator==
  *
  * A tile boundary is not a reduction. Each pass reads its own operands and writes its own results, and nothing is
  * carried between passes, so there is no accumulation register and none is wanted. Cross-beat state appears when a
  * reduction does -- see `samples.Accumulator` -- and that is a different pattern with a different hazard profile: a
  * reduction is a chaining barrier, an elementwise tile is not.
  *
  * ==loopCount is given, not derived==
  *
  * The unit counts passes; it does not compute how many passes a problem needs. If N elements are to be processed at
  * `lanes` per beat, the driver programs ceil(N / lanes) and the arithmetic happens where N is known. Putting a divider
  * here would duplicate a calculation SNAX-LOWER already makes for the accelerator's `n`, and would do it in the one
  * place with no access to N.
  *
  * ==Types (D143)==
  *
  * Each lane runs NumPy's arithmetic: `a` and `b` are lanes of their own types, each extended to `opType` (the
  * operands' result type, chosen by Python) by its own signedness, the op runs at `opType` (Min and Max comparing by
  * its signedness), and the result is extended or wrapped into `outType`. `ElementwiseAlu` keeps its `UInt` interface;
  * the casts sit at its boundary. Each type is fixed at elaboration.
  *
  * ==Handshake==
  *
  * One ready/valid pair per port, covering all `lanes` elements, joined as in `ElementwiseLoop`: a firing takes a beat
  * from `a` and `b` and gives one to `out` in the same cycle, an operand that arrives alone waits, and `valid` does not
  * depend on `ready`. The trip counter advances on each firing.
  *
  * @param inTypes
  *   `a`'s and `b`'s types, NumPy's names ("int32", "uint8"), from the BRM ports' dtypes
  * @param opType
  *   the type the op runs at: NumPy's result type of `inTypes`, at least as wide as each
  * @param outType
  *   `out`'s type, from the BRM port's dtype
  * @param lanes
  *   elements per beat, i.e. the spatial bound of the map the BRM replaces (W)
  * @param loopCountWidth
  *   bits of trip counter; must hold the largest T the driver will program
  * @param supportedOps
  *   encodings this instance can perform; a single entry emits fixed-function hardware and leaves `opSel` unused
  */
class ElementwiseTiledSpatial(
  val inTypes:        Seq[String] = Seq("int32", "int32"),
  val opType:         String      = "int32",
  val outType:        String      = "int32",
  val lanes:          Int         = 4,
  val loopCountWidth: Int         = 32,
  val supportedOps:   Seq[Int]    = ElementwiseOp.all
) extends Module {
  require(inTypes.size == 2, s"inTypes must name a and b, got ${inTypes.mkString(", ")}")
  require(lanes > 0, s"lanes must be positive, got $lanes")
  require(loopCountWidth > 0, s"loopCountWidth must be positive, got $loopCountWidth")

  private val ins = inTypes.map(IntType(_))
  private val op  = IntType(opType)
  private val res = IntType(outType)
  require(
    ins.forall(_.width <= op.width),
    s"opType $opType must be at least as wide as each input (${inTypes.mkString(", ")})"
  )

  override def desiredName: String =
    s"ElementwiseTiledSpatial_${ins.map(_.tag).mkString("x")}_${op.tag}_${res.tag}" +
      s"_n${lanes}_t${loopCountWidth}_${ElementwiseOp.tag(supportedOps)}"

  val io = IO(new Bundle {
    val opSel     = Input(UInt(ElementwiseOp.width.W))
    val loopCount = Input(UInt(loopCountWidth.W))
    val start     = Input(Bool())
    val busy      = Output(Bool())
    val done      = Output(Bool())

    val a   = Flipped(Decoupled(Vec(lanes, ins(0).lane)))
    val b   = Flipped(Decoupled(Vec(lanes, ins(1).lane)))
    val out = Decoupled(Vec(lanes, res.lane))
  })

  private val busy   = RegInit(false.B)
  private val count  = RegInit(0.U(loopCountWidth.W))
  private val target = RegInit(0.U(loopCountWidth.W))

  // Per lane: each operand extended to opType by its own signedness, the op at opType, the result into outType.
  private val alus = Seq.fill(lanes)(Module(new ElementwiseAlu(op.width, supportedOps, op.signed)))
  alus.zipWithIndex.foreach { case (alu, i) =>
    alu.io.opSel   := io.opSel
    alu.io.a       := IntType.cast(io.a.bits(i), ins(0), op).asUInt
    alu.io.b       := IntType.cast(io.b.bits(i), ins(1), op).asUInt
    io.out.bits(i) := IntType.cast(alu.io.out, op, res)
  }

  io.a.ready   := busy && io.b.valid && io.out.ready
  io.b.ready   := busy && io.a.valid && io.out.ready
  io.out.valid := busy && io.a.valid && io.b.valid

  // One tile per firing.
  private val advance = io.a.fire && io.b.fire
  private val last    = advance   && (count + 1.U) === target

  when(!busy) {
    when(io.start && io.loopCount =/= 0.U) {
      busy   := true.B
      target := io.loopCount
      count  := 0.U
    }
  }.otherwise {
    when(advance) {
      count := count + 1.U
      when(last) {
        busy  := false.B
        count := 0.U
      }
    }
  }

  io.busy := busy
  io.done := last
}
