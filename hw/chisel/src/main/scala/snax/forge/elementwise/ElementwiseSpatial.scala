package snax.forge.elementwise

import chisel3._
import chisel3.util._

/** `lanes` lanes, fed once -- the hardware image of the `spatial` variant.
  *
  * W = N, T = 1 in the descriptor's shape section: the fully spatial end of the family. The whole iteration space is
  * consumed in a single beat, so there is no trip count and no sequencing state. The registers `Module` provides go
  * unused here; the class extends `Module` regardless, so every unit in this package has the same skeleton.
  *
  * ==One handshake per port, not per lane==
  *
  * `Decoupled(Vec(lanes, ...))` puts a single ready/valid pair around the whole vector, which is the shape asked for:
  * all `lanes` elements of `A` arrive together under one `a.valid`, and likewise for `B` and `C`. `Vec(lanes,
  * Decoupled(...))` would give per-lane handshakes and a far more complicated control path, and would model a producer
  * that can deliver lanes independently -- which a streamer feeding a tile does not do.
  *
  * ==The constraint this module makes concrete==
  *
  * `lanes` is an elaboration-time parameter: it decides how much hardware exists. That is why a descriptor carrying
  * `lanes: N` with N still symbolic is not buildable and must be rejected at emit time -- there is no value to pass
  * here. Compare `loopCount` on the other two variants, which is a runtime port. The `bounded` flag in the descriptor
  * marks exactly this distinction.
  *
  * ==Handshake==
  *
  * `ready` passes straight through from the consumer and `out.valid` asserts when both operand transfers complete. See
  * `ElementwiseLoop` for the two properties this trades away: operands must arrive in lockstep or a beat is silently
  * dropped, and `valid` depends on `ready`, so a consumer whose `ready` is a function of `valid` deadlocks against it.
  *
  * @param dataWidth
  *   bits per element, from the DaCe dtype
  * @param lanes
  *   elements per beat -- for this variant, the entire iteration space
  * @param supportedOps
  *   encodings this instance can perform; a single entry emits fixed-function hardware and leaves `opSel` unused
  * @param signed
  *   interpret operands as two's complement for Min and Max
  */
class ElementwiseSpatial(
  val dataWidth:    Int      = 32,
  val lanes:        Int      = 4,
  val supportedOps: Seq[Int] = ElementwiseOp.all,
  val signed:       Boolean  = false
) extends Module {
  require(dataWidth > 0, s"dataWidth must be positive, got $dataWidth")
  require(lanes > 0, s"lanes must be positive, got $lanes")

  override def desiredName: String = s"ElementwiseSpatial_w${dataWidth}_n${lanes}_${ElementwiseOp.tag(supportedOps)}"

  val io = IO(new Bundle {
    val opSel = Input(UInt(ElementwiseOp.width.W))

    val a   = Flipped(Decoupled(Vec(lanes, UInt(dataWidth.W))))
    val b   = Flipped(Decoupled(Vec(lanes, UInt(dataWidth.W))))
    val out = Decoupled(Vec(lanes, UInt(dataWidth.W)))
  })

  // One ALU instance per lane, all driven from the same opSel. Sharing the
  // select is not an optimisation: the lanes are replications of a single SDFG
  // tasklet, so they cannot differ.
  private val alus = Seq.fill(lanes)(Module(new ElementwiseAlu(dataWidth, supportedOps, signed)))
  alus.zipWithIndex.foreach { case (alu, i) =>
    alu.io.opSel := io.opSel
    alu.io.a     := io.a.bits(i)
    alu.io.b     := io.b.bits(i)
  }
  io.out.bits := VecInit(alus.map(_.io.out))

  io.a.ready := io.out.ready
  io.b.ready := io.out.ready

  io.out.valid := io.a.valid && io.a.ready && io.b.valid && io.b.ready
}
