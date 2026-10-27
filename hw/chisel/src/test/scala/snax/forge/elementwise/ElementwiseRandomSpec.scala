package snax.forge.elementwise

import chisel3._
import chisel3.util.DecoupledIO
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Rand, Seeds, Sink, Source}

/** The three elementwise blocks under random data, operand gaps (`a` and `b` each its own) and output stalls, against
  * `ElementwiseModel`: output t is the op on beat t of `a` and beat t of `b`, lane by lane, by NumPy's rule (D143) at
  * the block's types.
  *
  * Per block, three variants at 64 bits (`ElementwiseTiledSpatial` int64, the other two `UInt`, uint64): fixed add and
  * fixed mul (as generated), and the selectable one over every op, drawn per run. Then `ElementwiseTiledSpatial` at the
  * narrow and mixed types of `Typed`, selectable, its runs going through every op. Per seed and variant, `Instances`
  * blocks with `lanes` drawn from {1, 2, 4, 8} (`ElementwiseLoop`: 1), each given `Runs` runs back to back (`Typed`:
  * one per op) of n in 1..64 beats, the gaps and the stall drawn per run. Checked every cycle: `a`, `b` and `out` fire
  * together or not at all (the join: a lone operand waits), and each output equals the model's next. The counted blocks
  * (`ElementwiseTiledSpatial`, `ElementwiseLoop`) are started with loopCount n and their sources hold n + `Extra`
  * beats: busy until the n-th firing and low the cycle after, so the trip count advances only on a fire of both, and no
  * beat is taken after the n-th.
  */
class ElementwiseRandomSpec extends AnyFlatSpec with ForgeTester {

  private val Instances = 2
  private val Runs      = 4
  private val Extra     = 3

  /** The types of `a`, `b`, the op and `out`. */
  private case class Types(a: IntType, b: IntType, op: IntType, out: IntType) {
    override def toString: String = s"${a.name} ${b.name} -> ${op.name} -> ${out.name}"
  }

  private object Types {

    /** `a` and `b` of their types, the op at NumPy's result type, `out` of `out`. */
    def apply(a: String, b: String, out: String): Types = {
      val (ta, tb) = (IntType(a), IntType(b))
      Types(ta, tb, ElementwiseModel.resultType(ta, tb), IntType(out))
    }
  }

  private val Int64  = Types("int64", "int64", "int64")
  private val UInt64 = Types("uint64", "uint64", "uint64")

  /** `ElementwiseTiledSpatial`'s narrow and mixed types: int32 and int8, signed and unsigned; int32 into int64; int8
    * with uint8 into int16 (the op at int16).
    */
  private val Typed: Seq[Types] = Seq(
    Types("int32", "int32", "int32"),
    Types("uint32", "uint32", "uint32"),
    Types("int8", "int8", "int8"),
    Types("uint8", "uint8", "uint8"),
    Types("int32", "int32", "int64"),
    Types("int8", "uint8", "int16")
  )

  /** A block's ports; `counted` its loopCount, start and busy when it has a trip count. */
  private case class Ports(
    clock:   Clock,
    opSel:   UInt,
    a:       DecoupledIO[Data],
    b:       DecoupledIO[Data],
    out:     DecoupledIO[Data],
    counted: Option[(UInt, Bool, Bool)]
  )

  private def run(p: Ports, lanes: Int, op: Int, t: Types, r: Rand, run: Int): Unit = {
    val n        = r.between(1, 64)
    val beats    = n + (if (p.counted.isDefined) Extra else 0)
    val as       = Seq.fill(beats)(Seq.fill(lanes)(r.word(t.a.width)))
    val bs       = Seq.fill(beats)(Seq.fill(lanes)(r.word(t.b.width)))
    val (ga, gb) = (r.probability(), r.probability())
    val stall    = r.probability()
    val a        = Source.random(as.map(LaneBits.pack(_, t.a.width)), ga, r)
    val b        = Source.random(bs.map(LaneBits.pack(_, t.b.width)), gb, r)
    val out      = Sink.random(stall, r)
    val want     = (0 until n).map { k =>
      val lanes = as(k).zip(bs(k)).map { case (x, y) => ElementwiseModel.numpy(op, x, y, t.a, t.b, t.op, t.out) }
      LaneBits.pack(lanes, t.out.width)
    }
    val clue     = s"lanes $lanes, $t, run $run: ${ElementwiseOp.name(op)}, n $n, gaps $ga $gb, stall $stall"

    def drive(): Unit = {
      a.cycle()
      b.cycle()
      p.a.valid.poke(a.valid.B)
      p.b.valid.poke(b.valid.B)
      LaneBits.pokeBeat(p.a.bits, a.bits)
      LaneBits.pokeBeat(p.b.bits, b.bits)
      p.out.ready.poke(out.cycle().B)
    }

    p.opSel.poke(op.U)
    p.counted.foreach { case (loopCount, start, busy) =>
      drive()
      assert(!busy.peek().litToBoolean, s"$clue: busy at start")
      start.poke(true.B)
      loopCount.poke(n.U)
      a.fired(p.a.ready.peek().litToBoolean)
      b.fired(p.b.ready.peek().litToBoolean)
      p.clock.step()
      start.poke(false.B)
    }

    var fires = 0
    var cycle = 1
    while (fires < n || p.counted.isDefined && cycle > 0) {
      drive()
      p.counted.map(_._3.peek().litToBoolean) match {
        case Some(busy) if fires == n =>
          assert(!busy, s"$clue, cycle $cycle: busy after the n-th firing")
          cycle = -1
        case Some(busy) if !busy      => fail(s"$clue, cycle $cycle: not busy after $fires firings")
        case _                        =>
          assert(cycle < 200 * n, s"$clue: no end after $cycle cycles ($fires firings)")
          val fo = out.fired(p.out.valid.peek().litToBoolean, LaneBits.peekBeat(p.out.bits))
          val fa = a.fired(p.a.ready.peek().litToBoolean)
          val fb = b.fired(p.b.ready.peek().litToBoolean)
          assert(fa == fb && fb == fo, s"$clue, cycle $cycle: a fired $fa, b $fb, out $fo")
          if (fo) {
            assert(
              out.got(fires) == want(fires),
              s"$clue, cycle $cycle: output $fires ${out.got(fires)}, the model ${want(fires)}"
            )
            fires += 1
          }
          p.clock.step()
          cycle += 1
      }
    }
    assert(a.taken == n                     && b.taken == n, s"$clue: ${a.taken} and ${b.taken} beats taken")
  }

  private val variants: Seq[(String, Seq[Int])] =
    Seq("add" -> Seq(ElementwiseOp.Add), "mul" -> Seq(ElementwiseOp.Mul), "every op" -> ElementwiseOp.all)

  private def suite(name: String)(body: (Seq[Int], Rand) => Unit): Unit = {
    behavior of s"$name under random data and stalls"
    for {
      (variant, ops) <- variants
      seed           <- Seeds()
    }
      it should s"give the model's outputs and fire a, b and out together, $variant (seed $seed)" in
        Seeds.run(seed)(r => for (_ <- 0 until Instances) body(ops, r))
  }

  private def tiled(t: Types, lanes: Int, ops: Seq[Int]): ElementwiseTiledSpatial =
    new ElementwiseTiledSpatial(Seq(t.a.name, t.b.name), t.op.name, t.out.name, lanes, 32, ops)

  private def tiledPorts(dut: ElementwiseTiledSpatial): Ports =
    Ports(dut.clock, dut.io.opSel, dut.io.a, dut.io.b, dut.io.out, Some((dut.io.loopCount, dut.io.start, dut.io.busy)))

  suite("ElementwiseTiledSpatial") { (ops, r) =>
    val lanes = r.pick(Seq(1, 2, 4, 8))
    test(tiled(Int64, lanes, ops)) { dut =>
      dut.io.start.poke(false.B)
      for (i <- 0 until Runs) run(tiledPorts(dut), lanes, r.pick(ops), Int64, r, i)
    }
  }

  behavior of "ElementwiseTiledSpatial at narrow and mixed types under random data and stalls"

  for {
    t    <- Typed
    seed <- Seeds()
  }
    it should s"give NumPy's outputs for every op, $t (seed $seed)" in
      Seeds.run(seed) { r =>
        for (_ <- 0 until Instances) {
          val lanes = r.pick(Seq(1, 2, 4, 8))
          test(tiled(t, lanes, ElementwiseOp.all)) { dut =>
            dut.io.start.poke(false.B)
            for ((op, i) <- ElementwiseOp.all.zipWithIndex) run(tiledPorts(dut), lanes, op, t, r, i)
          }
        }
      }

  suite("ElementwiseLoop") { (ops, r) =>
    test(new ElementwiseLoop(64, 32, ops)) { dut =>
      val p = Ports(
        dut.clock,
        dut.io.opSel,
        dut.io.a,
        dut.io.b,
        dut.io.out,
        Some((dut.io.loopCount, dut.io.start, dut.io.busy))
      )
      dut.io.start.poke(false.B)
      for (i <- 0 until Runs) run(p, 1, r.pick(ops), UInt64, r, i)
    }
  }

  suite("ElementwiseSpatial") { (ops, r) =>
    val lanes = r.pick(Seq(1, 2, 4, 8))
    test(new ElementwiseSpatial(64, lanes, ops)) { dut =>
      val p = Ports(dut.clock, dut.io.opSel, dut.io.a, dut.io.b, dut.io.out, None)
      for (i <- 0 until Runs) run(p, lanes, r.pick(ops), UInt64, r, i)
    }
  }
}
