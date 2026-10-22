package snax.forge.elementwise

import chisel3._
import chisel3.util.DecoupledIO
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Rand, Seeds, Sink, Source}

/** The three elementwise blocks at 64 bits under random data, operand gaps (`a` and `b` each its own) and output
  * stalls, against `ElementwiseModel`: output t is the op on beat t of `a` and beat t of `b`, lane by lane.
  *
  * Per block, three variants: fixed add and fixed mul (as generated), and the selectable one over every op, drawn per
  * run. Per seed and variant, `Instances` blocks with `lanes` drawn from {1, 2, 4, 8} (`ElementwiseLoop`: 1), each
  * given `Runs` runs back to back of n in 1..64 beats, the gaps and the stall drawn per run. Checked every cycle: `a`,
  * `b` and `out` fire together or not at all (the join: a lone operand waits), and each output equals the model's next.
  * The counted blocks (`ElementwiseTiledSpatial`, `ElementwiseLoop`) are started with loopCount n and their sources
  * hold n + `Extra` beats: busy until the n-th firing and low the cycle after, so the trip count advances only on a
  * fire of both, and no beat is taken after the n-th.
  */
class ElementwiseRandomSpec extends AnyFlatSpec with ForgeTester {

  private val Width     = 64
  private val Mask      = (BigInt(1) << Width) - 1
  private val Instances = 2
  private val Runs      = 4
  private val Extra     = 3

  /** A block's ports; `counted` its loopCount, start and busy when it has a trip count. */
  private case class Ports(
    clock:   Clock,
    opSel:   UInt,
    a:       DecoupledIO[Data],
    b:       DecoupledIO[Data],
    out:     DecoupledIO[Data],
    counted: Option[(UInt, Bool, Bool)]
  )

  private def lanesOf(d: Data): Seq[UInt] =
    d match {
      case v: Vec[_] => v.map(_.asInstanceOf[UInt])
      case u: UInt   => Seq(u)
      case other => throw new IllegalArgumentException(s"no lanes in $other")
    }

  private def poke(d: Data, v: BigInt): Unit =
    lanesOf(d).zipWithIndex.foreach { case (u, l) => u.poke(((v >> (Width * l)) & Mask).U) }

  private def peek(d: Data): BigInt =
    lanesOf(d).zipWithIndex.map { case (u, l) => u.peek().litValue << (Width * l) }.sum

  private def pack(lanes: Seq[BigInt]): BigInt = lanes.zipWithIndex.map { case (v, l) => v << (Width * l) }.sum

  private def run(p: Ports, lanes: Int, ops: Seq[Int], r: Rand, run: Int): Unit = {
    val n        = r.between(1, 64)
    val op       = r.pick(ops)
    val beats    = n + (if (p.counted.isDefined) Extra else 0)
    val as       = Seq.fill(beats)(Seq.fill(lanes)(r.word(Width)))
    val bs       = Seq.fill(beats)(Seq.fill(lanes)(r.word(Width)))
    val (ga, gb) = (r.probability(), r.probability())
    val stall    = r.probability()
    val a        = Source.random(as.map(pack), ga, r)
    val b        = Source.random(bs.map(pack), gb, r)
    val out      = Sink.random(stall, r)
    val want     = (0 until n).map(t => pack(as(t).zip(bs(t)).map { case (x, y) => ElementwiseModel(op, x, y, Width) }))
    val clue     = s"lanes $lanes, run $run: ${ElementwiseOp.name(op)}, n $n, gaps $ga $gb, stall $stall"

    def drive(): Unit = {
      a.cycle()
      b.cycle()
      p.a.valid.poke(a.valid.B)
      p.b.valid.poke(b.valid.B)
      poke(p.a.bits, a.bits)
      poke(p.b.bits, b.bits)
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
          val fo = out.fired(p.out.valid.peek().litToBoolean, peek(p.out.bits))
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

  suite("ElementwiseTiledSpatial") { (ops, r) =>
    val lanes = r.pick(Seq(1, 2, 4, 8))
    test(new ElementwiseTiledSpatial(Width, lanes, 32, ops)) { dut =>
      val p = Ports(
        dut.clock,
        dut.io.opSel,
        dut.io.a,
        dut.io.b,
        dut.io.out,
        Some((dut.io.loopCount, dut.io.start, dut.io.busy))
      )
      dut.io.start.poke(false.B)
      for (i <- 0 until Runs) run(p, lanes, ops, r, i)
    }
  }

  suite("ElementwiseLoop") { (ops, r) =>
    test(new ElementwiseLoop(Width, 32, ops)) { dut =>
      val p = Ports(
        dut.clock,
        dut.io.opSel,
        dut.io.a,
        dut.io.b,
        dut.io.out,
        Some((dut.io.loopCount, dut.io.start, dut.io.busy))
      )
      dut.io.start.poke(false.B)
      for (i <- 0 until Runs) run(p, 1, ops, r, i)
    }
  }

  suite("ElementwiseSpatial") { (ops, r) =>
    val lanes = r.pick(Seq(1, 2, 4, 8))
    test(new ElementwiseSpatial(Width, lanes, ops)) { dut =>
      val p = Ports(dut.clock, dut.io.opSel, dut.io.a, dut.io.b, dut.io.out, None)
      for (i <- 0 until Runs) run(p, lanes, ops, r, i)
    }
  }
}
