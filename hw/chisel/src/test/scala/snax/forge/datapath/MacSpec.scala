package snax.forge.datapath

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds}
import snax.forge.elementwise.ElementwiseModel

/** `Mac` in its three forms against a Scala model written from CONTRACTS, "Accelerator RTL spec": per lane p =
  * cast_out(cast_op(a) · cast_op(b)); accumulating, s ← first ? p : s + p on fire, `out` s; plain, `out` = p + c at
  * outType, combinational, or with regOut a register taken on `en`. Random beats and enables at three sizes (lanes 2,
  * 4, 8), int64 and int32, signed and unsigned, a wider outType too; and the fixed cases of snax-gvsoc's vectors.
  * Seeded (D187), s ← first ? seed + p : s + p, the seed of out's type and lanes: random types, lanes 1 to 8, a seed
  * each cycle; and the refusal of a seed on the plain form.
  */
class MacSpec extends AnyFlatSpec with ForgeTester {

  private def p(a: BigInt, b: BigInt, t: IntType, op: IntType, out: IntType): BigInt =
    ElementwiseModel.cast(
      (ElementwiseModel.cast(a, t, op) * ElementwiseModel.cast(b, t, op)) & LaneBits.mask(op.width),
      op,
      out
    )

  behavior of "Mac's plain form"

  it should "give a * b + c, (3, 5, -9, 5) * (-1, 0, 2, -4) + (8, -9, -4, -3) = (5, -9, -22, -23)" in {
    test(new Mac(Seq("int64", "int64", "int64"), "int64", "int64", 4, false, false)) { dut =>
      for ((v, l) <- Seq(3, 5, -9, 5).zipWithIndex) LaneBits.poke(dut.io.a(l), v)
      for ((v, l) <- Seq(-1, 0, 2, -4).zipWithIndex) LaneBits.poke(dut.io.b(l), v)
      for ((v, l) <- Seq(8, -9, -4, -3).zipWithIndex) LaneBits.poke(dut.io.c.get(l), v)
      val got = dut.io.out.map(x => LaneBits.signed(LaneBits.peek(x), 64))
      assert(got == Seq(5, -9, -22, -23).map(BigInt(_)))
    }
  }

  behavior of "Mac's accumulating form"

  it should "wrap its int32 partials, W = 2, T = 8: (-2147478991, 4)" in {
    test(new Mac(Seq("int32", "int32"), "int32", "int32", 2, true, false)) { dut =>
      val a = Seq((46341, 1)) ++ Seq.fill(3)((3, 1)) ++ Seq.fill(3)((3, 0)) :+ ((6, 0))
      val b = Seq((46341, 1)) ++ Seq.fill(6)((1, 1)) :+ ((1, 0))
      for (k <- 0 until 8) {
        dut.io.fire.get.poke(true.B)
        dut.io.first.get.poke((k == 0).B)
        LaneBits.poke(dut.io.a(0), a(k)._1); LaneBits.poke(dut.io.a(1), a(k)._2)
        LaneBits.poke(dut.io.b(0), b(k)._1); LaneBits.poke(dut.io.b(1), b(k)._2)
        dut.clock.step()
      }
      dut.io.fire.get.poke(false.B)
      assert(dut.io.out.map(x => LaneBits.signed(LaneBits.peek(x), 32)) == Seq(BigInt(-2147478991), BigInt(4)))
    }
  }

  behavior of "a seeded Mac"

  it should "start each job from its seed: lanes (10, -3), T = 2, (2·3 + 1·4, 5·-1 + 2·2) + seed = (20, -4)" in {
    test(new Mac(Seq("int32", "int32"), "int32", "int32", 2, true, false, true)) { dut =>
      val beats = Seq((Seq(2, 5), Seq(3, -1)), (Seq(1, 2), Seq(4, 2)))
      for (((a, b), k) <- beats.zipWithIndex) {
        dut.io.fire.get.poke(true.B)
        dut.io.first.get.poke((k == 0).B)
        for (l <- 0 until 2) {
          LaneBits.poke(dut.io.a(l), a(l)); LaneBits.poke(dut.io.b(l), b(l))
          LaneBits.poke(dut.io.seed.get(l), Seq(10, -3)(l))
        }
        dut.clock.step()
      }
      dut.io.fire.get.poke(false.B)
      assert(dut.io.out.map(x => LaneBits.signed(LaneBits.peek(x), 32)) == Seq(BigInt(20), BigInt(-4)))
    }
  }

  it should "refuse a seed on the plain form" in {
    val e = intercept[Exception](
      circt.stage.ChiselStage.emitCHIRRTL(
        new Mac(Seq("int32", "int32", "int32"), "int32", "int32", 2, false, false, true)
      )
    )
    assert(e.getMessage.contains("seed is the accumulating form's start"), e.getMessage)
  }

  for (seed <- Seeds())
    it should s"give the model's sums from each job's seed (seed $seed)" in Seeds.run(seed) { r =>
      for (lanes <- Seq(1, 2, 4, 8)) {
        val names    = IntType.names
        val (ta, tb) = (IntType(r.pick(names)), IntType(r.pick(names)))
        val op       = IntType(r.pick(names.filter(n => IntType(n).width >= (ta.width max tb.width))))
        val out      = IntType(r.pick(names))
        val clue     = s"${ta.name} x ${tb.name} at ${op.name} -> ${out.name}, $lanes lanes"
        test(new Mac(Seq(ta.name, tb.name), op.name, out.name, lanes, true, false, true)) { dut =>
          var s = Seq.fill(lanes)(BigInt(0))
          for (c <- 0 until 32) {
            val a = Seq.fill(lanes)(r.word(ta.width))
            val b = Seq.fill(lanes)(r.word(tb.width))
            val q = Seq.fill(lanes)(r.word(out.width))
            val e = r.chance(0.7)
            val f = c == 0 || r.chance(0.3)
            for (l <- 0 until lanes) {
              LaneBits.poke(dut.io.a(l), a(l)); LaneBits.poke(dut.io.b(l), b(l))
              LaneBits.poke(dut.io.seed.get(l), q(l))
            }
            dut.io.fire.get.poke(e.B); dut.io.first.get.poke(f.B)
            for (l <- 0 until lanes) assert(LaneBits.peek(dut.io.out(l)) == s(l), s"$clue, cycle $c, lane $l")
            if (e) s = (0 until lanes).map { l =>
              val pl = ElementwiseModel.cast(
                (ElementwiseModel.cast(a(l), ta, op) * ElementwiseModel.cast(b(l), tb, op)) & LaneBits.mask(op.width),
                op,
                out
              )
              ((if (f) q(l) else s(l)) + pl) & LaneBits.mask(out.width)
            }
            dut.clock.step()
          }
        }
      }
    }

  behavior of "Mac under random beats and enables"

  for (seed <- Seeds())
    it should s"give the model's outputs in every form (seed $seed)" in Seeds.run(seed) { r =>
      for {
        form  <- Seq("acc", "plain", "reg")
        lanes <- Seq(2, 4, 8)
      } {
        val t    = IntType(r.pick(Seq("int64", "uint64", "int32", "uint32")))
        val out  = if (form == "acc" && t.width == 32) IntType(r.pick(Seq(t.name, "int64"))) else t
        val acc  = form == "acc"
        val clue = s"$form, ${t.name} -> ${out.name}, $lanes lanes"
        val ins  = if (acc) Seq(t.name, t.name) else Seq(t.name, t.name, out.name)
        test(new Mac(ins, t.name, out.name, lanes, acc, form == "reg")) { dut =>
          var s = Seq.fill(lanes)(BigInt(0))
          for (c <- 0 until 24) {
            val a  = Seq.fill(lanes)(r.word(t.width))
            val b  = Seq.fill(lanes)(r.word(t.width))
            val cc = Seq.fill(lanes)(r.word(out.width))
            val e  = r.chance(0.7)
            val f  = c == 0 || r.chance(0.3)
            for (l <- 0 until lanes) {
              LaneBits.poke(dut.io.a(l), a(l)); LaneBits.poke(dut.io.b(l), b(l))
              dut.io.c.foreach(x => LaneBits.poke(x(l), cc(l)))
            }
            dut.io.fire.foreach(_.poke(e.B)); dut.io.first.foreach(_.poke(f.B)); dut.io.en.foreach(_.poke(e.B))
            val ps   = (0 until lanes).map(l => p(a(l), b(l), t, t, out))
            val ys   = (0 until lanes).map(l => (ps(l) + cc(l)) & LaneBits.mask(out.width))
            val want = if (form == "plain") ys else s
            for (l <- 0 until lanes) assert(LaneBits.peek(dut.io.out(l)) == want(l), s"$clue, cycle $c, lane $l")
            if (e) s = form match {
              case "acc" => if (f) ps else s.zip(ps).map { case (x, y) => (x + y) & LaneBits.mask(out.width) }
              case _     => ys
            }
            dut.clock.step()
          }
        }
      }
    }
}
