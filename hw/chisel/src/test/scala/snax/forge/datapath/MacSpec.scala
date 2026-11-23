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
  * each cycle; and the refusal of a seed on the plain form. A bank (D191), R rows of W sums: on fire row `row` ← first
  * ? p : row + p, `out` row `drain`, with random pins at R = 2, 4 and 16; and jobs as the control unit runs them, ROWS
  * ≤ R rows used, `row` the beat mod ROWS and `drain` the row before it, each of the job's last ROWS results its row's
  * sum; and the refusal of a bank on a seeded or plain Mac.
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

  behavior of "a banked Mac"

  /** The model of D191: R rows of `lanes` sums at `out`'s type. */
  private class Bank(rows: Int, lanes: Int, out: IntType) {
    var s:                                               Seq[Seq[BigInt]] = Seq.fill(rows)(Seq.fill(lanes)(BigInt(0)))
    def fire(row: Int, first: Boolean, ps: Seq[BigInt]): Unit             =
      s = s.updated(row, if (first) ps else s(row).zip(ps).map { case (x, y) => (x + y) & LaneBits.mask(out.width) })
  }

  private def poke(dut: Mac, a: Seq[BigInt], b: Seq[BigInt]): Unit =
    for (l <- a.indices) { LaneBits.poke(dut.io.a(l), a(l)); LaneBits.poke(dut.io.b(l), b(l)) }

  for (seed <- Seeds())
    it should s"give the model's rows under random pins at R 2, 4 and 16 (seed $seed)" in Seeds.run(seed) { r =>
      for (rows <- Seq(2, 4, 16)) {
        val names = IntType.names
        val t     = IntType(r.pick(names))
        val out   = IntType(r.pick(names.filter(n => IntType(n).width >= t.width)))
        val lanes = r.pick(Seq(1, 2, 4))
        val clue  = s"R $rows, ${t.name} -> ${out.name}, $lanes lanes"
        test(new Mac(Seq(t.name, t.name), out.name, out.name, lanes, true, false, false, rows)) { dut =>
          val m = new Bank(rows, lanes, out)
          for (c <- 0 until 4 * rows + 16) {
            val (a, b)       = (Seq.fill(lanes)(r.word(t.width)), Seq.fill(lanes)(r.word(t.width)))
            val (row, drain) = (r.pick(0 until rows), r.pick(0 until rows))
            val (e, f)       = (r.chance(0.7), r.chance(0.3))
            poke(dut, a, b)
            dut.io.fire.get.poke(e.B); dut.io.first.get.poke(f.B)
            dut.io.row.get.poke(row.U); dut.io.drain.get.poke(drain.U)
            for (l <- 0 until lanes)
              assert(LaneBits.peek(dut.io.out(l)) == m.s(drain)(l), s"$clue, cycle $c, row $drain, lane $l")
            if (e) m.fire(row, f, (0 until lanes).map(l => p(a(l), b(l), t, out, out)))
            dut.clock.step()
          }
        }
      }
    }

  for (seed <- Seeds())
    it should s"give each row's sum on a job's last ROWS firings, ROWS up to R (seed $seed)" in Seeds.run(seed) { r =>
      for (rows <- Seq(2, 4, 16)) {
        val used            = 1 + r.pick(0 until rows) // ROWS
        val (t, out, lanes) = (IntType("int8"), IntType("int32"), 4)
        val m               = 1 + r.pick(0 until 4)    // a job's passes over the rows: T = m · ROWS
        val clue            = s"R $rows, ROWS $used, T ${m * used}"
        test(new Mac(Seq(t.name, t.name), out.name, out.name, lanes, true, false, false, rows)) { dut =>
          for (job <- 0 until 2) {
            val beats = Seq.fill(m * used)((Seq.fill(lanes)(r.word(8)), Seq.fill(lanes)(r.word(8))))
            for (((a, b), d) <- beats.zipWithIndex) {
              val row = d % used
              poke(dut, a, b)
              dut.io.fire.get.poke(true.B); dut.io.first.get.poke((d < used).B)
              dut.io.row.get.poke(row.U); dut.io.drain.get.poke((if (row == 0) used - 1 else row - 1).U)
              if (d > m * used - used) { // the result of the firing before, its row's sum
                val prev = row - 1
                val want = (0 until lanes).map { l =>
                  beats.zipWithIndex.collect {
                    case ((x, y), e) if e % used == prev => p(x(l), y(l), t, out, out)
                  }.sum & LaneBits.mask(out.width)
                }
                for (l <- 0 until lanes)
                  assert(LaneBits.peek(dut.io.out(l)) == want(l), s"$clue, job $job, beat $d, row $prev, lane $l")
              }
              dut.clock.step()
            }
            // the job's last row, a cycle after its last firing
            dut.io.fire.get.poke(false.B); dut.io.row.get.poke(0.U); dut.io.drain.get.poke((used - 1).U)
            val want = (0 until lanes).map { l =>
              beats.zipWithIndex.collect {
                case ((x, y), e) if e % used == used - 1 => p(x(l), y(l), t, out, out)
              }.sum & LaneBits.mask(out.width)
            }
            for (l <- 0 until lanes) assert(LaneBits.peek(dut.io.out(l)) == want(l), s"$clue, job $job's last row")
            dut.clock.step()
          }
        }
      }
    }

  it should "refuse a bank on a seeded or a plain Mac" in {
    for (
      make <- Seq(
        () => new Mac(Seq("int32", "int32"), "int32", "int32", 2, true, false, true, 4),
        () => new Mac(Seq("int32", "int32", "int32"), "int32", "int32", 2, false, false, false, 4)
      )
    ) {
      val e = intercept[Exception](circt.stage.ChiselStage.emitCHIRRTL(make()))
      assert(e.getMessage.contains("a bank (D191) is an unseeded accumulating Mac's"), e.getMessage)
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
