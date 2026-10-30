package snax.forge.ctrl

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Seeds}

/** `GroupCtrl`'s folded counters (GroupRule's rules: a T register only from a JOB_BEATS, R_k = c · R_s a wire after a
  * fixed T c, no count of the sink's pushes) against `GroupModel`, which counts every firing as SnaxGroupRule does, on
  * random groups: chains of 1 to 3 units, half in a shuffled unit order, each of L 0-2 and T fixed (`params.T`) or from
  * the start, each T from {1, 2, 3, 4, 8} (1, powers of two and not); links of depth 0-2; back-to-back runs of random
  * JOBS (every run's head firings at most 96), random input gaps and output stalls. Every cycle: busy, the done pin
  * (high in E + 1, E the model's last cycle), each unit's fire and give, and the class.
  */
class GroupFoldSpec extends AnyFlatSpec with ForgeTester {

  private val Groups = 24
  private val Cycles = 500
  private val Most   = 96 // head firings per run
  private val Ts     = Seq(1, 2, 3, 4, 8)
  private var ended  = 0  // runs that reached their last cycle, every seed

  behavior of "GroupCtrl's folded counters against SnaxGroupRule's model"

  for (seed <- Seeds())
    it should s"end every run in the model's cycle, with its pushes and classes (seed $seed)" in Seeds.run(seed) { r =>
      for (g <- 0 until Groups) {
        val n     = 1 + g % 3
        val chain = { // the units in chain order, head first
          val a = (0 until n).toArray
          if (r.chance(0.5)) for (i <- n - 1 to 1 by -1) {
            val j = r.between(0, i); val t = a(i); a(i) = a(j); a(j) = t
          }
          a.toSeq
        }
        var fixed: Seq[Option[Int]] = Nil
        while (fixed.isEmpty || fixed.flatten.product > Most / 2)
          fixed = Seq.fill(n)(if (r.chance(0.5)) Some(r.pick(Ts)) else None)
        val units = (0 until n).map { k =>
          val ins = if (k == chain.head) r.between(1, 2) else r.between(0, 1)
          UnitDesc(s"u$k", r.between(0, 2), 1, 1, ins, fixed(k), if (k == chain.head) Nil else Seq(ins))
        }
        val links = chain.zip(chain.tail).map { case (a, b) => Link(a, b, units(b).inputs, r.between(0, 2)) }
        val desc  = GroupDesc("fold", units, links)
        val sink  = chain.last
        val clue0 = s"seed $seed, group $g: ${units.mkString(", ")}, links ${links.mkString(",")}"
        val model = new GroupModel(units, links)
        val pv    = r.probability(); val pr = r.probability()
        test(new GroupCtrl(desc, "t")) { dut =>
          var lastEnd = false
          for (c <- 0 until Cycles) {
            val clue   = s"$clue0, cycle $c"
            // a start of random JOBS and T, offered now and then while idle, at most Most head firings
            var jobs   = 0
            var opent  = Seq.empty[Int]
            var ts     = Seq.empty[Long]
            while (jobs == 0 || ts.product * jobs > Most) {
              jobs  = r.pick(Seq(1, 2, 3, 4, 7))
              opent = desc.open.map(_ => r.pick(Ts))
              var next = 0
              ts = units.map(_.fixedT.map(_.toLong).getOrElse { next += 1; opent(next - 1).toLong })
            }
            val offer  = model.idle && r.chance(0.3)
            dut.io.start.valid.poke(offer.B)
            dut.io.start.bits.jobs.poke(jobs.U)
            dut.io.start.bits.jobBeats.poke(opent.headOption.getOrElse(1).U)
            dut.io.start.bits.beats.foreach(v => v.zip(opent.drop(1)).foreach { case (p, t) => p.poke(t.U) })
            if (offer) assert(model.start(jobs, ts) == "taken", s"$clue: the model's start")
            val valids = units.map(u => Seq.fill(u.inputs)(!r.chance(pv)))
            valids.flatten.zipWithIndex.foreach { case (v, k) => dut.io.in(k).valid.poke(v.B) }
            val ready  = !r.chance(pr)
            dut.io.out.ready.poke(ready.B)
            assert(high(dut.io.busy) == model.busy, s"$clue: busy")
            assert(high(dut.io.done) == lastEnd, s"$clue: the done pin (E + 1)")
            lastEnd = false
            model.step(valids.map(_.forall(identity)), ready).foreach { s =>
              for (k <- 0 until n) {
                val e = dut.io.units(k); val m = s.units(k); val uc = s"$clue, ${units(k).name}"
                assert(high(e.fire) == m.fire, s"$uc: fire")
                assert(high(e.give) == m.push, s"$uc: give")
              }
              assert(high(dut.io.out.valid) == (s.units(sink).push || s.units(sink).frozen), s"$clue: out.valid")
              assert(dut.io.cls.peek().litValue.toInt == s.cls, s"$clue: class")
              if (s.end) { ended += 1; lastEnd = true }
            }
            dut.clock.step()
          }
        }
      }
    }

  it should "have ended enough random runs" in {
    info(s"$ended runs ended")
    assert(ended >= 150 * Seeds().size, s"$ended runs ended")
  }
}
