package snax.forge.ctrl

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Rand, Seeds}

/** `GroupCtrl` reading JOBS and each JOB_BEATS live from the csr's wires, against `GroupModel`, cycle by cycle: busy,
  * the done pulse (in E, and in a for an empty run), the done pin, each unit's fire and give, out.valid and the class.
  *
  *   - Random settings: chains of 1 to 3 units, half in a shuffled order, L 0-2, each T fixed or from a JOB_BEATS,
  *     links of depth 0-2, at job_width 3, 8 or 32; back-to-back starts whose JOBS and open Ts are drawn from {0, 1, 2,
  *     3, 5} (0 the empty run, 1 a job or a T of one beat), set on the wires only while idle and held through the run,
  *     as the csr holds their writes; random input gaps and output stalls.
  *   - The shared `+ 1` at its top value: job_width 8, JOBS 255 and every open T 255, so d + 1 = T and g + 1 = JOBS are
  *     reached at 255 = 2^8 − 1: a unit at L 0 (its done from g + 1), one at L 1 (its done from ¬left, g = JOBS), and a
  *     chain whose head at L 1 counts its results in mixed radix over its consumer's open T of 255 (the digit's + 1)
  *     then JOBS; each one run of 65025 head firings.
  */
class GroupRuleSettingsSpec extends AnyFlatSpec with ForgeTester {

  private val Groups = 18
  private val Cycles = 400
  private var ended  = 0 // random runs that reached their last cycle, every seed
  private var empty  = 0 // empty runs, every seed

  /** One test of `desc` at job_width `jw`: per cycle the settings `settings()` gives while the model is idle (JOBS and
    * the open Ts), a start when `offer(c)`, input valids and output ready from `pv` and `pr`; every cycle compared with
    * the model. Stops after `cycles` cycles, or once `runs` runs have ended. The runs that ended.
    */
  private def drive(
    desc:     GroupDesc,
    jw:       Int,
    cycles:   Int,
    r:        Rand,
    pv:       Double,
    pr:       Double,
    settings: () => (Long, Seq[Long]),
    offer:    Int => Boolean,
    runs:     Int = Int.MaxValue
  ): Int = {
    val units = desc.units
    val sink  = desc.output
    val model = new GroupModel(units, desc.links)
    var done  = 0
    test(new GroupCtrl(desc, "t", jw)) { dut =>
      var jobs    = 0L
      var opent   = desc.open.map(_ => 1L)
      var lastEnd = false
      var c       = 0
      while (c < cycles && done < runs) {
        val clue   = s"${units.mkString(", ")}, links ${desc.links.mkString(",")}, job_width $jw, cycle $c"
        if (model.idle) { val s = settings(); jobs = s._1; opent = s._2 }
        dut.io.settings.jobs.poke(jobs.U)
        dut.io.settings.beats.zip(opent).foreach { case (p, t) => p.poke(t.U) }
        val start  = model.idle && offer(c)
        dut.io.start.valid.poke(start.B)
        val answer = if (start) {
          var next = 0
          model.start(jobs, units.map(_.fixedT.map(_.toLong).getOrElse { next += 1; opent(next - 1) }))
        } else "none"
        if (answer == "ignored") empty += 1
        val valids = units.map(u => Seq.fill(u.inputs)(!r.chance(pv)))
        valids.flatten.zipWithIndex.foreach { case (v, k) => dut.io.in(k).valid.poke(v.B) }
        val ready  = !r.chance(pr)
        dut.io.out.ready.poke(ready.B)
        assert(high(dut.io.busy) == model.busy, s"$clue: busy")
        assert(high(dut.io.done) == lastEnd, s"$clue: the done pin (E + 1)")
        val step   = model.step(valids.map(_.forall(identity)), ready)
        assert(high(dut.io.pulse) == (answer == "ignored" || step.exists(_.end)), s"$clue: the done pulse")
        lastEnd = answer == "ignored"
        step.foreach { s =>
          for (k <- units.indices) {
            val e = dut.io.units(k); val m = s.units(k); val uc = s"$clue, ${units(k).name}"
            assert(high(e.fire) == m.fire, s"$uc: fire")
            assert(high(e.give) == m.push, s"$uc: give")
          }
          assert(high(dut.io.out.valid) == (s.units(sink).push || s.units(sink).frozen), s"$clue: out.valid")
          assert(int(dut.io.cls) == s.cls, s"$clue: class")
          if (s.end) { done += 1; lastEnd = true }
        }
        dut.clock.step()
        c += 1
      }
    }
    done
  }

  behavior of "GroupCtrl reading JOBS and JOB_BEATS live from the csr"

  for (seed <- Seeds())
    it should s"give the model's cycles at random JOBS and JOB_BEATS, 0 and 1 among them (seed $seed)" in Seeds.run(
      seed
    ) { r =>
      for (g <- 0 until Groups) {
        val n     = 1 + g % 3
        val chain = { // the units in chain order, head first
          val a = (0 until n).toArray
          if (r.chance(0.5)) for (i <- n - 1 to 1 by -1) {
            val j = r.between(0, i); val t = a(i); a(i) = a(j); a(j) = t
          }
          a.toSeq
        }
        val units = (0 until n).map { k =>
          val ins = if (k == chain.head) r.between(1, 2) else r.between(0, 1)
          val t   = if (r.chance(0.5)) Some(r.pick(Seq(1, 2, 3))) else None
          UnitDesc(s"u$k", r.between(0, 2), 1, 1, ins, t, if (k == chain.head) Nil else Seq(ins))
        }
        val links = chain.zip(chain.tail).map { case (a, b) => Link(a, b, units(b).inputs, r.between(0, 2)) }
        val desc  = GroupDesc("set", units, links)
        val jw    = r.pick(Seq(3, 8, 32)) // JOBS and T <= 5 fit 3 bits
        val vals  = Seq(0L, 1L, 1L, 2L, 3L, 5L)
        ended += drive(
          desc,
          jw,
          Cycles,
          r,
          r.probability(),
          r.probability(),
          () => (r.pick(vals), desc.open.map(_ => if (r.chance(0.1)) 0L else r.pick(vals.tail))),
          _ => r.chance(0.3)
        )
      }
    }

  it should "have ended enough random runs and empty ones" in {
    info(s"$ended runs ended, $empty empty")
    assert(ended >= 100 * Seeds().size && empty >= 20 * Seeds().size, s"$ended runs ended, $empty empty")
  }

  private val Top  = 255L // 2^8 − 1 at job_width 8
  private val tops = Seq(
    "a unit at L 0"                                -> GroupDesc("top", Seq(UnitDesc("u0", 0, 1, 1, 1, None)), Nil),
    "a unit at L 1"                                -> GroupDesc("top", Seq(UnitDesc("u0", 1, 1, 1, 1, None)), Nil),
    "a head at L 1 counting over its consumer's T" -> GroupDesc(
      "top",
      Seq(UnitDesc("u0", 1, 1, 1, 1, Some(1)), UnitDesc("u1", 0, 1, 1, 0, None, Seq(0))),
      Seq(Link(0, 1, 0, 0))
    )
  )

  for ((name, desc) <- tops)
    it should s"run JOBS 255 of T 255 at job_width 8 as the model, $name" in Seeds.run(Seeds().head) { r =>
      val runs = drive(desc, 8, 90000, r, 0.05, 0.05, () => (Top, desc.open.map(_ => Top)), _ == 2, runs = 1)
      assert(runs == 1, s"$name: the run did not end")
    }
}
