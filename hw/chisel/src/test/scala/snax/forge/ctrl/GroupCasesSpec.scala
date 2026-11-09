package snax.forge.ctrl

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, RunLines, SnaxGroupCases}

/** `GroupCtrl` alone (no datapath, as UnitCasesSpec drives shapes it has none for) through snax-gvsoc's 30 group cases
  * (`SnaxGroupCases`: dot 2-6 at T 2 and 4, J 1 and 2, dot 5 and 6 at W 1, backpressure, an input gap, JOBS 0, and the
  * fold's seed: two folds unseeded, the first or both seeded, a late seed on one fold and on a SerialAccumulate; their
  * data is checked by `tb.cpp`, and dot5_w1_j2's through the wrapper by ShellEndToEndSpec), cycle for cycle: per cycle
  * the outer valids and the output's ready poked; each input's ready (its unit's firing, C§17's per-unit join; a seed's
  * that firing on its job's first beat, the unit's firings counted mod its T), each unit's fire and give, each buffer's
  * occupancy after the cycle and busy compared; at the run's end its line from `cls` per busy cycle and the head's
  * firings. Each case runs twice (the rule does not tell the two apart, the control unit's counters do, D153): with
  * every unit's T fixed (`params.T`), the start carrying JOBS; and with every unit of T != 1 open, its T a JOB_BEATS of
  * the start, so dot 2's and dot 3's mul counts against or borrows from a sum whose T is a latch. Without
  * `SNAX_GVSOC_GROUP_CASES` the spec cancels.
  */
class GroupCasesSpec extends AnyFlatSpec with ForgeTester {

  /** The case's group: a linked input after the unit's outer ones (to_input = its outer inputs, as `tb.cpp`); with
    * `open`, every unit of T != 1 takes its T from the start.
    */
  def desc(c: SnaxGroupCases.Case, open: Boolean = false): GroupDesc = {
    def at(name: String) = c.units.indexWhere(_.name == name)
    GroupDesc(
      "dot",
      c.units.map { u =>
        val fed = c.links.exists(_.to == u.name)
        val t   = if (open && u.t != 1) None else Some(u.t)
        UnitDesc(u.name, u.latency, u.ii, u.span, u.inputs, t, if (fed) Seq(u.inputs) else Nil, u.seed)
      },
      c.links.map(l => Link(at(l.from), at(l.to), c.units(at(l.to)).inputs, l.depth))
    )
  }

  private def run(c: SnaxGroupCases.Case, open: Boolean): Unit = {
    val d     = desc(c, open)
    val beats = d.units.zip(c.units).collect { case (u, cu) if u.fixedT.isEmpty => cu.t }
    test(new GroupCtrl(d, "case")) { dut =>
      val runs  = new RunLines
      val fired = Array.fill(c.units.size)(0L) // each unit's firings so far: its job's first beat at 0 mod T
      for ((cyc, i) <- c.cycles.zipWithIndex) {
        val clue  = s"${c.name}${if (open) " (T from the start)" else ""}: cycle $i"
        val offer = i == c.start._1
        dut.io.start.valid.poke(offer.B)
        dut.io.start.bits.jobs.poke(c.jobs.U)
        dut.io.start.bits.beats.zip(if (beats.isEmpty) Seq(1) else beats).foreach { case (p, t) => p.poke(t.U) }
        if (offer) {
          val got = if (!high(dut.io.start.ready)) "refused" else if (c.jobs == 0) "ignored" else "taken"
          assert(got == c.start._2, s"$clue: start $got")
        }
        cyc.valids.zipWithIndex.foreach { case (v, k) => dut.io.in(k).valid.poke(v.B) }
        dut.io.out.ready.poke(cyc.ready.B)
        val busy  = high(dut.io.busy)
        assert(busy == cyc.busy, s"$clue: busy $busy")
        runs.cycle(busy, int(dut.io.cls), cyc.fire.head)
        if (busy) {
          for (k   <- c.units.indices) {
            val e = dut.io.units(k)
            assert(high(e.fire) == cyc.fire(k), s"$clue: ${c.units(k).name} fire")
            assert(high(e.give) == cyc.give(k), s"$clue: ${c.units(k).name} give")
          }
          var at = 0
          for {
            (u, k) <- c.units.zipWithIndex
            j      <- 0 until u.inputs
          } {
            val seed = u.seed      && j == u.inputs - 1
            val want = cyc.fire(k) && (!seed || fired(k) % u.t == 0)
            assert(high(dut.io.in(at).ready) == want, s"$clue: input $at's ready, ${u.name}'s firing")
            at += 1
          }
          for (k   <- c.units.indices if cyc.fire(k)) fired(k) += 1
          val occ = c.depths.filter(_ > 0).indices.map(i => dut.io.links(i).occupancy.peek().litValue.toInt)
          assert(occ == cyc.occupancy, s"$clue: occupancy $occ")
        }
        dut.clock.step()
      }
      val lines = runs.done
      assert(lines == c.runs, s"${c.name}: runs $lines, the case ${c.runs}")
    }
  }

  behavior of "GroupCtrl on snax-gvsoc's group cases"

  SnaxGroupCases.all match {
    case None      =>
      it should "give every group case's cycles and runs" in
        cancel(s"${SnaxGroupCases.Env} is not set (snax-gvsoc's tests/snax/group/cases)")
    case Some(all) =>
      it should "find the 30 group cases" in assert(all.size == 30, s"${all.size} cases")
      for (c <- all) {
        it should s"give ${c.name}'s cycles and runs" in run(c, open = false)
        it should s"give ${c.name}'s cycles and runs with its T from the start" in run(c, open = true)
      }
  }
}
