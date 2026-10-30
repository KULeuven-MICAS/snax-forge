package snax.forge.ctrl

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, RunLine, SnaxGroupCases}

/** `GroupCtrl` alone (no datapath, as UnitCasesSpec drives shapes it has none for) through snax-gvsoc's 23 group cases
  * (`SnaxGroupCases`: dot 2-6 at T 2 and 4, J 1 and 2, backpressure, an input gap, JOBS 0), cycle for cycle: per cycle
  * the outer valids and the output's ready poked; each input's ready (its unit's firing, C§17's per-unit join), each
  * unit's fire and give, each buffer's occupancy after the cycle and busy compared; at the run's end its line from
  * `cls` per busy cycle and the head's firings. Each unit's T is fixed (`params.T`), so the start carries JOBS. Without
  * `SNAX_GVSOC_GROUP_CASES` the spec cancels.
  */
class GroupCasesSpec extends AnyFlatSpec with ForgeTester {

  private val Classes = Seq("busy", "stall_in", "stall_out", "flush")

  /** The case's group: a linked input after the unit's outer ones (to_input = its outer inputs, as `tb.cpp`). */
  def desc(c: SnaxGroupCases.Case): GroupDesc = {
    def at(name: String) = c.units.indexWhere(_.name == name)
    GroupDesc(
      "dot",
      c.units.map { u =>
        val fed = c.links.exists(_.to == u.name)
        UnitDesc(u.name, u.latency, u.ii, u.span, u.inputs, Some(u.t), if (fed) Seq(u.inputs) else Nil)
      },
      c.links.map(l => Link(at(l.from), at(l.to), c.units(at(l.to)).inputs, l.depth))
    )
  }

  private def run(c: SnaxGroupCases.Case): Unit =
    test(new GroupCtrl(desc(c), "case")) { dut =>
      var counts  = Map.empty[String, Int].withDefaultValue(0)
      var heads   = 0
      var runs    = Vector.empty[RunLine]
      var wasBusy = false
      for ((cyc, i) <- c.cycles.zipWithIndex) {
        val clue  = s"${c.name}: cycle $i"
        val offer = i == c.start._1
        dut.io.start.valid.poke(offer.B)
        dut.io.start.bits.jobs.poke(c.jobs.U)
        dut.io.start.bits.jobBeats.poke(1.U)
        if (offer) {
          val got = if (!high(dut.io.start.ready)) "refused" else if (c.jobs == 0) "ignored" else "taken"
          assert(got == c.start._2, s"$clue: start $got")
        }
        cyc.valids.zipWithIndex.foreach { case (v, k) => dut.io.in(k).valid.poke(v.B) }
        dut.io.out.ready.poke(cyc.ready.B)
        val busy  = high(dut.io.busy)
        assert(busy == cyc.busy, s"$clue: busy $busy")
        if (wasBusy && !busy) {
          runs :+= RunLine(
            heads,
            counts("busy"),
            counts("stall_in"),
            counts("stall_out"),
            counts("flush")
          )
          counts = Map.empty[String, Int].withDefaultValue(0); heads = 0
        }
        wasBusy = busy
        if (busy) {
          for (k   <- c.units.indices) {
            val e = dut.io.units(k)
            assert(high(e.fire) == cyc.fire(k), s"$clue: ${c.units(k).name} fire")
            assert(high(e.give) == cyc.give(k), s"$clue: ${c.units(k).name} give")
          }
          var at = 0
          for {
            (u, k) <- c.units.zipWithIndex
            _      <- 0 until u.inputs
          } {
            assert(high(dut.io.in(at).ready) == cyc.fire(k), s"$clue: input $at's ready, ${u.name}'s firing")
            at += 1
          }
          val occ = c.depths.filter(_ > 0).indices.map(i => dut.io.links(i).occupancy.peek().litValue.toInt)
          assert(occ == cyc.occupancy, s"$clue: occupancy $occ")
          counts += Classes(dut.io.cls.peek().litValue.toInt) -> (counts(Classes(dut.io.cls.peek().litValue.toInt)) + 1)
          if (cyc.fire.head) heads += 1
        }
        dut.clock.step()
      }
      assert(runs == c.runs, s"${c.name}: runs $runs, the case ${c.runs}")
    }

  behavior of "GroupCtrl on snax-gvsoc's group cases"

  SnaxGroupCases.all match {
    case None      =>
      it should "give every group case's cycles and runs" in
        cancel(s"${SnaxGroupCases.Env} is not set (snax-gvsoc's tests/snax/group/cases)")
    case Some(all) =>
      it should "find the 23 group cases" in assert(all.size == 23, s"${all.size} cases")
      for (c <- all) it should s"give ${c.name}'s cycles and runs" in run(c)
  }
}
