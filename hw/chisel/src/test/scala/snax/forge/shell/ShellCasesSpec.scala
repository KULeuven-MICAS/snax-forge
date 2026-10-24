package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Sink, SnaxShellCases, Source}
import snax.forge.SnaxShellCases._

/** The shell cross-check (C§17): vecadd's `<tag>_shell_wrapper` (`CsrManager` + `CtrlAdapter` +
  * ElementwiseTiledSpatial, 4 lanes; CSRs from 0: loopCount, start, busy, counter) driven through snax-gvsoc's
  * `SnaxShell` cases (`SnaxShellCases`, read from `SNAX_GVSOC_SHELL_CASES`), which snax-gvsoc's `tb.cpp` checks
  * `snax_shell.hpp`'s rule against.
  *
  * Each case drives the wrapper cycle by cycle: each input offers a beat from the cycles its valid column is 1 (a
  * Decoupled valid stays up until its beat is taken, so a case may not drop it before), the output is ready as the case
  * says, and the cycle's access is offered to the CSR port (`ShellDriver`; a held write offered again on each `.`
  * line). Every access's result is compared, a write's cycle it was taken in (the CSR manager answers a write when it
  * takes it) and a read's value, and every cycle the unit's busy, tapped from the wrapper, against the busy column.
  * Without the variable the test is canceled.
  */
class ShellCasesSpec extends AnyFlatSpec with ForgeTester {

  private val fixture = RtlFixtures.all.find(_.name == "vecadd").get

  behavior of "vecadd's shell wrapper against SnaxShell's cases"

  private def run(c: Case): Unit = {
    require(
      c.fields == Seq("loopCount") && c.config == Seq("count") && c.waitKind == "lockstep" && c.inputs == 2,
      s"${c.name}: this spec drives vecadd's wrapper (fields loopCount, config count, lockstep, 2 inputs)"
    )
    test(new ShellHarness(fixture.spec, tapBusy = true)) { dut =>
      var d: ShellDriver = null
      val sources = (0 until c.inputs).map(k => new Source(1 << 20, _ => BigInt(0), () => c.cycles(d.cycle).valids(k)))
      val sinks   = dut.io.writers.map(_ => new Sink(() => c.cycles(d.cycle).ready))
      d = new ShellDriver(dut, sources, sinks)

      // The write in flight: the cycle it was first offered, its access and the cycle the case answers it in
      var pending: Option[(Int, Write, BigInt)] = None
      for ((cyc, i) <- c.cycles.zipWithIndex) {
        val clue      = s"${c.name}, cycle $i"
        // The clue of an assert is built before the check, so it may not read a pending that is None
        def heldSince = pending.map(_._1).getOrElse(-1)
        for ((src, k) <- sources.zipWithIndex)
          assert(!src.valid || cyc.valids(k), s"$clue: input $k's valid drops before its beat is taken")
        val req             = cyc.access match {
          case Write(o, v) =>
            assert(pending.isEmpty, s"$clue: a write while the one of cycle $heldSince is held")
            Some((o, v, true))
          case Held        =>
            val (_, w, _) = pending.getOrElse(fail(s"$clue: `.` with no held write: it was answered by then"))
            Some((w.offset, w.value, true))
          case Read(o)     =>
            assert(pending.isEmpty, s"$clue: a read while the write of cycle $heldSince is held")
            Some((o, BigInt(0), false))
          case NoAccess    =>
            assert(pending.isEmpty, s"$clue: no access while the write of cycle $heldSince is held")
            None
        }
        val busy            = dut.busy.get.peek().litToBoolean
        val (accepted, rsp) = d.tick(req)
        assert(busy == cyc.busy, s"$clue: busy $busy, the case says ${cyc.busy}")
        cyc.access match {
          case w: Write =>
            val expected = cyc.result.get
            if (accepted) assert(expected == i, s"$clue: write taken in cycle $i, the case says $expected")
            else pending = Some((i, w, expected))
          case Held =>
            val (offered, _, expected) = pending.get
            if (accepted) {
              assert(expected == i, s"$clue: the write of cycle $offered taken in cycle $i, the case says $expected")
              pending = None
            }
          case Read(o) =>
            assert(accepted, s"$clue: read $o not taken")
            val got = rsp.getOrElse(fail(s"$clue: read $o taken but not answered"))
            assert(cyc.result.contains(got), s"$clue: read $o gives $got, the case says ${cyc.result.mkString}")
          case NoAccess =>
        }
      }
      pending.foreach { case (offered, _, expected) =>
        fail(s"${c.name}: the write of cycle $offered still held after the last cycle (the case: taken in $expected)")
      }
    }
  }

  SnaxShellCases.all match {
    case None      =>
      it should "give every SnaxShell case's results and busy" in
        cancel(s"${SnaxShellCases.Env} is not set (snax-gvsoc's tests/snax/shell/cases)")
    case Some(all) =>
      it should "find SnaxShell's cases" in assert(all.nonEmpty, s"no case in ${sys.env(SnaxShellCases.Env)}")
      for (c <- all) it should s"give ${c.name}'s results and busy" in run(c)
  }
}
