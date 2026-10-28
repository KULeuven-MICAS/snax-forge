package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Sink, SnaxShellCases, Source}
import snax.forge.SnaxShellCases._

/** The shell cross-check (C§17): a group's CSRs in `<tag>_shell_wrapper` (`<tag>_csr`, the group's `<tag>_<group>_ctrl`
  * and its datapath) driven through snax-gvsoc's `SnaxShell` cases (`SnaxShellCases`, read from
  * `SNAX_GVSOC_SHELL_CASES`), which snax-gvsoc's `tb.cpp` checks `snax_shell.hpp`'s rule against.
  *
  * A case's `csrs` line and unit pick the wrapper: `START JOBS` with an ElementwiseLanes unit (L 0, two inputs) is
  * vecadd's (START 0, JOBS 1, STATUS 2, COUNTER 3); `START JOBS JOB_BEATS` with a fold (L 1, one input) is vsum's
  * (TreeAccumulate, one lane: START 0, JOBS 1, JOB_BEATS 2, STATUS 3, COUNTER 4).
  *
  * Each case drives the wrapper cycle by cycle: each input offers a beat from the cycles its valid column is 1 (a
  * Decoupled valid stays up until its beat is taken, so a case may not drop it before), the output is ready as the case
  * says, and the cycle's access is offered to the CSR port (`ShellDriver`; a held write offered again on each `.`
  * line). Every access's result is compared, a write's cycle it was taken in (the CSR manager answers a write when it
  * takes it) and a read's value, and every cycle the STATUS column against the group's pins: bit 0 its `busy` pin, bit
  * 1 done as the pins and the accepted starts give it (set from a cycle the `done` pin is high, the done pulse
  * registered, until the cycle after an accepted start after it). Without the variable the test is canceled.
  */
class ShellCasesSpec extends AnyFlatSpec with ForgeTester {

  private def fixture(c: Case): RtlFixtures.Fixture = {
    val name =
      if (c.csrs == Seq("START", "JOBS") && c.latency == 0 && c.ii == 1 && c.inputs == 2) "vecadd"
      else if (c.csrs == Seq("START", "JOBS", "JOB_BEATS") && c.latency == 1 && c.ii == 1 && c.inputs == 1) "vsum"
      else fail(s"${c.name}: no wrapper for csrs ${c.csrs.mkString(" ")}, L ${c.latency}, II ${c.ii}, ${c.inputs} in")
    RtlFixtures.all.find(_.name == name).get
  }

  behavior of "a group's CSRs in the shell wrapper against SnaxShell's cases"

  private def run(c: Case): Unit = {
    val f = fixture(c)
    test(new ShellHarness(f.spec)) { dut =>
      val g = f.spec.groups.head
      assert(g.window.map(_.name).take(c.csrs.size) == c.csrs, s"${c.name}: ${f.name}'s CSRs")
      var d: ShellDriver = null
      val sources = (0 until c.inputs).map(k => new Source(1 << 20, _ => BigInt(0), () => c.cycles(d.cycle).valids(k)))
      val sinks   = dut.io.writers.map(_ => new Sink(() => c.cycles(d.cycle).ready))
      d = new ShellDriver(dut, sources, sinks)

      var lastDone   = -1 // the last cycle the done pin was high
      var lastAccept = -1 // the last cycle a start was accepted
      // The write in flight: the cycle it was first offered, its access and the cycle the case answers it in
      var pending: Option[(Int, Write, BigInt)] = None
      for ((cyc, i) <- c.cycles.zipWithIndex) {
        val clue      = s"${c.name}, cycle $i"
        // The clue of an assert is built before the check, so it may not read a pending that is None
        def heldSince = pending.map(_._1).getOrElse(-1)
        for ((src, k) <- sources.zipWithIndex)
          assert(!src.valid || cyc.valids(k), s"$clue: input $k's valid drops before its beat is taken")
        val req  = cyc.access match {
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
        val busy = dut.io.bool(s"${g.name}_busy_o").peek().litToBoolean
        if (dut.io.bool(s"${g.name}_done_o").peek().litToBoolean) lastDone = i
        val done            = lastDone >= 0 && lastAccept < lastDone
        val status          = (if (busy) 1 else 0) | (if (done) 2 else 0)
        val (accepted, rsp) = d.tick(req)
        assert(status == cyc.status, s"$clue: STATUS from the pins $status, the case says ${cyc.status}")
        def start(w: Write) = w.offset == 0 && w.value.testBit(0)
        cyc.access match {
          case w: Write =>
            val expected = cyc.result.get
            if (accepted) {
              assert(expected == i, s"$clue: write taken in cycle $i, the case says $expected")
              if (start(w)) lastAccept = i
            } else pending = Some((i, w, expected))
          case Held =>
            val (offered, w, expected) = pending.get
            if (accepted) {
              assert(expected == i, s"$clue: the write of cycle $offered taken in cycle $i, the case says $expected")
              if (start(w)) lastAccept = i
              pending                  = None
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
      it should "give every SnaxShell case's results and STATUS" in
        cancel(s"${SnaxShellCases.Env} is not set (snax-gvsoc's tests/snax/shell/cases)")
    case Some(all) =>
      it should "find SnaxShell's cases" in assert(all.nonEmpty, s"no case in ${sys.env(SnaxShellCases.Env)}")
      for (c <- all) it should s"give ${c.name}'s results and STATUS" in run(c)
  }
}
