package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** The shell's rules on `CsrManager` + `CtrlAdapter` + a unit, through vecadd's `<tag>_shell_wrapper`
  * (ElementwiseTiledSpatial, 4 lanes; CSRs from 0: loopCount, start, busy, counter), each case worked by hand from
  * `SnaxShell`'s rules (snax-gvsoc `snax/snax_shell.cpp`, its header), c being the cycle of an access:
  *
  *   - a write to a field, or to start with bit 0 clear, is taken at once, and reads back as written;
  *   - a start goes to the unit when it is idle in c, accepted in c; while the unit is busy in c the write is held,
  *     then sent and answered in the cycle busy falls, E + 1 (E the run's last cycle);
  *   - busy reads the unit's busy in c (busy from the cycle after the start, to E);
  *   - the counter is the busy cycles after the accepted start and before c, held after the run; a start the unit
  *     ignores (count 0) still resets it.
  *
  * The streamer's inputs are always valid unless a case says otherwise; with them, a run of count n fires in each of
  * its n busy cycles.
  */
class ShellCasesSpec extends AnyFlatSpec with ChiselScalatestTester {

  private val fixture   = RtlFixtures.all.find(_.name == "vecadd").get
  private val LoopCount = 0
  private val Start     = 1
  private val Busy      = 2
  private val Counter   = 3

  private def shell(valid: Int => Boolean = _ => true)(body: ShellDriver => Unit): Unit =
    test(new ShellHarness(fixture.spec)) { dut =>
      body(new ShellDriver(dut, Seq(1 << 20, 1 << 20), (_, _) => BigInt(0), valid))
    }

  behavior of "the shell wrapper's CSRs"

  it should "read a field back as written" in shell() { d =>
    assert(d.write(LoopCount, BigInt("deadbeef", 16)) == 0)
    assert(d.read(LoopCount) == BigInt("deadbeef", 16))
    d.write(Start, 6)
    assert(d.read(Start) == 6)
  }

  it should "only write start with bit 0 clear" in shell() { d =>
    d.write(LoopCount, 4)
    assert(d.write(Start, 2) == 1) // taken at once
    assert(Seq.fill(3)(d.read(Busy)) == Seq(0, 0, 0))
    assert(d.read(Counter) == 0)
    assert(d.read(Start) == 2)
    assert(d.taken.sameElements(Seq(0, 0)), "no input taken")
  }

  // c0 loopCount 2, c1 start (accepted, busy from c2); inputs valid from c5: firings c5, c6 = E. The second start,
  // offered from c2, is held through c6 and taken in c7 = E + 1; it runs from c8 and fires in c8, c9.
  it should "hold a start while busy and answer it the cycle busy falls" in shell(valid = _ >= 5) { d =>
    d.write(LoopCount, 2)
    assert(d.write(Start, 1) == 1)
    assert(d.write(Start, 1) == 7)
    assert(d.held == (2 to 6))
    assert(d.read(Busy) == 1) // c8
    assert(d.read(Busy) == 1) // c9: the second run's last cycle
    assert(d.read(Busy) == 0) // c10
    assert(d.taken.sameElements(Seq(4, 4)))
  }

  // c0 busy 0; c1 loopCount 2, c2 start: busy from c3; inputs valid from c5, firings c5, c6 = E.
  it should "read the unit's busy" in shell(valid = _ >= 5) { d =>
    assert(d.read(Busy) == 0)
    d.write(LoopCount, 2)
    assert(d.write(Start, 1) == 2)
    assert(Seq.fill(5)(d.read(Busy)) == Seq(1, 1, 1, 1, 0)) // c3 .. c7
  }

  // c0 loopCount 3, c1 start: busy c2 .. c4. Counter at c: busy cycles in [c2, c - 1].
  // c8 loopCount 2, c9 start: busy c10, c11; counter from c10 the busy cycles in [c10, c - 1].
  it should "reset the counter on an accepted start, count busy cycles and hold it" in shell() { d =>
    d.write(LoopCount, 3)
    assert(d.write(Start, 1) == 1)
    assert(Seq.fill(6)(d.read(Counter)) == Seq(0, 1, 2, 3, 3, 3)) // c2 .. c7
    d.write(LoopCount, 2)
    assert(d.write(Start, 1) == 9)
    assert(Seq.fill(4)(d.read(Counter)) == Seq(0, 1, 2, 2)) // c10 .. c13
  }

  // c0 loopCount 2, c1 start: busy c2, c3, counter 2 at c4. c5 loopCount 0, c6 start: taken (idle), ignored by the
  // unit; busy 0 at c7 and the counter 0 at c8.
  it should "reset the counter on a start of count 0" in shell() { d =>
    d.write(LoopCount, 2)
    d.write(Start, 1)
    d.idle(2)
    assert(d.read(Counter) == 2)
    d.write(LoopCount, 0)
    assert(d.write(Start, 1) == 6)
    assert(d.read(Busy) == 0)
    assert(d.read(Counter) == 0)
  }
}
