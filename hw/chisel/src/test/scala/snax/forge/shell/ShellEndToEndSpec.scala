package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.ForgeTester

/** Each fixture end to end through its `<tag>_shell_wrapper`: per task the fields written in CSR order and start
  * written 1, as `main.c` does; the inputs fed by an always-ready streamer (each port's data.h array, lane i of beat t
  * element `t * lanes + i`, in 64-bit two's complement); busy polled until it reads 0; then the counter read. The
  * outputs must equal the goldens and the counter the fixture's value, worked from `SnaxShell`'s rules
  * (`vectors.json`'s `working`).
  */
class ShellEndToEndSpec extends AnyFlatSpec with ForgeTester {

  private val Mask = (BigInt(1) << 64) - 1

  behavior of "the shell wrapper end to end"

  for (f <- RtlFixtures.all) {
    it should s"give ${f.name}'s goldens and its counter" in {
      val unit       = f.spec.units.head
      val dirs       = snax.forge.Blocks(unit.block).ports.toMap
      val byPort     = f.ports.map(p => p.port -> p).toMap
      val brmDir     = unit.ports.map { case (brm, block) => brm -> dirs(block) }
      val ins        = brmDir.collect { case (p, "in") => byPort(p) }
      val outs       = brmDir.collect { case (p, "out") => byPort(p) }
      val busy       = unit.rw.size + 1
      val counterCsr = unit.rw.size + 2
      test(new ShellHarness(f.spec)) { dut =>
        val beat = (k: Int, t: Int) => {
          val p = ins(k)
          (0 until p.lanes).map(i => (p.values(t * p.lanes + i) & Mask) << (64 * i)).sum
        }
        val d    = ShellDriver(dut, ins.map(_.beats), beat)
        for (task   <- f.tasks) {
          val values = task.csrs.toMap
          unit.rw.zipWithIndex.foreach { case ((pin, _), i) => d.write(i, values(pin)) }
          d.write(unit.rw.size, 1)
          while (d.read(busy) != 0) {}
          assert(d.read(counterCsr) == task.counter, s"${task.name}: the counter")
        }
        for ((p, k) <- outs.zipWithIndex) {
          val want = (0 until p.beats).map { t =>
            (0 until p.lanes).map(i => (p.values(t * p.lanes + i) & Mask) << (64 * i)).sum
          }
          assert(d.outputs(k) == want, s"${p.port}: the outputs against ${p.array}")
        }
        assert(d.taken.toSeq == ins.map(_.beats), "every input beat taken")
      }
    }
  }
}
