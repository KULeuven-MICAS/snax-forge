package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits}

/** Each fixture end to end through its `<tag>_shell_wrapper`: per task its CSR values written by name (JOBS, and
  * JOB_BEATS for a fold) and START written 1, as `main.c` does; the inputs fed by an always-ready streamer (each port's
  * data.h array, lane i of beat t element `t * lanes + i`, in two's complement at the port's dtype width, D142); STATUS
  * polled until bit 0 reads 0; then COUNTER read, and STATUS again: done (2). The outputs must equal the goldens and
  * the counter the fixture's value, worked from the nest controller's rule (`vectors.json`'s `working`).
  */
class ShellEndToEndSpec extends AnyFlatSpec with ForgeTester {

  behavior of "the shell wrapper end to end"

  for (f <- RtlFixtures.all) {
    it should s"give ${f.name}'s goldens and its counter" in {
      val spec   = f.spec
      val unit   = spec.units.head
      val byPort = f.ports.map(p => p.port -> p).toMap
      val ins    = spec.inputs.map { case (_, p) => byPort(p) }
      val outs   = spec.outputs.map { case (_, p) => byPort(p) }
      test(new ShellHarness(spec)) { dut =>
        // Beat t of a port: its lanes' elements packed at the port's dtype width.
        def packed(p: RtlFixtures.Port, t: Int): BigInt = {
          val w = IntType(unit.port(p.port).dtype).width
          LaneBits.pack(p.values.slice(t * p.lanes, (t + 1) * p.lanes), w)
        }
        val d = ShellDriver(dut, ins.map(_.beats), (k, t) => packed(ins(k), t))
        for (task   <- f.tasks) {
          task.csrs.foreach { case (name, v) => d.write(dut.csr(name), v) }
          d.write(dut.csr("START"), 1)
          while ((d.read(dut.csr("STATUS")) & 1) != 0) {}
          assert(d.read(dut.csr("COUNTER")) == task.counter, s"${task.name}: the counter")
          assert(d.read(dut.csr("STATUS")) == 2, s"${task.name}: STATUS after the run: done, not busy")
        }
        for ((p, k) <- outs.zipWithIndex) {
          val want = (0 until p.beats).map(t => packed(p, t))
          assert(d.outputs(k) == want, s"${p.port}: the outputs against ${p.array}")
        }
        assert(d.taken.toSeq == ins.map(_.beats), "every input beat taken")
      }
    }
  }
}
