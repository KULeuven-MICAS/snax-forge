package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, LaneBits, RepoPaths, Sink, Source}

/** Several groups in one accelerator (D146, D148): dot 1 (C§17's table), the generated `accelerator.json` of
  * `recipes/dot_1.json` (the fixture `src/test/resources/rtl/dot_1/`): group mul (ElementwiseLanes, mul, 4 lanes of
  * int64; START 0, JOBS 1, STATUS 2, COUNTER 3) and group sum (TreeAccumulate, 4 lanes of int64 into one; START 4, JOBS
  * 5, JOB_BEATS 6, STATUS 7, COUNTER 8). Each group starts, runs and reports STATUS on its own: sum is started first
  * and starved (its input held back), mul then runs its 3 jobs to the end while sum stays busy, then sum's input flows
  * and it ends.
  */
class TwoGroupsSpec extends AnyFlatSpec with ForgeTester {

  private val file = RepoPaths.root.resolve("hw/chisel/src/test/resources/rtl/dot_1/accelerator.json")
  private val spec = AcceleratorSpec.load(file)

  behavior of "a two-group accelerator (dot 1)"

  it should "start, run and report STATUS per group" in {
    test(new ShellHarness(spec)) { dut =>
      val beat       = (k: Int, t: Int) => LaneBits.pack(Seq.tabulate(4)(l => BigInt(10 * k + t + l + 1)), 64)
      var release    = false // sum's input held back until then
      val sources    = Seq(
        new Source(3, beat(0, _), _ => true),
        new Source(3, beat(1, _), _ => true),
        new Source(2, beat(2, _), _ => release)
      )
      val d          = new ShellDriver(dut, sources, Seq(new Sink(_ => true), new Sink(_ => true)))
      val (mul, sum) = (0, 1)

      // sum: one job of two beats, started first, starved
      d.write(dut.csr("JOBS", sum), 1)
      d.write(dut.csr("JOB_BEATS", sum), 2)
      d.write(dut.csr("START", sum), 1)
      assert(d.read(dut.csr("STATUS", sum)) == 1, "sum busy")
      // mul: three jobs, which run to the end while sum waits
      d.write(dut.csr("JOBS", mul), 3)
      d.write(dut.csr("START", mul), 1)
      while ((d.read(dut.csr("STATUS", mul)) & 1) != 0) {}
      assert(d.read(dut.csr("STATUS", mul)) == 2, "mul done")
      assert(d.read(dut.csr("COUNTER", mul)) == 3, "mul's COUNTER: three busy cycles")
      assert(d.read(dut.csr("STATUS", sum)) == 1, "sum still busy, not done")
      assert(dut.io.bool("sum_busy_o").peek().litToBoolean && !dut.io.bool("mul_busy_o").peek().litToBoolean)
      release = true
      while ((d.read(dut.csr("STATUS", sum)) & 1) != 0) {}
      assert(d.read(dut.csr("STATUS", sum)) == 2, "sum done")
      assert(d.read(dut.csr("STATUS", mul)) == 2, "mul's done kept")

      val products = (0 until 3).map { t =>
        LaneBits.pack(Seq.tabulate(4)(l => BigInt(t + l + 1) * BigInt(10 + t + l + 1)), 64)
      }
      assert(d.outputs(0) == products, "mul's outputs")
      assert(d.outputs(1) == Seq((0 until 2).flatMap(t => (0 until 4).map(l => BigInt(20 + t + l + 1))).sum), "the sum")
    }
  }
}
