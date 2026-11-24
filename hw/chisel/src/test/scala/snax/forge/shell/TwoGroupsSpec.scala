package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, LaneBits, Sink, Source}

/** Several groups in one accelerator: the generated `accelerator.json` of matvec_acc_v2 (snax-gvsoc's
  * `forge/matvec_acc_v2/`), one accelerator `matvec` in one streamer set: group mul (SimdLanes, mul, 4 lanes of int64;
  * START 0, JOBS 1, STATUS 2, BUSY_COUNTER 3) and group sum (SerialAccumulate, one lane of int64, seeded; START 4, JOBS
  * 5, JOB_BEATS 6, STATUS 7, BUSY_COUNTER 8). Each group starts, runs and reports STATUS on its own: sum is started
  * first and starved (its input and seed held back), mul then runs its 3 jobs to the end while sum stays busy, then
  * sum's input flows and it ends.
  */
class TwoGroupsSpec extends AnyFlatSpec with ForgeTester {

  private lazy val spec = RtlFixtures.all.find(_.name == "matvec_acc_v2").get.spec

  behavior of "a two-group accelerator (matvec_acc_v2)"

  it should "start, run and report STATUS per group" in {
    test(new ShellHarness(spec)) { dut =>
      val beat       = (k: Int, t: Int) => LaneBits.pack(Seq.tabulate(4)(l => BigInt(10 * k + t + l + 1)), 64)
      var release    = false // sum's input and seed held back until then
      val sources    = Seq(
        new Source(3, beat(0, _), _ => true),
        new Source(3, beat(1, _), _ => true),
        new Source(2, t => BigInt(21 + t), _ => release),
        new Source(1, _ => BigInt(5), _ => release)
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
      assert(d.read(dut.csr("BUSY_COUNTER", mul)) == 3, "mul's BUSY_COUNTER: three busy cycles")
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
      assert(d.outputs(1) == Seq(BigInt(5 + 21 + 22)), "the sum, from its seed")
    }
  }
}
