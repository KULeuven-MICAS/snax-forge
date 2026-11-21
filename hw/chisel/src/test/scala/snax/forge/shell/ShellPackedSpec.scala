package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, LaneBits}

/** A packed outer output in the wrapper (D174), the smallest case: the fixture vsum_v1_1_1 (TreeAccumulate, 4 lanes of
  * int32 into one) with its sum's output changed by hand to int32 at d_out 2, as `accelerator.json` gives a packed
  * port. The writer's port is one 64-bit word, and 3 jobs give two words: the first two sums, then the third alone,
  * zero above it.
  */
class ShellPackedSpec extends AnyFlatSpec with ForgeTester {

  private lazy val spec = {
    val base = RtlFixtures.all.find(_.name == "vsum_v1_1_1").get.spec
    base.copy(groups = base.groups.map { g =>
      g.copy(units = g.units.map { u =>
        u.copy(ports = u.ports.map {
          case ("out", p) => "out" -> p.copy(dtype = "int32", dOut = 2)
          case other      => other
        })
      })
    })
  }

  behavior of "a packed outer output (vsum_v1_1_1's sum at int32, d_out 2)"

  it should "have a 64-bit writer and send 3 sums as two words, the last short" in {
    test(new ShellHarness(spec)) { dut =>
      assert(dut.io.writers == Seq(64), s"writers ${dut.io.writers}")
      val mask          = LaneBits.mask(32)
      val value         = (t: Int, l: Int) => BigInt(-(10 * t + l + 1))
      val (jobs, beats) = (3, 2)
      val d = ShellDriver(dut, Seq(jobs * beats), (_, t) => LaneBits.pack(Seq.tabulate(4)(l => value(t, l) & mask), 32))
      d.write(dut.csr("JOBS"), jobs)
      d.write(dut.csr("JOB_BEATS"), beats)
      d.write(dut.csr("START"), 1)
      while ((d.read(dut.csr("STATUS")) & 1) != 0) {}
      val sums =
        (0 until jobs).map(j => (0 until beats).flatMap(t => (0 until 4).map(value(j * beats + t, _))).sum & mask)
      assert(
        d.outputs(0) == Seq(sums(0) | (sums(1) << 32), sums(2)),
        s"words ${d.outputs(0).map(_.toString(16))}, sums ${sums.map(_.toString(16))}"
      )
    }
  }
}
