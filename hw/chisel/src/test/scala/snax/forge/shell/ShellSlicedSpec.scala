package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, LaneBits}

/** A sliced outer input in the wrapper (D180), the smallest case: the fixture vsum (TreeAccumulate, one lane of int64)
  * with its input changed by hand to int32 at d_in 2, as `accelerator.json` gives a sliced port. The reader's port is
  * one 64-bit word of two elements, low first; 3 jobs of 3 beats pop 9 elements from 5 words, the last word short, and
  * each sum is its job's elements, sign-extended.
  */
class ShellSlicedSpec extends AnyFlatSpec with ForgeTester {

  private lazy val spec = {
    val base = RtlFixtures.all.find(_.name == "vsum").get.spec
    base.copy(groups = base.groups.map { g =>
      g.copy(units = g.units.map { u =>
        u.copy(ports = u.ports.map {
          case ("a", p) => "a" -> p.copy(dtype = "int32", dIn = 2)
          case other    => other
        })
      })
    })
  }

  behavior of "a sliced outer input (vsum's sum at int32, d_in 2)"

  it should "have a 64-bit reader and sum 9 elements from 5 words, the last short" in {
    test(new ShellHarness(spec)) { dut =>
      assert(dut.io.readers == Seq(64), s"readers ${dut.io.readers}")
      val (jobs, beats) = (3, 3)
      val elem          = (e: Int) => BigInt(if (e % 2 == 0) -(7 * e + 3) else 11 * e + 1)
      val words         = (jobs * beats + 1) / 2
      val d             = ShellDriver(
        dut,
        Seq(words),
        (_, t) => LaneBits.pack(Seq(2 * t, 2 * t + 1).map(e => elem(e) & LaneBits.mask(32)), 32)
      )
      d.write(dut.csr("JOBS"), jobs)
      d.write(dut.csr("JOB_BEATS"), beats)
      d.write(dut.csr("START"), 1)
      while ((d.read(dut.csr("STATUS")) & 1) != 0) {}
      val sums          = (0 until jobs).map(j => (0 until beats).map(t => elem(j * beats + t)).sum & LaneBits.mask(64))
      assert(d.taken == Seq(words), s"words taken ${d.taken}")
      assert(d.outputs(0) == sums, s"sums ${d.outputs(0).map(_.toString(16))}, want ${sums.map(_.toString(16))}")
    }
  }
}
