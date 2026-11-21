package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, LaneBits}

/** A sliced outer input in the wrapper (D180), the smallest case: the fixture vecadd_v1_1_w1 (SimdLanes add, one lane of
  * int32), its inputs a and b sliced 2 from a word and its output packed 2 to one, as the flow places them. Each
  * reader's and the writer's port is one 64-bit word; 3 jobs pop 3 elements from 2 words a reader, the last word short,
  * and push 3 sums into 2 words, the last zero above its slice. Its 64 jobs, goldens and counter are
  * ShellEndToEndSpec's.
  */
class ShellSlicedSpec extends AnyFlatSpec with ForgeTester {

  private lazy val spec = RtlFixtures.all.find(_.name == "vecadd_v1_1_w1").get.spec

  behavior of "a sliced outer input (vecadd_v1_1_w1: a and b at int32, d_in 2; out d_out 2)"

  it should "have 64-bit readers and add 3 elements from 2 words each, the last short" in {
    test(new ShellHarness(spec)) { dut =>
      assert(dut.io.readers == Seq(64, 64) && dut.io.writers == Seq(64), s"${dut.io.readers} ${dut.io.writers}")
      val mask  = LaneBits.mask(32)
      val a     = (e: Int) => BigInt(-(7 * e + 3))
      val b     = (e: Int) => BigInt(11 * e + 1)
      val jobs  = 3
      val words = (jobs + 1) / 2
      val d     = ShellDriver(
        dut,
        Seq(words, words),
        (k, t) => LaneBits.pack(Seq(2 * t, 2 * t + 1).map(e => (if (k == 0) a(e) else b(e)) & mask), 32)
      )
      d.write(dut.csr("JOBS"), jobs)
      d.write(dut.csr("START"), 1)
      while ((d.read(dut.csr("STATUS")) & 1) != 0) {}
      val sums  = (0 until jobs).map(e => (a(e) + b(e)) & mask)
      assert(d.taken == Seq(words, words), s"words taken ${d.taken}")
      assert(
        d.outputs(0) == Seq(sums(0) | (sums(1) << 32), sums(2)),
        s"words ${d.outputs(0).map(_.toString(16))}, sums ${sums.map(_.toString(16))}"
      )
      assert(d.read(dut.csr("BUSY_COUNTER")) == jobs + 1, "a firing a cycle from busy cycle 2")
    }
  }
}
