package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds}
import snax.forge.shell.AcceleratorSpec.{CsrSpec, GroupSpec, PortSpec, UnitSpec}

/** A unit's scalar CSRs through `<tag>_shell_wrapper` (C§17): axpy's one SimdLanes unit at axpby with k1 from its CSR,
  * as `accelerator.json` gives it (`scalars` `{"k1": <opType>}`), at int32 (K1 at offset 2, 32 bits; STATUS 3, BUSY_COUNTER
  * 4) and int64 (K1_LO 2 and K1_HI 3, 32 bits each, low word first; STATUS 4, BUSY_COUNTER 5). Per seed: K1 written and read
  * back, then runs of random a and b with a random k1 each, written as `main.c` does (K1, or K1_LO then K1_HI) before
  * START, the outputs NumPy's `k1 * a + b` at the op type, so the datapath's pin takes what the program last wrote;
  * with k2 from its CSR too (`K2` after `K1`), `k1 * a + k2 * b`.
  */
class ShellScalarSpec extends AnyFlatSpec with ForgeTester {

  private val Lanes = 2

  /** axpy's description at `dtype`, its scalars `slots` from CSRs. */
  private def axpy(dtype: String, slots: Seq[String] = Seq("k1")): AcceleratorSpec = {
    var at     = 0
    def next() = { at += 1; at - 1 }
    val start  = CsrSpec("START", "rw", next(), 1)
    val jobs   = CsrSpec("JOBS", "rw", next(), 32)
    val wide   = IntType(dtype).width == 64
    val own    = slots.flatMap { s =>
      val n = s.toUpperCase
      if (wide) Seq(s"${n}_LO", s"${n}_HI").map(CsrSpec(_, "rw", next(), 32, Some(s)))
      else Seq(CsrSpec(n, "rw", next(), IntType(dtype).width, Some(s)))
    }
    val status = CsrSpec("STATUS", "ro", next(), 2)
    val count  = CsrSpec("BUSY_COUNTER", "ro", next(), 32)
    val ports  = Seq("a", "b", "out").map(p => p -> PortSpec(p, if (p == "out") "out" else "in", dtype, Lanes))
    val unit   = UnitSpec(
      "acc",
      "simd",
      "SimdLanes",
      ports,
      Some(dtype),
      Map("op" -> ujson.Str("axpby")),
      0,
      1,
      1,
      own,
      slots.map(_ -> dtype)
    )
    AcceleratorSpec(
      "p",
      "snax_forge_axpy",
      Seq("acc" -> "a", "acc" -> "b"),
      Seq("acc" -> "out"),
      Seq(
        GroupSpec("acc", Seq(start, jobs, status, count), Seq(unit), Seq.empty)
      )
    )
  }

  behavior of "a unit's scalar CSRs in the shell"

  it should "put K1 after JOBS at int32, 32 bits, and K1_LO, K1_HI at int64" in {
    val i32 = axpy("int32")
    assert(
      i32.csrs.map(c => (c.name, c.offset, c.width, c.pin)) == Seq(
        ("START", 0, 1, None),
        ("JOBS", 1, 32, None),
        ("K1", 2, 32, Some("k1")),
        ("STATUS", 3, 2, None),
        ("BUSY_COUNTER", 4, 32, None)
      )
    )
    val i64 = axpy("int64")
    assert(
      i64.csrs.map(c => (c.name, c.offset, c.width)) == Seq(
        ("START", 0, 1),
        ("JOBS", 1, 32),
        ("K1_LO", 2, 32),
        ("K1_HI", 3, 32),
        ("STATUS", 4, 2),
        ("BUSY_COUNTER", 5, 32)
      )
    )
    test(new ShellHarness(i64)) { dut =>
      val d = ShellDriver(dut, Seq(0, 0), (_, _) => BigInt(0))
      d.write(dut.csr("K1_LO"), BigInt("89abcdef", 16))
      d.write(dut.csr("K1_HI"), BigInt("01234567", 16))
      assert(d.read(dut.csr("K1_LO")) == BigInt("89abcdef", 16))
      assert(d.read(dut.csr("K1_HI")) == BigInt("01234567", 16))
    }
  }

  /** Runs of `jobs` beats with a random k (and k2) each: K1 (K2) written as main.c does, then START; the outputs
    * NumPy's.
    */
  private def runs(spec: AcceleratorSpec, r: snax.forge.Rand, runs: Int, jobs: Int): Unit = {
    val t      = IntType(spec.units.head.opType.get)
    val slots  = spec.units.head.scalars.map(_._1)
    val inputs = Seq.fill(runs, 2)(Seq.fill(jobs * Lanes)(r.word(t.width)))
    val ks     = Seq.fill(runs, slots.size)(r.word(t.width))
    test(new ShellHarness(spec)) { dut =>
      val beats = (0 until runs).map(_ => jobs)
      // input k's beat t over every run, run by run
      val d     = ShellDriver(
        dut,
        Seq(runs * jobs, runs * jobs),
        (k, b) => LaneBits.pack((0 until Lanes).map(i => inputs(b / jobs)(k)((b % jobs) * Lanes + i)), t.width)
      )
      for (run <- 0 until runs) {
        d.write(dut.csr("JOBS"), jobs)
        for ((s, i) <- slots.zipWithIndex) {
          val n = s.toUpperCase
          if (t.width == 64) {
            d.write(dut.csr(s"${n}_LO"), ks(run)(i) & LaneBits.mask(32))
            d.write(dut.csr(s"${n}_HI"), ks(run)(i) >> 32)
          } else {
            d.write(dut.csr(n), ks(run)(i))
            assert(d.read(dut.csr(n)) == ks(run)(i), s"run $run: $n read back")
          }
        }
        d.write(dut.csr("START"), 1)
        while ((d.read(dut.csr("STATUS")) & 1) != 0) {}
        assert(d.read(dut.csr("BUSY_COUNTER")) == jobs, s"run $run: the counter")
      }
      val want  = for {
        run <- 0 until runs
        b   <- 0 until jobs
      } yield LaneBits.pack(
        (0 until Lanes).map { i =>
          val (x, y) = (inputs(run)(0)(b * Lanes + i), inputs(run)(1)(b * Lanes + i))
          val k2     = if (slots.size == 2) ks(run)(1) else BigInt(1)
          (ks(run)(0) * x + k2 * y) & LaneBits.mask(t.width)
        },
        t.width
      )
      val got   = d.outputs(0).map(_ & LaneBits.mask(Lanes * t.width))
      assert(got == want, s"${t.name}: the outputs")
      assert(beats.sum == d.taken.head)
    }
  }

  for (seed <- Seeds())
    it should s"give k1 * a + b at the k1 last written, at int32 and int64 (seed $seed)" in Seeds.run(seed) { r =>
      runs(axpy("int32"), r, 3, 4)
      runs(axpy("int64"), r, 3, 4)
    }

  for (seed <- Seeds().take(2))
    it should s"give k1 * a + k2 * b with both from CSRs (seed $seed)" in Seeds.run(seed) { r =>
      runs(axpy("int32", Seq("k1", "k2")), r, 2, 3)
      runs(axpy("int64", Seq("k1", "k2")), r, 2, 3)
    }
}
