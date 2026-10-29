package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds, SnaxGroupCases}

/** Each fixture end to end through its `<tag>_shell_wrapper`: per task its CSR values written by name (JOBS, and
  * JOB_BEATS for a fold) and START written 1, as `main.c` does; the inputs fed by an always-ready streamer (each port's
  * data.h array, lane i of beat t element `t * lanes + i` modulo the array's size, in two's complement at the port's
  * dtype width, D142); STATUS polled until bit 0 reads 0; then COUNTER read, and STATUS again: done (2). The outputs
  * must equal the goldens (NumPy's, `vectors.json`) and the counter the fixture's value, worked from the nest
  * controller's rule or, for a linked group, C§17's Cycles (`vectors.json`'s `working`).
  *
  * Then each linked variant (dot 2–6, matvec 4–6) at N 8 and 16, W 4, on random int64 data: dot J = 1, matvec J = 2
  * rows (q_i = Σ_j A_ij · p_j, p read again for each row), the results NumPy's (int64, wrapping), the counter the busy
  * cycles of snax-gvsoc's matching group case (`dot<k>_n<N>_j<J>`: its run line's classes added up; cancelled without
  * `SNAX_GVSOC_GROUP_CASES`). And matvec 4–6 at int32 (every port and the product retyped), N 8, J 2, on the rows
  * (46341, 3, 3, 3, 3, 3, 3, 6) and (0, 1, 1, 1, 1, 0, 0, 0) against p = (46341, 1, ..., 1): 46341² wraps at int32, q =
  * (-2147478991, 4), the tb's MAC vector `mac_int32_wrap_w2_t8` through the whole accelerator.
  */
class ShellEndToEndSpec extends AnyFlatSpec with ForgeTester {

  /** A run: the CSR values, each input's elements and each output's expected elements, by outer port in order. */
  case class Run(csrs: Seq[(String, Long)], ins: Seq[Seq[BigInt]], beats: Seq[Int], outs: Seq[Seq[BigInt]])

  /** Run `run` on `spec`'s wrapper: the outputs and the counter (`counter`, when given) checked. */
  private def drive(spec: AcceleratorSpec, run: Run, counter: Option[Long], clue: String): Unit = {
    val ports = (spec.inputs ++ spec.outputs).map { case (u, p) => spec.unit(u).port(p) }
    val ins   = ports.take(spec.inputs.size)
    val outs  = ports.drop(spec.inputs.size)
    test(new ShellHarness(spec)) { dut =>
      // Beat t of a port: its lanes' elements packed at the port's dtype width.
      def packed(p: AcceleratorSpec.PortSpec, values: Seq[BigInt], t: Int): BigInt = {
        val w = IntType(p.dtype).width
        LaneBits.pack((0 until p.lanes).map(i => values((t * p.lanes + i) % values.size)), w)
      }
      val d = ShellDriver(dut, run.beats, (k, t) => packed(ins(k), run.ins(k), t))
      run.csrs.foreach { case (name, v) => d.write(dut.csr(name), v) }
      d.write(dut.csr("START"), 1)
      while ((d.read(dut.csr("STATUS")) & 1) != 0) {}
      counter.foreach(c => assert(d.read(dut.csr("COUNTER")) == c, s"$clue: the counter"))
      assert(d.read(dut.csr("STATUS")) == 2, s"$clue: STATUS after the run: done, not busy")
      for (((p, want), k) <- outs.zip(run.outs).zipWithIndex) {
        val beats = want.size / p.lanes
        val w     = IntType(p.dtype).width
        val got   = d.outputs(k).map(_ & LaneBits.mask(p.lanes * w))
        assert(got == (0 until beats).map(t => packed(p, want, t)), s"$clue: output $k")
      }
      assert(d.taken.toSeq == run.beats, s"$clue: every input beat taken")
    }
  }

  behavior of "the shell wrapper end to end"

  for (f <- RtlFixtures.driven)
    it should s"give ${f.name}'s goldens and its counter" in {
      val spec  = f.spec
      val outer = spec.inputs ++ spec.outputs
      assert(f.ports.map(_.port) == outer.map(_._2), s"vectors.json's ports in accelerator.json's order: $outer")
      val ins   = f.ports.take(spec.inputs.size)
      val outs  = f.ports.drop(spec.inputs.size)
      for (task <- f.tasks) {
        val run = Run(task.csrs, ins.map(_.values), ins.map(_.beats), outs.map(p => p.values.take(p.beats * p.lanes)))
        drive(spec, run, Some(task.counter), s"${f.name}, ${task.name}")
      }
    }

  /** The linked variants' fixtures: dot 2–6, matvec 4–6. */
  private val variants = RtlFixtures.all.filter(f => f.spec.groups.exists(_.links.nonEmpty))

  /** NumPy's q = A p at `t`, A `rows` × n row-major, wrapping. */
  private def matvec(a: Seq[BigInt], p: Seq[BigInt], rows: Int, t: IntType): Seq[BigInt] =
    (0 until rows).map { i =>
      val n = p.size
      LaneBits.signed((0 until n).map(j => a(i * n + j) * p(j)).sum & LaneBits.mask(t.width), t.width)
    }

  /** The busy cycles of the group case `name`: its run line's classes added up. */
  private def caseCycles(name: String): Option[Long] =
    SnaxGroupCases.all.map { all =>
      val c = all.find(_.name == name).getOrElse(fail(s"no group case $name"))
      val r = c.runs.head
      (r.busy + r.stallIn + r.stallOut + r.flush).toLong
    }

  private def group(f: RtlFixtures.Fixture, n: Int, jobs: Int): String = s"dot${f.name.split("_").last}_n${n}_j$jobs"

  it should "have the linked variants dot 2–6 and matvec 4–6" in
    assert(variants.map(_.name) == Seq("dot_2", "dot_3", "dot_4", "dot_5", "dot_6", "matvec_4", "matvec_5", "matvec_6"))

  for {
    f    <- variants
    n    <- Seq(8, 16)
    seed <- Seeds()
  }
    it should s"give ${f.name}'s NumPy results and its group case's cycles at N $n (seed $seed)" in Seeds.run(seed) {
      r =>
        val jobs  = if (f.name.startsWith("matvec")) 2 else 1
        val lanes = f.spec.units.head.port("a").lanes
        val t     = n / lanes
        val cyc   = caseCycles(group(f, n, jobs)).getOrElse(cancel(s"${SnaxGroupCases.Env} is not set"))
        val a     = Seq.fill(jobs * n)(LaneBits.signed(r.word(64), 64))
        val p     = Seq.fill(n)(LaneBits.signed(r.word(64), 64))
        val run   =
          Run(
            Seq("JOBS" -> jobs.toLong, "JOB_BEATS" -> t.toLong),
            Seq(a, p),
            Seq(jobs * t, jobs * t),
            Seq(matvec(a, p, jobs, IntType("int64")))
          )
        drive(f.spec, run, Some(cyc), s"${f.name}, N $n, J $jobs")
    }

  /** `spec` with every port, and the product, at `dtype`. */
  private def retyped(spec: AcceleratorSpec, dtype: String): AcceleratorSpec =
    spec.copy(groups = spec.groups.map { g =>
      g.copy(units = g.units.map { u =>
        u.copy(ports = u.ports.map { case (b, x) => b -> x.copy(dtype = dtype) }, opType = u.opType.map(_ => dtype))
      })
    })

  for (f <- variants if f.name.startsWith("matvec"))
    it should s"wrap ${f.name}'s int32 products: q = (-2147478991, 4)" in {
      val spec = retyped(f.spec, "int32")
      val a    = Seq(46341, 3, 3, 3, 3, 3, 3, 6, 0, 1, 1, 1, 1, 0, 0, 0).map(BigInt(_))
      val p    = (46341 +: Seq.fill(7)(1)).map(BigInt(_))
      val q    = matvec(a, p, 2, IntType("int32"))
      assert(q == Seq(BigInt(-2147478991), BigInt(4)))
      val cyc  = caseCycles(group(f, 8, 2)).getOrElse(cancel(s"${SnaxGroupCases.Env} is not set"))
      drive(
        spec,
        Run(Seq("JOBS" -> 2L, "JOB_BEATS" -> 2L), Seq(a, p), Seq(4, 4), Seq(q)),
        Some(cyc),
        s"${f.name} int32"
      )
    }
}
