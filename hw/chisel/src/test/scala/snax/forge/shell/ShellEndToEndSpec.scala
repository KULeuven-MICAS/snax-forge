package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds, Sink, SnaxGroupCases, Source}

/** Each fixture end to end through its `<tag>_shell_wrapper`: per task its CSR values written by name (JOBS, and
  * JOB_BEATS for a fold) and START written 1, as `main.c` does; the inputs fed by an always-ready streamer (each port's
  * data.h array, lane i of beat t element `t * lanes + i` modulo the array's size, in two's complement at the port's
  * dtype width; a sliced input's word d_in beats of them, a packed output's d_out results, the last word short, zero
  * above its slices); STATUS polled until bit 0 reads 0; then BUSY_COUNTER read, and STATUS again: done (2). The
  * outputs must equal the goldens (NumPy's, `vectors.json`) and the counter the fixture's value, worked from the nest
  * controller's rule or, for a linked group, the Cycles of CONTRACTS, "Accelerator RTL spec" (`vectors.json`'s
  * `working`).
  *
  * Then each linked fixture without a fork at W 4 (dot 4) at N 8 and 16, on random int64 data of one seed per N (the
  * cycles do not depend on the data, which ShellRandomSpec varies): J = 1, the results NumPy's (int64, wrapping), the
  * counter the busy cycles of snax-gvsoc's matching group case (`dot<k>_n<N>_j<J>`: its run line's classes added up;
  * cancelled without `SNAX_GVSOC_GROUP_CASES`).
  *
  * And dot_v1_2: its two groups on one set's movers through `<tag>_dot_port_mux`, `main.c`'s CSR writes in its order
  * (vmul's JOBS, PORT_MUX_SEL 0, vmul's START, vsum's JOBS and JOB_BEATS while vmul runs, vmul's STATUS polled and its
  * BUSY_COUNTER read, PORT_MUX_SEL 1, vsum's START, its STATUS polled and BUSY_COUNTER read), an always-ready streamer
  * per mover and task: task 1's readers vmul's a and b (random int64, one seed), task 2's reader 0 the products task
  * 1's writer gave and reader 1 idle. The products NumPy's (wrapping), the sum on the writer's low word with the other
  * three 0, each counter the nest controller's rule at L 0 and L 1: one busy cycle a beat for vmul, one more for vsum.
  */
class ShellEndToEndSpec extends AnyFlatSpec with ForgeTester {

  /** A run: the CSR values, each input's elements and beats (pops) and each output's expected elements, by outer port
    * in order; `results`, per output the elements its writer gets when more than it expects (a carried output's earlier
    * visits, D192, its last the expected); without `values` only the words are counted.
    */
  case class Run(
    csrs:    Seq[(String, Long)],
    ins:     Seq[Seq[BigInt]],
    beats:   Seq[Int],
    outs:    Seq[Seq[BigInt]],
    results: Seq[Int] = Nil,
    values:  Boolean  = true
  )

  /** Run `run` on `spec`'s wrapper: the outputs and the counter (`counter`, when given) checked. */
  private def drive(spec: AcceleratorSpec, run: Run, counter: Option[Long], clue: String): Unit = {
    val ports = (spec.inputs ++ spec.outputs).map { case (u, p) => spec.unit(u).port(p) }
    val ins   = ports.take(spec.inputs.size)
    val outs  = ports.drop(spec.inputs.size)
    // A port's elements per word on its streamer: its stream's lanes (one for a broadcast), d_in (sliced) or d_out
    // (packed) times
    def per(p: AcceleratorSpec.PortSpec): Int = p.streamLanes * p.dIn * p.dOut
    val words = run.beats.zip(ins).map { case (b, p) => -(-b * p.streamLanes / per(p)) }
    test(new ShellHarness(spec)) { dut =>
      // Word t of a port: its elements packed at the port's dtype width; past the last, an input's wrap and an
      // output's are zero (a short last word)
      def packed(p: AcceleratorSpec.PortSpec, values: Seq[BigInt], t: Int, zero: Boolean = false): BigInt = {
        val w = IntType(p.dtype).width
        LaneBits.pack(
          (0 until per(p)).map { i =>
            val e = t * per(p) + i
            if (zero && e >= values.size) BigInt(0) else values(e % values.size)
          },
          w
        )
      }
      val d = ShellDriver(dut, words, (k, t) => packed(ins(k), run.ins(k), t))
      run.csrs.foreach { case (name, v) => d.write(dut.csr(name), v) }
      d.write(dut.csr("START"), 1)
      while ((d.read(dut.csr("STATUS")) & 1) != 0) {}
      counter.foreach(c => assert(d.read(dut.csr("BUSY_COUNTER")) == c, s"$clue: the counter"))
      assert(d.read(dut.csr("STATUS")) == 2, s"$clue: STATUS after the run: done, not busy")
      for (((p, want), k) <- outs.zip(run.outs).zipWithIndex) {
        val beats = -(-want.size / per(p))
        val all   = -(-run.results.lift(k).getOrElse(want.size) / per(p))
        val w     = IntType(p.dtype).width
        val got   = d.outputs(k).map(_ & LaneBits.mask(per(p) * w))
        assert(got.size == all, s"$clue: output $k's words")
        if (run.values)
          assert(got.takeRight(beats) == (0 until beats).map(t => packed(p, want, t, zero = true)), s"$clue: output $k")
      }
      assert(d.taken.toSeq == words, s"$clue: every input word taken")
    }
  }

  /** The fixtures whose readers do not stream their arrays in memory order, which `vectors.json` gives and `drive`
    * streams: bicg_v4_1's strips (D192) read A, p and r a strip at a time, and its seed reads back q's writes through
    * L1. Their counter, STATUS, input words and output words are checked; ShellRandomSpec checks their values on its
    * model.
    */
  private val Strided = Set("bicg_v4_1")

  behavior of "the shell wrapper end to end"

  for (f <- RtlFixtures.driven)
    it should s"give ${f.name}'s goldens and its counter" in {
      val spec  = f.spec
      val outer = spec.inputs ++ spec.outputs
      assert(f.ports.map(_.port) == outer.map(_._2), s"vectors.json's ports in accelerator.json's order: $outer")
      val ins   = f.ports.take(spec.inputs.size)
      val outs  = f.ports.drop(spec.inputs.size)
      for (task <- f.tasks) {
        val run = Run(
          task.csrs,
          ins.map(_.values),
          ins.map(_.beats),
          outs.map(p => p.values.take(p.beats * p.lanes)),
          outs.map(p => p.beats * p.lanes),
          !Strided(f.name)
        )
        drive(spec, run, Some(task.counter), s"${f.name}, ${task.name}")
      }
    }

  /** The linked fixtures without a fork at `lanes` lanes (W): at 4, dot 4. */
  private def linked(lanes: Int) =
    RtlFixtures.all.filter { f =>
      f.spec.groups.exists(_.links.nonEmpty) && f.spec.forks.isEmpty && f.spec.units.head.port("a").lanes == lanes
    }
  private val variants           = linked(4)

  /** NumPy's q = A p at `t`, A `rows` × n row-major, wrapping. */
  private def matvec(a: Seq[BigInt], p: Seq[BigInt], rows: Int, t: IntType): Seq[BigInt] =
    (0 until rows).map { i =>
      val n = p.size
      LaneBits.signed(LaneBits.dot(a.slice(i * n, (i + 1) * n), p, t.width), t.width)
    }

  /** The busy cycles of the group case `name`: its run line's classes added up. */
  private def caseCycles(name: String): Option[Long] =
    SnaxGroupCases.all.map { all =>
      val c = all.find(_.name == name).getOrElse(fail(s"no group case $name"))
      val r = c.runs.head
      (r.busy + r.stallIn + r.stallOut + r.flush).toLong
    }

  private def group(f: RtlFixtures.Fixture, n: Int, jobs: Int): String =
    s"dot${f.name.split("_").last.stripPrefix("v")}_n${n}_j$jobs"

  it should "have the linked fixture dot 4" in assert(variants.map(_.name) == Seq("dot_v4"))

  for {
    f      <- variants
    (n, k) <- Seq(8, 16).zipWithIndex
    seed = Seeds()(k % Seeds().size)
  }
    it should s"give ${f.name}'s NumPy results and its group case's cycles at N $n (seed $seed)" in Seeds.run(seed) {
      r =>
        val jobs  = 1
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

  it should "run dot_v1_2's two tasks through its port mux as main.c does" in Seeds.run(Seeds().head) { r =>
    val f                        = RtlFixtures.all.find(_.name == "dot_v1_2").get
    val spec                     = f.spec
    val sel                      = spec.portMuxCsrs.head.offset
    val (n, w)                   = (64, 4)
    val beats                    = n / w
    val (mulCounter, sumCounter) = (beats.toLong, beats + 1L) // L 0: a busy cycle a beat; L 1: one more
    val a                        = Seq.fill(n)(LaneBits.signed(r.word(64), 64))
    val b                        = Seq.fill(n)(LaneBits.signed(r.word(64), 64))
    val prods                    = a.zip(b).map { case (x, y) => (x * y) & LaneBits.mask(64) }
    def beat(v: Seq[BigInt])(t: Int) = LaneBits.pack(v.slice(t * w, (t + 1) * w).map(LaneBits.signed(_, 64)), 64)
    val (mul, sum) = (0, 1)
    test(new ShellHarness(spec)) { dut =>
      val one = new ShellDriver(
        dut,
        Seq(new Source(beats, beat(a), _ => true), new Source(beats, beat(b), _ => true)),
        Seq(new Sink(_ => true))
      )
      one.write(dut.csr("JOBS", mul), beats)
      one.write(sel, 0)
      one.write(dut.csr("START", mul), 1)
      one.write(dut.csr("JOBS", sum), 1)
      one.write(dut.csr("JOB_BEATS", sum), beats)
      while ((one.read(dut.csr("STATUS", mul)) & 1) != 0) {}
      assert(one.read(dut.csr("BUSY_COUNTER", mul)) == mulCounter, "vmul's counter")
      assert(one.outputs.head == (0 until beats).map(beat(prods)), "vmul's products on writer 0")
      assert(one.taken == Seq(beats, beats), "every beat of a and b taken")

      val two = new ShellDriver(
        dut,
        Seq(new Source(beats, t => one.outputs.head(t), _ => true), new Source(0, _ => BigInt(0), _ => true)),
        Seq(new Sink(_ => true))
      )
      two.write(sel, 1)
      assert(two.read(sel) == 1, "PORT_MUX_SEL read back")
      two.write(dut.csr("START", sum), 1)
      while ((two.read(dut.csr("STATUS", sum)) & 1) != 0) {}
      assert(two.read(dut.csr("BUSY_COUNTER", sum)) == sumCounter, "vsum's counter")
      assert(two.outputs.head == Seq(prods.sum & LaneBits.mask(64)), "the sum on writer 0's low word, the rest 0")
      assert(two.taken == Seq(beats, 0), "every product taken, reader 1 idle")
      assert(two.read(dut.csr("STATUS", mul)) == 2 && two.read(dut.csr("STATUS", sum)) == 2, "both done")
    }
  }
}
