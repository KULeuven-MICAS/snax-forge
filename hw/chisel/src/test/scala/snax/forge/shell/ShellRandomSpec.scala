package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds, Sink, Source}
import snax.forge.elementwise.{ElementwiseModel, ElementwiseOp}

/** Each fixture's `<tag>_shell_wrapper` (built as `Emit --spec` builds it: vecadd and vecmul, accumulate at W = 1, 4
  * and 8, and the linked groups dot 2–6, dot 5 at W = 1 (SerialAccumulate at lanes 1) and matvec 4–6) under random
  * data, configurations, streamer gaps and output stalls, through its CSR port as `main.c` drives it.
  *
  * Per seed, `Runs` runs back to back on one wrapper. Per run: J jobs in 1..32 (and for a fold T in 1..8 beats each; an
  * SIMD unit has T = 1), one run in eight the empty run (JOBS 0), JOBS (and JOB_BEATS) written, then START; the gaps
  * and the stall drawn when it starts. The streamer's inputs hold every run's beats in turn. Then, at random, either
  *   - STATUS is read every cycle (`rsp` always ready) until bit 0 reads 0, and BUSY_COUNTER read: it must equal the
  *     reads whose bit 0 was 1, and the cycles from the accepted start to the run's last output (E); STATUS then reads
  *     2; or
  *   - the next run's CSRs and START are written at once: the START is held (`req.ready` low) until E and taken in E +
  *     1, or at once when the run has already ended.
  * An empty run reads STATUS 2 and BUSY_COUNTER 0 from the cycle after its start.
  *
  * The outputs of all runs must equal the model at the fixture's types (its `accelerator.json` ports; each port's lanes
  * drawn and packed at its dtype's width): `ElementwiseModel.numpy` per lane for a SIMD unit, the lane sums of every T
  * inputs for a fold, each lane cast to the output type first; for a linked group (a dot product: a product per lane,
  * then the sum of a job's lanes and beats, T its JOB_BEATS) the sum over each job's T beats of a · b, wrapping at the
  * output type. A sliced input (D180) takes its run's beats d_in to a word, the run's last word short, and offers a
  * run's words only once its START is taken, as `main.c` starts a run's streamer after the last one ended (a word
  * offered in the run's last cycle would be taken and dropped at the next start); a packed output (D174) gives its
  * run's results d_out to a word, the last short, zero above.
  *
  * A group of two chains on a forked input (bicg_v3's one pass, D189): per run M rows in 1..16 (one run in eight the
  * empty run, a JOBS 0), the sum's T and the Mac's each a divisor of M, the chains' JOBS M / T so both heads fire M
  * times; A's reader gives row i to the product's and the Mac's a, p's its b, r's (broadcast, one lane a beat, sliced)
  * r_i to every lane of the Mac's b. The model: q, each job's Σ over its rows of A_i · p_i, and s, each job's Σ over
  * its rows of A_i · r_i lane by lane, wrapping at int32; q packed two to a word. Both outputs are checked, the run's
  * end E is the last chain's.
  */
class ShellRandomSpec extends AnyFlatSpec with ForgeTester {

  private val Runs = 6

  /** A run: its CSR writes by name, whether it is the empty run, per input its beats (each its stream's lanes), per
    * output its words.
    */
  private case class Run(csrs: Seq[(String, Long)], empty: Boolean, ins: Seq[Seq[Seq[BigInt]]], outs: Seq[Seq[BigInt]])

  /** bicg_v3's run of `rows` rows (0: the empty run), T `ts` on the sum and `tm` on the Mac: its CSRs, A, p and r, q
    * packed and s.
    */
  private def forkRun(spec: AcceleratorSpec, r: snax.forge.Rand, rows: Int, ts: Int, tm: Int): Run = {
    val n         = spec.unit("mul").port("a").lanes
    val (i8, i32) = (IntType("int8"), IntType("int32"))
    def draw(lanes: Int) = Seq.fill(rows)(Seq.fill(lanes)(r.word(8)))
    val (a, p, x) = (draw(n), draw(n), draw(1))
    def wide(v: BigInt) = ElementwiseModel.cast(v, i8, i32)
    val q      = (0 until rows).grouped(ts).toSeq.map { job =>
      LaneBits.dot(job.flatMap(a(_).map(wide)), job.flatMap(p(_).map(wide)), 32)
    }
    val s      = (0 until rows).grouped(tm).toSeq.map { job =>
      LaneBits
        .pack((0 until n).map(l => LaneBits.dot(job.map(i => wide(a(i)(l))), job.map(i => wide(x(i).head)), 32)), 32)
    }
    val jobs   = if (rows == 0) Seq(0L, 0L) else Seq((rows / ts).toLong, (rows / tm).toLong)
    val csrs   =
      Seq("JOBS_SUM" -> jobs(0), "JOBS_MAC" -> jobs(1), "SUM_JOB_BEATS" -> ts.toLong, "MAC_JOB_BEATS" -> tm.toLong)
    val packed = q.grouped(2).map(_.zipWithIndex.map { case (o, k) => o << (k * 32) }.sum).toSeq
    Run(csrs, rows == 0, Seq(a, p, x), Seq(packed, s))
  }

  behavior of "the shell wrapper under random data and stalls"

  for {
    f    <- RtlFixtures.driven
    seed <- Seeds()
  }
    it should s"give ${f.name}'s model outputs, count its busy cycles and hold a start while busy (seed $seed)" in
      Seeds.run(seed) { r =>
        val spec    = f.spec
        val unit    = spec.units.head
        val linked  = spec.groups.head.units.size > 1
        val fold    = linked || unit.datapath == "TreeAccumulate"
        val inPorts = spec.inputs.map { case (u, p) => spec.unit(u).port(p) }
        val outPort = spec.outputs.head match { case (u, p) => spec.unit(u).port(p) }
        val lanes   = inPorts.head.lanes
        val inTypes = inPorts.map(p => IntType(p.dtype))
        val outType = IntType(outPort.dtype)
        val opType  = unit.opType.map(IntType(_)).getOrElse(outType)
        val op      = unit.params.get("op").map(v => ElementwiseOp.byName(v.str))
        val dIn     = inPorts.map(_.dIn)
        val sliced  = dIn.exists(_ > 1)
        val forked  = spec.forks.nonEmpty

        val runs = if (forked) Seq.tabulate(Runs) { i =>
          val rows      = if (i > 0 && r.chance(0.125)) 0 else r.between(1, 16)
          def divisor() = r.pick((1 to (rows max 1)).filter(t => (rows max 1) % t == 0))
          forkRun(spec, r, rows, divisor(), divisor())
        }
        else
          Seq.tabulate(Runs) { i =>
            val jobs  = if (i > 0 && r.chance(0.125)) 0 else r.between(1, 32)
            val t     = if (fold) r.between(1, 8) else 1
            val n     = jobs * t
            val ins   = inTypes.map(ty => Seq.fill(n)(Seq.fill(lanes)(r.word(ty.width))))
            val outs  =
              if (linked)
                (0 until n).grouped(t max 1).toSeq.map { job =>
                  def at(i: Int) = job.flatMap(k => ins(i)(k).map(ElementwiseModel.cast(_, inTypes(i), outType)))
                  LaneBits.dot(at(0), at(1), outType.width)
                }
              else if (!fold)
                (0 until n).map { k =>
                  LaneBits.pack(
                    ins(0)(k).zip(ins(1)(k)).map { case (a, b) =>
                      ElementwiseModel.numpy(op.get, a, b, inTypes(0), inTypes(1), opType, outType)
                    },
                    outType.width
                  )
                }
              else
                ins(0)
                  .map(_.map(ElementwiseModel.cast(_, inTypes(0), outType)).sum)
                  .grouped(t max 1)
                  .map(_.sum & LaneBits.mask(outType.width))
                  .toSeq
            // a packed output: d_out results (each its lanes) to a word, the last short
            val bits  = outPort.lanes * outType.width
            val words = outs.grouped(outPort.dOut).map(_.zipWithIndex.map { case (o, k) => o << (k * bits) }.sum).toSeq
            Run(Seq("JOBS" -> jobs.toLong) ++ Option.when(fold)("JOB_BEATS" -> t.toLong), jobs == 0, ins, Seq(words))
          }

        var gap   = 0.0
        var stall = 0.0
        // each run's words per input (d_in beats a word, the last short) and how many runs' are offered
        def words(run: Run, k: Int): Seq[BigInt] =
          run.ins(k).grouped(dIn(k)).map(g => LaneBits.pack(g.flatten, inTypes(k).width)).toSeq
        var offered = 0
        def allowed(k: Int) = runs.take(offered).map(words(_, k).size).sum
        lazy val sources: Seq[Source] = inTypes.indices.map { k =>
          val beats = runs.flatMap(words(_, k)).toIndexedSeq
          new Source(beats.size, beats, _ => !r.chance(gap) && (!sliced || sources(k).taken < allowed(k)))
        }
        test(new ShellHarness(spec)) { dut =>
          val d                 = new ShellDriver(dut, sources, spec.outputs.map(_ => new Sink(_ => !r.chance(stall))))
          val (status, counter) = (dut.csr("STATUS"), dut.csr("BUSY_COUNTER"))
          def config(run: Run): Unit     = run.csrs.foreach { case (name, v) => d.write(dut.csr(name), v) }
          // the outputs given by the runs up to i, per output
          def given(i:    Int): Seq[Int] = spec.outputs.indices.map(o => runs.take(i + 1).map(_.outs(o).size).sum)
          def outputs: Seq[Int] = d.outputs.map(_.size)

          /** START, held until taken: the cycle it was taken; the next run's gaps from there. */
          def begin(): Int = {
            val taken = d.write(dut.csr("START"), 1)
            gap   = r.probability()
            stall = r.probability()
            offered += 1
            taken
          }

          config(runs.head)
          var accepted = begin()
          for ((run, i) <- runs.zipWithIndex) {
            val clue = s"run $i: ${run.csrs.map { case (k, v) => s"$k $v" }.mkString(", ")}, gap $gap, stall $stall"
            val done = given(i)
            if (run.empty) {
              assert(d.read(status) == 2, s"$clue: STATUS after the empty run")
              assert(d.read(counter) == 0, s"$clue: BUSY_COUNTER after the empty run")
              if (i + 1 < Runs) { config(runs(i + 1)); accepted = begin() }
            } else if (i + 1 < Runs && r.chance(0.5)) {
              config(runs(i + 1))
              val offered = d.cycle
              val taken   = begin()
              assert(outputs == done, s"$clue: next start taken at $taken with outputs left")
              assert(
                taken == math.max(offered, d.lastOut + 1),
                s"$clue: start offered $offered, taken $taken, E ${d.lastOut}"
              )
              assert(d.held.filter(_ >= offered) == (offered until taken), s"$clue: start not held in every cycle to E")
              accepted = taken
            } else {
              var ones  = 0
              var s     = d.read(status)
              while ((s & 1) == 1) { ones += 1; s = d.read(status) }
              assert(s == 2, s"$clue: STATUS $s once not busy: done")
              val count = d.read(counter)
              assert(outputs == done, s"$clue: busy fell with outputs left")
              assert(count == ones, s"$clue: BUSY_COUNTER $count, STATUS read busy $ones times")
              assert(
                count == d.lastOut - accepted,
                s"$clue: BUSY_COUNTER $count, start taken $accepted, E ${d.lastOut}"
              )
              if (i + 1 < Runs) {
                config(runs(i + 1))
                accepted = begin()
              }
            }
          }
          for (o        <- spec.outputs.indices) {
            val want = runs.flatMap(_.outs(o))
            for ((got, k) <- d.outputs(o).zipWithIndex)
              assert(got == want(k), s"output $o, word $k: $got, the model ${want(k)}")
            assert(d.outputs(o).size == want.size, s"every word of output $o")
          }
          assert(d.taken == sources.map(_.count), "every input beat taken")
        }
      }
}
