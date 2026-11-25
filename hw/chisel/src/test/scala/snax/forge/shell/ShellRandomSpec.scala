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
  * output type. A sliced input takes its run's beats d_in to a word, the run's last word short, and offers a run's
  * words only once its START is taken, as `main.c` starts a run's streamer after the last one ended (a word offered in
  * the run's last cycle would be taken and dropped at the next start); a packed output gives its run's results d_out to
  * a word, the last short, zero above.
  *
  * A group of two chains on a forked input (bicg_v3's one pass): per run M rows in 1..16 (one run in eight the empty
  * run, a JOBS 0), the sum's T and the Mac's each a divisor of M, the chains' JOBS M / T so both heads fire M times;
  * A's reader gives row i to the product's and the Mac's a, p's its b, r's (broadcast, one lane a beat, sliced) r_i to
  * every lane of the Mac's b. The model: q, each job's Σ over its rows of A_i · p_i, and s, each job's Σ over its rows
  * of A_i · r_i lane by lane, wrapping at int32; q packed two to a word. Both outputs are checked, the run's end E is
  * the last chain's.
  *
  * The same with s's sums in a bank (bicg_v4): per run ROWS in 1..R, the Mac's T a multiple of it, the heads' firings n
  * a multiple of that and of the sum's T (its JOB_BEATS, or its fixed T); r is read once a row, a beat on each firing
  * where n mod ROWS = 0, and the Mac's sums are ROWS rows, firing f adding into row f mod ROWS, each job giving its
  * rows in order. With q carried (bicg_v4_1) the sum is seeded once a job from its seed's reader, the seed 0 in the sum
  * on its first CARRY jobs (CARRY in 0..its JOBS). mvt's one pass (mvt_v6): the sum seeded every job (x1), and the bank
  * seeded a beat a row (x2), each job's row ρ starting from its ρ-th seed beat. atax's one group (atax_v4): t over a
  * held link into the bank and A forked to it through a fork link (`ataxRun`).
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

  /** bicg_v4's run (the bank, its held r, and with a carried q bicg_v4_1's; with both seeds mvt_v6's), or the empty
    * run: its CSRs, every input's beats in the spec's order and q packed and s. The units by their roles: the product
    * (`SimdLanes`), the sum (`TreeAccumulate`) and the bank (a `Mac` of rows above 1).
    */
  private def bankRun(spec: AcceleratorSpec, r: snax.forge.Rand, empty: Boolean): Run = {
    val mac           = spec.units.find(_.rows > 1).get
    val sum           = spec.units.find(_.datapath == "TreeAccumulate").get
    val mul           = spec.units.find(_.datapath == "SimdLanes").get
    val (i8, i32)     = (IntType("int8"), IntType("int32"))
    val w             = mac.port("a").lanes
    val carried       = sum.ports.exists(_._2.carry)
    val sumSeeded     = sum.inPorts.contains("seed")
    val macSeeded     = mac.inPorts.contains("seed")
    var (rows, tm, n) = (1, 1, 1)
    do {
      rows = r.between(1, mac.rows)
      tm   = rows * r.between(1, 4)
      n    = tm * r.between(1, 3)
    } while (sum.fixedT.exists(t => t != 1 && n % t != 0))
    val ts            = sum.fixedT.filter(_ != 1).getOrElse(r.pick((1 to n).filter(n % _ == 0)))
    val js            = n / ts
    val carry         = if (carried) r.between(0, js) else 0
    def draw(k: Int, lanes: Int, bits: Int) = if (empty) Nil else Seq.fill(k)(Seq.fill(lanes)(r.word(bits)))
    val (a, p, x, seeds) = (draw(n, w, 8), draw(n, w, 8), draw(n / rows, 1, 8), draw(js, 1, 32))
    val rowSeeds         = draw(n / tm * rows, w, 32) // the bank's seed, a beat a row of each job
    def wide(v: BigInt) = ElementwiseModel.cast(v, i8, i32)
    val q        =
      if (empty) Nil
      else
        (0 until n).grouped(ts).toSeq.zipWithIndex.map { case (job, j) =>
          val dot = LaneBits.dot(job.flatMap(a(_).map(wide)), job.flatMap(p(_).map(wide)), 32)
          (dot + (if (sumSeeded && (!carried || j >= carry)) seeds(j).head else BigInt(0))) & LaneBits.mask(32)
        }
    val s        =
      if (empty) Nil
      else
        (0 until n).grouped(tm).toSeq.zipWithIndex.flatMap { case (job, j) => // its row f mod ROWS
          (0 until rows).map { row =>
            val fs = job.filter(_ % rows == row)
            LaneBits.pack(
              (0 until w).map { l =>
                val dot = LaneBits.dot(fs.map(f => wide(a(f)(l))), fs.map(f => wide(x(f / rows).head)), 32)
                (dot + (if (macSeeded) rowSeeds(j * rows + row)(l) else BigInt(0))) & LaneBits.mask(32)
              },
              32
            )
          }
        }
    val jobs     = if (empty) Seq(0L, 0L) else Seq(js.toLong, (n / tm).toLong)
    val (su, mu) = (sum.name.toUpperCase, mac.name.toUpperCase)
    val csrs     = Seq(s"JOBS_$su" -> jobs(0), s"JOBS_$mu" -> jobs(1)) ++
      Option.when(sum.fixedT.isEmpty)(s"${su}_JOB_BEATS" -> ts.toLong) ++
      Seq(s"${mu}_JOB_BEATS" -> tm.toLong, s"${mu}_ROWS" -> rows.toLong) ++
      Option.when(carried)(s"${su}_CARRY" -> carry.toLong)
    val beats    = Map(
      (mul.name, "a")    -> a,
      (mul.name, "b")    -> p,
      (sum.name, "seed") -> seeds,
      (mac.name, "b")    -> x,
      (mac.name, "seed") -> rowSeeds
    )
    val packed   = q.grouped(2).map(_.zipWithIndex.map { case (o, k) => o << (k * 32) }.sum).toSeq
    Run(csrs, empty, spec.inputs.map(beats), spec.outputs.map(Map((sum.name, "out") -> packed, (mac.name, "out") -> s)))
  }

  /** atax's one group (atax_v4): its run, or the empty run. Per run ROWS in 1..R, the sum's T = ROWS (a row of A, so
    * the product and the bank fire alike, as LOWER makes them), the bank's T = M · ROWS for M rows of A in 1..4, JOBS
    * 1..2; A's reader gives beat f (row f / ROWS, slice f mod ROWS) to the product's a and, through the fork link (or
    * its own reader), the bank's; x's a beat a firing on the product's b. The model: t_i = Σ over row i's beats of A ·
    * x (int8 to int32, wrapping), held over the link for the row and on every lane of the bank's b; the bank's row c of
    * a job Σ over its rows i of A_(i · ROWS + c) · t_i lane by lane at int32, each job giving its rows in order.
    */
  private def ataxRun(spec: AcceleratorSpec, r: snax.forge.Rand, empty: Boolean): Run = {
    val mac       = spec.units.find(_.rows > 1).get
    val sum       = spec.units.find(_.datapath == "TreeAccumulate").get
    val mul       = spec.units.find(_.datapath == "SimdLanes").get
    val (i8, i32) = (IntType("int8"), IntType("int32"))
    val w         = mul.port("a").lanes
    val rows      = r.between(1, mac.rows)
    val m         = r.between(1, 4)
    val js        = if (empty) 0 else r.between(1, 2)
    val n         = js * m * rows
    def draw(lanes: Int) = Seq.fill(n)(Seq.fill(lanes)(r.word(8)))
    val (a, x) = (draw(w), draw(w))
    def wide(v: BigInt) = ElementwiseModel.cast(v, i8, i32)
    val t        = (0 until n).grouped(rows).toSeq.map { row =>
      LaneBits.dot(row.flatMap(a(_).map(wide)), row.flatMap(x(_).map(wide)), 32)
    }
    val y        = (0 until n).grouped(m * rows).toSeq.flatMap { job =>
      (0 until rows).map { c =>
        val fs = job.filter(_ % rows == c)
        LaneBits.pack((0 until w).map(l => LaneBits.dot(fs.map(f => wide(a(f)(l))), fs.map(f => t(f / rows)), 32)), 32)
      }
    }
    val (su, mu) = (sum.name.toUpperCase, mac.name.toUpperCase)
    val csrs     =
      Seq("JOBS" -> js.toLong, s"${su}_JOB_BEATS" -> rows.toLong, s"${mu}_JOB_BEATS" -> (m * rows).toLong) ++
        Seq(s"${mu}_ROWS" -> rows.toLong)
    val beats    = Map((mul.name, "a") -> a, (mul.name, "b") -> x, (mac.name, "a") -> a)
    Run(csrs, empty, spec.inputs.map(beats), Seq(y))
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

        val banked = spec.units.exists(_.rows > 1)
        val held   = spec.groups.exists(g => g.links.exists(l => spec.unit(l.to._1).port(l.to._2).hold))

        val runs =
          if (held) Seq.tabulate(Runs)(i => ataxRun(spec, r, i > 0 && r.chance(0.125)))
          else if (banked) Seq.tabulate(Runs)(i => bankRun(spec, r, i > 0 && r.chance(0.125)))
          else if (forked) Seq.tabulate(Runs) { i =>
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
              val words =
                outs.grouped(outPort.dOut).map(_.zipWithIndex.map { case (o, k) => o << (k * bits) }.sum).toSeq
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
