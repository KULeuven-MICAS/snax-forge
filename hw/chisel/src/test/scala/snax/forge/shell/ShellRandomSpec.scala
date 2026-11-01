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
  * elementwise unit has T = 1), one run in eight the empty run (JOBS 0), JOBS (and JOB_BEATS) written, then START; the
  * gaps and the stall drawn when it starts. The streamer's inputs hold every run's beats in turn. Then, at random,
  * either
  *   - STATUS is read every cycle (`rsp` always ready) until bit 0 reads 0, and COUNTER read: it must equal the reads
  *     whose bit 0 was 1, and the cycles from the accepted start to the run's last output (E); STATUS then reads 2; or
  *   - the next run's CSRs and START are written at once: the START is held (`req.ready` low) until E and taken in E +
  *     1, or at once when the run has already ended.
  * An empty run reads STATUS 2 and COUNTER 0 from the cycle after its start.
  *
  * The outputs of all runs must equal the model at the fixture's types (its `accelerator.json` ports; each port's lanes
  * drawn and packed at its dtype's width): `ElementwiseModel.numpy` per lane for an elementwise unit, the lane sums of
  * every T inputs for a fold, each lane cast to the output type first; for a linked group (a dot product: a product per
  * lane, then the sum of a job's lanes and beats, T its JOB_BEATS) the sum over each job's T beats of a · b, wrapping
  * at the output type.
  */
class ShellRandomSpec extends AnyFlatSpec with ForgeTester {

  private val Runs = 6

  private case class Run(jobs: Int, t: Int, ins: Seq[Seq[Seq[BigInt]]], outs: Seq[BigInt]) {
    def empty: Boolean = jobs == 0
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

        val runs = Seq.tabulate(Runs) { i =>
          val jobs = if (i > 0 && r.chance(0.125)) 0 else r.between(1, 32)
          val t    = if (fold) r.between(1, 8) else 1
          val n    = jobs * t
          val ins  = inTypes.map(ty => Seq.fill(n)(Seq.fill(lanes)(r.word(ty.width))))
          val outs =
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
          Run(jobs, t, ins, outs)
        }

        var gap     = 0.0
        var stall   = 0.0
        val sources = inTypes.indices.map { k =>
          val beats = runs.flatMap(_.ins(k)).map(LaneBits.pack(_, inTypes(k).width)).toIndexedSeq
          new Source(beats.size, beats, _ => !r.chance(gap))
        }
        test(new ShellHarness(spec)) { dut =>
          val d                 = new ShellDriver(dut, sources, Seq(new Sink(_ => !r.chance(stall))))
          val (status, counter) = (dut.csr("STATUS"), dut.csr("COUNTER"))
          def config(run: Run): Unit = {
            d.write(dut.csr("JOBS"), run.jobs)
            if (fold) d.write(dut.csr("JOB_BEATS"), run.t)
          }

          /** START, held until taken: the cycle it was taken; the next run's gaps from there. */
          def begin(): Int = {
            val taken = d.write(dut.csr("START"), 1)
            gap   = r.probability()
            stall = r.probability()
            taken
          }

          config(runs.head)
          var accepted = begin()
          for ((run, i) <- runs.zipWithIndex) {
            val clue = s"run $i: J ${run.jobs}, T ${run.t}, gap $gap, stall $stall"
            val done = runs.take(i + 1).map(_.outs.size).sum
            if (run.empty) {
              assert(d.read(status) == 2, s"$clue: STATUS after the empty run")
              assert(d.read(counter) == 0, s"$clue: COUNTER after the empty run")
              if (i + 1 < Runs) { config(runs(i + 1)); accepted = begin() }
            } else if (i + 1 < Runs && r.chance(0.5)) {
              config(runs(i + 1))
              val offered = d.cycle
              val taken   = begin()
              assert(d.outputs.head.size == done, s"$clue: next start taken at $taken with outputs left")
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
              assert(d.outputs.head.size == done, s"$clue: busy fell with outputs left")
              assert(count == ones, s"$clue: COUNTER $count, STATUS read busy $ones times")
              assert(count == d.lastOut - accepted, s"$clue: COUNTER $count, start taken $accepted, E ${d.lastOut}")
              if (i + 1 < Runs) {
                config(runs(i + 1))
                accepted = begin()
              }
            }
          }
          val want = runs.flatMap(_.outs)
          for ((got, k) <- d.outputs.head.zipWithIndex)
            assert(got == want(k), s"output $k: $got, the model ${want(k)}")
          assert(d.outputs.head.size == want.size, "every output")
          assert(d.taken == sources.map(_.count), "every input beat taken")
        }
      }
}
