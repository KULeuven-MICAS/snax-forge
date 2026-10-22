package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{Blocks, ForgeTester, Rand, Seeds, Sink, Source}
import snax.forge.elementwise.ElementwiseModel

/** Each fixture's `<tag>_shell_wrapper` (built as `Emit --spec` builds it: vecadd and vecmul, accumulate at W = 1, 4
  * and 8) under random data, configurations, streamer gaps and output stalls, through its CSR port as `main.c` drives
  * it.
  *
  * Per seed, `Runs` runs back to back on one wrapper. Per run: n in 1..64 (and for the accumulator T in 1..n, dividing
  * n in half the runs), the fields written in CSR order, then start; the gaps and the stall drawn when it starts. The
  * streamer's inputs hold every run's beats in turn. Then, at random, either
  *   - busy is read every cycle (`rsp` always ready) until it reads 0, and the counter read: it must equal the reads of
  *     busy that gave 1, and the cycles from the accepted start to the run's last output (E); or
  *   - the next run's fields and start are written at once: the start is held (`req.ready` low) until E and taken in E
  *     + 1, or at once when the run has already ended.
  *
  * The outputs of all runs must equal the model: `ElementwiseModel` per lane for the elementwise blocks, the lane sums
  * every T inputs for the accumulator (as `AccumulatorRandomSpec`'s).
  */
class ShellRandomSpec extends AnyFlatSpec with ForgeTester {

  private val Width = 64
  private val Mask  = (BigInt(1) << Width) - 1
  private val Runs  = 6

  private case class Run(n: Int, t: Int, ins: Seq[Seq[Seq[BigInt]]], outs: Seq[BigInt])

  private def pack(lanes: Seq[BigInt]): BigInt = lanes.zipWithIndex.map { case (v, l) => v << (Width * l) }.sum

  behavior of "the shell wrapper under random data and stalls"

  for {
    f    <- RtlFixtures.all
    seed <- Seeds()
  }
    it should s"give ${f.name}'s model outputs, count its busy cycles and hold a start while busy (seed $seed)" in
      Seeds.run(seed) { r =>
        val unit            = f.spec.units.head
        val block           = Blocks(unit.block)
        val lanes           = Blocks.Params(unit.params).int("lanes")
        val dirs            = block.ports.toMap
        val numIns          = unit.ports.count { case (_, b) => dirs(b) == "in" }
        val start           = unit.rw.size
        val (busy, counter) = (start + 1, start + 2)
        val elementwise     = unit.block == "ElementwiseTiledSpatial"

        val runs = Seq.tabulate(Runs) { i =>
          val n    = r.between(1, 64)
          val ts   = (1 to n).filter(t => (n % t == 0) == (i % 2 == 0))
          val t    = if (elementwise || ts.isEmpty) 1 else r.pick(ts)
          val ins  = Seq.fill(numIns)(Seq.fill(n)(Seq.fill(lanes)(r.word(Width))))
          val outs =
            if (elementwise) {
              val op = unit.tie.toMap.apply("opSel")
              (0 until n).map(k =>
                pack(ins(0)(k).zip(ins(1)(k)).map { case (a, b) => ElementwiseModel(op, a, b, Width) })
              )
            } else ins(0).map(_.sum).grouped(t).map(_.sum & Mask).toSeq
          Run(n, t, ins, outs)
        }

        var gap     = 0.0
        var stall   = 0.0
        val sources = (0 until numIns).map { k =>
          val beats = runs.flatMap(_.ins(k)).map(pack).toIndexedSeq
          new Source(beats.size, beats, () => !r.chance(gap))
        }
        test(new ShellHarness(f.spec)) { dut =>
          val d = new ShellDriver(dut, sources, Seq(new Sink(() => !r.chance(stall))))
          def config(run: Run): Unit =
            unit.rw.zipWithIndex.foreach { case ((pin, _), i) => d.write(i, if (pin == "len") run.t else run.n) }

          /** Start, held until taken: the cycle it was taken; the next run's gaps from there. */
          def begin(): Int = {
            val taken = d.write(start, 1)
            gap   = r.probability()
            stall = r.probability()
            taken
          }

          config(runs.head)
          var accepted = begin()
          for ((run, i) <- runs.zipWithIndex) {
            val clue = s"run $i: n ${run.n}, T ${run.t}, gap $gap, stall $stall"
            val done = runs.take(i + 1).map(_.outs.size).sum
            if (i + 1 < Runs && r.chance(0.5)) {
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
              while (d.read(busy) == 1) ones += 1
              val count = d.read(counter)
              assert(d.outputs.head.size == done, s"$clue: busy fell with outputs left")
              assert(count == ones, s"$clue: counter $count, busy read 1 $ones times")
              assert(count == d.lastOut - accepted, s"$clue: counter $count, start taken ${accepted}, E ${d.lastOut}")
              if (i + 1 < Runs) {
                config(runs(i + 1))
                accepted = begin()
              }
            }
          }
          val want = runs.flatMap(_.outs)
          for ((got, k) <- d.outputs.head.zipWithIndex)
            assert(got == want(k), s"output $k: $got, the model ${want(k)}")
          assert(d.taken == sources.map(_.count), "every input beat taken")
        }
      }
}
