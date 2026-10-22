package snax.forge.samples

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Rand, Seeds, Sink, Source}

/** `Accumulator` (64-bit elements and sum) under random data, configurations, input gaps and output stalls, against a
  * model written from its docstring: a run of n inputs and T inputs per sum gives the lane sums of the inputs, every T
  * of them summed, the last sum ending at the n-th input, all wrapping at 64 bits.
  *
  * Per seed, `Instances` blocks with `lanes` drawn from {1, 2, 4, 8}, each given `Runs` runs back to back: n in 1..64,
  * T in 1..n (dividing n in half the runs), gap and stall drawn per run. The source holds n + `Extra` beats. Checked
  * every cycle: each output equal to the model's next, no beat taken after the n-th, busy from the cycle after the
  * start until the cycle the last output fires and low the cycle after.
  */
class AccumulatorRandomSpec extends AnyFlatSpec with ForgeTester {

  private val Width     = 64
  private val Mask      = (BigInt(1) << Width) - 1
  private val Instances = 3
  private val Runs      = 5
  private val Extra     = 3

  /** n and T, T dividing n when `divides`, else not (n drawn again until it has such a T). */
  private def config(r: Rand, divides: Boolean): (Int, Int) = {
    val n  = r.between(1, 64)
    val ts = (1 to n).filter(t => (n % t == 0) == divides)
    if (ts.isEmpty) config(r, divides) else (n, r.pick(ts))
  }

  /** The model: the beats' lane sums, every T of the first n summed. */
  private def model(beats: Seq[Seq[BigInt]], n: Int, t: Int): Seq[BigInt] =
    beats.take(n).map(_.sum).grouped(t).map(_.sum & Mask).toSeq

  behavior of "Accumulator under random data and stalls"

  for (seed <- Seeds())
    it should s"give the model's sums, take n beats and end busy after the last output (seed $seed)" in
      Seeds.run(seed) { r =>
        for (i <- 0 until Instances) {
          val lanes = r.pick(Seq(1, 2, 4, 8))
          test(new Accumulator(Width, Width, lanes)) { dut =>
            dut.io.start.poke(false.B)
            for (run <- 0 until Runs) {
              val (n, t)  = config(r, divides = run % 2 == 0)
              val beats   = Seq.fill(n + Extra)(Seq.fill(lanes)(r.word(Width)))
              val packed  = beats.map(_.zipWithIndex.map { case (v, l) => v << (Width * l) }.sum)
              val (g, s)  = (r.probability(), r.probability())
              val src     = Source.random(packed, g, r)
              val snk     = Sink.random(s, r)
              val want    = model(beats, n, t)
              val clue    = s"lanes $lanes, run $run: n $n, T $t, gap $g, stall $s"
              def drive() = {
                src.cycle()
                dut.io.in.valid.poke(src.valid.B)
                for (l <- 0 until lanes) dut.io.in.bits(l).poke(((src.bits >> (Width * l)) & Mask).U)
                dut.io.out.ready.poke(snk.cycle().B)
              }

              drive()
              assert(!dut.io.busy.peek().litToBoolean, s"$clue: busy at start")
              dut.io.start.poke(true.B)
              dut.io.loopCount.poke(n.U)
              dut.io.len.poke(t.U)
              src.fired(dut.io.in.ready.peek().litToBoolean)
              dut.clock.step()
              dut.io.start.poke(false.B)

              var ended = false
              var cycle = 1
              while (!ended) {
                drive()
                val busy = dut.io.busy.peek().litToBoolean
                if (snk.got.size == want.size) {
                  assert(!busy, s"$clue, cycle $cycle: busy after the last output fired")
                  ended = true
                } else {
                  assert(busy, s"$clue, cycle $cycle: not busy with ${want.size - snk.got.size} outputs to go")
                  assert(cycle < 200 * n, s"$clue: no end after $cycle cycles")
                  val before = src.taken
                  if (src.fired(dut.io.in.ready.peek().litToBoolean))
                    assert(before < n, s"$clue, cycle $cycle: beat $before taken after the n-th")
                  if (snk.fired(dut.io.out.valid.peek().litToBoolean, dut.io.out.bits.peek().litValue)) {
                    val k = snk.got.size - 1
                    assert(snk.got(k) == want(k), s"$clue, cycle $cycle: output $k ${snk.got(k)}, the model ${want(k)}")
                  }
                  dut.clock.step()
                  cycle += 1
                }
              }
              assert(src.taken == n, s"$clue: ${src.taken} beats taken")
            }
          }
        }
      }
}
