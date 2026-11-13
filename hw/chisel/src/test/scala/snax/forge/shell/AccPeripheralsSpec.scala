package snax.forge.shell

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, LaneBits, Seeds}

/** `<tag>_peripherals` (`AccPeripherals`) against a model of its docstring. Per seed, `Layouts` layouts of 1 to 3
  * groups, each BUSY_COUNTER of 1, 2, 3 or 8 bits, driven for `Cycles` cycles as a group's control unit drives it:
  * after a random idle gap an accepted start in a (busy low), then busy from a + 1 to E for a run of 0 to 20 cycles (0
  * the empty run, E = a), then idle again; a start may come in E + 1. Checked every cycle: each BUSY_COUNTER against
  * the model (cleared by the accepted start, + 1 per busy cycle, held, saturating at 2^w - 1); after every run, in E +
  * 1, that it holds min(E - a, 2^w - 1), and in a + 1 that the start cleared it.
  */
class AccPeripheralsSpec extends AnyFlatSpec with ForgeTester {

  private val Layouts = 4
  private val Cycles  = 400
  private var checked = 0 // runs whose E - a was checked, every seed

  behavior of "<tag>_peripherals' BUSY_COUNTER"

  for (seed <- Seeds())
    it should s"count each group's busy cycles from its accepted start, saturating (seed $seed)" in Seeds.run(seed) {
      r =>
        for (l <- 0 until Layouts) {
          val widths = Seq.fill(r.between(1, 3))(r.pick(Seq(1, 2, 3, 8)))
          val n      = widths.size
          val pStart = r.pick(Seq(0.05, 0.2, 0.5))
          test(new AccPeripherals("t", widths.zipWithIndex.map { case (w, g) => s"g$g" -> w })) { dut =>
            val model = Array.fill(n)(BigInt(0))
            val left  = Array.fill(n)(-1) // busy cycles left in the run, -1 idle
            val start = Array.fill(n)(-1) // the run's a
            val ended = Array.fill(n)(-1) // the run's E, checked in E + 1
            for (c <- 0 until Cycles) {
              val clue = s"layout $l (widths ${widths.mkString(" ")}), cycle $c"
              // E + 1: the run's count; a + 1: the start cleared it
              for (g <- 0 until n) {
                val max = LaneBits.mask(widths(g))
                val got = dut.io.busyCounter(g).peek().litValue
                assert(got == model(g), s"$clue, group $g: BUSY_COUNTER $got, the model ${model(g)}")
                if (ended(g) == c - 1) {
                  assert(got == (BigInt(ended(g) - start(g)) min max), s"$clue, group $g: E - a after the run")
                  checked += 1
                }
                if (start(g) == c - 1) assert(got == 0, s"$clue, group $g: cleared by the start")
              }
              val starts = (0 until n).map(g => left(g) < 0 && r.chance(pStart))
              val busy = (0 until n).map(g => left(g) > 0)
              for (g <- 0 until n) {
                dut.io.start(g).poke(starts(g).B)
                dut.io.busy(g).poke(busy(g).B)
                val max = LaneBits.mask(widths(g))
                if (starts(g)) model(g) = 0 else if (busy(g) && model(g) < max) model(g) += 1
                if (starts(g)) {
                  start(g) = c; left(g) = r.pick(Seq(0, 1, 2, 3, 7, 8, 9, 20))
                  if (left(g) == 0) { ended(g) = c; left(g) = -1 }
                } else if (busy(g)) {
                  left(g) -= 1
                  if (left(g) == 0) { ended(g) = c; left(g) = -1 }
                }
              }
              dut.clock.step()
            }
          }
        }
    }

  it should "have checked enough runs" in {
    info(s"$checked runs checked")
    assert(checked >= 150 * Seeds().size, s"$checked runs checked")
  }
}
