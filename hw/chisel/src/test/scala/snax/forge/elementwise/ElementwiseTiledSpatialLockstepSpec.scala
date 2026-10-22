package snax.forge.elementwise

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{Blocks, SnaxUnitCases}
import snax.forge.SnaxUnitCases.Start
import snax.forge.shell.RtlFixtures

/** The lockstep half of the unit cross-check: ElementwiseTiledSpatial driven through snax-gvsoc's `SnaxUnit` cases.
  *
  * Every case with `wait lockstep` (`SnaxUnitCases`, read from `SNAX_GVSOC_UNIT_CASES`) drives the Chisel block the
  * same way, cycle by cycle: each input's valid and the output's ready as the case line gives them, a start poked in
  * the cycle the case offers it (taken by the next clock edge, so busy from the cycle after), and fire (both operand
  * transfers), push (`out.fire`) and busy compared every cycle. A start's answer is checked against the block's state
  * when it comes: `taken` idle with a count, `ignored` idle with count 0, `refused` busy. The `expect run` lines
  * (SnaxUnit's trace) have no Chisel counterpart here; the pipelined folds are `AccumulatorRateSpec`'s. Without the
  * variable the test is canceled.
  */
class ElementwiseTiledSpatialLockstepSpec extends AnyFlatSpec with ChiselScalatestTester {

  private val Env = SnaxUnitCases.Env

  private def cases = SnaxUnitCases.all.map(_.filter(_.waitKind == "lockstep"))

  behavior of "ElementwiseTiledSpatial against SnaxUnit's lockstep cases"

  /** The block's configurations: 64-bit, 4 lanes, add; then each fixture's of this block (GEN1, `RtlFixtures`). */
  private val configs: Seq[(String, () => ElementwiseTiledSpatial)] =
    (
      "w64 n4 add",
      () => new ElementwiseTiledSpatial(dataWidth = 64, lanes = 4, supportedOps = Seq(ElementwiseOp.Add))
    ) +:
      RtlFixtures.all.map(f => f -> f.spec.units.head).collect {
        case (f, u) if u.block == "ElementwiseTiledSpatial" =>
          val p = Blocks.Params(u.params)
          f.name -> (() =>
            new ElementwiseTiledSpatial(
              p.int("dataWidth"),
              p.int("lanes"),
              p.int("loopCountWidth"),
              p.ops("supportedOps"),
              p.bool("signed")
            )
          )
      }

  for ((config, block) <- configs)
    it should s"fire, push and be busy in the cycles every lockstep case gives ($config)" in {
      val all = cases.getOrElse(cancel(s"$Env is not set (snax-gvsoc's tests/snax/unit/cases)"))
      assert(all.nonEmpty, s"no lockstep case in ${sys.env(Env)}")
      for (c <- all) {
        require(c.cycles.forall(_.valids.size == 2), s"${c.name}: ElementwiseTiledSpatial has two inputs")
        test(block()) { dut =>
          for ((cyc, i) <- c.cycles.zipWithIndex) {
            val clue = s"${c.name}, cycle $i"
            val busy = dut.io.busy.peek().litToBoolean
            c.starts.get(i) match {
              case Some(st @ Start(_, answer)) =>
                val n   = c.word(st, "count", 0)
                val got = if (busy) "refused" else if (n == 0) "ignored" else "taken"
                assert(got == answer, s"$clue: start $n $got, the case says $answer")
                dut.io.start.poke(true.B)
                dut.io.loopCount.poke(n.U)
              case None                        =>
                dut.io.start.poke(false.B)
            }
            dut.io.a.valid.poke(cyc.valids(0).B)
            dut.io.b.valid.poke(cyc.valids(1).B)
            dut.io.out.ready.poke(cyc.ready.B)
            val fire = dut.io.a.valid.peek().litToBoolean && dut.io.a.ready.peek().litToBoolean &&
              dut.io.b.valid.peek().litToBoolean && dut.io.b.ready.peek().litToBoolean
            val push = dut.io.out.valid.peek().litToBoolean && cyc.ready
            assert(fire == cyc.fire, s"$clue: fire $fire, the case says ${cyc.fire}")
            assert(push == cyc.push, s"$clue: push $push, the case says ${cyc.push}")
            assert(busy == cyc.busy, s"$clue: busy $busy, the case says ${cyc.busy}")
            dut.clock.step()
          }
        }
      }
    }
}
