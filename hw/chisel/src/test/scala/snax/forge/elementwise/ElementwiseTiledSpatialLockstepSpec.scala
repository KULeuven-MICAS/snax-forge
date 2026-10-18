package snax.forge.elementwise

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._

/** The lockstep half of GEN3's unit cross-check: ElementwiseTiledSpatial driven through snax-gvsoc's `SnaxUnit` cases.
  *
  * snax-gvsoc's case files (`tests/snax/unit/cases/`, one `.txt` each) are worked by hand from `SnaxUnit`'s rule
  * (`snax/snax_unit.hpp`); its `tb.cpp` checks the rule against them. Here every case with `wait lockstep` drives the
  * Chisel block the same way, cycle by cycle: each input's valid and the output's ready as the case line gives them, a
  * start poked in the cycle the case offers it (taken by the next clock edge, so busy from the cycle after), and fire
  * (both operand transfers), push (`out.fire`) and busy compared every cycle. A start's answer is checked against the
  * block's state when it comes: `taken` idle with a count, `ignored` idle with count 0, `refused` busy. The `expect
  * run` lines (SnaxUnit's trace) and the pipelined cases have no Chisel counterpart here.
  *
  * The cases are read from the folder `SNAX_GVSOC_UNIT_CASES` names (e.g. `gvsoc/snax-gvsoc/tests/snax/unit/cases`);
  * without it the test is canceled, since CI never fetches snax-gvsoc (D128).
  */
class ElementwiseTiledSpatialLockstepSpec extends AnyFlatSpec with ChiselScalatestTester {

  private val Env = "SNAX_GVSOC_UNIT_CASES"

  private case class Start(count: Int, answer: String)
  private case class Cycle(valids: Seq[Boolean], ready: Boolean, fire: Boolean, push: Boolean, busy: Boolean)
  private case class Case(name: String, waitKind: String, starts: Map[Int, Start], cycles: Seq[Cycle])

  /** One case file in tb.cpp's format: header lines, then `cycles` and one line per cycle; a file of `like` lines only
    * (`pipelined_l0`, the lockstep cases rerun as pipelined) has no cycles of its own and gives None.
    */
  private def parse(path: Path): Option[Case] = {
    val lines    = Files.readAllLines(path).asScala.map(_.split("#")(0).trim).filter(_.nonEmpty).toSeq
    val at       = lines.indexOf("cycles")
    var waitKind = "lockstep"
    var inputs   = 2
    var starts   = Map.empty[Int, Start]
    if (at < 0) {
      require(lines.exists(_.startsWith("like ")), s"$path: no `cycles` line and no `like` line")
      return None
    }
    for (line <- lines.take(at)) line.split("\\s+").toList match {
      case "wait" :: w :: Nil                 => waitKind = w
      case "inputs" :: n :: Nil               => inputs   = n.toInt
      case "start" :: c :: n :: answer :: Nil => starts += c.toInt -> Start(n.toInt, answer)
      case _                                  => // latency, ii, like, expect: not read here
    }
    val cycles = lines.drop(at + 1).map { line =>
      val v = line.split("\\s+").map(_ == "1").toSeq
      require(v.size == inputs + 4, s"$path: `$line` has ${v.size} columns, not ${inputs + 4}")
      Cycle(v.take(inputs), v(inputs), v(inputs + 1), v(inputs + 2), v(inputs + 3))
    }
    Some(Case(path.getFileName.toString.stripSuffix(".txt"), waitKind, starts, cycles))
  }

  private def cases: Option[Seq[Case]] =
    sys.env.get(Env).map { dir =>
      val files = Files.list(Paths.get(dir)).iterator.asScala.filter(_.toString.endsWith(".txt")).toSeq.sorted
      files.flatMap(parse).filter(_.waitKind == "lockstep")
    }

  behavior of "ElementwiseTiledSpatial against SnaxUnit's lockstep cases"

  it should "fire, push and be busy in the cycles every lockstep case gives" in {
    val all = cases.getOrElse(cancel(s"$Env is not set (snax-gvsoc's tests/snax/unit/cases)"))
    assert(all.nonEmpty, s"no lockstep case in ${sys.env(Env)}")
    for (c <- all) {
      require(c.cycles.forall(_.valids.size == 2), s"${c.name}: ElementwiseTiledSpatial has two inputs")
      test(new ElementwiseTiledSpatial(dataWidth = 64, lanes = 4, supportedOps = Seq(ElementwiseOp.Add))) { dut =>
        for ((cyc, i) <- c.cycles.zipWithIndex) {
          val clue = s"${c.name}, cycle $i"
          val busy = dut.io.busy.peek().litToBoolean
          c.starts.get(i) match {
            case Some(Start(n, answer)) =>
              val got = if (busy) "refused" else if (n == 0) "ignored" else "taken"
              assert(got == answer, s"$clue: start $n $got, the case says $answer")
              dut.io.start.poke(true.B)
              dut.io.loopCount.poke(n.U)
            case None                   =>
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
