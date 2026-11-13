package snax.forge

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.shell.RtlFixtures

/** The estimate's control-unit registers against the emitted RTL: every RTL fixture through `Emit --spec` once, and per
  * `<tag>_<group>_ctrl` module the registers its SystemVerilog declares (firtool's `reg [n:m] name;` lines, a Vec
  * register one per element, as `util/ctrl_regs.py` reads them) equal, by name and bits, to that module's entry of
  * `ctrl_registers.json` beside the fixtures: `snax_forge/estimate/bits.py`'s `ctrl_registers` on each fixture's
  * `accelerator.json`, which `pixi run python util/ctrl_regs.py --json` writes and `tests/estimate/test_bits.py` checks
  * is current. Two registers firtool names its own way: the registered done pulse `io_done_REG`, the estimate's `done`,
  * and a unit's valid bits, a Vec of one-bit registers `<u>_v_0` … `<u>_v_<L-1>`, summed into the estimate's `<u>_v`.
  */
class CtrlRegistersSpec extends AnyFlatSpec {

  type Registers = Map[String, Int]

  private val Json = RtlFixtures.dir.resolve("ctrl_registers.json")

  /** Per fixture, per control unit, the estimate's registers. */
  private lazy val estimate: Map[String, Map[String, Registers]] =
    ujson.read(Files.readString(Json)).obj.toMap.map { case (f, mods) =>
      f -> mods.obj.toMap.map { case (m, regs) =>
        m -> regs.arr.map(r => r(0).str -> r(1).num.toInt).toMap
      }
    }

  private val Module = raw"(?ms)^module\s+(\w+)\s*[(;#].*?^endmodule".r
  private val Reg    = raw"(?m)^\s*reg\s*(?:\[(\d+):(\d+)\])?\s*(\w+)\s*(?:\[(\d+):(\d+)\])?\s*;".r
  private val Valid  = raw"(\w+_v)_\d+".r

  /** The estimate's name for a register firtool declares. */
  private def named(reg: String): String =
    reg match {
      case "io_done_REG" => "done"
      case Valid(v)      => v
      case other         => other
    }

  /** Per `*_ctrl` module of the SystemVerilog in `sv/` that declares a register, its registers by the estimate's name.
    */
  def emitted(sv: Path): Map[String, Registers] =
    Files
      .list(sv)
      .iterator()
      .asScala
      .filter(_.toString.endsWith(".sv"))
      .flatMap(f => Module.findAllMatchIn(Files.readString(f)))
      .filter(_.group(1).endsWith("_ctrl"))
      .map { m =>
        val regs = Reg.findAllMatchIn(m.group(0)).toSeq.map { r =>
          val width = if (r.group(1) == null) 1 else (r.group(1).toInt - r.group(2).toInt).abs + 1
          val depth = if (r.group(4) == null) 1 else (r.group(4).toInt - r.group(5).toInt).abs + 1
          named(r.group(3)) -> width * depth
        }
        m.group(1) -> regs.groupMapReduce(_._1)(_._2)(_ + _)
      }
      .filter(_._2.nonEmpty) // <tag>_ctrl holds the groups' control units only
      .toMap

  behavior of "estimate/bits.py's ctrl_registers"

  it should "list every RTL fixture in ctrl_registers.json" in
    assert(estimate.keySet == RtlFixtures.all.map(_.name).toSet)

  for (f <- RtlFixtures.all)
    it should s"be the registers of ${f.name}'s emitted control units" in {
      val out  = Files.createTempDirectory(s"ctrl-regs-${f.name}")
      val done = Emit.emitSpec(f.file("accelerator.json"), out)
      val got  = emitted(out.resolve("sv"))
      val want = estimate(f.name)
      assert(got.keySet == want.keySet, s"${done.top}: the emitted control units")
      for ((m, regs) <- want)
        assert(
          got(m) == regs,
          s"$m\nonly the SV: ${(got(m).toSet -- regs).toSeq.sorted}\nonly the estimate: ${(regs.toSet -- got(m)).toSeq.sorted}"
        )
    }
}
