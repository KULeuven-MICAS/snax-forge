package snax.forge.shell

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{Datapaths, Emit}

/** A block alone (`Emit --block`) is the module it is inside an accelerator (`Emit --spec`): the same SystemVerilog,
  * byte for byte, for every file the block emits.
  *
  * Compared on blocks whose pins the control unit drives: vsum_tree_w8's `TreeAccumulate` (int64, lanes 8, add) and
  * dot_5_w1's `Mac` (int64, lanes 1, accumulating), each at the params `blocks.json`'s fill rule gives its unit. A pin
  * the accelerator ties to a constant is propagated into the module by firtool, so there the two differ by design:
  * dot_5_w1's `SerialAccumulate`, whose `first` is constant at T = 1 (snax-forge-experiments' PITFALLS, DC on
  * SerialAccumulate), is emitted both ways and its difference reported, not checked.
  */
class BlockInShellSpec extends AnyFlatSpec {

  /** A unit of a fixture, the params it is emitted at, and whether the two texts must be equal. */
  case class Case(fixture: String, unit: String, same: Boolean, params: ujson.Obj)

  private val cases = Seq(
    Case(
      "vsum_tree_w8",
      "acc",
      true,
      ujson.Obj("inType" -> "int64", "outType" -> "int64", "lanes" -> 8, "op" -> "add")
    ),
    Case(
      "dot_5_w1",
      "mac",
      true,
      ujson.Obj(
        "inTypes"    -> ujson.Arr("int64", "int64"),
        "opType"     -> "int64",
        "outType"    -> "int64",
        "lanes"      -> 1,
        "accumulate" -> true,
        "regOut"     -> false
      )
    ),
    Case("dot_5_w1", "sum", false, ujson.Obj("inType" -> "int64", "outType" -> "int64", "lanes" -> 1))
  )

  private def fixture(n: String): RtlFixtures.Fixture = RtlFixtures.all.find(_.name == n).get

  /** Each fixture's accelerator, emitted once. */
  private lazy val shells: Map[String, Path] = cases
    .map(_.fixture)
    .distinct
    .map { n =>
      val out = Files.createTempDirectory(s"emit-$n")
      Emit.emitSpec(fixture(n).file("accelerator.json"), out)
      n -> out.resolve("sv")
    }
    .toMap

  private def text(p: Path): Seq[String] = Files.readAllLines(p).asScala.toSeq

  behavior of "a block alone"

  for (c <- cases)
    it should s"be ${if (c.same) "" else "reported against "}${c.fixture}'s ${c.unit} inside its accelerator" in {
      val unit  = fixture(c.fixture).spec.unit(c.unit)
      val d     = Datapaths(unit.datapath)
      val p     = c.params.value.toMap
      assert(d.fill(unit) == p, s"${c.unit}: the fill rule gives ${d.fill(unit)}")
      val out   = Files.createTempDirectory(s"block-${c.unit}")
      val done  = Emit.emitBlock(d, p, out)
      val alone = done.files.filter(_.toString.endsWith(".sv"))
      for (f <- alone) {
        val inside = shells(c.fixture).resolve(f.getFileName)
        assert(Files.isRegularFile(inside), s"${f.getFileName} is not in ${c.fixture}'s sv/")
        val (a, b) = (text(f), text(inside))
        if (c.same) assert(a == b, s"${f.getFileName}:\n${diff(a, b)}")
        else info(s"${f.getFileName}, alone (-) and inside ${c.fixture} (+):\n${diff(a, b)}")
      }
    }

  /** The lines only in `a` (`-`) and only in `b` (`+`), each in its file's order. */
  private def diff(a: Seq[String], b: Seq[String]): String =
    (a.diff(b).map("- " + _) ++ b.diff(a).map("+ " + _)).mkString("\n")
}
