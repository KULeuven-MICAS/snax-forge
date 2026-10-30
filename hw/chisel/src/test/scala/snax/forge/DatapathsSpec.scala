package snax.forge

import java.nio.file.Files

import chisel3._
import _root_.circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.Datapaths.{BoolParam, DtypeParam, DtypesParam, IntParam, OpParam}

/** The datapath catalogue: `Emit --describe` (the registry, `Datapaths`) equals the checked-in `hw/chisel/blocks.json`,
  * the file the Python side checks `accelerator.json` against (D130). Compared as JSON, key and list order included, so
  * the file may be laid out by hand. Each datapath's builder also gives an IO with exactly the catalogue's ports and
  * pins, and no valid, ready or counter (D147).
  */
class DatapathsSpec extends AnyFlatSpec {

  behavior of "the datapath catalogue"

  it should "equal blocks.json" in {
    val file = RepoPaths.root.resolve("hw/chisel/blocks.json")
    val want = ujson.read(Files.readString(file))
    assert(ujson.write(ujson.read(Emit.describe)) == ujson.write(want))
  }

  /** The least params each type takes: an int's min (8 lanes, whole words of the first type), the first op, `flag` for
    * a bool (and its opposite for `regOut`, so a Mac is accumulating, or plain with its output register), the first
    * type (two of it for a list, three for a plain Mac's a, b and c).
    */
  private def least(d: Datapaths.Datapath, flag: Boolean): Map[String, ujson.Value] = {
    val plain = d.name == "Mac" && !flag
    d.params.map {
      case (n, IntParam(min), _) => n -> ujson.Num(min.max(if (n == "lanes") 8 else min))
      case (n, OpParam(vs), _)   => n -> ujson.Str(vs.head)
      case (n, BoolParam, _)     => n -> ujson.Bool(if (n == "regOut") !flag else flag)
      case (n, DtypeParam, _)    => n -> ujson.Str(IntType.names.head)
      case (n, DtypesParam, _)   => n -> ujson.Arr.from(Seq.fill(if (plain) 3 else 2)(IntType.names.head))
    }.toMap
  }

  it should "name every live port and pin of each datapath's IO, and nothing else" in {
    for {
      d    <- Datapaths.all
      flag <- Seq(false, true)
    } {
      val p     = least(d, flag)
      var names = Set.empty[String]
      ChiselStage.emitCHIRRTL(new Module {
        val built = d.build(p)
        names = built.elements.keySet.toSet
        built := DontCare
      })
      val want  = (d.ports.filter(x => Datapaths.holds(x.when, p)).map(_.name) ++ d.livePins(p).map(_.name)).toSet
      assert(want == names, s"${d.name}: catalogue $want, IO ${names.toSeq.sorted}")
    }
  }
}
